# Copyright (c) 2025 ByteDance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import AutoModel, AutoModelForCausalLM
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutput, CausalLMOutput
from transformers.modeling_utils import PreTrainedModel
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.utils import logging

from .configuration_illada import ILLaDAConfig

try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
except ImportError:
    flash_attn_func = None
    flash_attn_varlen_func = None


logger = logging.get_logger(__name__)


class ILLaDARMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


ALL_LAYERNORM_LAYERS.append(ILLaDARMSNorm)


class ILLaDARotaryEmbedding(nn.Module):
    def __init__(self, config: ILLaDAConfig):
        super().__init__()
        rope_scaling = config.rope_scaling or {}
        rope_type = rope_scaling.get("rope_type", rope_scaling.get("type", "default"))
        factor = rope_scaling.get("factor", 1.0)
        if rope_type != "default" or factor != 1.0:
            raise ValueError("This iLLaDA checkpoint expects default RoPE without scaling.")

        head_dim = config.hidden_size // config.num_attention_heads
        inv_freq = 1.0 / (
            config.rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    @torch.no_grad()
    def forward(self, x, position_ids):
        inv_freq = self.inv_freq[None, :, None].to(device=x.device)
        position_ids = position_ids[:, None, :].to(dtype=torch.float32, device=x.device)
        device_type = x.device.type if isinstance(x.device.type, str) and x.device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq.float() @ position_ids.float()).transpose(1, 2)
            emb = torch.cat((freqs, freqs), dim=-1)
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class ILLaDAMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=config.mlp_bias)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=config.mlp_bias)
        self.act_fn = ACT2FN[config.hidden_act]
        self.dropout = nn.Dropout(config.resid_pdrop)

    def forward(self, x):
        return self.dropout(self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x)))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


def _prepare_4d_attention_mask(attention_mask, dtype, device):
    if attention_mask is None:
        return None

    attention_mask = attention_mask.to(device=device)
    min_dtype = torch.finfo(dtype).min

    if attention_mask.dim() == 2:
        allowed = attention_mask.to(torch.bool)[:, None, None, :]
        additive_mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
        return additive_mask.masked_fill(~allowed, min_dtype)

    if attention_mask.dim() == 3:
        attention_mask = attention_mask[:, None, :, :]

    if attention_mask.dim() != 4:
        raise ValueError("attention_mask must have shape (batch, seq), (batch, q, k), or (batch, 1, q, k)")

    if attention_mask.dtype == torch.bool:
        additive_mask = torch.zeros(attention_mask.shape, dtype=dtype, device=device)
        return additive_mask.masked_fill(~attention_mask, min_dtype)

    return attention_mask.to(dtype=dtype)


class ILLaDAAttention(nn.Module):
    def __init__(self, config: ILLaDAConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=config.attention_bias)
        self.k_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.v_proj = nn.Linear(
            self.hidden_size,
            self.num_key_value_heads * self.head_dim,
            bias=config.attention_bias,
        )
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=config.attention_bias)
        self.resid_dropout = nn.Dropout(config.resid_pdrop)

    def _project_qkv(self, hidden_states, position_embeddings):
        batch_size, seq_len, _ = hidden_states.size()
        query_states = self.q_proj(hidden_states)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        key_states = key_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)
        value_states = value_states.view(batch_size, seq_len, self.num_key_value_heads, self.head_dim).transpose(1, 2)

        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        return query_states, key_states, value_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        batch_size, seq_len, _ = hidden_states.size()
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)

        if cu_seqlens is not None:
            if flash_attn_varlen_func is None:
                raise RuntimeError(
                    "Packed iLLaDA training requires flash-attn with flash_attn_varlen_func."
                )
            if batch_size != 1:
                raise ValueError(
                    f"Packed iLLaDA input must have batch_size=1, got {batch_size}."
                )
            if max_seqlen is None:
                raise ValueError("max_seqlen is required when cu_seqlens is provided.")
            query_flat = query_states.squeeze(0).transpose(0, 1).contiguous()
            key_flat = key_states.squeeze(0).transpose(0, 1).contiguous()
            value_flat = value_states.squeeze(0).transpose(0, 1).contiguous()
            attn_output = flash_attn_varlen_func(
                query_flat,
                key_flat,
                value_flat,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
                dropout_p=self.attention_dropout if self.training else 0.0,
                causal=False,
            )
            attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
            attn_output = self.resid_dropout(self.o_proj(attn_output))
            return attn_output, None

        attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask[:, :, :, : key_states.shape[-2]]

        attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query_states.dtype)
        attn_weights = F.dropout(attn_weights, p=self.attention_dropout, training=self.training)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (batch_size, self.num_heads, seq_len, self.head_dim):
            raise ValueError(
                f"attn_output should have shape {(batch_size, self.num_heads, seq_len, self.head_dim)}, "
                f"got {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous().reshape(batch_size, seq_len, -1)
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, attn_weights if output_attentions else None


class ILLaDASdpaAttention(ILLaDAAttention):
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        if cu_seqlens is not None:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=False,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        if output_attentions:
            logger.warning_once(
                "torch SDPA does not return attention weights; falling back to the eager attention implementation."
            )
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        batch_size, seq_len, _ = hidden_states.size()
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)

        if query_states.device.type == "cuda" and attention_mask is not None:
            query_states = query_states.contiguous()
            key_states = key_states.contiguous()
            value_states = value_states.contiguous()

        attn_output = F.scaled_dot_product_attention(
            query_states,
            key_states,
            value_states,
            attn_mask=attention_mask,
            dropout_p=self.attention_dropout if self.training else 0.0,
            is_causal=False,
        )
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, -1)
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, None


class ILLaDAFlashAttention2(ILLaDASdpaAttention):
    """Match LLaDA's full-attention FlashAttention dispatch and masked SDPA fallback."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: bool = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # The parent already handles packed FlashAttention, masked SDPA and
        # eager attention when the caller requests attention weights.
        if cu_seqlens is not None or attention_mask is not None or output_attentions:
            return super().forward(
                hidden_states=hidden_states,
                attention_mask=attention_mask,
                output_attentions=output_attentions,
                position_embeddings=position_embeddings,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        if flash_attn_func is None:
            raise RuntimeError("flash_attention_2 requires flash-attn with flash_attn_func.")

        batch_size, seq_len, _ = hidden_states.size()
        query_states, key_states, value_states = self._project_qkv(hidden_states, position_embeddings)
        attn_output = flash_attn_func(
            query_states.transpose(1, 2),
            key_states.transpose(1, 2),
            value_states.transpose(1, 2),
            dropout_p=self.attention_dropout if self.training else 0.0,
            causal=False,
        )
        attn_output = attn_output.reshape(batch_size, seq_len, self.hidden_size)
        attn_output = self.resid_dropout(self.o_proj(attn_output))
        return attn_output, None


ILLADA_ATTENTION_CLASSES = {
    "eager": ILLaDAAttention,
    "sdpa": ILLaDASdpaAttention,
    "flash_attention_2": ILLaDAFlashAttention2,
}


class ILLaDADecoderLayer(nn.Module):
    def __init__(self, config: ILLaDAConfig, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        attn_impl = getattr(config, "_attn_implementation", "sdpa")
        if attn_impl not in ILLADA_ATTENTION_CLASSES:
            attn_impl = "eager"
        self.self_attn = ILLADA_ATTENTION_CLASSES[attn_impl](config=config, layer_idx=layer_idx)
        self.mlp = ILLaDAMLP(config)
        if config.layer_norm_eps is not None:
            self.input_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
            self.post_attention_layernorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.input_layernorm = ILLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
            self.post_attention_layernorm = ILLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = False,
        position_embeddings: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
    ) -> Tuple[torch.FloatTensor, Optional[torch.FloatTensor]]:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, self_attn_weights = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            position_embeddings=position_embeddings,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        outputs = (hidden_states,)
        if output_attentions:
            outputs += (self_attn_weights,)
        return outputs


class ILLaDAPreTrainedModel(PreTrainedModel):
    config_class = ILLaDAConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = ["ILLaDADecoderLayer"]
    _supports_sdpa = True
    _supports_flash_attn_2 = True

    def post_init(self):
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and any(substring in name for substring in ["o_proj", "down_proj"]):
                module.std = self.config.initializer_range / math.sqrt(2 * self.config.num_hidden_layers)
        super().post_init()

    def _init_weights(self, module):
        std = getattr(module, "std", self.config.initializer_range)
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            std = math.sqrt(1 / (2 * self.config.hidden_size))
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class ILLaDAModel(ILLaDAPreTrainedModel):
    def __init__(self, config: ILLaDAConfig):
        super().__init__(config)
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.vocab_size
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [ILLaDADecoderLayer(config, layer_idx) for layer_idx in range(config.num_hidden_layers)]
        )
        if config.layer_norm_eps is not None:
            self.norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        else:
            self.norm = ILLaDARMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = ILLaDARotaryEmbedding(config=config)
        self.gradient_checkpointing = False
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        **kwargs,
    ) -> Union[Tuple, BaseModelOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if input_ids is not None and input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        elif inputs_embeds.dim() == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)

        batch_size, seq_len, _ = inputs_embeds.shape
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

        # Like LLaDA, an all-valid 2D padding mask needs no attention bias.
        # Preserve real padding and arbitrary attention_bias for the SDPA fallback.
        if (
            self.config._attn_implementation == "flash_attention_2"
            and attention_mask is not None
            and attention_mask.dim() == 2
            and bool(attention_mask.bool().all())
        ):
            attention_mask = None

        prepared_attention_mask = _prepare_4d_attention_mask(
            attention_mask,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        prepared_attention_bias = _prepare_4d_attention_mask(
            attention_bias,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        if prepared_attention_mask is None:
            prepared_attention_mask = prepared_attention_bias
        elif prepared_attention_bias is not None:
            prepared_attention_mask = prepared_attention_mask + prepared_attention_bias

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        all_hidden_states = () if output_hidden_states else None
        all_self_attns = () if output_attentions else None

        for decoder_layer in self.layers:
            if output_hidden_states:
                all_hidden_states += (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    decoder_layer.__call__,
                    hidden_states,
                    prepared_attention_mask,
                    output_attentions,
                    position_embeddings,
                    cu_seqlens,
                    max_seqlen,
                )
            else:
                layer_outputs = decoder_layer(
                    hidden_states,
                    attention_mask=prepared_attention_mask,
                    output_attentions=output_attentions,
                    position_embeddings=position_embeddings,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
                )

            hidden_states = layer_outputs[0]
            if output_attentions:
                all_self_attns += (layer_outputs[1],)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_self_attns] if v is not None)

        return BaseModelOutput(
            last_hidden_state=hidden_states,
            hidden_states=all_hidden_states,
            attentions=all_self_attns,
        )


class ILLaDAForCausalLM(ILLaDAPreTrainedModel):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = ILLaDAModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.loss_fct = CrossEntropyLoss()
        self.post_init()

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def set_input_embeddings(self, value):
        self.model.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    def set_decoder(self, decoder):
        self.model = decoder

    def get_decoder(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        attention_bias: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        orig_keep_mask: Optional[torch.BoolTensor] = None,
        cu_seqlens: Optional[torch.Tensor] = None,
        max_seqlen: Optional[int] = None,
        use_cache: Optional[bool] = None,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutput]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            attention_bias=attention_bias,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        hidden_states = outputs[0]
        sparse_ce = labels is not None and orig_keep_mask is not None
        if sparse_ce:
            if labels.shape != orig_keep_mask.shape:
                raise ValueError(
                    f"labels shape {labels.shape} must match orig_keep_mask shape {orig_keep_mask.shape}."
                )
            hidden_states_sparse = hidden_states[orig_keep_mask]
            logits_sparse = self.lm_head(hidden_states_sparse)
            sparse_labels = labels[orig_keep_mask].clamp(min=0).to(logits_sparse.device)
            sparse_log_probs = -F.cross_entropy(
                logits_sparse.float(),
                sparse_labels,
                reduction="none",
            )
            logits = torch.zeros(
                orig_keep_mask.shape,
                dtype=sparse_log_probs.dtype,
                device=hidden_states.device,
            )
            logits[orig_keep_mask] = sparse_log_probs
        elif isinstance(logits_to_keep, int) and logits_to_keep > 0:
            hidden_states_for_logits = hidden_states[:, -logits_to_keep:, :]
            logits = self.lm_head(hidden_states_for_logits)
        elif isinstance(logits_to_keep, torch.Tensor):
            if logits_to_keep.dtype == torch.bool:
                if logits_to_keep.shape != hidden_states.shape[:2]:
                    raise ValueError(
                        f"Boolean logits_to_keep shape {logits_to_keep.shape} must match "
                        f"hidden-state prefix shape {hidden_states.shape[:2]}."
                    )
                hidden_states_sparse = hidden_states[logits_to_keep]
                logits_sparse = self.lm_head(hidden_states_sparse)
                logits = torch.zeros(
                    *hidden_states.shape[:2],
                    self.vocab_size,
                    dtype=logits_sparse.dtype,
                    device=hidden_states.device,
                )
                logits[logits_to_keep] = logits_sparse
            else:
                hidden_states_for_logits = hidden_states[:, logits_to_keep, :]
                logits = self.lm_head(hidden_states_for_logits)
        else:
            logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None and not sparse_ce:
            logits = logits.float()
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous().to(shift_logits.device)
            loss = self.loss_fct(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1))

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

AutoModel.register(ILLaDAConfig, ILLaDAForCausalLM)
AutoModelForCausalLM.register(ILLaDAConfig, ILLaDAForCausalLM)
