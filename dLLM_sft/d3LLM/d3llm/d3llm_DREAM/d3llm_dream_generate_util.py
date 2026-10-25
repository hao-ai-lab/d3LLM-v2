# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream

import warnings
import copy
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.distributions as dists
from torch.nn import functional as F
from transformers import __version__
from transformers.generation.configuration_utils import (
    GenerationConfig
)
from transformers.cache_utils import DynamicCache
from transformers.utils import (
    ModelOutput,
    is_torchdynamo_compiling,
    logging,
)

logger = logging.get_logger(__name__)

def create_full_block_attention_mask(prompt_length, max_length, block_size, device=None, dtype=None):
    """
    Creates a complete attention mask for the entire sequence with block-based causal attention.
    
    Args:
        prompt_length: Length of the prompt (first irregular block)
        max_length: Maximum total sequence length
        block_size: Size of each regular block
        device: Device to create tensor on
        dtype: Data type for the attention mask
        
    Returns:
        attention_mask: Tensor of shape [1, 1, max_length, max_length]
    """
    # Use the provided dtype or default to bfloat16
    if dtype is None:
        dtype = torch.bfloat16
    
    # Initialize mask with -inf (no attention)
    attention_mask = torch.full((1, 1, max_length, max_length), -torch.inf, device=device, dtype=dtype)
    
    # Block 0: Prompt (can see itself)
    attention_mask[:, :, :prompt_length, :prompt_length] = 0
    
    # Calculate the number of regular blocks after prompt
    remaining_length = max_length - prompt_length
    num_blocks = (remaining_length + block_size - 1) // block_size
    
    # Process each regular block
    for b in range(num_blocks):
        block_start = prompt_length + b * block_size
        block_end = min(prompt_length + (b + 1) * block_size, max_length)
        
        # Current block can see the prompt
        attention_mask[:, :, block_start:block_end, :prompt_length] = 0
        
        # Current block can see all previous regular blocks
        for prev_b in range(b):
            prev_start = prompt_length + prev_b * block_size
            prev_end = min(prompt_length + (prev_b + 1) * block_size, max_length)
            attention_mask[:, :, block_start:block_end, prev_start:prev_end] = 0
        
        # Current block can see itself (full attention within block)
        attention_mask[:, :, block_start:block_end, block_start:block_end] = 0
    
    return attention_mask

def extract_attention_mask(full_mask, start_pos, input_length, cache_length):
    """
    Extract the relevant portion of attention mask for current forward pass.
    
    Args:
        full_mask: Complete attention mask [1, 1, max_length, max_length]
        start_pos: Starting position in the full sequence
        input_length: Length of current input sequence
        cache_length: Length of cached sequence
        
    Returns:
        attention_mask: Extracted mask [1, 1, input_length, cache_length + input_length]
    """
    end_pos = start_pos + input_length
    total_length = cache_length + input_length
    
    # Extract the relevant rows (current input positions)
    # and columns (cache + current input positions)
    extracted_mask = torch.full((1, 1, input_length, total_length), -torch.inf,
                               device=full_mask.device, dtype=full_mask.dtype)
    
    # Copy cache columns (0 to cache_length in the extracted mask corresponds to 0 to cache_length in full mask)
    extracted_mask[:, :, :, :cache_length] = full_mask[:, :, start_pos:end_pos, :cache_length]
    
    # Copy current input columns
    extracted_mask[:, :, :, cache_length:] = full_mask[:, :, start_pos:end_pos, start_pos:end_pos]
    
    return extracted_mask

def get_num_transfer_tokens(mask_index, steps):
    '''
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    '''
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = torch.zeros(mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64) + base

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, :remainder[i]] += 1

    return num_transfer_tokens


def top_p_logits(logits, top_p=None):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the indices to the right to keep the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    mask = torch.zeros_like(logits, dtype=torch.bool, device=logits.device)
    mask = mask.scatter_(-1, sorted_indices, sorted_indices_to_remove)
    logits = logits.masked_fill(mask, torch.finfo(logits.dtype).min)
    return logits

def top_k_logits(logits, top_k=None):
    top_k = min(top_k, logits.size(-1))  # Safety check
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
    logits = logits.masked_fill(indices_to_remove, torch.finfo(logits.dtype).min)
    return logits


def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    probs = torch.softmax(logits, dim=-1)

    if temperature > 0:
        try:
            x0 = dists.Categorical(probs=probs).sample()
            initial_confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
        except:
            initial_confidence, x0 = probs.max(dim=-1)
    else:
        initial_confidence, x0 = probs.max(dim=-1)
    
    # Save initial confidence
    confidence = initial_confidence.clone()
    
    if margin_confidence:
        sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
        # Extract top1 and top2 probabilities
        top1_probs = sorted_probs[:, 0]
        top2_probs = sorted_probs[:, 1]
        # Calculate confidence as top1 - top2
        confidence = top1_probs - top2_probs
    
    if neg_entropy:
        epsilon = 1e-10
        log_probs = torch.log(probs + epsilon)
        confidence = torch.sum(probs * log_probs, dim=-1)
    
    return confidence, x0, initial_confidence


# def sample_tokens(logits, temperature=0.0, top_p=None, top_k=None, margin_confidence=False, neg_entropy=False):

#     if temperature > 0:
#         logits = logits / temperature
#     if top_p is not None and top_p < 1:
#         logits = top_p_logits(logits, top_p)
#     if top_k is not None:
#         logits = top_k_logits(logits, top_k)
#     probs = torch.softmax(logits, dim=-1)

#     if temperature > 0:
#         try:
#             x0 = dists.Categorical(probs=probs).sample()
#             confidence = torch.gather(probs, -1, x0.unsqueeze(-1)).squeeze(-1)
#         except:
#             confidence, x0 = probs.max(dim=-1)
#     else:
#         confidence, x0 = probs.max(dim=-1)
    
#     if margin_confidence:
#         sorted_probs, _ = torch.sort(probs, dim=-1, descending=True)
#         # Extract top1 and top2 probabilities
#         top1_probs = sorted_probs[:, 0]
#         top2_probs = sorted_probs[:, 1]
#         # Calculate confidence as top1 - top2
#         confidence = top1_probs - top2_probs
    
#     if neg_entropy:
#         epsilon = 1e-10
#         log_probs = torch.log(probs + epsilon)
#         confidence = torch.sum(probs * log_probs, dim=-1)
    
#     return confidence, x0

def sample_tokens_with_entropy(logits, temperature=1.0):
    original_probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(original_probs + 1e-8)
    entropy = -torch.sum(original_probs * log_probs, dim=-1)
    
    if temperature == 0:
        samples = torch.argmax(logits, dim=-1)
    else:
        scaled_logits = logits / temperature
        # Convert to probabilities and sample
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return entropy, samples


def handle_early_stop(x, block_states, eos_token_id, prompt_length, mask_token_id=None, debug=False):
    """
    Early stop: find first EOS, set all tokens after it to EOS (not mask), mark subsequent blocks as complete.
    Returns: (has_eos, first_eos_abs_pos)
    """
    if eos_token_id is None:
        return False, None
    
    # Only check non-mask tokens in generation region
    gen_region = x[:, prompt_length:]
    eos_mask = (gen_region == eos_token_id)
    
    if not eos_mask.any():
        return False, None
    
    # Find first EOS position (only among decoded, non-mask tokens)
    pos = torch.arange(gen_region.shape[1], device=x.device).unsqueeze(0)
    first_eos_rel = torch.where(eos_mask, pos, gen_region.shape[1]).amin(dim=1)
    first_eos_abs = prompt_length + first_eos_rel[0].item()
    
    if debug:
        print(f"[EarlyStop] Found first EOS at position {first_eos_abs} (relative: {first_eos_rel[0].item()})")
        # Count non-mask tokens before EOS
        if mask_token_id is not None:
            decoded_before_eos = ((x[:, prompt_length:first_eos_abs+1] != mask_token_id).sum().item())
            print(f"[EarlyStop] Decoded tokens before (and including) EOS: {decoded_before_eos}")
    
    # Set all tokens after first EOS to EOS (NOT mask!)
    x[:, first_eos_abs+1:] = eos_token_id
    
    # Update block states based on EOS position
    blocks_marked = 0
    blocks_updated = 0
    for bid in sorted(block_states.keys()):
        if bid > 0:
            start, end = block_states[bid]["start"], block_states[bid]["end"]
            
            if start > first_eos_abs:
                # Block entirely after EOS - mark as complete with no masks
                if block_states[bid]["mask_count"] != 0:
                    block_states[bid]["mask_count"] = 0
                    block_states[bid]["is_complete"] = True
                    blocks_marked += 1
            elif end > first_eos_abs and start <= first_eos_abs:
                # Block CONTAINS EOS - recalculate mask_count for portion before EOS
                if mask_token_id is not None:
                    # Only count masks from block start to EOS position (inclusive)
                    masks_before_eos = (x[:, start:first_eos_abs+1] == mask_token_id).sum().item()
                    old_count = block_states[bid]["mask_count"]
                    if masks_before_eos != old_count:
                        block_states[bid]["mask_count"] = masks_before_eos
                        blocks_updated += 1
                        if masks_before_eos == 0:
                            block_states[bid]["is_complete"] = True
    
    if debug:
        if blocks_marked > 0:
            print(f"[EarlyStop] Marked {blocks_marked} blocks after EOS as complete")
        if blocks_updated > 0:
            print(f"[EarlyStop] Updated {blocks_updated} blocks containing EOS")
    
    return True, first_eos_abs


@dataclass
class DreamModelOutput(ModelOutput):
    sequences: torch.LongTensor = None
    history: Optional[Tuple[torch.FloatTensor]] = None


class DreamGenerationConfig(GenerationConfig):
    def __init__(self, **kwargs):
        self.temperature: float = kwargs.pop("temperature", 0.0)
        self.top_p: Optional[float] = kwargs.pop("top_p", None)
        self.top_k: Optional[int] = kwargs.pop("top_k", None)
        self.max_length = kwargs.pop("max_length", 20)
        self.max_new_tokens = kwargs.pop("max_new_tokens", None)
        # diffusion specific params
        self.eps: float = kwargs.pop("eps", 1e-3)
        self.steps: int = kwargs.pop("steps", 512)
        self.alg: str = kwargs.pop("alg", 'origin')
        self.alg_temp: Optional[float] = kwargs.pop("alg_temp", None)

        # Parameters that define the output variables of `generate`
        self.num_return_sequences: int = kwargs.pop("num_return_sequences", 1)
        self.return_dict_in_generate: bool = kwargs.pop("return_dict_in_generate", False)
        self.output_history: bool = kwargs.pop("output_history", False)

        # Special tokens that can be used at generation time
        self.mask_token_id = kwargs.pop("mask_token_id", None)
        self.pad_token_id = kwargs.pop("pad_token_id", None)
        self.bos_token_id = kwargs.pop("bos_token_id", None)
        self.eos_token_id = kwargs.pop("eos_token_id", None)

        # Wild card
        self.generation_kwargs = kwargs.pop("generation_kwargs", {})

        # The remaining attributes do not parametrize `.generate()`, but are informative and/or used by the hub
        # interface.
        self._from_model_config = kwargs.pop("_from_model_config", False)
        self._commit_hash = kwargs.pop("_commit_hash", None)
        self.transformers_version = kwargs.pop("transformers_version", __version__)

        # Additional attributes without default values
        if not self._from_model_config:
            # we don't want to copy values from the model config if we're initializing a `GenerationConfig` from a
            # model's default configuration file
            for key, value in kwargs.items():
                try:
                    setattr(self, key, value)
                except AttributeError as err:
                    logger.error(f"Can't set {key} with value {value} for {self}")
                    raise err

        # Validate the values of the attributes
        self.validate(is_init=True)

    def validate(self, is_init=False):
        pass

class DreamGenerationMixin:
    @staticmethod
    def _expand_inputs_for_generation(
        expand_size: int = 1,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None
    ) -> Tuple[torch.LongTensor, Dict[str, Any]]:
        """Expands tensors from [batch_size, ...] to [batch_size * expand_size, ...]"""
        # Do not call torch.repeat_interleave if expand_size is 1 because it clones
        # the input tensor and thus requires more memory although no change is applied
        if expand_size == 1:
            return input_ids, attention_mask
        if input_ids is not None:
            input_ids = input_ids.repeat_interleave(expand_size, dim=0)
        if attention_mask is not None:
            attention_mask = attention_mask.repeat_interleave(expand_size, dim=0)
        return input_ids, attention_mask

    def _validate_generated_length(self, generation_config, input_ids_length, has_default_max_length):
        """Performs validation related to the resulting generated length"""

        # Can't throw warnings/exceptions during compilation
        if is_torchdynamo_compiling():
            return

        # 1. Max length warnings related to poor parameterization
        if has_default_max_length and generation_config.max_new_tokens is None and generation_config.max_length == 20:
            # 20 is the default max_length of the generation config
            warnings.warn(
                f"Using the model-agnostic default `max_length` (={generation_config.max_length}) to control the "
                "generation length. We recommend setting `max_new_tokens` to control the maximum length of the "
                "generation.",
                UserWarning,
            )
        if input_ids_length >= generation_config.max_length:
            input_ids_string = "input_ids"
            raise ValueError(
                f"Input length of {input_ids_string} is {input_ids_length}, but `max_length` is set to"
                f" {generation_config.max_length}. This can lead to unexpected behavior. You should consider"
                " increasing `max_length` or, better yet, setting `max_new_tokens`."
            )

    def _prepare_generated_length(
        self,
        generation_config,
        has_default_max_length,
        input_ids_length,
    ):
        """Prepared max and min length in generation configs to avoid clashes between similar attributes"""

        if generation_config.max_new_tokens is not None:
            if not has_default_max_length and generation_config.max_length is not None:
                logger.warning(
                    f"Both `max_new_tokens` (={generation_config.max_new_tokens}) and `max_length`(="
                    f"{generation_config.max_length}) seem to have been set. `max_new_tokens` will take precedence. "
                    "Please refer to the documentation for more information. "
                    "(https://huggingface.co/docs/transformers/main/en/main_classes/text_generation)"
                )
            generation_config.max_length = generation_config.max_new_tokens + input_ids_length

        elif has_default_max_length:
            if generation_config.max_length == DreamGenerationConfig().max_length:
                generation_config.max_length = generation_config.max_length + input_ids_length
                max_position_embeddings = getattr(self.config, "max_position_embeddings", None)
                if max_position_embeddings is not None:
                    generation_config.max_length = min(generation_config.max_length, max_position_embeddings)

        return generation_config

    def _prepare_generation_config(
        self, generation_config: Optional[DreamGenerationConfig], **kwargs: Dict
    ) -> DreamGenerationConfig:
        """
        Prepares the base generation config, then applies any generation configuration options from kwargs. This
        function handles retrocompatibility with respect to configuration files.
        """
        # priority: `generation_config` argument > `model.generation_config` (the default generation config)
        using_model_generation_config = False
        if generation_config is None:
            generation_config = DreamGenerationConfig.from_model_config(self.config)
            using_model_generation_config = True

        # `torch.compile` can't compile `copy.deepcopy`, arguments in `kwargs` that are part of `generation_config`
        # will mutate the object with `.update`. As such, passing these arguments through `kwargs` is disabled -- an
        # exception will be raised in `_validate_model_kwargs`
        if not is_torchdynamo_compiling():
            generation_config = copy.deepcopy(generation_config)
            _kwargs = generation_config.update(**kwargs)
            # If `generation_config` is provided, let's fallback ALL special tokens to the default values for the model
            if not using_model_generation_config:
                if generation_config.bos_token_id is None:
                    generation_config.bos_token_id = self.generation_config.bos_token_id
                if generation_config.eos_token_id is None:
                    generation_config.eos_token_id = self.generation_config.eos_token_id
                if generation_config.pad_token_id is None:
                    generation_config.pad_token_id = self.generation_config.pad_token_id
                if generation_config.mask_token_id is None:
                    generation_config.mask_token_id = self.generation_config.mask_token_id

        return generation_config

    def _prepare_special_tokens(
        self,
        generation_config: DreamGenerationConfig,
        device: Optional[Union[torch.device, str]] = None,
    ):
        """
        Prepares the special tokens for generation, overwriting the generation config with their processed versions
        converted to tensor.
        Note that `generation_config` is changed in place and stops being serializable after this method is called.
        That is no problem if called within `generate` (`generation_config` is a local copy that doesn't leave the
        function). However, if called outside `generate`, consider creating a copy of `generation_config` first.
        """

        # Convert special tokens to tensors
        def _tensor_or_none(token, device=None):
            if token is None:
                return token

            device = device if device is not None else self.device
            if isinstance(token, torch.Tensor):
                return token.to(device)
            return torch.tensor(token, device=device, dtype=torch.long)

        bos_token_tensor = _tensor_or_none(generation_config.bos_token_id, device=device)
        eos_token_tensor = _tensor_or_none(generation_config.eos_token_id, device=device)
        pad_token_tensor = _tensor_or_none(generation_config.pad_token_id, device=device)
        mask_token_tensor = _tensor_or_none(generation_config.mask_token_id, device=device)

        # We can have more than one eos token. Always treat it as a 1D tensor (when it exists).
        if eos_token_tensor is not None and eos_token_tensor.ndim == 0:
            eos_token_tensor = eos_token_tensor.unsqueeze(0)

        # Set pad token if unset (and there are conditions to do so)
        if pad_token_tensor is None and eos_token_tensor is not None:
            pad_token_tensor = eos_token_tensor[0]
            logger.warning(f"Setting `pad_token_id` to `eos_token_id`:{pad_token_tensor} for open-end generation.")

        # Update generation config with the updated special tokens tensors
        # NOTE: this must be written into a different attribute name than the one holding the original special tokens
        # (in their non-tensor form), in order to enable end-to-end compilation. See
        # https://pytorch.org/docs/stable/torch.compiler_cudagraph_trees.html#limitations
        generation_config._bos_token_tensor = bos_token_tensor
        generation_config._eos_token_tensor = eos_token_tensor
        generation_config._pad_token_tensor = pad_token_tensor
        generation_config._mask_token_tensor = mask_token_tensor

    def _prepare_inputs(self, inputs, generation_config, **kwargs):
        """Common input preparation for all generation methods"""
        generation_config = self._prepare_generation_config(generation_config, **kwargs)
        
        assert inputs is not None
        input_ids = inputs
        device = input_ids.device
        attention_mask = kwargs.pop("attention_mask", None)
        self._prepare_special_tokens(generation_config, device=device)
        
        input_ids_length = input_ids.shape[-1]
        has_default_max_length = kwargs.get("max_length") is None and generation_config.max_length is not None
        generation_config = self._prepare_generated_length(
            generation_config=generation_config,
            has_default_max_length=has_default_max_length,
            input_ids_length=input_ids_length,
        )
        
        self._validate_generated_length(generation_config, input_ids_length, has_default_max_length)
        
        if not is_torchdynamo_compiling() and self.device.type != input_ids.device.type:
            warnings.warn(
                "You are calling .generate() with the `input_ids` being on a device type different"
                f" than your model's device. `input_ids` is on {input_ids.device.type}, whereas the model"
                f" is on {self.device.type}. You may experience unexpected behaviors or slower generation.",
                UserWarning,
            )
        if (
            hasattr(generation_config, "pad_token_id") and
            torch.any(input_ids == generation_config.pad_token_id) and
            attention_mask is None
        ):
            warnings.warn(
                "Padding was detected but no attention mask is passed here. For correct "
                "generation results, please set `attention_mask` when batch-padding inputs.",
                UserWarning,
            )
        
        input_ids, attention_mask = self._expand_inputs_for_generation(
            expand_size=generation_config.num_return_sequences,
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        return input_ids, attention_mask, generation_config

    @torch.no_grad()
    def diffusion_generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        input_ids, attention_mask, generation_config = self._prepare_inputs(inputs, generation_config, **kwargs)
        threshold = kwargs.get("threshold", 0.5)
        block_length = kwargs.get("block_length", 32)
        
        result, nfe = self._sample(
            input_ids,
            attention_mask=attention_mask,
            generation_config=generation_config,
            threshold=threshold,
            block_length=block_length,
        )
        return result, nfe

    @torch.no_grad()
    def generate_multi_block(
        self,
        inputs: Optional[torch.Tensor] = None,
        generation_config: Optional[DreamGenerationConfig] = None,
        **kwargs,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        input_ids, attention_mask, generation_config = self._prepare_inputs(inputs, generation_config, **kwargs)
        threshold = kwargs.get("threshold", 0.9)
        block_size = kwargs.get("block_size", 32)
        block_add_threshold = kwargs.get("block_add_threshold", 0.5)
        decoded_token_threshold = kwargs.get("decoded_token_threshold", 0.5)
        cache_delay_iter = kwargs.get("cache_delay_iter", 10000)    # how many steps to delay before KV-caching, default to 10000 steps (no KV-cache)
        early_stop = kwargs.get("early_stop", False)
        
        if cache_delay_iter >= 10000:  
            # no KV-cache
            result, nfe = self._sample_multi_block(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                threshold=threshold,
                block_size=block_size,
                block_add_threshold=block_add_threshold,
                decoded_token_threshold=decoded_token_threshold,
                early_stop=early_stop,
            )
        else:
            # KV-cache
            result, nfe = self._sample_multi_block_kv_cache(
                input_ids,
                attention_mask=attention_mask,
                generation_config=generation_config,
                threshold=threshold,
                block_size=block_size,
                block_add_threshold=block_add_threshold,
                decoded_token_threshold=decoded_token_threshold,
                cache_delay_iter=cache_delay_iter,
                refresh_interval=kwargs.get("refresh_interval", 10000),
                early_stop=early_stop,
            )
        return result, nfe

    def _sample(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: Optional[float] = 0.5,
        block_length: Optional[int] = 32,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        steps = generation_config.steps
        temperature = generation_config.temperature
        alg = generation_config.alg

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id)
        gen_length = max_length - input_ids.shape[1]
        
   
        # Handle block configuration
        if block_length is None:
            block_length = gen_length  # Default: single block (original behavior)
        
        assert gen_length % block_length == 0, f"gen_length ({gen_length}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length // block_length
        
        assert steps % num_blocks == 0, f"steps ({steps}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Process each block
        i = 0
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
            
            while True:
                i += 1
                mask_index = (x == mask_token_id)

                model_output = self(x, attention_mask, tok_idx)

                mask_index[:, current_block_end:] = 0
                
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                if alg == 'entropy_threshold':
                    mask_logits = logits[mask_index]
                    
                    entropy, x0 = sample_tokens_with_entropy(mask_logits, temperature=temperature)
                    
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id
                    full_entropy = torch.full_like(x, torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_index] = x0.clone()
                    full_entropy[mask_index] = entropy
                    
                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id).sum()
                    
                    selected_entropy, select_index = torch.topk(full_entropy, current_transfer_tokens, largest=False)
                    transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        if selected_entropy[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = True
                        else:
                            transfer_index[0, select_index[0, k]] = False
                    x[transfer_index] = x_[transfer_index].clone()


                if (x[:, current_block_start:current_block_end] == mask_token_id).sum() == 0:
                    break

        
        if return_dict_in_generate:
            return DreamModelOutput(sequences=x,history=histories,), i
        else:
            return x, i



    def _sample_multi_block(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: float = 0.9,
        block_size: int = 32,
        block_add_threshold: float = 0.5,
        decoded_token_threshold: float = 0.5,
        early_stop: bool = False,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """
        Pipelined parallel decoding without cache.
        
        Args:
            block_add_threshold: Add new block when last block progress >= this threshold.
                                Set to 1.0 for fully sequential processing.
            decoded_token_threshold: Block becomes fully activated when previous block progress >= this threshold.
                                    Set to 1.0 for fully sequential processing.
            threshold: Entropy threshold for decoding (lower entropy = higher confidence).
        
        When block_add_threshold=1.0 and decoded_token_threshold=1.0, this method behaves
        identically to standard generation with sequential block processing.
        """
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        temperature = generation_config.temperature
        alg = generation_config.alg
        eos_token_id = generation_config.eos_token_id if early_stop else None
        
        max_new_tokens = max_length - input_ids.shape[1]
        prompt_length = input_ids.shape[1]
        x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)
        
        # Prepare attention mask
        if attention_mask is not None and torch.any(attention_mask == 0.0):
            attention_mask_padded = F.pad(attention_mask, (0, max_new_tokens), value=1.0)
            tok_idx = attention_mask_padded.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask_padded == 0, 1)
            attn_mask_4d = torch.logical_and(
                attention_mask_padded.unsqueeze(1).unsqueeze(-2),
                attention_mask_padded.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attn_mask_4d = "full"
        
        # Track block states: {block_id: {start, end, mask_count, total_masks, is_complete}}
        # Initialize with prompt block
        block_states = {
            0: {
                "start": 0,
                "end": input_ids.shape[1],
                "mask_count": 0,
                "total_masks": input_ids.shape[1],
                "is_complete": True,
            }
        }
        
        # Create first generation block
        num_blocks = max_new_tokens // block_size
        next_block_id = 1
        if next_block_id <= num_blocks:
            block_start = input_ids.shape[1] + (next_block_id - 1) * block_size
            block_end = min(block_start + block_size, input_ids.shape[1] + max_new_tokens)
            # First block should be immediately activated since prompt (block 0) is already complete
            should_activate = 1.0 >= decoded_token_threshold  # prompt progress is always 1.0
            block_states[next_block_id] = {
                "start": block_start,
                "end": block_end,
                "mask_count": block_end - block_start,
                "total_masks": block_end - block_start,
                "is_complete": should_activate,
            }
            next_block_id += 1
        
        nfe = 0
        has_eos = False
        first_eos_abs = None
        
        while True:
            # Check if all blocks are exhausted AND no more blocks to create
            mask_index = x == mask_token_id
            total_masks = mask_index[:, prompt_length:].sum()
            
            if total_masks == 0 and next_block_id > num_blocks:
                break
            
            nfe += 1
            
            # Early stop: handle EOS tokens (check every iteration as EOS position may change)
            if early_stop and eos_token_id is not None:
                debug_this_step = False
                has_eos, current_eos_pos = handle_early_stop(x, block_states, eos_token_id, prompt_length, 
                                                             mask_token_id=mask_token_id, debug=debug_this_step)
                
                # Track EOS position changes
                if has_eos:
                    first_eos_abs = current_eos_pos  # Update to current position
                
                # Recalculate mask_count for all blocks after early_stop modifies the sequence
                if has_eos:
                    # Recalculate actual mask counts based on current sequence state
                    mask_index_updated = (x == mask_token_id)
                    blocks_updated = 0
                    for bid in sorted(block_states.keys()):
                        if bid > 0:
                            start, end = block_states[bid]["start"], block_states[bid]["end"]
                            old_mask_count = block_states[bid]["mask_count"]
                            actual_mask_count = mask_index_updated[:, start:end].sum().item()
                            
                            if actual_mask_count != old_mask_count:
                                block_states[bid]["mask_count"] = actual_mask_count
                                blocks_updated += 1
                                if actual_mask_count == 0:
                                    block_states[bid]["is_complete"] = True
                    
                    # Create all missing blocks after EOS and mark them as complete
                    blocks_created = 0
                    while next_block_id <= num_blocks:
                        block_start = prompt_length + (next_block_id - 1) * block_size
                        block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                        
                        if block_start > first_eos_abs:
                            # Block is entirely after EOS, create it as complete with no masks
                            block_states[next_block_id] = {
                                "start": block_start,
                                "end": block_end,
                                "mask_count": 0,
                                "total_masks": block_end - block_start,
                                "is_complete": True,
                            }
                            blocks_created += 1
                            next_block_id += 1
                        else:
                            break
                    
                    # Recalculate total_masks after updating block states
                    total_masks = mask_index_updated[:, prompt_length:].sum()
                    
                    if total_masks == 0:
                        break
            
            # Update block activation states
            def update_block_activation_states():
                """Update which blocks should be fully activated based on previous block progress."""
                for bid in sorted(block_states.keys()):
                    if bid > 0 and not block_states[bid]["is_complete"]:
                        prev_progress = (
                            1 - block_states[bid - 1]["mask_count"] / block_states[bid - 1]["total_masks"]
                        )
                        if prev_progress >= decoded_token_threshold:
                            block_states[bid]["is_complete"] = True
            
            update_block_activation_states()
            
            # Add new block dynamically based on last block's progress (skip if EOS detected)
            if next_block_id <= num_blocks and not has_eos:
                last_bid = max(block_states.keys())
                if last_bid > 0:  # Not just prompt
                    last_progress = (
                        1 - block_states[last_bid]["mask_count"] / block_states[last_bid]["total_masks"]
                    )
                    # Create next block when:
                    # 1. Last block progress >= block_add_threshold (for parallel processing), OR
                    # 2. Last block is complete (mask_count == 0) for sequential processing
                    should_add_block = (last_progress >= block_add_threshold) or (block_states[last_bid]["mask_count"] == 0)
                    
                    if should_add_block:
                        # Add next block
                        block_start = input_ids.shape[1] + (next_block_id - 1) * block_size
                        block_end = min(block_start + block_size, input_ids.shape[1] + max_new_tokens)
                        if block_end > block_start:
                            # Check how many positions in this block are actually masked
                            actual_mask_count = (x[:, block_start:block_end] == mask_token_id).sum().item()
                            
                            # Determine if this block should be immediately activated
                            prev_bid = next_block_id - 1
                            prev_progress = (
                                1 - block_states[prev_bid]["mask_count"] / block_states[prev_bid]["total_masks"]
                            )
                            should_activate = prev_progress >= decoded_token_threshold
                            
                            block_states[next_block_id] = {
                                "start": block_start,
                                "end": block_end,
                                "mask_count": actual_mask_count,
                                "total_masks": block_end - block_start,
                                "is_complete": should_activate,
                            }
                            next_block_id += 1
            
            # Find the rightmost block that should be processed
            rightmost_active_bid = 0
            for bid in sorted(block_states.keys()):
                if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                    rightmost_active_bid = bid
            
            if rightmost_active_bid == 0:
                break
            
            active_end = block_states[rightmost_active_bid]["end"]
            
            # Always do forward pass on entire sequence (like generate() does)
            model_output = self(x, attn_mask_4d, None)
            logits = model_output.logits
            logits = torch.cat([logits[:, :1], logits[:, :-1]], dim=1)
            
            # Mask out future blocks (positions after active_end)
            mask_index_for_decode = mask_index.clone()
            mask_index_for_decode[:, active_end:] = 0
            
            # Decode with unified logic across all active blocks
            if alg == 'entropy_threshold':
                # Calculate entropy for all positions at once
                p = F.softmax(logits.to(torch.float64), dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
                
                # Sample tokens for all masked positions
                x0 = torch.argmax(logits, dim=-1)
                if temperature > 0:
                    p_temp = F.softmax(logits / temperature, dim=-1)
                    x0 = torch.multinomial(p_temp.view(-1, p_temp.shape[-1]), 1).view(x.shape)
                
                # Create transfer index based on entropy threshold
                transfer_index = (entropy < threshold) & mask_index_for_decode
                
                # For fully activated blocks, ensure at least one token is decoded (guaranteed progress)
                first_fully_activated_bid = None
                for bid in sorted(block_states.keys()):
                    if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                        first_fully_activated_bid = bid
                        break
                
                if first_fully_activated_bid is not None:
                    start, end = block_states[first_fully_activated_bid]["start"], block_states[first_fully_activated_bid]["end"]
                    block_transfer = transfer_index[:, start:end]
                    
                    if not block_transfer.any():
                        # Force decode the lowest entropy token in this fully activated block
                        block_mask = mask_index_for_decode[:, start:end]
                        block_entropy = entropy[:, start:end]
                        block_entropy = torch.where(block_mask, block_entropy, torch.inf)
                        best_idx = block_entropy[0].argmin()
                        transfer_index[0, start + best_idx] = True
                
                # Apply the decoded tokens
                x[transfer_index] = x0[transfer_index]
                
                # Update block states based on which positions were decoded
                for bid in sorted(block_states.keys()):
                    if bid > 0 and block_states[bid]["mask_count"] > 0:
                        start, end = block_states[bid]["start"], block_states[bid]["end"]
                        block_decoded = transfer_index[:, start:end].sum().item()
                        if block_decoded > 0:
                            block_states[bid]["mask_count"] -= block_decoded
            
            if nfe > 10000:
                # # print(f"[DEBUG-MB] Breaking: nfe > 10000")
                break
        
        if return_dict_in_generate:
            return DreamModelOutput(sequences=x), nfe
        return x, nfe
    
    def _sample_multi_block_kv_cache(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config: DreamGenerationConfig,
        threshold: float = 0.9,
        block_size: int = 32,
        block_add_threshold: float = 0.5,
        decoded_token_threshold: float = 0.5,
        cache_delay_iter: int = 10000,
        refresh_interval: int = 10000,
        early_stop: bool = False,
    ) -> Union[DreamModelOutput, torch.LongTensor]:
        """
        Pipelined parallel decoding with Delayed KV-Cache.
        
        Strategy: Wait cache_delay_iter steps after a block is fully decoded before caching it.
        This allows token representations to stabilize in the diffusion process.
        All blocks (including prompt) are delayed by cache_delay_iter before caching.
        """
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id = generation_config.mask_token_id
        temperature = generation_config.temperature
        alg = generation_config.alg
        eos_token_id = generation_config.eos_token_id if early_stop else None
        
        max_new_tokens = max_length - input_ids.shape[1]
        prompt_length = input_ids.shape[1]
        num_blocks = (max_new_tokens + block_size - 1) // block_size
        
        # Initialize sequence with masks
        x = F.pad(input_ids, (0, max_new_tokens), value=mask_token_id)
        
        # Track block states with delayed caching support
        # {block_id: {start, end, mask_count, total_masks, is_complete, completed_at_nfe, is_cached}}
        block_states = {
            0: {
                "start": 0, 
                "end": prompt_length, 
                "mask_count": 0, 
                "total_masks": prompt_length, 
                "is_complete": True,
                "completed_at_nfe": 0,  # Prompt completed at NFE=0, will be cached after delay
                "is_cached": False,
            },
        }
        
        # Create first generation block
        next_block_id = 1
        if next_block_id <= num_blocks:
            block_start = prompt_length + (next_block_id - 1) * block_size
            block_end = min(block_start + block_size, prompt_length + max_new_tokens)
            # First block should be immediately activated since prompt (block 0) is already complete
            should_activate = 1.0 >= decoded_token_threshold  # prompt progress is always 1.0
            block_states[next_block_id] = {
                "start": block_start,
                "end": block_end,
                "mask_count": block_end - block_start,
                "total_masks": block_end - block_start,
                "is_complete": should_activate,
                "completed_at_nfe": None,
                "is_cached": False,
            }
            next_block_id += 1
        
        # Initialize cache and manually track cache length
        past_key_values = None
        manual_cache_length = 0  # Manually track cache length instead of relying on DynamicCache
        nfe = 0
        has_eos = False
        first_eos_abs = None
        last_eos_pos = None  # Track EOS position changes
        
        def update_block_activation_states():
            """Update which blocks should be fully activated based on previous block progress."""
            for bid in sorted(block_states.keys()):
                if bid > 0 and not block_states[bid]["is_complete"]:
                    prev_progress = 1 - block_states[bid - 1]["mask_count"] / block_states[bid - 1]["total_masks"]
                    if prev_progress >= decoded_token_threshold:
                        block_states[bid]["is_complete"] = True
        
        # print(f"\n[DEBUG-KV] Init: prompt_len={prompt_length}, max_new_tokens={max_new_tokens}, block_size={block_size}")
        # print(f"[DEBUG-KV] Params: threshold={threshold}, alg={alg}, temp={temperature}")
        # print(f"[DEBUG-KV] Params: block_add_threshold={block_add_threshold}, decoded_token_threshold={decoded_token_threshold}, cache_delay_iter={cache_delay_iter}")
        # print(f"[DEBUG-KV] Initial block_states: {[(bid, s['is_complete'], s['mask_count']) for bid, s in block_states.items()]}")
        
        while True:
            # Check termination
            mask_index = x == mask_token_id
            total_masks = mask_index[:, prompt_length:].sum()
            
            # print(f"\n[DEBUG-KV] === NFE={nfe} ===")
            # print(f"[DEBUG-KV] total_masks={total_masks}, next_block_id={next_block_id}/{num_blocks+1}")
            
            if total_masks == 0 and next_block_id > num_blocks:
                break
            
            nfe += 1
            
            # Early stop: handle EOS tokens (check every iteration as EOS position may change)
            if early_stop and eos_token_id is not None:
                # debug_this_step = (nfe <= 5 or nfe % 50 == 0)
                debug_this_step = False
                has_eos, current_eos_pos = handle_early_stop(x, block_states, eos_token_id, prompt_length,
                                                            mask_token_id=mask_token_id, debug=debug_this_step)
                
                # Track EOS position changes
                if has_eos:
                    # if last_eos_pos is not None and current_eos_pos != last_eos_pos:
                        # print(f"[EarlyStop-KV] EOS position changed: {last_eos_pos} -> {current_eos_pos} (delta: {current_eos_pos - last_eos_pos})")
                    last_eos_pos = current_eos_pos
                    first_eos_abs = current_eos_pos  # Update to current position
                
                # Recalculate mask_count for all blocks after early_stop modifies the sequence
                if has_eos:
                    # Recalculate actual mask counts based on current sequence state
                    mask_index_updated = (x == mask_token_id)
                    blocks_updated = 0
                    for bid in sorted(block_states.keys()):
                        if bid > 0:
                            start, end = block_states[bid]["start"], block_states[bid]["end"]
                            old_mask_count = block_states[bid]["mask_count"]
                            actual_mask_count = mask_index_updated[:, start:end].sum().item()
                            
                            if actual_mask_count != old_mask_count:
                                block_states[bid]["mask_count"] = actual_mask_count
                                blocks_updated += 1
                                if actual_mask_count == 0:
                                    block_states[bid]["is_complete"] = True
                            
                            # Mark blocks after EOS as cached to skip them
                            if block_states[bid]["start"] > first_eos_abs:
                                block_states[bid]["is_cached"] = True
                                if block_states[bid]["completed_at_nfe"] is None:
                                    block_states[bid]["completed_at_nfe"] = nfe
                    
                    # Create all missing blocks after EOS and mark them as complete
                    blocks_created = 0
                    while next_block_id <= num_blocks:
                        block_start = prompt_length + (next_block_id - 1) * block_size
                        block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                        
                        if block_start > first_eos_abs:
                            # Block is entirely after EOS, create it as complete with no masks
                            block_states[next_block_id] = {
                                "start": block_start,
                                "end": block_end,
                                "mask_count": 0,
                                "total_masks": block_end - block_start,
                                "is_complete": True,
                                "completed_at_nfe": nfe,
                                "is_cached": True,  # Mark as cached since it's after EOS
                            }
                            blocks_created += 1
                            next_block_id += 1
                        else:
                            # Block might contain valid content, stop creating
                            break
                    
                    # Recalculate total_masks after updating block states
                    total_masks = mask_index_updated[:, prompt_length:].sum()
                    
                    if total_masks == 0:
                        print(f"[EarlyStop-KV] Exiting at NFE={nfe}, all masks cleared after EOS")
                        break
            
            # Update block activation states
            update_block_activation_states()
            
            # Add new block dynamically based on last block's progress (skip if EOS detected)
            if next_block_id <= num_blocks and not has_eos:
                last_bid = max(block_states.keys())
                if last_bid > 0:
                    last_progress = 1 - block_states[last_bid]["mask_count"] / block_states[last_bid]["total_masks"]
                    should_add_block = (last_progress >= block_add_threshold) or (block_states[last_bid]["mask_count"] == 0)
                    
                    if should_add_block:
                        block_start = prompt_length + (next_block_id - 1) * block_size
                        block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                        if block_end > block_start:
                            actual_mask_count = (x[:, block_start:block_end] == mask_token_id).sum().item()
                            prev_progress = 1 - block_states[next_block_id - 1]["mask_count"] / block_states[next_block_id - 1]["total_masks"]
                            should_activate = prev_progress >= decoded_token_threshold
                            
                            block_states[next_block_id] = {
                                "start": block_start,
                                "end": block_end,
                                "mask_count": actual_mask_count,
                                "total_masks": block_end - block_start,
                                "is_complete": should_activate,
                                "completed_at_nfe": None,
                                "is_cached": False,
                            }
                            next_block_id += 1
            
            # Find rightmost active block (has masks or is complete)
            rightmost_active_bid = 0
            for bid in sorted(block_states.keys()):
                if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                    rightmost_active_bid = bid
            
            if rightmost_active_bid == 0:
                break
            
            active_end = block_states[rightmost_active_bid]["end"]
            
            refresh_output = None
            
            # Delayed KV-Cache strategy: cache blocks after they stabilize
            cache_length = manual_cache_length
            
            # Update completion timestamps and find blocks ready to cache
            blocks_to_cache = []
            for bid in sorted(block_states.keys()):
                if block_states[bid]["end"] <= cache_length:
                    continue  # Already cached
                
                if block_states[bid]["mask_count"] == 0:  # Fully decoded
                    # Mark completion time if just finished
                    if block_states[bid]["completed_at_nfe"] is None:
                        block_states[bid]["completed_at_nfe"] = nfe
                    
                    # Check if ready to cache after delay
                    delay = nfe - block_states[bid]["completed_at_nfe"]
                    if delay >= cache_delay_iter and not block_states[bid]["is_cached"]:
                        blocks_to_cache.append(bid)
                        block_states[bid]["is_cached"] = True
                else:
                    break  # Stop at first incomplete block
            
            # Determine update_kvcache: how many new tokens to add to cache
            update_kvcache = 0
            if blocks_to_cache:
                latest_to_cache = max(blocks_to_cache)
                update_kvcache = block_states[latest_to_cache]["end"] - cache_length
            
            # Determine input sequence and forward strategy
            # Find the earliest position that needs to be forwarded (not cached)
            forward_start_pos = cache_length  # Start from end of cached region
            
            # Check if there are completed but not cached blocks (in stabilization period)
            has_stabilizing_blocks = False
            for bid in sorted(block_states.keys()):
                if (block_states[bid]["mask_count"] == 0 and 
                    block_states[bid]["completed_at_nfe"] is not None and
                    not block_states[bid]["is_cached"]):
                    # This block is completed but not cached (stabilizing)
                    has_stabilizing_blocks = True
                    forward_start_pos = min(forward_start_pos, block_states[bid]["start"])
            
            # Also include active blocks
            for bid in sorted(block_states.keys()):
                if block_states[bid]["mask_count"] > 0:
                    forward_start_pos = min(forward_start_pos, block_states[bid]["start"])
            
            # If forward_start_pos >= active_end, no blocks need forwarding
            if forward_start_pos >= active_end:
                break
            
            # Determine forward strategy
            if update_kvcache > 0:
                # Need to cache new blocks: forward from cache_length to end
                input_seq = x[:, cache_length:]
                process_start_pos = cache_length
                use_kv_cache = True
            elif past_key_values is not None and not has_stabilizing_blocks and nfe % refresh_interval != 0:
                # Have cache and no stabilizing blocks: can use cache
                input_seq = x[:, forward_start_pos:]
                process_start_pos = forward_start_pos
                use_kv_cache = True
            elif past_key_values is not None and (has_stabilizing_blocks or nfe % refresh_interval == 0):
                # Refresh cache when: 1) has stabilizing blocks, or 2) periodic refresh interval
                if has_stabilizing_blocks:
                    latest_stabilizing_bid = max([bid for bid in block_states.keys()
                                                 if block_states[bid]["mask_count"] == 0 and 
                                                    block_states[bid]["completed_at_nfe"] is not None and
                                                    not block_states[bid]["is_cached"]])
                    refresh_pos = block_states[latest_stabilizing_bid]["start"]
                else:
                    # Periodic refresh: refresh up to cache_length
                    refresh_pos = cache_length
                
                # Full forward to get fresh cache (only once for all completed blocks)
                refresh_output = self(x, None, None, use_cache=True)
                refresh_cache = refresh_output.past_key_values
                
                # Truncate cache to refresh_pos (mimics prefix_cache strategy)
                past_key_values = DynamicCache()
                for layer_idx in range(len(refresh_cache)):
                    k, v = refresh_cache[layer_idx]
                    past_key_values.update(k[:, :, :refresh_pos, :], v[:, :, :refresh_pos, :], layer_idx)
                
                input_seq = x
                process_start_pos = 0
                use_kv_cache = False
            else:
                # No cache OR has stabilizing blocks: forward full sequence without cache
                input_seq = x
                process_start_pos = 0
                use_kv_cache = False
            
            stabilizing_blocks = []
            for bid in sorted(block_states.keys()):
                if block_states[bid]["mask_count"] == 0 and block_states[bid]["completed_at_nfe"] is not None:
                    delay = nfe - block_states[bid]["completed_at_nfe"]
                    # completed_blocks.append(f"B{bid}(delay={delay},cached={block_states[bid]['is_cached']})")
                    if not block_states[bid]["is_cached"]:
                        stabilizing_blocks.append(bid)
            # if completed_blocks:
                # print(f"[DEBUG-KV] Completed blocks: {', '.join(completed_blocks)}")
            # if stabilizing_blocks:
                # print(f"[DEBUG-KV] Stabilizing blocks (not cached yet): {stabilizing_blocks}")
            
            # print(f"[DEBUG-KV] has_stabilizing_blocks={has_stabilizing_blocks}, forward_start_pos={forward_start_pos}")
            # print(f"[DEBUG-KV] cache_length={cache_length}, update_kvcache={update_kvcache}, input_seq.shape={input_seq.shape}, process_start_pos={process_start_pos}, use_kv_cache={use_kv_cache}")
            
            # Forward pass: reuse refresh_output if possible to avoid redundant forward
            if refresh_output is not None and process_start_pos == 0:
                # Refresh forward already computed full sequence, reuse it
                outputs = refresh_output
                logits = torch.cat([refresh_output.logits[:, :1], refresh_output.logits[:, :-1]], dim=1)
            elif use_kv_cache:
                # With KV cache
                cache_position = torch.arange(
                    manual_cache_length, 
                    manual_cache_length + input_seq.shape[1], 
                    device=self.device
                )
                outputs = self(
                    input_seq,
                    attention_mask=None,
                    past_key_values=past_key_values,
                    use_cache=True,
                    update_kvcache=update_kvcache if update_kvcache > 0 else 0,
                    cache_position=cache_position
                )
                logits = torch.cat([outputs.logits[:, :1], outputs.logits[:, :-1]], dim=1)
                # Note: Cached region is always fully decoded (no masks) by design,
                # so we don't need to compute entropy for it
            else:
                # Without KV cache (like _sample_multi_block)
                outputs = self(input_seq, None, None)
                logits = torch.cat([outputs.logits[:, :1], outputs.logits[:, :-1]], dim=1)
            
            # Update cache and manual cache length only when we have blocks to cache
            if update_kvcache > 0:
                past_key_values = outputs.past_key_values
                manual_cache_length += update_kvcache
            
            # Mask out future blocks (positions after active_end)
            mask_index_for_decode = mask_index.clone()
            mask_index_for_decode[:, active_end:] = 0
            
            # Decode with entropy-based threshold (unified logic matching _sample_multi_block)
            if alg == 'entropy_threshold':
                # Calculate entropy for all positions at once
                p = F.softmax(logits.to(torch.float64), dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
                
                # Sample tokens for all masked positions
                x0 = torch.argmax(logits, dim=-1)
                if temperature > 0:
                    p_temp = F.softmax(logits / temperature, dim=-1)
                    x0 = torch.multinomial(p_temp.view(-1, p_temp.shape[-1]), 1).view(logits.shape[:2])
                
                # Map entropy to full sequence based on what was actually forwarded
                entropy_full = torch.full_like(x, torch.inf, dtype=entropy.dtype)
                logits_end = process_start_pos + logits.shape[1]
                entropy_full[:, process_start_pos:logits_end] = entropy
                
                # Map sampled tokens to full sequence
                x0_full = x.clone()
                x0_full[:, process_start_pos:logits_end] = x0
                
                # Create transfer index based on entropy threshold (global application)
                transfer_index = (entropy_full < threshold) & mask_index_for_decode
                
                # For fully activated blocks, ensure at least one token is decoded (guaranteed progress)
                first_fully_activated_bid = None
                for bid in sorted(block_states.keys()):
                    if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                        first_fully_activated_bid = bid
                        break
                
                if first_fully_activated_bid is not None:
                    start, end = block_states[first_fully_activated_bid]["start"], block_states[first_fully_activated_bid]["end"]
                    block_transfer = transfer_index[:, start:end]
                    # print(f"[DEBUG-KV] first_fully_activated_bid={first_fully_activated_bid}, block_transfer.any()={block_transfer.any()}")
                    
                    if not block_transfer.any():
                        # Force decode the lowest entropy token in this fully activated block
                        block_mask = mask_index_for_decode[:, start:end]
                        block_entropy = entropy_full[:, start:end]
                        block_entropy = torch.where(block_mask, block_entropy, torch.inf)
                        best_idx = block_entropy[0].argmin()
                        transfer_index[0, start + best_idx] = True
                        # print(f"[DEBUG-KV] Forced decode at position {start + best_idx}")
                
                # Apply the decoded tokens
                x[transfer_index] = x0_full[transfer_index]
                
                # Update block mask counts
                for bid in sorted(block_states.keys()):
                    if bid > 0:
                        start, end = block_states[bid]["start"], block_states[bid]["end"]
                        block_decoded = transfer_index[:, start:end].sum().item()
                        if block_decoded > 0:
                            block_states[bid]["mask_count"] = max(0, block_states[bid]["mask_count"] - block_decoded)
                
                # Check EOS token
                if eos_token_id is not None and transfer_index.any():
                    decoded_positions = torch.where(transfer_index[0])[0]
                    for pos in decoded_positions:
                        if x[0, pos].item() == eos_token_id:
                            break
            
            # Safety check
            if nfe > 10000:
                break
        
        if return_dict_in_generate:
            return DreamModelOutput(sequences=x), nfe
        return x, nfe
