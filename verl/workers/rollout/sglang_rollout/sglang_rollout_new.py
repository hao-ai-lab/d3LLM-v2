# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Unified rollout: merges sglang_rollout.py (AR) and sglang_rollout_dllm.py (dLLM) into one file.
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
from __future__ import annotations

import asyncio
import logging
import math
import os
import time
from contextlib import contextmanager
from copy import deepcopy
from json import JSONDecodeError
from typing import TYPE_CHECKING, Union
from uuid import uuid4

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine

try:
    from sglang.srt.openai_api.protocol import Tool
except ImportError:
    from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.sampling.sampling_params import SamplingParams

try:
    from sglang.srt.utils import get_ip, get_open_port
except ImportError:
    from sglang.srt.utils import get_local_ip_auto as get_ip
    from sglang.srt.utils import get_open_port
from tensordict import TensorDict
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.third_party.sglang import parallel_state as sglang_ps
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import (
    OpenAIFunctionCallSchema,
    OpenAIFunctionParsedSchema,
    OpenAIFunctionToolCall,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.model import compute_position_id_with_mask
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import (
    get_response_mask,
    pad_sequence_to_length,
)
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.workers.rollout.sglang_rollout.utils import (
    broadcast_pyobj,
    compute_batch_wall_times,
)

try:
    from sglang.srt.function_call.function_call_parser import FunctionCallParser
except ImportError:
    from sglang.srt.function_call_parser import FunctionCallParser


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


# NOTE(sgm): add for verl. We can optimize it by making
#  the dataloader yield List[int] without padding.
def _pre_process_inputs(
    pad_token_id,
    prompt_token_ids: torch.Tensor,
) -> list[int]:
    # remove the left padding in the prompt token_id
    non_pad_index = torch.nonzero(prompt_token_ids != pad_token_id, as_tuple=False)[0][0]
    token_ids = prompt_token_ids[non_pad_index:].tolist()
    return token_ids


# NOTE(linjunrong): adhoc
def _sequentialize_decode_order(do_list: list) -> list:
    """Convert SGLang's multi-token-per-step decode_order to strictly sequential 1→2→...→N.

    SGLang dLLM assigns the same step number to tokens unmasked simultaneously
    (e.g., [1,1,1,3,2,1,4,...]).  For d3LLM training we need each response token
    to have a unique ascending step so threshold-based masking works correctly.
    Within each step, ties are broken by position (left-to-right).
    """
    if not do_list:
        return do_list
    # (position, original_step) sorted by (step, position) → stable sequential order
    indexed = sorted(
        ((pos, step) for pos, step in enumerate(do_list) if step > 0),
        key=lambda x: (x[1], x[0]),
    )
    out = [0] * len(do_list)
    for rank, (pos, _) in enumerate(indexed, start=1):
        out[pos] = rank
    return out


def _make_global_monotonic_nfe(resp_do_raw: list, nested_windows, window_size: int = None) -> list:
    """Build a globally monotonic decode_orders_nfe for the response.

    For LowConfidence / LowConfidenceMultiBlock (per DLLM_EXTEND window):
        Each extend window decodes with an independent NFE counter resetting to 1.
        Raw per-window values like [1,3,2,4, 2,1,3,4, ...] make max(nfe_tensor)
        equal to max_T_window (≤ block_size) instead of the true total forward count.

        Fix: add cumulative offset to each window so the global max equals the
        total denoising forwards for retained content tokens:
            offset_w = T_w0 + T_w1 + ... + T_{w-1}
            nfe_global[pos] = nfe_local[pos] + offset_w  (if nfe_local > 0)

        With this, max(decode_orders_nfe) = sum(T_window_i) for retained windows.

    For FullAttnMultiBlock:
        decode_order is already globally monotonic within a single run().
        Do NOT re-chunk; window_size must be None.

        The noisy backends (block_trajectory, all_block_trajectory) use the
        decode_orders_nfe per-window, applying local re-sequentialization inside
        the function, so this global offset does not affect their correctness.

    Args
    ----
    resp_do_raw    : flat 1-D list of raw NFE values for the response portion.
    nested_windows : list-of-lists (one sub-list per LCMB window) if the scheduler
                     delivered the decode_order nested; None otherwise.
    window_size    : LCMB decode window size (block_size).  When ``nested_windows``
                     is None but this is provided, the response is re-chunked into
                     ``window_size``-token windows so the per-window offset can still
                     be applied (the scheduler sometimes flattens LCMB decode_order
                     before it reaches us — same data, just un-nested).  None ⇒ treat
                     the response as a single globally-monotonic stream (LowConfidence).

    Returns
    -------
    Globally monotonic list of the same length as resp_do_raw.
    """
    windows = nested_windows
    if windows is None:
        if not window_size or window_size <= 0:
            # LowConfidence (flat): already globally monotonic.
            return list(resp_do_raw)
        # LCMB delivered flat: re-chunk into fixed-size windows.  The recorder writes
        # the whole extend window (block_size tokens) per call, so consecutive
        # window_size-sized chunks of resp_do_raw reconstruct the original windows.
        windows = [
            resp_do_raw[k : k + window_size]
            for k in range(0, len(resp_do_raw), window_size)
        ]

    # Accumulate per-window offset.
    global_nfe = []
    offset = 0
    for window_do in windows:
        T_win = max((v for v in window_do if v > 0), default=0)
        for v in window_do:
            global_nfe.append(v + offset if v > 0 else 0)
        offset += T_win

    # LCMB windows are block-size aligned; the last window may extend beyond the
    # actual response boundary (e.g., due to ceiling-division padding).  Truncate
    # to exactly len(resp_do_raw) to prevent shape mismatches when storing into
    # the pre-allocated do_nfe_tensor (total_length).
    expected = len(resp_do_raw)
    if len(global_nfe) > expected:
        global_nfe = global_nfe[:expected]
    elif len(global_nfe) < expected:
        global_nfe.extend([0] * (expected - len(global_nfe)))
    return global_nfe


def _flatten_meta_nfe(v) -> list:
    """Flatten meta_info['nfe'] to one scalar per DLLM_EXTEND window."""
    if isinstance(v, (int, float)):
        return [int(v)]
    out: list = []
    for x in v:
        if isinstance(x, (list, tuple)):
            out.extend(_flatten_meta_nfe(x))
        else:
            out.append(int(x))
    return out


def _sum_nfe_list(v) -> int:
    """Sum all per-window nfe values from meta_info (may be nested from streaming)."""
    return sum(_flatten_meta_nfe(v))


def _retained_nfe_count(nfe_values) -> int:
    """True NFE for retained content = number of DISTINCT positive denoising steps.

    ``nfe_values`` is a row of decode_orders_nfe (global-monotonic step index per
    position, 0 = prompt/padding/post-EOS-trimmed).  Each distinct positive value
    is one denoising forward that unmasked >= 1 *retained* token, so:

        resp_tokens (= count of positive entries) >= NFE (= count of distinct values)
        => TPF = resp_tokens / NFE  is always >= 1.

    Counting distinct steps — not max(), not sum(meta_info nfe) — is the only
    measure robust to BOTH skew sources:
      * EOS trim zeroes a window's later positions, so the forwards that produced
        only post-EOS tokens vanish from the set (meta-sum would still add them,
        inflating NFE and pushing TPF below 1);
      * a multi-block forward that accepts 0 tokens for THIS row leaves a gap in the
        step numbering, so it is absent from the set (max() spans the gap and would
        over-count it).
    """
    return len({int(v) for v in nfe_values if v > 0})


def _post_process_outputs(tokenizer, output, keep_trailing_mask: bool = False):
    # Detect dLLM mask token for filtering residual mask tokens from output
    _mask_token_id = getattr(tokenizer, "mask_token_id", None)
    _eos_id = tokenizer.eos_token_id or 2
    # Use space token as neutral replacement (won't trigger EOS-based truncation)
    _space_id = tokenizer.encode(" ", add_special_tokens=False)[0] if _mask_token_id else 0

    def _map_each_response(resp):
        output_token_logprobs = resp["meta_info"]["output_token_logprobs"]
        pairs = [(log_prob, token_ids) for log_prob, token_ids, _ in output_token_logprobs]
        if not pairs:
            # dLLM fallback: output_token_logprobs is empty because dLLM generates all
            # response tokens in parallel (not autoregressively). Use output_ids directly.
            ids = resp.get("output_ids", [])
            if not ids:
                # Final fallback: re-encode text (should rarely happen)
                text = resp.get("text", "")
                logger.warning(f"[dLLM DEBUG] No output_ids, re-encoding text: {text[:100]}...")
                ids = tokenizer.encode(text, add_special_tokens=False) if text else []
            # Replace residual mask tokens.
            # Default: trailing → <|endoftext|>, middle → space (for EOS-based attention truncation).
            # When keep_trailing_mask=True (dllm_full_response_mask): keep trailing <|mdm_mask|>.
            if _mask_token_id is not None:
                n_mask = sum(1 for t in ids if t == _mask_token_id)
                if n_mask > 0:
                    logger.info(f"[dLLM] Filtered {n_mask}/{len(ids)} residual mask tokens")
                    # Find last non-mask position
                    last_content = len(ids) - 1
                    while last_content >= 0 and ids[last_content] == _mask_token_id:
                        last_content -= 1
                    if keep_trailing_mask:
                        ids = [
                            (_space_id if (t == _mask_token_id and i <= last_content) else t)
                            for i, t in enumerate(ids)
                        ]
                    else:
                        ids = [
                            (_space_id if i <= last_content else _eos_id) if t == _mask_token_id else t
                            for i, t in enumerate(ids)
                        ]
            # Measure before pad_sequence: for LLaDA eos_token_id == pad_token_id,
            # so once pad_sequence is called the two are indistinguishable.
            n_eos = sum(1 for t in ids if t == _eos_id)
            # first_eos_pos = response-relative index of the first EOS token.
            # Always scan for actual EOS regardless of keep_trailing_mask, so that
            # length-penalty and metrics always reflect real content length.
            first_eos_pos = next((i for i, t in enumerate(ids) if t == _eos_id), len(ids))
            dummy_lp = [0.0] * len(ids)
            return torch.tensor(ids, dtype=torch.long), torch.tensor(dummy_lp, dtype=torch.float), n_eos, first_eos_pos
        log_probs, output_token_ids = zip(*pairs)
        n_eos = sum(1 for t in output_token_ids if t == _eos_id)
        first_eos_pos = next((i for i, t in enumerate(output_token_ids) if t == _eos_id), len(output_token_ids))
        return torch.tensor(output_token_ids), torch.tensor(log_probs), n_eos, first_eos_pos

    out_map = list(map(lambda x: _map_each_response(x), output))
    batched_output_token_ids = []
    batched_logprobs = []
    # Per-sequence stats collected before pad_sequence.
    batched_eos_counts = []
    # first_eos_pos[i] = number of content tokens before the first EOS in sequence i.
    # Unlike (raw_length - eos_count), this correctly handles EOS in the middle of the sequence.
    batched_first_eos_pos = []
    for output_token_ids, log_probs, n_eos, first_eos_pos in out_map:
        batched_output_token_ids.append(output_token_ids)
        batched_logprobs.append(log_probs)
        batched_eos_counts.append(n_eos)
        batched_first_eos_pos.append(first_eos_pos)
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    batched_output_token_ids = pad_sequence(batched_output_token_ids, batch_first=True, padding_value=pad_token_id)
    if len(batched_logprobs) > 0:
        batched_logprobs = pad_sequence(batched_logprobs, batch_first=True, padding_value=0.0)
    return batched_output_token_ids, batched_logprobs, batched_eos_counts, batched_first_eos_pos


def get_tool_call_parser_type(tokenizer: PreTrainedTokenizer) -> str:
    items = FunctionCallParser.ToolCallParserEnum.items()
    for parser_type, parser_cls in items:
        parser = parser_cls()
        if parser.bot_token in tokenizer.get_vocab() and (parser.eot_token == "" or parser.eot_token in tokenizer.get_vocab()):
            return parser_type
    else:
        raise ValueError(f"No tool call parser found for tokenizer {tokenizer}")


class SGLangRollout(BaseRollout):
    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer,
        model_hf_config,
        port=None,
        trust_remote_code: bool = False,
        device_mesh: DeviceMesh | None = None,
        **kwargs,
    ):
        """Synchronized SGLang rollout engine.

        Args:
            actor_module: Huggingface model name or path to the model. The
                model should be supported by SGLang.
            config: A DictConfig object containing SGLang-specific operational
                parameters and rollout settings.
                Refer to https://docs.sglang.ai/backend/server_arguments.html
            tokenizer: The tokenizer instance compatible with the actor_module.
            model_hf_config: The Hugging Face model's configuration (e.g.,
                `transformers.PretrainedConfig`). It provides architectural
                details and hyperparameters like `max_position_embeddings`,
                used by SGLang for correct model initialization. This is
                the model's inherent design, not SGLang's runtime behavior.
            port: Optional port for multi-node initialization when nnodes > 1.
            trust_remote_code: Whether or not to allow for custom models
                defined on the Hub in their own modeling files.
            device_mesh: Optional `DeviceMesh` object for distributed setup.
            **kwargs: Additional keyword arguments, primarily `train_tp` for
                Megatron Backend integration to initialize hybrid engine
                process groups.
        """
        super().__init__()
        self.config = config
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        (
            self._tool_schemas,
            self._tool_map,
            self._tool_call_parser_type,
            self._sgl_tools,
            self._function_call_parser,
        ) = self._initialize_tools(config, tokenizer)
        # If turn on `free_cache_engine`, SGLang engine's KV cache
        # will be freed after each `generate_sequences` call.
        assert not (not config.enforce_eager and config.free_cache_engine), "disable CUDA graph (enforce_eager = False) if free cache engine"

        logger.info(f"tool_schemas: {self._tool_schemas}, tool_map: {self._tool_map}, tool_call_parser_type: {self._tool_call_parser_type}, sgl_tools: {self._sgl_tools}, function_call_parser: {self._function_call_parser}")

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)

        self._verify_config(model_hf_config=model_hf_config)
        # Store tokenizer early so _init_inference_engine can access vocab size.
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        # For dLLM (LLaDA/DREAM) rollouts: read mask_token_id from the model's
        # HF config, same source as dllm_actor.py (actor_module.config.mask_token_id).
        # Standard HF tokenizers don't carry this attribute, so model_hf_config
        # is the canonical source.
        self._dllm_mask_token_id = getattr(model_hf_config, "mask_token_id", None)
        # LCMB decode window size: each LCMB extend window decodes `block_size` tokens
        # with an NFE counter that resets to 1 per window.  Needed to reconstruct the
        # globally-monotonic NFE (and correct TPF) when the scheduler delivers the
        # decode_order flattened instead of nested.  Read from the rollout dllm
        # algorithm config yaml; falls back to None (treated as single window) for
        # non-LCMB algorithms.
        self._dllm_decode_window_size = self._read_dllm_block_size()
        _algo_name = self.config.get("dllm_algorithm", None) if hasattr(self.config, "get") else None
        # LC / LCMB reset the NFE counter on every DLLM_EXTEND window; FAM uses a
        # single run() with globally monotonic decode_order already.
        self._dllm_nfe_resets_per_window = bool(
            _algo_name and "FullAttn" not in str(_algo_name)
        )
        # Backward-compat alias (LCMB-only checks elsewhere).
        self._dllm_is_multiblock = bool(_algo_name and "MultiBlock" in str(_algo_name))
        # initialize the inference engine
        self._init_inference_engine(trust_remote_code, actor_module, port)

        self._init_sampling_params(**kwargs)

    def _read_dllm_block_size(self):
        """Read the LCMB decode window size (`block_size`) from the rollout dllm
        algorithm config yaml.  Returns int or None when unavailable / not LCMB."""
        try:
            from omegaconf import OmegaConf

            algo = self.config.get("dllm_algorithm", None) if hasattr(self.config, "get") else None
            cfg_path = self.config.get("dllm_algorithm_config", None) if hasattr(self.config, "get") else None
            if not algo or not cfg_path or not os.path.isfile(cfg_path):
                return None
            algo_cfg = OmegaConf.load(cfg_path)
            bs = algo_cfg.get("block_size", None)
            return int(bs) if bs else None
        except Exception as e:  # noqa: BLE001 - config read is best-effort
            logger.warning(f"[dLLM] could not read LCMB block_size from config: {e}")
            return None

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), "tensor parallel size should be less than or equal to the world size"
        self.train_tp = kwargs.get("train_tp", None)
        if self.train_tp is not None:
            # deployed with megatron
            os.environ["CUDA_TIMER_STREAM_KAFKA_ENABLE"] = "0"
            os.environ["MEGATRON_IMPORT_TIMERS"] = "0"
            train_tp = kwargs.get("train_tp", None)
            num_tp_per_train_tp = train_tp // self.tensor_parallel_size
            sglang_ps.initialize_parallel_state(
                tensor_model_parallel_size=self.tensor_parallel_size,
                num_tp_per_train_tp=num_tp_per_train_tp,
            )

        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))

        # init device mesh
        if self._device_mesh_cpu is None:
            device_mesh_kwargs = dict(
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )

            self._device_mesh_cpu = init_device_mesh("cpu", **device_mesh_kwargs)

        self._rank = self._device_mesh_cpu.get_rank()
        # Support both "tp" (AR worker default) and "infer_tp" (dLLM worker) mesh dim names.
        _dim_names = self._device_mesh_cpu.mesh_dim_names
        if "tp" in _dim_names:
            self._mesh_tp_name = "tp"
        elif "infer_tp" in _dim_names:
            self._mesh_tp_name = "infer_tp"
        else:
            raise ValueError(
                f"SGLangRollout: device_mesh must contain 'tp' or 'infer_tp' dim, "
                f"got mesh_dim_names={tuple(_dim_names)}"
            )
        _tp_mesh = self._device_mesh_cpu[self._mesh_tp_name]
        self._tp_rank = _tp_mesh.get_local_rank()
        self._tp_size = _tp_mesh.size()
        if self._rank == 0:
            logger.info(f"_init_distributed_env: tp_world: {self._tp_size}, global_world: {world_size}, mesh_tp_name: {self._mesh_tp_name}")
        # get tp_rank of this process in this tp group
        visible_devices = [None] * self._device_mesh_cpu.size(1)

        torch.distributed.all_gather_object(visible_devices, os.environ["CUDA_VISIBLE_DEVICES"], self._device_mesh_cpu.get_group(self._mesh_tp_name))
        self.visible_devices_set = set(",".join(visible_devices).split(","))
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert self.config.max_model_len >= self.config.prompt_length + self.config.response_length, f"""max_model_len should be greater than total sequence length (prompt_length + response_length): 
            {self.config.max_model_len} >= {self.config.prompt_length} + {self.config.response_length}"""
        # For composite configs (e.g. Qwen3.5), max_position_embeddings is in text_config
        max_pos = getattr(model_hf_config, "max_position_embeddings", None)
        if max_pos is None and hasattr(model_hf_config, "text_config"):
            max_pos = getattr(model_hf_config.text_config, "max_position_embeddings", None)
        if max_pos is not None:
            assert max_pos >= self.config.max_model_len, "model context length should be greater than total sequence length"
        # currently max_turns stand for max number of tool calls
        if self.config.multi_turn.max_turns is None:
            self.config.multi_turn.max_turns = self.config.max_model_len // 3

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        # initialize the inference engine
        nnodes = -(-self._tp_size // len(self.visible_devices_set))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group(self._mesh_tp_name),
                src=self._device_mesh_cpu[self._mesh_tp_name].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if self.config.load_format.startswith("dummy") else self.config.load_format
        logger.info(f"[SGLang] config.load_format={self.config.load_format}, actual load_format={load_format}, model_path={actor_module}")
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        # CRITICAL FIX for dLLM with tp_size > 1:
        # SGLang Engine internally manages TP workers when tp_size > 1.
        # Only the FIRST rank in the ENTIRE TP group should create the Engine.
        # Other ranks should set self._engine = None and rely on broadcast.
        dllm_algo = self.config.get("dllm_algorithm", None) if hasattr(self.config, "get") else None
        if dllm_algo and self._tp_size > 1:
            should_create_engine = (self._tp_rank == 0)
            logger.info(f"[dLLM] tp_rank={self._tp_rank}, tp_size={self._tp_size}, should_create_engine={should_create_engine} (dLLM TP>1: only rank 0)")
        elif dllm_algo and self._tp_size == 1:
            should_create_engine = True
            logger.info(f"[dLLM] tp_rank={self._tp_rank}, tp_size=1, each rank creates its own engine (data parallel)")
        else:
            should_create_engine = first_rank_in_node

        if should_create_engine:
            rank = dist.get_rank()
            os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"

            # dLLM support: pass algorithm params if configured
            dllm_kwargs = {}
            dllm_algo = self.config.get("dllm_algorithm", None) if hasattr(self.config, "get") else None
            logger.warning(f"[dLLM DEBUG] dllm_algorithm from config: {dllm_algo}")
            if dllm_algo:
                dllm_kwargs["dllm_algorithm"] = dllm_algo
                dllm_algo_cfg = self.config.get("dllm_algorithm_config", None)
                if dllm_algo_cfg:
                    dllm_kwargs["dllm_algorithm_config"] = dllm_algo_cfg
                # Pass mask_invalid_token config through ServerArgs fields so SGLang TP workers
                # receive it even when spawned as separate Ray actors (os.environ not inherited).
                _mask_invalid = bool(self.config.get("mask_invalid_token", False))
                print(
                    f"\033[33m[sglang_rollout _init_inference_engine]\033[0m "
                    f"mask_invalid_token={_mask_invalid} tokenizer_vocab_size={len(self.tokenizer)}",
                    flush=True,
                )
                if _mask_invalid:
                    dllm_kwargs["dllm_mask_invalid_token"] = True
                    dllm_kwargs["dllm_tokenizer_vocab_size"] = len(self.tokenizer)
                # dLLM scheduler defaults max_running_requests=1 (serial). Set it to
                # max_num_seqs so all requests in a batch are processed concurrently.
                dllm_kwargs["max_running_requests"] = self.config.get("max_num_seqs", 256)
                logger.warning(f"[dLLM DEBUG] dllm_kwargs: {dllm_kwargs}")

            # Attention backend override for dLLM (bidirectional attention)
            # Also forward any extra sglang-specific kwargs from engine_kwargs.sglang config
            # (e.g. cuda_graph_bs, cuda_graph_max_bs, max_model_len, disable_cuda_graph, ...)
            attn_backend = None
            extra_sglang_kwargs = {}
            if hasattr(self.config, "get"):
                sglang_kwargs = self.config.get("engine_kwargs", {})
                if sglang_kwargs:
                    sglang_kwargs = sglang_kwargs.get("sglang", {}) or {}
                    attn_backend = sglang_kwargs.get("attention_backend", None)
                    # Forward all remaining sglang kwargs to the Engine
                    extra_sglang_kwargs = {k: v for k, v in sglang_kwargs.items() if k != "attention_backend"}
            attn_kwargs = {"attention_backend": attn_backend} if attn_backend else {}
            logger.warning(f"[dLLM DEBUG] attn_kwargs: {attn_kwargs}, extra_sglang_kwargs: {extra_sglang_kwargs}")

            # For dLLM with CUDA graph enabled, pre-capture graphs for all batch sizes
            # up to max_num_seqs (mirrors try_LLaDA2.1_GSM8K.py: cuda_graph_bs=list(range(1, BS+1)))
            # When enforce_eager=True, disable CUDA graph in SGLang to avoid OOM during capture.
            cuda_graph_kwargs = {}
            if self.config.enforce_eager:
                # enforce_eager=True means no CUDA graph; propagate this to SGLang explicitly
                # so it doesn't auto-select a large cuda_graph_max_bs for high-HBM GPUs.
                cuda_graph_kwargs["disable_cuda_graph"] = True
            elif dllm_kwargs:
                _max_seqs = self.config.get("max_num_seqs", 256)
                cuda_graph_kwargs["cuda_graph_bs"] = list(range(1, _max_seqs + 1))
            logger.warning(f"[dLLM DEBUG] cuda_graph_kwargs: {cuda_graph_kwargs}, enforce_eager: {self.config.enforce_eager}")

            logger.warning(f"[dLLM DEBUG] Creating Engine with model_path={actor_module}, tp_size={self._tp_size}")
            logger.warning(f"[dLLM DEBUG] Engine kwargs - dllm: {dllm_kwargs}, attn: {attn_kwargs}, cuda_graph: {cuda_graph_kwargs}")

            # dLLM always needs real weights; override dummy load_format
            if dllm_algo and load_format == "dummy":
                logger.info(f"[dLLM] Overriding load_format 'dummy' → 'auto' for dLLM (tp_size={self._tp_size})")
                load_format = "auto"

            engine_kwargs = {
                "model_path": actor_module,
                "tp_size": self._tp_size,
                "dtype": self.config.dtype,
                "context_length": self.config.max_model_len,
                "mem_fraction_static": self.config.gpu_memory_utilization,
                "trust_remote_code": trust_remote_code,
                "enable_memory_saver": True,
                "disable_cuda_graph": self.config.enforce_eager,
                "base_gpu_id": 0,
                "gpu_id_step": 1,
                "load_format": load_format,
                **dllm_kwargs,
                **attn_kwargs,
                # extra_sglang_kwargs can override context_length if explicitly set in config
                **extra_sglang_kwargs,
                # cuda_graph_kwargs last so disable_cuda_graph can override extra_sglang_kwargs
                **cuda_graph_kwargs,
            }

            if self.config.get("fixed_teacher", False):
                # The engine releases weights immediately after initialization and
                # between rollout batches. No student sync will restore them, so
                # preserve the initial teacher weights across memory-saver sleeps.
                engine_kwargs["enable_weights_cpu_backup"] = True

            # Add distributed params for multi-node, or multi-TP dLLM (needs dist init even on single node)
            if nnodes > 1 or (dllm_algo and self._tp_size > 1):
                engine_kwargs["dist_init_addr"] = dist_init_addr
                engine_kwargs["nnodes"] = nnodes
                engine_kwargs["node_rank"] = node_rank

            # Add port for multi-worker setup
            if dist.get_world_size() > 1:
                engine_kwargs["port"] = 30000 + rank

            logger.warning(f"[dLLM DEBUG] Final engine_kwargs: {engine_kwargs}")
            self._engine = Engine(**engine_kwargs)
            logger.warning(f"[dLLM DEBUG] Engine created successfully")
        else:
            self._engine = None

        self.sharding_manager = None
        # offload: in rollout_only mode, only release KV cache (not weights) since we need to keep
        # the model weights loaded. In normal mode, release all memory as FSDP will sync weights later.
        if self._engine is not None and self._tp_rank == 0:
            if self.config.get("rollout_only", False):
                self._engine.release_memory_occupation(tags=["kv_cache"])
            else:
                self._engine.release_memory_occupation()
        self.is_sleep = True
        
    def _init_sampling_params(self, **kwargs):
        kwargs = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        # supporting adding any sampling params from the config file
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                kwargs[k] = self.config.get(k)
        self.sampling_params = kwargs

        
    def _initialize_tools(self, config, tokenizer):
        """Initialize tools from configuration.

        Args:
            config: Configuration object containing tool-related settings,
                    specifically `config.multi_turn.tool_config_path`.
            tokenizer: The tokenizer instance used for parsing tool calls from
                       the model's generated text.

        Returns:
            tuple: A tuple containing:
                - tool_schemas (list[dict]): OpenAI-formatted JSON schemas
                  defining each tool's capabilities.
                - tool_map (dict[str, BaseTool]): A dictionary mapping tool
                  names to their executable `BaseTool` objects.
                - tool_call_parser_type (str): The identifier for the specific
                  parser type (e.g., 'json_mode', 'tool_code') used to extract
                  tool calls.
                - sgl_tools (list[sglang.srt.openai_api.protocol.Tool]): Tool
                  definitions optimized for SGLang's internal engine.
                - function_call_parser (sglang.srt.function_call_parser.FunctionCallParser):
                  The active parser instance responsible for extracting
                  structured tool calls from model outputs.
        """
        if config.multi_turn.tool_config_path is None:
            return [], {}, None, [], None

        import importlib.util
        import sys

        from omegaconf import OmegaConf

        from verl.tools.schemas import OpenAIFunctionToolSchema

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []

            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)

                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]

                tool_cls = getattr(module, class_name)

                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)

                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)

            return tool_list

        tools_config_file = config.multi_turn.tool_config_path
        tools_config = OmegaConf.load(tools_config_file)
        tool_list = initialize_tools_from_config(tools_config)
        logger.info(f"Initialize tools from configuration.: tool_list: {tool_list}")
        tool_schemas = [tool.get_openai_tool_schema().model_dump() for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}
        tool_call_parser_type = get_tool_call_parser_type(tokenizer)
        sgl_tools = [Tool.model_validate(tool_schema) for tool_schema in tool_schemas]
        function_call_parser = FunctionCallParser(
            sgl_tools,
            tool_call_parser_type,
        )

        return (
            tool_schemas,
            tool_map,
            tool_call_parser_type,
            sgl_tools,
            function_call_parser,
        )

    @contextmanager
    def update_sampling_params(self, **kwargs):
        """
        Temporarily updates the model's sampling parameters for the
        duration of a `with` block. Parameters are automatically fall
          back to their original values upon exiting the block.

        Args:
            **kwargs: Keyword arguments representing sampling parameters
                    to be updated. Only parameters that already exist in
                    `self.sampling_params` will be updated.
        """
        # Store original values of parameters that will be updated
        old_sampling_params_args = {key: self.sampling_params[key] for key in kwargs if key in self.sampling_params}

        # Update sampling parameters with new values
        for key, value in kwargs.items():
            if key in self.sampling_params:
                self.sampling_params[key] = value

        try:
            yield
            # Yield and execute the code within the 'with' block
        finally:
            # Always restore original values, even if an error
            # occurred in the `with` block
            for key, value in old_sampling_params_args.items():
                self.sampling_params[key] = value

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        if self.config.multi_turn.enable:
            return self._req_level_generate_sequences(prompts, **kwargs)
        return self._batch_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _batch_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        """Generates sequences for a batch of prompts.
        For single-turn generation, all prompts are processed in one request.
        For multi-turn generation, each prompt is processed separately via
        `_generate_req_level_sequences` for better tool calling control.
        `_generate_batch_level_sequences` involves:
        1.  Extracting and pre-processing prompt token IDs from the input
            `prompts`. This includes handling padding and preparing raw
            token ID lists.
        2.  Preparing inputs for the SGLang engine, including multi-modal
            data if present.
        3.  Invoking the SGLang engine (`self._engine.async_generate`,
            an async coroutine) with the batch of processed inputs and
            specified sampling parameters on the master TP rank.
        4.  Broadcasting the results from the master TP rank to all
            other TP ranks.
        5.  Post-processing the engine's output to format the generated
            token IDs and (if applicable) log probabilities.
        6.  Constructing the final sequences by concatenating original
            prompts with the generated responses.
        7.  Updating attention masks and position IDs to reflect the full
            concatenated sequences.
        8.  If `self.config.free_cache_engine` is true, the SGLang engine's
            KV cache is flushed after generation on the master TP rank.
        Args:
            prompts: A `DataProto` object containing the batch of
              input prompts, including tensor data (like `input_ids`,
              `attention_mask`) and meta-information (like `eos_token_id`,
              `do_sample`).
            **kwargs: Additional keyword arguments that can override the
              default sampling parameters (e.g., `temperature`, `top_p`,
              `max_new_tokens`). These are temporarily applied using
              `update_sampling_params`.
        Returns:
            DataProto: A `DataProto` object containing the batch of
              generated sequences. This includes tensors for `prompts`
              (original input IDs), `responses` (generated token IDs),
              `input_ids` (concatenated prompt and response),
              `attention_mask`, and `position_ids` for the full
              sequences.
        Note that when `n > 1`, each prompt generates multiple sequences,
        so we need to replicate its non-tensor data (i.e. raw prompts,
        messages, reward scores, etc.) n times to match the expanded
        tensor data. This is done in the `_non_tensor_batch` dictionary.
        """
        # input ids: (bs, prompt_length), left-padded
        idx = prompts.batch["input_ids"]
        # attention_mask: (bs, seq_length), left-padded
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]

        # used to generate attention mask for the
        # response based on EOS token position
        eos_token_id = prompts.meta_info["eos_token_id"]

        batch_size = idx.size(0)

        # Extract non-tensor data
        non_tensor_batch = prompts.non_tensor_batch
        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = []
            for raw_prompt_ids, multi_modal_data in zip(
                non_tensor_batch.pop("raw_prompt_ids"),
                non_tensor_batch.pop("multi_modal_data"),
            ):
                sglang_inputs.append(
                    {
                        "prompt_token_ids": raw_prompt_ids,
                        "multi_modal_data": multi_modal_data,
                        "image_data": (multi_modal_data.get("image", None) if isinstance(multi_modal_data, dict) else None),
                    }
                )
        else:
            sglang_inputs = [{"prompt_token_ids": raw_prompt_ids} for raw_prompt_ids in non_tensor_batch.pop("raw_prompt_ids")]

        # Ensure token IDs are lists or numpy arrays
        for input_data in sglang_inputs:
            if isinstance(input_data["prompt_token_ids"], np.ndarray):
                input_data["prompt_token_ids"] = input_data["prompt_token_ids"].tolist()
            elif not isinstance(input_data["prompt_token_ids"], list):
                raise TypeError(f"prompt_token_ids must be a list or numpy array, got {type(input_data['prompt_token_ids'])}")

        # Extract token IDs and image data for SGLang Engine
        idx_list = [input_data["prompt_token_ids"] for input_data in sglang_inputs]
        image_list = [input_data.get("image_data", None) for input_data in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            kwargs = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,  # if validate, already repeat in ray_trainer
            )
        else:
            # Training mode: use config temperature (default 1.0) for exploration
            kwargs = dict(
                temperature=self.config.get("temperature", 1.0),
                top_p=self.config.get("top_p", 1.0),
                top_k=self.config.get("top_k", -1),
                n=1,
            )

        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            # print(f"{self.sampling_params=}")
            logger.warning(f"[dLLM DEBUG] Generating with sampling_params: {self.sampling_params}")
            logger.warning(f"[dLLM DEBUG] Batch size: {len(idx_list)}, first prompt length: {len(idx_list[0]) if idx_list else 0}")
            if idx_list and len(idx_list) > 0:
                logger.warning(f"[dLLM DEBUG] First prompt token IDs (last 20): {idx_list[0][-20:]}")
                logger.warning(f"[dLLM DEBUG] First prompt decoded: {self.tokenizer.decode(idx_list[0][-100:])}")
            if self._tp_rank == 0:
                t_start = time.perf_counter()
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,  # because we have already convert it to prompt token id
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        input_ids=idx_list,
                        image_data=image_list,
                    )
                )
                t_elapsed = time.perf_counter() - t_start
                # Accurate token counts from SGLang output (before any padding)
                n_prompt = sum(len(ids) for ids in idx_list) if idx_list else 0
                n_resp = sum(len(o.get("output_ids", [])) for o in output) if output else 0
                nfe = (
                    sum(_sum_nfe_list(o.get("meta_info", {}).get("nfe", 0)) for o in output)
                    if output
                    else 0
                )
                decode_forward_count = (
                    sum(
                        _sum_nfe_list(
                            o.get("meta_info", {}).get("decode_forward_count", 0)
                        )
                        for o in output
                    )
                    if output
                    else 0
                )
                # Request timings overlap within a batch, so use their shared
                # wall-clock timeline instead of summing per-request durations.
                prefill_time_s, decode_time_s = compute_batch_wall_times(output or [])
                tpf = n_resp / nfe if nfe > 0 else 0.0
                equivalent_tpf = (
                    n_resp / decode_forward_count if decode_forward_count > 0 else 0.0
                )
                tps = n_resp / t_elapsed if t_elapsed > 0 else 0.0
                decode_tps = n_resp / decode_time_s if decode_time_s > 0 else 0.0
                bs = len(idx_list)
                logger.warning(
                    f"[dLLM TPS] bs={bs}, "
                    f"prompt={n_prompt} (avg={n_prompt/bs:.0f}), "
                    f"resp={n_resp} (avg={n_resp/bs:.0f}), "
                    f"nfe={nfe} tpf={tpf:.6f}, "
                    f"decode_forwards={decode_forward_count} equivalent_tpf={equivalent_tpf:.6f}, "
                    f"prefill={prefill_time_s:.6f}s decode={decode_time_s:.6f}s "
                    f"decode_tps={decode_tps:.6f}, "
                    f"elapsed={t_elapsed:.6f}s tps={tps:.6f}"
                )
                logger.warning(f"[dLLM DEBUG] Generated {len(output)} outputs")
                if output and len(output) > 0:
                    logger.warning(f"[dLLM DEBUG] First output text: {output[0].get('text', '')[:200]}")
                    logger.warning(f"[dLLM DEBUG] First output_ids: {output[0].get('output_ids', [])[:20]}")
                    logger.warning(f"[dLLM DEBUG] First output meta_info keys: {output[0].get('meta_info', {}).keys()}")
            else:
                output = None
            # Most naive implementation, can extract tensor and send via gloo if too slow
            [output] = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu[self._mesh_tp_name].get_group(),
                src=self._device_mesh_cpu[self._mesh_tp_name].mesh[0].item(),
                force_cpu_device=False,
            )
            # Ensure tokenizer carries mask_token_id so _post_process_outputs
            # can detect residual MASK tokens (standard HF tokenizers lack this attr).
            if self._dllm_mask_token_id is not None and not hasattr(self.tokenizer, "mask_token_id"):
                self.tokenizer.mask_token_id = self._dllm_mask_token_id
            out = _post_process_outputs(
                self.tokenizer,
                output,
                keep_trailing_mask=bool(self.config.get("dllm_full_response_mask", False)),
            )

            response = out[0].to(idx.device)
            rollout_log_probs = out[1].to(idx.device)
            # out[2]: EOS counts per sequence (before pad_sequence)
            # out[3]: position of first EOS per sequence = content tokens before first EOS
            _eos_counts = out[2]       # list[int], length = batch_size
            _first_eos_pos = out[3]    # list[int], length = batch_size
            _eos_tensor = torch.tensor(_eos_counts, dtype=torch.long, device=idx.device)
            _first_eos_tensor = torch.tensor(_first_eos_pos, dtype=torch.long, device=idx.device)
            logger.info(
                f"[dLLM EOS] eos_count={_eos_counts} | "
                f"first_eos_pos={_first_eos_pos} | "
                f"mean_eos={_eos_tensor.float().mean():.2f} | "
                f"mean_content_len(before_EOS)={_first_eos_tensor.float().mean():.1f}"
            )

            # Extract decode_order from dLLM meta_info (if available)
            # decode_order[i] = step when position i was unmasked (0 = prompt token)
            dllm_decode_orders = None
            dllm_nfe = None
            if output and isinstance(output, list):
                first_meta = output[0].get("meta_info", {})
                if "decode_order" in first_meta:
                    dllm_decode_orders = [o["meta_info"]["decode_order"] for o in output]
                else:
                    print(
                        f"\033[91m\033[1m"
                        f"[DECODE_ORDER DEBUG] 'decode_order' NOT in meta_info!\n"
                        f"  meta_info keys : {list(first_meta.keys())}\n"
                        f"  output[0] keys : {list(output[0].keys())}\n"
                        f"  full meta_info : {first_meta}"
                        f"\033[0m",
                        flush=True,
                    )
                # nfe = TRUE forward-pass count recorded by the dLLM algorithm itself.
                # Each algorithm run() appends one nfe value per decode window (one entry
                # per DLLM_EXTEND call), so meta_info["nfe"] is the per-window nfe list for
                # the whole response.  Summing it gives the total forward passes used to
                # generate the response — the correct TPF denominator, independent of any
                # decode_order alignment / global-monotonic reconstruction.
                if "nfe" in first_meta:
                    dllm_nfe = [o["meta_info"]["nfe"] for o in output]

            if response.shape[1] < self.config.response_length:
                response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
                rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, 0.0)

            # ── Replace trailing EOS with MASK (response + decode_orders) ─────────
            # Save nested decode_order structure BEFORE EOS trim (trim may flatten).
            _pre_eos_nested: list = []
            if dllm_decode_orders is not None:
                for _i in range(len(dllm_decode_orders)):
                    _raw = dllm_decode_orders[_i]
                    if _raw and isinstance(_raw[0], list):
                        _pre_eos_nested.append([list(w) for w in _raw])
                    else:
                        _pre_eos_nested.append(None)

            # SGLang block-denoising continues after the first EOS, filling each
            # subsequent block with EOS tokens (valid decode_order > 0).  Those
            # positions carry no real training signal ("EOS → predict EOS").
            # Treat them as unresolved mask positions, matching d3LLM's actual_len
            # which counts only content up to (and including) the first EOS.
            #
            # Done HERE — after response is padded to response_length but BEFORE
            # seq / attention_mask / do_tensor are built — so every derived tensor
            # sees the corrected data from the start.
            #
            # NOTE: We search for the first EOS directly in the response tensor
            # rather than using _first_eos_pos[i].  When keep_trailing_mask=False
            # (the default path), _post_process_outputs converts trailing MASK→EOS
            # and returns first_eos_pos=len(ids) (full length), so using it would
            # make _fe+1 >= response.size(1) and skip the replacement entirely.
            # Scanning the response tensor directly is always correct.
            #
            # TODO: remove once SGLang supports native EOS early-stop in block denoising.
            _mask_tok_id_r = self._dllm_mask_token_id
            _eos_tok_id_r  = self.tokenizer.eos_token_id
            if _mask_tok_id_r is not None and _eos_tok_id_r is not None:
                for _i in range(response.shape[0]):
                    # Find first EOS in the padded response tensor.
                    _eos_positions = (response[_i] == _eos_tok_id_r).nonzero(as_tuple=True)[0]
                    if _eos_positions.numel() == 0:
                        continue  # no EOS yet; nothing to replace
                    _fe = int(_eos_positions[0].item())  # response-relative index of first EOS
                    if _fe + 1 < response.size(1):
                        response[_i, _fe + 1:] = _mask_tok_id_r
                    # Zero raw decode_orders for the same trailing positions so that
                    # do_tensor / do_nfe_tensor are already correct when built below.
                    if dllm_decode_orders is not None and _i < len(dllm_decode_orders):
                        _do_list = dllm_decode_orders[_i]
                        if _do_list and isinstance(_do_list[0], list):
                            _do_list = [s for _t in _do_list for s in _t]
                        # Locate response start: first non-zero entry in do_list.
                        _resp_start = next(
                            (j for j, v in enumerate(_do_list) if v > 0),
                            len(_do_list),
                        )
                        _trail_in_list = _resp_start + _fe + 1
                        if _trail_in_list < len(_do_list):
                            dllm_decode_orders[_i] = (
                                list(_do_list[:_trail_in_list])
                                + [0] * (len(_do_list) - _trail_in_list)
                            )

            # ── Recompute _first_eos_tensor from the corrected response tensor ──────
            # _post_process_outputs computed first_eos_pos on raw ids (before our
            # trailing-EOS→MASK replacement), so it may be stale.  Scan the
            # (now-corrected) response tensor directly so that
            # batch["response_first_eos_pos"] and all consumers (length penalty,
            # metrics, debug) see the real first-EOS position.
            if "_eos_tensor" in dir() and _eos_tok_id_r is not None:
                _fep_corrected = []
                for _i in range(response.shape[0]):
                    _ep = (response[_i] == _eos_tok_id_r).nonzero(as_tuple=True)[0]
                    _fep_corrected.append(int(_ep[0].item()) if _ep.numel() > 0 else response.size(1))
                _first_eos_tensor = torch.tensor(_fep_corrected, dtype=torch.long, device=idx.device)

            # utilize current sampling params
            if self.sampling_params.get("n", 1) > 1 and do_sample:
                idx = idx.repeat_interleave(self.sampling_params["n"], dim=0)
                attention_mask = attention_mask.repeat_interleave(self.sampling_params["n"], dim=0)
                position_ids = position_ids.repeat_interleave(self.sampling_params["n"], dim=0)
                batch_size = batch_size * self.sampling_params["n"]
                _non_tensor_batch = {}
                for key, val in non_tensor_batch.items():
                    _non_tensor_batch[key] = np.repeat(val, self.sampling_params["n"], axis=0)
            else:
                _non_tensor_batch = non_tensor_batch
            seq = torch.cat([idx, response], dim=-1)

        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:  # qwen2vl mrope (batch size, 4, seq len)
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)

        # TODO(sgm): fix position_ids on right_pad
        # prompt: left pad + response: right pad
        # attention_mask: [0,0,0,0,1,1,1,1, | 1,1,1,0,0,0,0,0]
        # position_ids:   [0,0,0,0,0,1,2,3, | 4,5,6,7,8,9,10,11]
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        if self.config.get("dllm_full_response_mask", False):
            # Treat all response tokens (including trailing EOS/MASK) as valid,
            # consistent with DARE-style training where the full response region participates in attention.
            response_attention_mask = torch.ones(batch_size, response_length, dtype=attention_mask.dtype, device=attention_mask.device)
        else:
            response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        # all the tp ranks should contain the same data here. data in all ranks are valid
        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,  # here input_ids become the whole sentences
                "rollout_log_probs": rollout_log_probs,  # we will recompute old log prob with actor
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        # Store per-sequence EOS stats (measured before pad_sequence, so reliable even
        # when eos_token_id == pad_token_id as in LLaDA).  Shape: [batch_size].
        if "_eos_tensor" in dir():
            n_repeat = self.sampling_params.get("n", 1)
            batch["response_eos_count"] = _eos_tensor.repeat_interleave(n_repeat)
            # first_eos_pos[i] = number of content tokens before the first EOS in sequence i.
            # This is the correct "effective content length" for full-attention dLLMs.
            batch["response_first_eos_pos"] = _first_eos_tensor.repeat_interleave(n_repeat)

        # Attach dLLM decode_orders if available from SGLang dLLM rollout
        if dllm_decode_orders is not None:
            prompt_length = idx.size(1)
            total_length = seq.size(1)
            do_tensor = torch.zeros(batch_size, total_length, dtype=torch.long, device=idx.device)
            # decode_orders_nfe: raw NFE step numbers before sequentialization.
            # decode_orders_nfe[i, j] = NFE step when pos j was unmasked (same NFE step value
            # for tokens unmasked simultaneously).  Used by the "trajectory" noisy backend to
            # sample t at NFE-step granularity rather than per-token-rank granularity.
            do_nfe_tensor = torch.zeros(batch_size, total_length, dtype=torch.long, device=idx.device)
            _orig_max_steps = []
            for i, do_list in enumerate(dllm_decode_orders):
                # decode_order from SGLang may arrive nested (list-of-lists) when the
                # scheduler streams partial results — one sub-list per LCMB extend
                # window.  The same data can also arrive flattened (a single flat list)
                # depending on how many intermediate sends happened, so downstream code
                # must not rely on the nesting being present.  We flatten here and, for
                # LCMB, reconstruct windows from the known window size when needed.
                _nested_windows = _pre_eos_nested[i] if i < len(_pre_eos_nested) else None
                if do_list and isinstance(do_list[0], list):
                    do_list = [s for turn_do in do_list for s in turn_do]

                if not do_list:
                    resp_am = attention_mask[i, -response_length:] if attention_mask.size(1) > response_length else attention_mask[i]
                    resp_mask = (response[i] != self.pad_token_id) if response is not None else resp_am.bool()
                    n_resp = int(resp_mask.sum().item())
                    if n_resp > 0:
                        _raw_resp_ids = response[i].tolist() if response is not None else []
                        _raw_do_from_sglang = dllm_decode_orders[i] if dllm_decode_orders is not None and i < len(dllm_decode_orders) else "N/A"
                        _raw_meta = output[i].get("meta_info", {}) if output is not None and i < len(output) else {}
                        print(
                            f"\033[91m\033[1m"
                            f"[DECODE_ORDER EMPTY] seq {i} has n_resp={n_resp} but empty do_list!\n"
                            f"  raw dllm_decode_orders[{i}]     = {_raw_do_from_sglang!r}\n"
                            f"  meta_info['decode_order'] raw  = {_raw_meta.get('decode_order', 'KEY_MISSING')!r}\n"
                            f"  meta_info keys                 = {list(_raw_meta.keys())}\n"
                            f"  output[{i}] keys               = {list(output[i].keys()) if output and i < len(output) else 'N/A'}\n"
                            f"\033[0m",
                            flush=True,
                        )
                        raise RuntimeError(
                            f"[decode_order missing] seq {i}: SGLang returned empty decode_order "
                            f"but n_resp={n_resp} (non-empty response).  "
                            f"Make sure the rollout algorithm has record_decode_order: true and "
                            f"supports decode_order recording (FullAttnMultiBlock, LowConfidence, "
                            f"EntropyThreshold).  "
                            f"raw response token_ids (first 20): {_raw_resp_ids[:20]}"
                        )
                    else:
                        _orig_max_steps.append(0)
                        # response is all-padding (eos==pad for LLaDA): DLLM returned
                        # all-mask output_ids, converted to all-EOS by _map_each_response.
                        _raw_resp_ids = response[i].tolist() if response is not None else []
                        _raw_text = self.tokenizer.decode(_raw_resp_ids, skip_special_tokens=False)
                        _sglang_do = dllm_decode_orders[i]
                        print(
                            f"[T=0/empty] seq {i}: decode_order=[] AND n_resp=0 → T=0.\n"
                            f"  raw response token_ids: {_raw_resp_ids}\n"
                            f"  raw response text: {repr(_raw_text)}\n"
                            f"  SGLang decode_order: {_sglang_do}"
                        )
                elif max(do_list) == 0:
                    # SGLang returned a non-empty decode_order but ALL values are 0.
                    # This is a bug in the algorithm — log it and mark T=0 for filtering.
                    resp_mask = (response[i] != self.pad_token_id) if response is not None else torch.zeros(response_length, dtype=torch.bool)
                    n_resp = int(resp_mask.sum().item())
                    print(
                        f"[T=0/allzero] seq {i}: decode_order non-empty but all-zero "
                        f"(len={len(do_list)}), n_resp={n_resp}. "
                        f"do_list[:10]={do_list[:10]}"
                    )
                    # Do NOT fabricate a sequential decode_order — the decode trajectory
                    # is genuinely unknown. Keep T=0 so this sample gets filtered out.
                    _orig_max_steps.append(0)
                else:
                    _orig_max_steps.append(max(do_list))

                # ── Align decode_order to the right-padded response (length-agnostic) ──
                # SGLang returns decode_order covering only the actual decode window:
                #   do_list = [leading zeros: prompt remainder]
                #           + [per-token NFE, 1:1 with each generated response token]
                #           + [trailing zeros: post-EOS positions, zeroed above]
                # Its length is (actual_prompt_remainder + num_generated), NOT total_length.
                # The response tensor is RIGHT-padded (real tokens start at prompt_length),
                # so we strip the leading prompt-remainder zeros and place the response
                # decode-orders left-aligned beginning at prompt_length.  Branching on
                # len(do_list) (the old code) right-aligned the values into the response
                # tail and broke every sample whose do_list was shorter than total_length.
                resp_start = next((j for j, v in enumerate(do_list) if v > 0), len(do_list))

                # Global-monotonic NFE: per-window counters reset on each DLLM_EXTEND.
                # Nested decode_order: re-chunk using the nested sub-lists (includes
                # prompt-remainder zeros inside the first window).
                # Flat decode_order: re-chunk only the response portion (after
                # resp_start) into block_size windows — do NOT re-chunk the leading
                # prompt-remainder zeros with the first gen block.
                _win_sz = self._dllm_decode_window_size if self._dllm_nfe_resets_per_window else None
                if _nested_windows is not None:
                    full_nfe = _make_global_monotonic_nfe(do_list, _nested_windows, window_size=None)
                else:
                    _resp_part = do_list[resp_start:]
                    _resp_nfe = _make_global_monotonic_nfe(_resp_part, None, window_size=_win_sz)
                    full_nfe = [0] * resp_start + _resp_nfe
                # Zero NFE wherever decode_order was cleared (e.g. post-EOS trailing trim).
                full_nfe = [nf if v > 0 else 0 for nf, v in zip(full_nfe, do_list)]

                resp_do_raw = do_list[resp_start:]
                resp_nfe    = full_nfe[resp_start:]
                resp_do     = _sequentialize_decode_order(resp_do_raw)
                resp_len    = min(len(resp_do), response_length)
                if resp_len > 0:
                    do_tensor[i, prompt_length:prompt_length + resp_len] = torch.tensor(
                        resp_do[:resp_len], dtype=torch.long, device=idx.device
                    )
                    do_nfe_tensor[i, prompt_length:prompt_length + resp_len] = torch.tensor(
                        resp_nfe[:resp_len], dtype=torch.long, device=idx.device
                    )
            batch["decode_orders"] = do_tensor
            batch["decode_orders_nfe"] = do_nfe_tensor
            batch["_orig_max_decode_steps"] = torch.tensor(_orig_max_steps, dtype=torch.long, device=idx.device)

        # ── Total NFE (denoising forward-pass count) for TPF ────────────────────
        # nfe_total = number of DISTINCT positive denoising steps in decode_orders_nfe
        # (content up to first EOS).  Derived from the SAME tensor that consumers use
        # to count resp_tokens, so resp_tokens >= nfe_total and TPF = resp/nfe >= 1 by
        # construction.  Robust to EOS-trim (post-EOS forwards drop out of the set) and
        # to multi-block forwards that accept 0 tokens for a row (gap, not counted).
        #
        # nfe_meta_sum = sum(all meta nfe) — diagnostic, includes post-EOS extends.
        # nfe_max_do   = max(decode_orders_nfe) — diagnostic, spans accept-0 gaps.
        if dllm_decode_orders is not None:
            _nfe_content_totals = [
                _retained_nfe_count(do_nfe_tensor[i].tolist())
                for i in range(do_nfe_tensor.size(0))
            ]
            _nfe_from_do = do_nfe_tensor.max(dim=-1).values
            _nfe_tensor = torch.tensor(
                _nfe_content_totals, dtype=torch.long, device=idx.device
            )
            batch["nfe_total"] = _nfe_tensor.clamp(min=0)
            batch["nfe_decode"] = batch["nfe_total"]
            batch["nfe_max_do"] = _nfe_from_do  # diagnostic: max(decode_orders_nfe)
            if dllm_nfe is not None:
                batch["nfe_meta_sum"] = torch.tensor(
                    [_sum_nfe_list(v) for v in dllm_nfe],
                    dtype=torch.long,
                    device=idx.device,
                )
        elif dllm_nfe is not None:
            _tot = [max(_sum_nfe_list(v), 1) for v in dllm_nfe]
            _nfe_tensor = torch.tensor(_tot, dtype=torch.long, device=idx.device)
            batch["nfe_total"] = _nfe_tensor
            batch["nfe_decode"] = _nfe_tensor

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None:
            self._engine.flush_cache()
            torch.cuda.empty_cache()

        return DataProto(batch=batch, non_tensor_batch=_non_tensor_batch)

    async def _async_rollout_a_request(
        self,
        req: AsyncRolloutRequest,
        do_sample: bool = True,
        is_validate: bool = False,
        **kwargs,
    ) -> AsyncRolloutRequest:
        assert self._tp_rank == 0, "only the master process can call this function"
        _req = deepcopy(req)
        finish_reason_type = None
        output = None

        current_turns = 0
        while current_turns < self.config.multi_turn.max_turns:
            if _req.state == AsyncRolloutRequestStateEnum.PENDING:
                await self._handle_pending_state(_req)
                _req.state = AsyncRolloutRequestStateEnum.RUNNING
            elif _req.state == AsyncRolloutRequestStateEnum.TOOL_CALLING:
                if _req.messages[-1].tool_calls is not None:
                    parsed_tool_calls = _req.messages[-1].tool_calls
                    tool_call_results = await asyncio.gather(
                        *[
                            self._tool_map[tool_call.function.name].execute(
                                _req.request_id,
                                tool_call.function.arguments,
                                **_req.tools_kwargs[tool_call.function.name].get("execute_kwargs", {}),
                            )
                            for tool_call in parsed_tool_calls
                        ]
                    )
                    for i, (tool_call, (resp, reward, metrics)) in enumerate(zip(parsed_tool_calls, tool_call_results)):
                        _req.add_tool_response_message(
                            self.tokenizer,
                            resp,
                            (i == len(parsed_tool_calls) - 1),
                            format=self.config.multi_turn.format,
                        )
                        _req.update_metrics(metrics, tool_call.function.name)
                        if len(_req.input_ids) >= self.config.max_model_len:
                            break
                    if len(_req.input_ids) >= self.config.max_model_len:
                        finish_reason_type = FinishReasonTypeEnum.STOP
                        break
                    _req.state = AsyncRolloutRequestStateEnum.RUNNING
                else:
                    raise ValueError(f"Unexpected tool calling last message state: {_req.messages[-1]}")
            elif _req.state == AsyncRolloutRequestStateEnum.RUNNING:
                output = await self._handle_engine_call(_req, do_sample, is_validate, **kwargs)
                content = output["text"]
                finish_reason_type = FinishReasonTypeEnum.from_str(output["meta_info"]["finish_reason"]["type"])
                current_turns += 1
                if finish_reason_type == FinishReasonTypeEnum.LENGTH:
                    _req.add_assistant_message(
                        self.tokenizer,
                        content,
                        already_over_long=True,
                        format=self.config.multi_turn.format,
                    )
                    break
                else:
                    if self._function_call_parser and self._function_call_parser.has_tool_call(content):
                        finish_reason_type = FinishReasonTypeEnum.TOOL_CALL
                        _req.state = AsyncRolloutRequestStateEnum.TOOL_CALLING
                        try:
                            normed_content, tool_calls = self._function_call_parser.parse_non_stream(content)
                        except JSONDecodeError:
                            normed_content = content
                            tool_calls = []
                        except AttributeError:
                            normed_content = content
                            tool_calls = []
                        parsed_tool_calls = []
                        for tool_call in tool_calls:
                            function, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                                OpenAIFunctionParsedSchema(
                                    name=tool_call.name,
                                    arguments=tool_call.parameters,
                                )
                            )
                            # Drop the tool call if its arguments has decode error
                            if has_decode_error:
                                continue
                            parsed_tool_calls.append(
                                OpenAIFunctionToolCall(
                                    id=str(tool_call.tool_index),
                                    function=function,
                                )
                            )
                        if len(parsed_tool_calls) > 0:
                            _req.add_assistant_message(
                                self.tokenizer,
                                normed_content,
                                tool_calls=parsed_tool_calls,
                                format=self.config.multi_turn.format,
                            )
                        else:
                            _req.add_assistant_message(
                                self.tokenizer,
                                content,
                                format=self.config.multi_turn.format,
                            )
                            finish_reason_type = FinishReasonTypeEnum.STOP
                            _req.state = AsyncRolloutRequestStateEnum.COMPLETED
                            break
                    else:
                        _req.add_assistant_message(
                            self.tokenizer,
                            content,
                            format=self.config.multi_turn.format,
                        )
                        break

        if current_turns >= self.config.multi_turn.max_turns:
            finish_reason_type = FinishReasonTypeEnum.STOP

        # Calculate the reward for each tool
        async def calc_reward_and_release_fn(name: str, tool: BaseTool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        tool_reward_tasks = []
        for name in _req.tools_kwargs.keys():
            tool = self._tool_map[name]
            tool_reward_tasks.append(calc_reward_and_release_fn(name, tool))
        tool_reward_scores = await asyncio.gather(*tool_reward_tasks)
        tool_reward_scores = dict(tool_reward_scores)
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type)

        return _req

    async def _handle_engine_call(self, _req: AsyncRolloutRequest, do_sample: bool, is_validate: bool, **kwargs) -> dict:
        generation_prompt_ids = _req.get_generation_prompt(self.tokenizer)
        max_new_tokens = min(self.config.response_length, self.config.max_model_len - len(generation_prompt_ids) - 1)
        if not do_sample:
            kwargs = dict(
                n=1,
                presence_penalty=0.0,
                frequency_penalty=0.0,
                repetition_penalty=1.0,
                temperature=0,
                top_p=1,
                top_k=-1,
                ignore_eos=False,
                min_new_tokens=0,
                max_new_tokens=self.config.response_length,
                skip_special_tokens=True,
                spaces_between_special_tokens=True,
            )
        elif is_validate:
            # TODO: try **
            kwargs = {
                "top_k": self.config.val_kwargs.top_k,
                "top_p": self.config.val_kwargs.top_p,
                "temperature": self.config.val_kwargs.temperature,
                "n": 1,  # if validate, already repeat in ray_trainer
            }
        else:
            # Training mode: use config temperature for exploration
            kwargs = {
                "temperature": self.config.get("temperature", 1.0),
                "top_p": self.config.get("top_p", 1.0),
                "top_k": self.config.get("top_k", -1),
                "n": 1,
            }
        kwargs["max_new_tokens"] = max_new_tokens
        if "n" not in kwargs or kwargs["n"] > 1:  # group size is supported in preprocess
            kwargs["n"] = 1
        # users can customize different sampling_params at different run
        with self.update_sampling_params(**kwargs):
            output = await self._engine.async_generate(
                input_ids=generation_prompt_ids,
                sampling_params=self.sampling_params,
                return_logprob=False,
            )
        return output

    async def _handle_pending_state(self, _req: AsyncRolloutRequest) -> AsyncRolloutRequest:
        if _req.tools is not None:
            tool_creation_coroutines = []
            for tool_schema in _req.tools:
                tool = self._tool_map[tool_schema.function.name]
                create_kwargs = _req.tools_kwargs[tool.name].get("create_kwargs", {})
                tool_creation_coroutines.append(tool.create(_req.request_id, **create_kwargs))
            await asyncio.gather(*tool_creation_coroutines)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def generate_sequences_with_tools(self, prompts: DataProto, **kwargs) -> DataProto:
        logger.warning(
            "`generate_sequences_with_tools` is deprecated, please use `generate_sequences(...)`",
            DeprecationWarning,
            stacklevel=2,
        )
        return self._req_level_generate_sequences(prompts, **kwargs)

    @GPUMemoryLogger(role="sglang rollout", logger=logger)
    @torch.no_grad()
    def _req_level_generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        # Async rollout with tools support
        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        tgt_device = prompts.batch["input_ids"].device
        if self._tp_rank == 0:
            req_list = self._preprocess_prompt_to_async_rollout_requests(
                prompts,
                n=1 if is_validate else self.config.n,
            )
            loop = asyncio.get_event_loop()
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[self._async_rollout_a_request(req, do_sample, is_validate, **kwargs) for req in req_list],
                )
            )
            sorted_output_req_list = sorted(output_req_list, key=lambda x: (x.batch_data_id, x.rollout_offset))
        else:
            sorted_output_req_list = None

        [sorted_output_req_list] = broadcast_pyobj(
            data=[sorted_output_req_list],
            rank=self._rank,
            dist_group=self._device_mesh_cpu[self._mesh_tp_name].get_group(),
            src=self._device_mesh_cpu[self._mesh_tp_name].mesh[0].item(),
            force_cpu_device=False,
        )
        # Construct the batch data
        prompt_ids, response_ids = [], []
        prompt_attention_mask, response_attention_mask = [], []
        prompt_position_ids, response_position_ids = [], []
        prompt_loss_mask, response_loss_mask = [], []
        messages = []
        reward_scores = []
        for req in sorted_output_req_list:
            assert req.state == AsyncRolloutRequestStateEnum.COMPLETED, f"Request {req.request_id} is not completed"
            assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), f"""Request {req.request_id} has different length of 
                {len(req.input_ids)=}, {len(req.attention_mask)=}, {len(req.position_ids)=}, {len(req.loss_mask)=}"""
            error_message_lines = [
                f"""Request {req.request_id} has input_ids length {len(req.input_ids)}
                    greater than max_model_len {self.config.max_model_len}""",
                f"Decoded input_ids: {self.tokenizer.decode(req.input_ids)}",
                f"Decoded prompt_ids: {self.tokenizer.decode(req.prompt_ids)}",
                f"Decoded response_ids: {self.tokenizer.decode(req.response_ids)}",
                f"Messages: {req.messages}",
                f"Max model length: {req.max_model_len}",
            ]
            error_message = "\n".join(error_message_lines)
            assert len(req.input_ids) <= self.config.max_model_len, error_message

            prompt_ids.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
            response_ids.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
            if len(req.response_ids) > self.config.response_length:
                logger.warning(
                    f"""{req.request_id=} has response_ids length {len(req.response_ids)} 
                    greater than max_response_len {self.config.response_length},\n{req=}"""
                )
            prompt_attention_mask.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
            response_attention_mask.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
            prompt_position_ids.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
            response_position_ids.append(torch.tensor(req.response_position_ids, dtype=torch.int, device=tgt_device))
            prompt_loss_mask.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
            response_loss_mask.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
            messages.append({"messages": req.messages})
            reward_scores.append(req.reward_scores)

        prompt_ids = pad_sequence(
            prompt_ids,
            batch_first=True,
            padding_value=self.pad_token_id,
            padding_side="left",
        )
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)
        response_ids = pad_sequence(response_ids, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)
        prompt_attention_mask = pad_sequence(
            prompt_attention_mask,
            batch_first=True,
            padding_value=0,
            padding_side="left",
        )
        if prompt_attention_mask.shape[1] < self.config.prompt_length:
            prompt_attention_mask = pad_sequence_to_length(prompt_attention_mask, self.config.prompt_length, 0, left_pad=True)
        response_attention_mask = pad_sequence(response_attention_mask, batch_first=True, padding_value=0)
        if response_attention_mask.shape[1] < self.config.response_length:
            response_attention_mask = pad_sequence_to_length(response_attention_mask, self.config.response_length, 0)
        prompt_position_ids = pad_sequence(prompt_position_ids, batch_first=True, padding_value=0, padding_side="left")
        if prompt_position_ids.shape[1] < self.config.prompt_length:
            prompt_position_ids = pad_sequence_to_length(prompt_position_ids, self.config.prompt_length, 0, left_pad=True)
        response_length = response_ids.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=response_ids.device)
        delta_position_id = delta_position_id.unsqueeze(0).repeat(len(sorted_output_req_list), 1)
        response_position_ids = prompt_position_ids[:, -1:] + delta_position_id
        prompt_loss_mask = pad_sequence(prompt_loss_mask, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss_mask.shape[1] < self.config.prompt_length:
            prompt_loss_mask = pad_sequence_to_length(prompt_loss_mask, self.config.prompt_length, 0, left_pad=True)
        response_loss_mask = pad_sequence(response_loss_mask, batch_first=True, padding_value=0)
        if response_loss_mask.shape[1] < self.config.response_length:
            response_loss_mask = pad_sequence_to_length(response_loss_mask, self.config.response_length, 0)

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)
        position_ids = torch.cat((prompt_position_ids, response_position_ids), dim=-1)
        loss_mask = torch.cat((prompt_loss_mask, response_loss_mask), dim=-1)

        # Construct the batch data
        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,  # here input_ids become the whole sentences
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=len(sorted_output_req_list),
        )

        # free cache engine
        if self.config.free_cache_engine and self._engine is not None and self._tp_rank == 0:
            self._engine.flush_cache()
            torch.cuda.empty_cache()

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "messages": np.array(messages),
                "reward_scores": np.array(reward_scores),
            },
        )

    def _preprocess_prompt_to_async_rollout_requests(self, prompts: DataProto, n: int) -> list[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, "need data.return_raw_chat=True, due to no official way do parse_messages"
        req_list = []
        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            for rollout_offset in range(n):
                if self._tool_schemas:
                    _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]
                    _tool_schemas = []
                    for k in _tools_kwargs.keys():
                        _tool_schemas.append(self._tool_map[k].get_openai_tool_schema())
                    prompt_with_chat_template = self.tokenizer.apply_chat_template(
                        conversation=raw_prompt,
                        tools=[tool.model_dump() for tool in _tool_schemas],
                        add_generation_prompt=True,
                        tokenize=False,
                        return_tensors="pt",
                    )
                    input_data = self.tokenizer(
                        prompt_with_chat_template,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    _input_ids = input_data["input_ids"][0].tolist()
                    _attention_mask = input_data["attention_mask"][0].tolist()
                    _position_ids = compute_position_id_with_mask(input_data["attention_mask"][0]).tolist()
                    if len(_input_ids) > self.config.prompt_length:
                        logger.warning(
                            "Prompt {} has length {} greater than max_prompt_len {}",
                            data_idx,
                            len(_input_ids),
                            self.config.prompt_length,
                        )
                        _input_ids = _input_ids[: self.config.prompt_length]
                        _attention_mask = _attention_mask[: self.config.prompt_length]
                        _position_ids = _position_ids[: self.config.prompt_length]
                else:
                    _input_ids = _pre_process_inputs(self.pad_token_id, prompts.batch["input_ids"][data_idx])
                    _attention_mask = _pre_process_inputs(0, prompts.batch["attention_mask"][data_idx])
                    _position_ids = compute_position_id_with_mask(torch.tensor(_attention_mask)).tolist()
                    _tool_schemas = []
                    _tools_kwargs = {}

                req = AsyncRolloutRequest(
                    batch_data_id=data_idx,
                    rollout_offset=rollout_offset,
                    request_id=str(uuid4()),
                    state=AsyncRolloutRequestStateEnum.PENDING,
                    messages=[Message.model_validate(msg) for msg in raw_prompt],
                    tools=_tool_schemas,
                    tools_kwargs=_tools_kwargs,
                    input_ids=_input_ids,
                    prompt_ids=_input_ids,
                    response_ids=[],
                    attention_mask=_attention_mask,
                    prompt_attention_mask=_attention_mask,
                    response_attention_mask=[],
                    position_ids=_position_ids,
                    prompt_position_ids=_position_ids,
                    response_position_ids=[],
                    loss_mask=[0] * len(_input_ids),
                    prompt_loss_mask=[0] * len(_input_ids),
                    response_loss_mask=[],
                    reward_scores={},
                    max_response_len=self.config.response_length,
                    max_model_len=min(
                        self.config.max_model_len,
                        self.config.prompt_length + self.config.response_length,
                    ),
                )

                error_message = f"Request {req.request_id} has mismatched lengths: input_ids={len(req.input_ids)}, attention_mask={len(req.attention_mask)}, position_ids={len(req.position_ids)}, loss_mask={len(req.loss_mask)}"
                assert len(req.input_ids) == len(req.attention_mask) == len(req.position_ids) == len(req.loss_mask), error_message

                req_list.append(req)

        return req_list

    def execute_method(self, method: Union[str, bytes], *args, **kwargs):
        if method == "chat_completion":
            json_request = args[0]

            formatted_messages = []
            for msg in json_request["messages"]:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                formatted_messages.append(f"{role}: {content}")
            prompt_str = "\n".join(formatted_messages)

            sampling_params_dict = {
                "n": json_request.get("n", 1),
                "max_new_tokens": json_request.get("max_completion_tokens", self.config.response_length),
                "temperature": json_request.get("temperature", 1.0),
                "top_p": json_request.get("top_p", 1.0),
            }
            output = None
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=prompt_str,
                        sampling_params=sampling_params_dict,
                        return_logprob=True,
                    )
                )
            output = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu[self._mesh_tp_name].get_group(),
                src=self._device_mesh_cpu[self._mesh_tp_name].mesh[0].item(),
                force_cpu_device=False,
            )

            # only return value from master rank
            if self._tp_rank != 0:
                return None
            # build openai chat completion format
            choices = []
            id = None
            for i, content in enumerate(output):
                choices.append(
                    {
                        "index": i,
                        "message": {
                            "role": "assistant",
                            "content": content["text"],
                        },
                        "finish_reason": content["meta_info"]["finish_reason"]["type"],
                    }
                )
                id = content["meta_info"]["id"]

            return {
                "id": "chatcmpl-" + id,
                "object": "chat.completion",
                "created": int(time.time()),
                "model": json_request.get("model", "sglang_model"),
                "choices": choices,
            }
        else:
            raise ValueError(f"not supported method : {method}")

        # this function is left for uniform train-inference resharding

    def resume(self):
        if not self.is_sleep:
            return
        self.sharding_manager.__enter__()  # pylint: disable=C2801

        self.is_sleep = False

    # this function is left for uniform train-inference resharding
    def offload(self):
        if self.is_sleep:
            return

        self.sharding_manager.__exit__(None, None, None)
        self.is_sleep = True
