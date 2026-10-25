# Copyright 2025 Nanyang Technological University (NTU), Singapore and the verl-agent team.
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
"""
SGLang target rollout for speculative planning (eval-only, rollout-only).
No FSDP model, no dLLM/decode_order, minimal logging. Used only as the
target model for async generate in multi-turn speculative rollout.
"""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import contextmanager
from copy import deepcopy

import numpy as np
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.nn.utils.rnn import pad_sequence
from tensordict import TensorDict

from omegaconf import DictConfig
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.sampling.sampling_params import SamplingParams
from transformers import PreTrainedTokenizer

try:
    from sglang.srt.utils import get_ip, get_open_port
except ImportError:
    from sglang.srt.utils import get_local_ip_auto as get_ip
    from sglang.srt.utils import get_open_port

from verl import DataProto
from verl.utils.net_utils import is_ipv6
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.sglang_rollout.utils import broadcast_pyobj

from .sglang_rollout import _pre_process_inputs, _post_process_outputs

logger = logging.getLogger(__name__)


class SGLangTargetRollout(BaseRollout):
    """
    Eval-only, rollout-only SGLang rollout for speculative planning target.
    Loads only the SGLang engine (no FSDP). Single-turn batch generate only,
    no dLLM/decode_order, no verbose [dLLM DEBUG] / [MEMORY DEBUG] logs.
    """

    def __init__(
        self,
        actor_module: str,
        config: DictConfig,
        tokenizer: PreTrainedTokenizer,
        model_hf_config,
        trust_remote_code: bool = True,
        device_mesh=None,
        port=None,
        **kwargs,
    ):
        super().__init__()
        self.config = deepcopy(config)
        self._device_mesh_cpu = device_mesh
        os.environ.setdefault("SGL_DISABLE_TP_MEMORY_INBALANCE_CHECK", "true")

        self._init_distributed_env(device_mesh_cpu=device_mesh, **kwargs)
        self._verify_config(model_hf_config=model_hf_config)
        self._init_inference_engine(trust_remote_code, actor_module, port)
        self._init_sampling_params(**kwargs)

        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
        self.sharding_manager = None
        self.is_sleep = True

    def _init_distributed_env(self, device_mesh_cpu, **kwargs):
        self._device_mesh_cpu = device_mesh_cpu
        self.tensor_parallel_size = self.config.get("tensor_model_parallel_size", 1)
        assert self.tensor_parallel_size <= dist.get_world_size(), (
            "tensor parallel size should be less than or equal to the world size"
        )
        tp_size = self.tensor_parallel_size
        world_size = int(os.getenv("WORLD_SIZE", "-1"))
        if self._device_mesh_cpu is None:
            self._device_mesh_cpu = init_device_mesh(
                "cpu",
                mesh_shape=(world_size // tp_size, tp_size, 1),
                mesh_dim_names=["dp", "tp", "pp"],
            )
        self._rank = self._device_mesh_cpu.get_rank()
        self._tp_rank = self._device_mesh_cpu["tp"].get_local_rank()
        self._tp_size = self._device_mesh_cpu["tp"].size()
        visible_devices = [None] * self._device_mesh_cpu.size(1)
        torch.distributed.all_gather_object(
            visible_devices, os.environ.get("CUDA_VISIBLE_DEVICES", ""), self._device_mesh_cpu.get_group("tp")
        )
        self.visible_devices_set = set(",".join(visible_devices).split(",")) if visible_devices[0] else set()
        if self.visible_devices_set:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(sorted(list(self.visible_devices_set)))

    def _verify_config(self, model_hf_config):
        if not self.config.get("max_model_len", None):
            self.config.max_model_len = self.config.prompt_length + self.config.response_length
        assert self.config.max_model_len >= self.config.prompt_length + self.config.response_length
        max_pos = getattr(model_hf_config, "max_position_embeddings", None)
        if max_pos is None and hasattr(model_hf_config, "text_config"):
            max_pos = getattr(model_hf_config.text_config, "max_position_embeddings", None)
        if max_pos is not None:
            assert max_pos >= self.config.max_model_len

    def _init_inference_engine(self, trust_remote_code, actor_module, port):
        nnodes = max(1, -(-self._tp_size // max(len(self.visible_devices_set), 1)))
        if nnodes > 1:
            ip = get_ip()
            port = get_open_port() if port is None else port
            [ip, port] = broadcast_pyobj(
                [ip, port],
                rank=self._rank,
                dist_group=self._device_mesh_cpu.get_group("tp"),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            dist_init_addr = f"[{ip}]:{port}" if is_ipv6(ip) else f"{ip}:{port}"
        else:
            dist_init_addr = None

        load_format = "dummy" if getattr(self.config, "load_format", "auto").startswith("dummy") else getattr(self.config, "load_format", "auto")
        tp_size_per_node = self._tp_size // nnodes
        node_rank = self._tp_rank // tp_size_per_node
        first_rank_in_node = self._tp_rank % tp_size_per_node == 0

        if not first_rank_in_node:
            self._engine = None
            return

        rank = dist.get_rank()
        os.environ["SGLANG_BLOCK_NONZERO_RANK_CHILDREN"] = "0"
        self._engine = Engine(
            model_path=actor_module,
            dtype=self.config.dtype,
            mem_fraction_static=self.config.gpu_memory_utilization,
            enable_memory_saver=True,
            disable_cuda_graph=getattr(self.config, "enforce_eager", False),
            base_gpu_id=0,
            gpu_id_step=1,
            tp_size=self._tp_size,
            node_rank=node_rank,
            load_format=load_format,
            dist_init_addr=dist_init_addr,
            nnodes=nnodes,
            trust_remote_code=trust_remote_code,
            port=30000 + rank,
        )
        if self.config.get("rollout_only", False):
            self._engine.release_memory_occupation(tags=["kv_cache"])

    def _init_sampling_params(self, **kwargs):
        params = dict(
            n=1,
            max_new_tokens=self.config.response_length,
            presence_penalty=0.0,
            frequency_penalty=0.0,
            repetition_penalty=1.0,
        )
        for k in self.config.keys():
            if hasattr(SamplingParams(), str(k)):
                params[k] = self.config.get(k)
        self.sampling_params = params

    @property
    def inference_engine(self):
        return self._engine

    @contextmanager
    def update_sampling_params(self, **kwargs):
        old = {k: self.sampling_params[k] for k in kwargs if k in self.sampling_params}
        for k, v in kwargs.items():
            if k in self.sampling_params:
                self.sampling_params[k] = v
        try:
            yield
        finally:
            for k, v in old.items():
                self.sampling_params[k] = v

    def resume(self):
        self.is_sleep = False

    def offload(self):
        self.is_sleep = True

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto, **kwargs) -> DataProto:
        idx = prompts.batch["input_ids"]
        attention_mask = prompts.batch["attention_mask"]
        position_ids = prompts.batch["position_ids"]
        eos_token_id = prompts.meta_info["eos_token_id"]
        batch_size = idx.size(0)
        non_tensor_batch = prompts.non_tensor_batch

        if "raw_prompt_ids" not in non_tensor_batch:
            non_tensor_batch["raw_prompt_ids"] = np.array(
                [_pre_process_inputs(self.pad_token_id, idx[i]) for i in range(batch_size)],
                dtype=object,
            )

        if "multi_modal_data" in non_tensor_batch:
            sglang_inputs = [
                {
                    "prompt_token_ids": raw_prompt_ids,
                    "multi_modal_data": mmd,
                    "image_data": mmd.get("image", None) if isinstance(mmd, dict) else None,
                }
                for raw_prompt_ids, mmd in zip(
                    non_tensor_batch.pop("raw_prompt_ids"),
                    non_tensor_batch.pop("multi_modal_data"),
                )
            ]
        else:
            sglang_inputs = [{"prompt_token_ids": rid} for rid in non_tensor_batch.pop("raw_prompt_ids")]

        for inp in sglang_inputs:
            if isinstance(inp["prompt_token_ids"], np.ndarray):
                inp["prompt_token_ids"] = inp["prompt_token_ids"].tolist()

        idx_list = [inp["prompt_token_ids"] for inp in sglang_inputs]
        image_list = [inp.get("image_data", None) for inp in sglang_inputs]

        do_sample = prompts.meta_info.get("do_sample", True)
        is_validate = prompts.meta_info.get("validate", False)
        if not do_sample:
            gen_kwargs = dict(
                n=1, presence_penalty=0.0, frequency_penalty=0.0, repetition_penalty=1.0,
                temperature=0, top_p=1, top_k=-1, ignore_eos=False, min_new_tokens=0,
                max_new_tokens=self.config.response_length, skip_special_tokens=True, spaces_between_special_tokens=True,
            )
        elif is_validate:
            gen_kwargs = dict(
                top_k=self.config.val_kwargs.top_k,
                top_p=self.config.val_kwargs.top_p,
                temperature=self.config.val_kwargs.temperature,
                n=1,
            )
        else:
            gen_kwargs = dict(temperature=self.config.get("temperature", 1.0), n=1)

        with self.update_sampling_params(**gen_kwargs):
            if self._tp_rank == 0:
                loop = asyncio.get_event_loop()
                output = loop.run_until_complete(
                    self._engine.async_generate(
                        prompt=None,
                        sampling_params=self.sampling_params,
                        return_logprob=True,
                        input_ids=idx_list,
                        image_data=image_list,
                    )
                )
            else:
                output = None

            [output] = broadcast_pyobj(
                data=[output],
                rank=self._rank,
                dist_group=self._device_mesh_cpu["tp"].get_group(),
                src=self._device_mesh_cpu["tp"].mesh[0].item(),
                force_cpu_device=False,
            )
            out = _post_process_outputs(self.tokenizer, output)
            response = out[0].to(idx.device)
            rollout_log_probs = out[1].to(idx.device)

        if response.shape[1] < self.config.response_length:
            response = pad_sequence_to_length(response, self.config.response_length, self.pad_token_id)
            rollout_log_probs = pad_sequence_to_length(rollout_log_probs, self.config.response_length, self.pad_token_id)

        if self.sampling_params.get("n", 1) > 1 and do_sample:
            n = self.sampling_params["n"]
            idx = idx.repeat_interleave(n, dim=0)
            attention_mask = attention_mask.repeat_interleave(n, dim=0)
            position_ids = position_ids.repeat_interleave(n, dim=0)
            batch_size = batch_size * n
            non_tensor_batch = {k: np.repeat(v, n, axis=0) for k, v in non_tensor_batch.items()}

        seq = torch.cat([idx, response], dim=-1)
        response_length = response.size(1)
        delta_position_id = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).repeat(batch_size, 1)
        if position_ids.dim() == 3:
            delta_position_id = delta_position_id.view(batch_size, 1, -1).expand(batch_size, position_ids.size(1), -1)
        response_position_ids = position_ids[:, -1:] + delta_position_id
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)
        response_attention_mask = get_response_mask(response_id=response, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": idx,
                "responses": response,
                "input_ids": seq,
                "rollout_log_probs": rollout_log_probs,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )

        if self.config.get("free_cache_engine", False) and self._engine is not None:
            self._engine.flush_cache()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        return DataProto(batch=batch, non_tensor_batch=non_tensor_batch)
