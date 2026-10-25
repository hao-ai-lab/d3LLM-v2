# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import logging
import os

import torch
import torch.distributed as dist
from sglang.srt.entrypoints.engine import Engine
from sglang.srt.model_executor.model_runner import LocalSerializedTensor
from sglang.srt.utils import MultiprocessingSerializer
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.fsdp.api import FullStateDictConfig, ShardedStateDictConfig, StateDictType
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

from verl import DataProto
from verl.protocol import all_gather_data_proto
from verl.utils.debug import GPUMemoryLogger, log_gpu_memory_usage
from verl.utils.fsdp_utils import fsdp_version, load_fsdp_model_to_gpu, offload_fsdp_model_to_cpu
from verl.utils.torch_functional import check_cuda_is_available

from .base import BaseShardingManager

# from vllm.distributed import parallel_state as sglang_ps
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _preprocess_tensor_for_update_weights(tensor: torch.Tensor):
    if isinstance(tensor, DTensor):
        return tensor.full_tensor()
    return tensor


def _merge_lora_state_dict(params, lora_scaling):
    """Merge LoRA adapters into base weights and strip PEFT prefix for SGLang.

    Moves tensors to CPU during merge to minimize GPU memory usage.
    Deletes original GPU tensors as we go to avoid holding both copies.
    """
    merged, lora_a, lora_b = {}, {}, {}

    keys = list(params.keys())
    for name in keys:
        tensor = params.pop(name)  # pop to free GPU memory early
        clean = name.removeprefix("base_model.model.")
        if ".lora_A." in clean:
            lora_a[clean.split(".lora_A.")[0]] = tensor.cpu() if tensor.is_cuda else tensor
        elif ".lora_B." in clean:
            lora_b[clean.split(".lora_B.")[0]] = tensor.cpu() if tensor.is_cuda else tensor
        else:
            merged[clean.replace(".base_layer.", ".")] = tensor

    for mod in lora_a:
        key = mod + ".weight"
        if key in merged and mod in lora_b:
            w = merged[key]
            w = w.cpu() if w.is_cuda else w
            A = lora_a[mod].to(dtype=w.dtype)
            B = lora_b[mod].to(dtype=w.dtype)
            merged[key] = w + lora_scaling * (B @ A)

    return merged


class FSDPSGLangShardingManager(BaseShardingManager):
    @check_cuda_is_available()
    def __init__(
        self,
        module: FSDP,
        inference_engine: Engine,
        model_config,
        full_params: bool = False,
        device_mesh: DeviceMesh = None,
        offload_param: bool = False,
        skip_weight_sync: bool = False,
        teacher_model_path: str = None,
    ):
        self.skip_weight_sync = skip_weight_sync
        self.teacher_model_path = teacher_model_path
        self._validation_mode = False
        self._teacher_needs_restore = False
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.offload_param = offload_param

        # Detect PeftModel for LoRA → merged weight sync
        self._is_peft = False
        self._lora_scaling = 1.0
        try:
            from peft import PeftModel
            # FSDP v1: module._fsdp_wrapped_module is PeftModel
            # FSDP v2: module itself is PeftModel (fully_shard wraps in-place)
            unwrapped = module._fsdp_wrapped_module if hasattr(module, '_fsdp_wrapped_module') else module
            # Also check peft_config attr (works for both FSDP v1 and v2)
            if isinstance(unwrapped, PeftModel) or hasattr(unwrapped, 'peft_config'):
                self._is_peft = True
                peft_cfg = unwrapped.peft_config.get('default') if hasattr(unwrapped, 'peft_config') else None
                if peft_cfg:
                    self._lora_scaling = peft_cfg.lora_alpha / peft_cfg.r
                logger.warning(f"[LoRA] Detected PeftModel, lora_scaling={self._lora_scaling:.2f}")
        except ImportError:
            pass

        # Full params — don't force for LoRA; merge handles ShardedTensors on CPU
        self.full_params = full_params
        if self.full_params and fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(self.module, state_dict_type=StateDictType.FULL_STATE_DICT, state_dict_config=FullStateDictConfig())
        elif fsdp_version(self.module) == 1:
            FSDP.set_state_dict_type(
                self.module,
                state_dict_type=StateDictType.SHARDED_STATE_DICT,
                state_dict_config=ShardedStateDictConfig(),
            )

        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        # When True, __enter__/__exit__ skip weight sync and engine reload/unload.
        # Use start_multi_step_rollout() / stop_multi_step_rollout() to manage.
        self._engine_alive = False

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def _get_params(self):
        """Extract model parameters. For LoRA: use FULL_STATE_DICT to get all params
        in one bulk all-gather, then merge LoRA adapters on CPU for SGLang."""
        if self._is_peft and fsdp_version(self.module) == 1:
            # Temporarily switch to FULL_STATE_DICT for LoRA merge (one fast all-gather).
            # Keep on GPU (faster) — the merge moves only LoRA tensors to CPU.
            prev = FSDP.set_state_dict_type(self.module, StateDictType.FULL_STATE_DICT,
                                             FullStateDictConfig(offload_to_cpu=False, rank0_only=False))
            params = self.module.state_dict()
            FSDP.set_state_dict_type(self.module, prev.state_dict_type, prev.state_dict_config)
            return _merge_lora_state_dict(params, self._lora_scaling)
        else:
            params = self.module.state_dict()
            device = torch.cuda.current_device()
            params = {k: v.to(device, non_blocking=True) if fsdp_version(self.module) == 2 else v for k, v in params.items()}
            # Fallback: detect PEFT from state_dict keys
            if any(k.startswith("base_model.model.") for k in list(params.keys())[:5]):
                params = _merge_lora_state_dict(params, self._lora_scaling)
            return params

    @GPUMemoryLogger(role="FSDPSGLangShardingManager enter", logger=logger)
    def __enter__(self):
        if self._engine_alive:
            # Engine already loaded (multi-step rollout mode); skip weight sync/reload.
            # Just swap rng states so generation uses the inference rng.
            if self.device_mesh is not None:
                self.torch_random_states = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(self.gen_random_states)
            return
        torch.cuda.empty_cache()
        if self.skip_weight_sync and not self._validation_mode:
            # Fixed teacher: resume SGLang engine without syncing FSDP weights
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                self.inference_engine.resume_memory_occupation()
            torch.cuda.empty_cache()
        else:
            log_gpu_memory_usage("Before state_dict() in sharding manager memory", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            params = self._get_params()
            log_gpu_memory_usage("After state_dict() in sharding manager memory", logger=logger)
            # Copy, not share memory
            self.update_weights(params)
            log_gpu_memory_usage("After sync model weights in sharding manager", logger=logger)

            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()
            log_gpu_memory_usage("After del state_dict and empty_cache in sharding manager", logger=logger)

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    @GPUMemoryLogger(role="FSDPSGLangShardingManager exit", logger=logger)
    def __exit__(self, exc_type, exc_value, traceback):
        if self._engine_alive:
            # Keep engine loaded; just restore rng state for training code.
            self.module.train()
            torch.cuda.empty_cache()
            if self.device_mesh is not None:
                self.gen_random_states = torch.cuda.get_rng_state()
                torch.cuda.set_rng_state(self.torch_random_states)
            return
        log_gpu_memory_usage("Before SGLang offload in sharding manager", logger=logger)
        self.release_memory()
        log_gpu_memory_usage("After SGLang offload in sharding manager", logger=logger)

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def update_weights(self, params):
        if self.skip_weight_sync:
            # Validation overwrites the fixed teacher, including its CPU backup
            # on the next sleep. Restore from the original checkpoint afterward.
            self._teacher_needs_restore = True
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self.inference_engine.resume_memory_occupation()

        # Most naive implementation, can optimize a lot if it is bottleneck from sglang Engine weight update
        named_tensors = [(k, v) for k, v in params.items()]
        load_format = None
        for tensor_index, (name, tensor) in enumerate(named_tensors):
            serialized_tensor = MultiprocessingSerializer.serialize(_preprocess_tensor_for_update_weights(tensor))

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                gathered_serialized_tensors = [None for _ in range(self.device_mesh["infer_tp"].mesh.size()[0])]
            else:
                gathered_serialized_tensors = None
            dist.gather_object(
                obj=serialized_tensor,
                object_gather_list=gathered_serialized_tensors,
                dst=self.device_mesh["infer_tp"].mesh.tolist()[0],
                group=self.device_mesh["infer_tp"].get_group(),
            )

            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                self.inference_engine.update_weights_from_tensor(
                    named_tensors=[
                        (
                            name,
                            LocalSerializedTensor(values=gathered_serialized_tensors),
                        )
                    ],
                    load_format=load_format,
                    flush_cache=tensor_index == len(named_tensors) - 1,
                )

    def release_memory(self):
        if self.device_mesh["infer_tp"].get_local_rank() == 0:
            self.inference_engine.release_memory_occupation()

    def set_validation_mode(self, enabled: bool):
        """Use the student during validation, then restore the fixed teacher.

        Called around the entire validation pass, while the engine is asleep.
        Reloading the checkpoint avoids retaining a second full CPU weight copy.
        """
        if not self.skip_weight_sync:
            return
        if self._engine_alive:
            raise RuntimeError("Cannot switch validation mode during an active rollout")
        if not self.teacher_model_path:
            raise ValueError("Student validation with a fixed teacher requires teacher_model_path")
        self._validation_mode = enabled
        if not enabled and self._teacher_needs_restore:
            if self.tp_rank == 0:
                self.inference_engine.resume_memory_occupation()
                try:
                    result = self.inference_engine.update_weights_from_disk(
                        model_path=self.teacher_model_path, load_format="auto"
                    )
                    if not result[0]:
                        raise RuntimeError(f"Failed to restore fixed teacher: {result[1]}")
                finally:
                    self.inference_engine.release_memory_occupation()
            self._teacher_needs_restore = False

    def start_multi_step_rollout(self):
        """Sync FSDP weights into SGLang ONCE and keep engine alive for the whole rollout batch.

        Call this before a multi-step rollout loop (e.g., multi_turn_loop). Each
        generate_sequences call inside the loop will skip the expensive weight-sync
        and just reuse the already-loaded engine.
        """
        if self._engine_alive:
            logger.warning("start_multi_step_rollout: engine already alive, skipping")
            return
        torch.cuda.empty_cache()
        if self.skip_weight_sync and not self._validation_mode:
            # Fixed teacher: just resume without weight sync
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            if self.device_mesh["infer_tp"].get_local_rank() == 0:
                self.inference_engine.resume_memory_occupation()
            torch.cuda.empty_cache()
        else:
            # Policy model: sync weights once before rollout loop
            log_gpu_memory_usage("start_multi_step_rollout: before state_dict", logger=logger)
            if self.offload_param:
                load_fsdp_model_to_gpu(self.module)
            params = self._get_params()
            self.update_weights(params)
            del params
            if self.offload_param:
                offload_fsdp_model_to_cpu(self.module)
            torch.cuda.empty_cache()
            log_gpu_memory_usage("start_multi_step_rollout: weights synced", logger=logger)
        self._engine_alive = True

    def stop_multi_step_rollout(self):
        """Release SGLang engine after the multi-step rollout batch is done."""
        if not self._engine_alive:
            logger.warning("stop_multi_step_rollout: engine not alive, skipping")
            return
        self._engine_alive = False
        self.release_memory()
        self.module.train()
        torch.cuda.empty_cache()
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        log_gpu_memory_usage("stop_multi_step_rollout: engine released", logger=logger)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        group = self.device_mesh["infer_tp"].get_group()

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]


class SGLangOnlyShardingManager(BaseShardingManager):
    """Sharding manager for rollout_only mode (no FSDP model).

    SGLang engine starts in sleep state after init (release_memory_occupation was called).
    This manager resumes/releases memory occupation around each generate call,
    without any weight synchronization.
    """

    def __init__(self, inference_engine: Engine, device_mesh: DeviceMesh):
        self.inference_engine = inference_engine
        self.device_mesh = device_mesh
        self.tp_size = self.device_mesh["infer_tp"].size()
        self.tp_rank = self.device_mesh["infer_tp"].get_local_rank()

        self.torch_random_states = torch.cuda.get_rng_state()
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh["dp"].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        if self.tp_rank == 0:
            # Only resume KV cache, weights are already loaded and should stay in GPU
            self.inference_engine.resume_memory_occupation(tags=["kv_cache"])
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        if self.tp_rank == 0:
            # Only release KV cache, keep weights in GPU
            self.inference_engine.release_memory_occupation(tags=["kv_cache"])
        torch.cuda.empty_cache()
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group so each rank has identical input."""
        if self.tp_size == 1:
            return data
        group = self.device_mesh["infer_tp"].get_group()
        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data
        return data.chunk(chunks=self.tp_size)[self.tp_rank]
