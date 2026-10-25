# Copyright 2025 Shanghai AI Lab
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
The main entry point to run the PPO algorithm
"""

import logging
import os
import warnings
from typing import Union

import psutil
import torch
import torch.distributed

torch._dynamo.config.optimize_ddp = False
from codetiming import Timer
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.distributed.device_mesh import init_device_mesh

import verl.utils.torch_functional as verl_F
from verl.utils.py_functional import convert_to_regular_types
from verl import DataProto
from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.single_controller.base import Worker
from verl.single_controller.base.decorator import Dispatch, register
from verl.utils import hf_processor, hf_tokenizer
from verl.utils.activation_offload import enable_activation_offloading
from verl.utils.checkpoint.fsdp_checkpoint_manager import FSDPCheckpointManager
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.flops_counter import FlopsCounter
from verl.utils.fs import copy_to_local
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    fsdp_version,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
    load_fsdp_model_to_gpu,
    load_fsdp_optimizer,
    offload_fsdp_model_to_cpu,
    offload_fsdp_optimizer,
    layered_summon_lora_params,
)
from verl.utils.import_utils import import_external_libs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.sharding_manager.fsdp_ulysses import FSDPUlyssesShardingManager
from verl.utils.device import (
    get_device_name,
    get_torch_device,
    is_cuda_available,
    is_npu_available,
)
from verl.workers.fsdp_workers import get_sharding_strategy, ActorRolloutRefWorker
from verl.workers.actor.dllm_actor import DLLMDataParallelPPOActor

from peft import LoraConfig, TaskType, get_peft_model
from codetiming import Timer

import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from peft import PeftModel
from safetensors.torch import save_file
from dataclasses import asdict
import json
from tensordict import TensorDict

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

device_name = get_device_name()


def _merge_dllm_config(parent_config, target_config):
    """Merge ``parent_config.dllm_config`` into *target_config* (actor or ref).

    Fields defined in ``dllm_config`` are written into *target_config* so that
    ``DLLMDataParallelPPOActor`` sees them when it reads its config.  Any key
    that is already present in *target_config* (i.e. explicitly overridden at the
    actor/ref level) is **not** overwritten – the per-role value takes precedence.

    This must be called inside an ``open_dict`` context so that struct-mode
    OmegaConf configs accept new keys.
    """
    dllm_cfg = parent_config.get("dllm_config", None)
    if dllm_cfg is None:
        return
    for key, value in OmegaConf.to_container(dllm_cfg, resolve=True).items():
        if target_config.get(key) is None:
            target_config[key] = value


def _is_b200() -> bool:
    """Return True only on B200/NVL72 (compute capability >= 10.0)."""
    try:
        import torch

        return torch.cuda.get_device_capability()[0] >= 10
    except Exception:
        return False


class DLLMActorRolloutRefWorker(ActorRolloutRefWorker):
    @staticmethod
    def _disable_triton_kernels(model):
        """Disable liger-kernel / flash-attn Triton ops that produce NaN on GB200/NVL72.

        Patches SDARMLP.forward to use standard PyTorch SwiGLU and
        SDARRMSNorm.forward to use standard PyTorch RMSNorm.
        """
        import types

        patched_mlp = patched_rms = 0
        for module in model.modules():
            name = type(module).__name__
            if name == "SDARMLP":

                def _std_mlp_forward(self, x):
                    return self.down_proj(
                        self.act_fn(self.gate_proj(x)) * self.up_proj(x)
                    )

                module.forward = types.MethodType(_std_mlp_forward, module)
                patched_mlp += 1
            elif name == "SDARRMSNorm":

                def _std_rms_forward(self, hidden_states):
                    input_dtype = hidden_states.dtype
                    hidden_states = hidden_states.to(torch.float32)
                    variance = hidden_states.pow(2).mean(-1, keepdim=True)
                    hidden_states = hidden_states * torch.rsqrt(
                        variance + self.variance_epsilon
                    )
                    return self.weight * hidden_states.to(input_dtype)

                module.forward = types.MethodType(_std_rms_forward, module)
                patched_rms += 1
        if patched_mlp or patched_rms:
            logger.warning(
                f"[init_model] Disabled Triton kernels: {patched_mlp} MLP, {patched_rms} RMSNorm"
            )

    def _build_model_optimizer(
        self,
        model_path,
        fsdp_config,
        optim_config,
        override_model_config,
        use_remove_padding=False,
        use_fused_kernels=False,
        enable_gradient_checkpointing=False,
        trust_remote_code=False,
        use_liger=False,
        role="actor",
        enable_activation_offload=False,
        attn_implementation="eager",  # Safe default; override via config for models that support flash-attn
    ):
        from torch import optim
        from torch.distributed.fsdp import CPUOffload, MixedPrecision
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from transformers import AutoConfig, AutoModel, AutoModelForCausalLM

        # transformers v5.3+ 移除 AutoModelForVision2Seq，改用 AutoModelForImageTextToText；v4 仍保留旧名
        try:
            from transformers import AutoModelForVision2Seq
        except ImportError:
            try:
                from transformers import (
                    AutoModelForImageTextToText as AutoModelForVision2Seq,
                )
            except ImportError:
                AutoModelForVision2Seq = None

        from verl.utils.model import (
            get_generation_config,
            print_model_size,
            update_model_config,
        )
        from verl.utils.torch_dtypes import PrecisionType

        assert role in ["actor", "ref"]

        log_gpu_memory_usage(f"Before init {role} from HF AutoModel", logger=logger)
        local_path = model_path

        # note that we have to create model in fp32. Otherwise, the optimizer is in bf16, which is incorrect
        # TODO(zhangchi.usc1992): 1. support create from random initialized model. 2. Support init with FSDP directly
        self.tokenizer = hf_tokenizer(local_path, trust_remote_code=trust_remote_code)
        self.processor = hf_processor(local_path, trust_remote_code=trust_remote_code)

        torch_dtype = fsdp_config.get("model_dtype", None)
        if torch_dtype is None:
            torch_dtype = torch.float32 if self._is_actor else torch.bfloat16
        else:
            torch_dtype = PrecisionType.to_dtype(torch_dtype)

        # override model kwargs
        actor_model_config = AutoConfig.from_pretrained(
            local_path,
            trust_remote_code=trust_remote_code,
            attn_implementation=attn_implementation,
        )  # LNY: LLaDA does not support flash-attn?

        # patch for kimi-vl
        if getattr(actor_model_config, "model_type", None) == "kimi_vl":
            actor_model_config.text_config.topk_method = "greedy"

        self.generation_config = get_generation_config(
            local_path, trust_remote_code=trust_remote_code
        )

        override_config_kwargs = {
            "bos_token_id": self.tokenizer.bos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        if getattr(self.tokenizer, "mask_token_id", None) is not None:
            override_config_kwargs["mask_token_id"] = self.tokenizer.mask_token_id
        override_config_kwargs.update(override_model_config)
        update_model_config(
            actor_model_config, override_config_kwargs=override_config_kwargs
        )
        if self.rank == 0:
            print(f"Model config after override: {actor_model_config}")

        # NOTE(fix me): tie_word_embedding causes meta_tensor init to hang
        init_context = get_init_weight_context_manager(
            use_meta_tensor=not actor_model_config.tie_word_embeddings,
            mesh=self.device_mesh,
        )

        with init_context(), warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if (
                AutoModelForVision2Seq is not None
                and type(actor_model_config)
                in AutoModelForVision2Seq._model_mapping.keys()
            ):
                actor_module_class = AutoModelForVision2Seq
            else:
                # actor_module_class = AutoModelForCausalLM # AutoModel
                model_name = self.config.model.name
                if model_name in ["sdar"]:
                    actor_module_class = AutoModelForCausalLM
                elif model_name in ["dream", "llada"]:
                    actor_module_class = AutoModel
                else:
                    try:
                        actor_module_class = AutoModelForCausalLM
                    except Exception:
                        actor_module_class = AutoModel

            # # LNY: ref model enable quantization
            # bnb_config = None
            # if role == "ref":
            #     from transformers import BitsAndBytesConfig
            #     bnb_config = BitsAndBytesConfig(
            #         load_in_4bit=True,
            #         bnb_4bit_use_double_quant=True,
            #         bnb_4bit_quant_type="nf4",
            #         bnb_4bit_compute_dtype=torch.bfloat16,
            #     )

            actor_module = actor_module_class.from_pretrained(
                pretrained_model_name_or_path=local_path,
                torch_dtype=torch_dtype,
                config=actor_model_config,
                trust_remote_code=trust_remote_code,
                # quantization_config=bnb_config,  # LNY: ref model enable quantization
            )

            # Apply Liger kernel to the model if use_liger is set to True
            if use_liger:
                from liger_kernel.transformers.monkey_patch import (
                    _apply_liger_kernel_to_instance,
                )

                _apply_liger_kernel_to_instance(model=actor_module)

            apply_monkey_patch(
                model=actor_module,
                use_remove_padding=use_remove_padding,
                ulysses_sp_size=self.ulysses_sequence_parallel_size,
                use_fused_kernels=use_fused_kernels,
            )

            # some parameters may not in torch_dtype. TODO(zhangchi.usc1992) remove this after we switch to fsdp2
            actor_module.to(torch_dtype)

            model_name = self.config.model.get("name", "")
            if enable_gradient_checkpointing:
                # LLaDA uses a custom ActivationCheckpointingStrategy instead of the
                # standard HuggingFace gradient_checkpointing flag.  It also does not
                # advertise supports_gradient_checkpointing, so calling the HF API
                # would raise ValueError.  We route LLMs to their own path here.
                if model_name in ["llada"]:
                    from verl.models.d3LLM_LLaDA.modeling_llada import (
                        ActivationCheckpointingStrategy,
                    )
                    actor_module.model.set_activation_checkpointing(
                        ActivationCheckpointingStrategy.whole_layer
                    )
                else:
                    actor_module.gradient_checkpointing_enable(
                        gradient_checkpointing_kwargs={"use_reentrant": False}
                    )
            if self._is_lora:
                print("Applying LoRA to actor module")
                actor_module.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    "task_type": TaskType.CAUSAL_LM,
                    "r": self.config.model.lora_rank,
                    "lora_alpha": self.config.model.lora_alpha,
                    "target_modules": convert_to_regular_types(
                        self.config.model.target_modules
                    ),
                    "bias": "none",
                    "lora_dropout": self.config.model.get("lora_dropout", 0.0),
                }
                actor_module = get_peft_model(actor_module, LoraConfig(**lora_config))
                actor_module.print_trainable_parameters()

                # LNY: After injecting LoRA, ensure all parameters use torch_dtype
                for param in actor_module.parameters():
                    if param.dtype != torch_dtype:
                        param.data = param.data.to(torch_dtype)
        torch.distributed.barrier()

        if self.rank == 0:
            print_model_size(actor_module)

        log_gpu_memory_usage(f"After init {role} from HF AutoModel", logger=logger)

        # We wrap FSDP for rollout as well
        mixed_precision_config = fsdp_config.get("mixed_precision", None)
        if mixed_precision_config is not None:
            param_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("param_dtype", "bf16")
            )
            reduce_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("reduce_dtype", "fp32")
            )
            buffer_dtype = PrecisionType.to_dtype(
                mixed_precision_config.get("buffer_dtype", "fp32")
            )
        else:
            param_dtype = torch.bfloat16
            reduce_dtype = torch.float32
            buffer_dtype = torch.float32

        mixed_precision = MixedPrecision(
            param_dtype=param_dtype,
            reduce_dtype=reduce_dtype,
            buffer_dtype=buffer_dtype,
        )

        auto_wrap_policy = get_fsdp_wrap_policy(
            module=actor_module,
            config=fsdp_config.get("wrap_policy", None),
            is_lora=self.config.model.get("lora_rank", 0) > 0,
        )

        print(f"wrap_policy: {auto_wrap_policy}")

        fsdp_mesh = self.device_mesh
        sharding_strategy = get_sharding_strategy(fsdp_mesh)

        fsdp_strategy = self.config.actor.strategy
        if fsdp_strategy == "fsdp":
            if role == "actor":
                # Actor does NOT use FSDP CPUOffload — it conflicts with manual
                # optimizer_offload (load_fsdp_optimizer / offload_fsdp_optimizer).
                # When FSDP CPUOffload offloads params to CPU between clip_grad_norm_
                # and optimizer.step(), Adam sees optimizer state on GPU but params
                # on CPU, causing a device mismatch. Instead, param offloading is
                # handled manually via _is_offload_param (read from config by parent).
                cpu_offload = None
            else:
                # ref: default True matches legacy "always offload ref" behavior.
                use_param_offload = bool(fsdp_config.get("param_offload", True))
                cpu_offload = (
                    CPUOffload(offload_params=True) if use_param_offload else None
                )
            actor_module_fsdp = FSDP(
                actor_module,
                cpu_offload=cpu_offload,
                param_init_fn=init_fn,
                use_orig_params=self._is_lora,  # LNY: When using LoRA, use_orig_params=True must be set, because the LoRA adapter needs to access the original parameters
                auto_wrap_policy=auto_wrap_policy,
                device_id=get_torch_device().current_device(),
                sharding_strategy=sharding_strategy,  # zero3
                mixed_precision=mixed_precision,
                sync_module_states=True,
                device_mesh=self.device_mesh,
                forward_prefetch=False,
            )
        elif fsdp_strategy == "fsdp2":
            assert (
                CPUOffloadPolicy is not None
            ), "PyTorch version >= 2.4 is required for using fully_shard API (FSDP2)"
            mp_policy = MixedPrecisionPolicy(
                param_dtype=param_dtype,
                reduce_dtype=reduce_dtype,
                cast_forward_inputs=True,
            )
            if role == "actor" and fsdp_config.offload_policy:
                cpu_offload = CPUOffloadPolicy(pin_memory=True)
                self._is_offload_param = False
                self._is_offload_optimizer = False
            else:
                cpu_offload = (
                    None if role == "actor" else CPUOffloadPolicy(pin_memory=True)
                )

            fsdp_kwargs = {
                "mesh": fsdp_mesh,
                "mp_policy": mp_policy,
                "offload_policy": cpu_offload,
                "reshard_after_forward": fsdp_config.reshard_after_forward,
            }
            full_state = actor_module.state_dict()
            apply_fsdp2(actor_module, fsdp_kwargs, fsdp_config)
            fsdp2_load_full_state_dict(actor_module, full_state, fsdp_mesh, cpu_offload)
            actor_module_fsdp = actor_module
        else:
            raise NotImplementedError(f"not implement {fsdp_strategy}")

        if enable_activation_offload:
            enable_activation_offloading(
                actor_module_fsdp, fsdp_strategy, enable_gradient_checkpointing
            )

        log_gpu_memory_usage(f"After {role} FSDP init", logger=logger)

        # TODO: add more optimizer args into config
        if role == "actor" and optim_config is not None:
            from verl.utils.torch_functional import (
                get_constant_schedule_with_warmup,
                get_cosine_schedule_with_warmup,
            )

            actor_optimizer = optim.AdamW(
                actor_module_fsdp.parameters(),
                lr=optim_config.lr,
                betas=optim_config.get("betas", (0.9, 0.999)),
                weight_decay=optim_config.get("weight_decay", 1e-2),
            )

            total_steps = optim_config.get("total_training_steps", 0)
            num_warmup_steps = int(optim_config.get("lr_warmup_steps", -1))
            warmup_style = optim_config.get("warmup_style", "constant")
            min_lr_ratio = optim_config.get("min_lr_ratio", 0.0)
            num_cycles = optim_config.get("num_cycles", 0.5)
            if num_warmup_steps < 0:
                num_warmup_steps_ratio = optim_config.get("lr_warmup_steps_ratio", 0.0)
                num_warmup_steps = int(num_warmup_steps_ratio * total_steps)

            print(f"Total steps: {total_steps}, num_warmup_steps: {num_warmup_steps}")

            if warmup_style == "constant":
                actor_lr_scheduler = get_constant_schedule_with_warmup(
                    optimizer=actor_optimizer, num_warmup_steps=num_warmup_steps
                )
            elif warmup_style == "cosine":
                actor_lr_scheduler = get_cosine_schedule_with_warmup(
                    optimizer=actor_optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=total_steps,
                    min_lr_ratio=min_lr_ratio,
                    num_cycles=num_cycles,
                )
            else:
                raise NotImplementedError(
                    f"Warmup style {warmup_style} is not supported"
                )

            log_gpu_memory_usage(f"After {role} optimizer init", logger=logger)
        else:
            actor_optimizer = None
            actor_lr_scheduler = None

        return (
            actor_module_fsdp,
            actor_optimizer,
            actor_lr_scheduler,
            actor_model_config,
        )

    def _build_rollout(self, trust_remote_code=False):
        from torch.distributed.device_mesh import init_device_mesh

        # TODO(sgm): support FSDP hybrid shard for larger model
        infer_tp = self.config.rollout.tensor_model_parallel_size
        dp = self.world_size // infer_tp
        assert (
            self.world_size % infer_tp == 0
        ), f"rollout world_size: {self.world_size} is not divisible by infer_tp: {infer_tp}"
        rollout_device_mesh = init_device_mesh(
            device_name, mesh_shape=(dp, infer_tp), mesh_dim_names=["dp", "infer_tp"]
        )
        rollout_name = self.config.rollout.name
        if rollout_name == "vllm":  # !!! We have not yet adapted dLLM to vllm !!!
            from verl.workers.rollout.vllm_rollout import vllm_mode, vLLMRollout
            from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager

            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            local_path = copy_to_local(
                self.config.model.path, use_shm=self.config.model.get("use_shm", False)
            )
            lora_kwargs = (
                {
                    "lora_kwargs": {
                        "enable_lora": True,
                        "max_loras": 1,
                        "max_lora_rank": self._lora_rank,
                    }
                }
                if self._is_lora
                else {}
            )
            # lora_kwargs = {}
            if vllm_mode == "customized":
                rollout = vLLMRollout(
                    actor_module=self.actor_module_fsdp,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            elif vllm_mode == "spmd":
                from verl.workers.rollout.vllm_rollout import vLLMAsyncRollout

                vllm_rollout_cls = (
                    vLLMRollout
                    if self.config.rollout.mode == "sync"
                    else vLLMAsyncRollout
                )
                rollout = vllm_rollout_cls(
                    model_path=local_path,
                    config=self.config.rollout,
                    tokenizer=self.tokenizer,
                    model_hf_config=self.actor_model_config,
                    device_mesh=rollout_device_mesh,
                    trust_remote_code=trust_remote_code,
                    **lora_kwargs,
                )
            else:
                raise NotImplementedError("vllm_mode must be 'customized' or 'spmd'")

            log_gpu_memory_usage(
                f"After building {rollout_name} rollout", logger=logger
            )
            full_params = torch.distributed.get_world_size() == 1
            rollout_sharding_manager = FSDPVLLMShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout.inference_engine,
                model_config=self.actor_model_config,
                full_params=full_params,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                load_format=self.config.rollout.load_format,
                layered_summon=self.config.rollout.get("layered_summon", False),
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        elif rollout_name in ["sglang", "sglang_async"]:
            if rollout_name == "sglang_async":
                warnings.warn(
                    "'sglang_async' has been deprecated and merged into 'sglang'. "
                    "Please use 'sglang' going forward.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            # NOTE(linjunrong): Due to recent fp8 support in SGLang. Now importing any symbol relate to
            # SGLang's model_runner would check CUDA device capability. However, due to verl's setting,
            # the main process of ray can not find any CUDA device, which would potentially lead to:
            # "RuntimeError: No CUDA GPUs are available".
            # For this reason, sharding_manager.__init__ should not import FSDPSGLangShardingManager and
            # we import it here use the abs path.
            # check: https://github.com/sgl-project/sglang/blob/00f42707eaddfc2c0528e5b1e0094025c640b7a0/python/sglang/srt/layers/quantization/fp8_utils.py#L76
            from verl.workers.sharding_manager.fsdp_sglang import (
                FSDPSGLangShardingManager,
            )

            from verl.workers.rollout.sglang_rollout.sglang_rollout_new import (
                SGLangRollout,
            )

            if self.config.model.name == "sdar":
                with open_dict(self.config.rollout):
                    if self.config.rollout.get("mask_token_id") is None:
                        self.config.rollout.mask_token_id = 151669
                    if self.config.rollout.get("block_length") is None:
                        self.config.rollout.block_length = 4
                    if self.config.rollout.get("dllm_algorithm") is None:
                        self.config.rollout.dllm_algorithm = "LowConfidence"

            local_path = copy_to_local(
                self.config.model.path, use_shm=self.config.model.get("use_shm", False)
            )
            log_gpu_memory_usage(
                f"Before building {rollout_name} rollout", logger=logger
            )
            rollout = SGLangRollout(
                actor_module=local_path,
                config=self.config.rollout,
                tokenizer=self.tokenizer,
                model_hf_config=self.actor_model_config,
                trust_remote_code=trust_remote_code,
                device_mesh=rollout_device_mesh,
            )
            log_gpu_memory_usage(
                f"After building {rollout_name} rollout", logger=logger
            )

            if torch.distributed.get_world_size() == 1:
                self.config.rollout.load_format = "dummy_hf"
            rollout_sharding_manager = FSDPSGLangShardingManager(
                module=self.actor_module_fsdp,
                inference_engine=rollout._engine,
                model_config=self.actor_model_config,
                full_params="hf" in self.config.rollout.load_format,
                device_mesh=rollout_device_mesh,
                offload_param=self._is_offload_param,
                skip_weight_sync=self.config.rollout.get("fixed_teacher", False),
                teacher_model_path=local_path,
            )
            log_gpu_memory_usage("After building sharding manager", logger=logger)

        else:
            raise NotImplementedError(
                f"rollout.name must be 'sglang' or 'vllm' for dLLM, got {rollout_name!r}."
            )

        return rollout, rollout_sharding_manager

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def init_model(self):
        # This is used to import external_lib into the huggingface systems
        import_external_libs(self.config.model.get("external_lib", None))

        from omegaconf import OmegaConf

        override_model_config = OmegaConf.to_container(
            self.config.model.get("override_config", OmegaConf.create())
        )

        use_remove_padding = self.config.model.get("use_remove_padding", False)
        use_shm = self.config.model.get("use_shm", False)
        use_fused_kernels = self.config.model.get("use_fused_kernels", False)

        if self._is_actor or self._is_rollout:
            # we need the model for actor and rollout
            if self._is_actor:
                optim_config = self.config.actor.optim
                fsdp_config = self.config.actor.fsdp_config
            else:
                optim_config = None
                fsdp_config = OmegaConf.create()

            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            (
                self.actor_module_fsdp,
                self.actor_optimizer,
                self.actor_lr_scheduler,
                self.actor_model_config,
            ) = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=fsdp_config,
                optim_config=optim_config,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                enable_gradient_checkpointing=self.config.model.get(
                    "enable_gradient_checkpointing", False
                ),
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="actor",
                enable_activation_offload=self.config.model.get(
                    "enable_activation_offload", False
                ),
                attn_implementation=self.config.model.get(
                    "attn_implementation", "eager"
                ),
            )
            print(
                f"[Actor] attn_implementation="
                f"{self.config.model.get('attn_implementation', 'eager')}"
            )

            # SDAR: liger/Triton SwiGLU & flash RMSNorm can NaN on GB200/NVL72 — use PyTorch 路径
            if getattr(self.config.model, "name", None) == "sdar" and _is_b200():
                print(
                    "\033[31m[SDAR] B200 detected — disabling Triton kernels for actor_module_fsdp\033[0m"
                )
                self._disable_triton_kernels(self.actor_module_fsdp)
            elif getattr(self.config.model, "name", None) == "sdar":
                print(
                    "\033[32m[SDAR] non-B200 GPU — Triton kernels kept enabled for actor_module_fsdp ✓\033[0m"
                )

            # get the original unwrapped module
            if fsdp_version(self.actor_module_fsdp) == 1:
                self.actor_module = self.actor_module_fsdp._fsdp_wrapped_module

            if self._is_offload_param:
                offload_fsdp_model_to_cpu(self.actor_module_fsdp)
                log_gpu_memory_usage(
                    "After offload actor model during init", logger=logger
                )

            if self._is_offload_optimizer:
                offload_fsdp_optimizer(optimizer=self.actor_optimizer)
                log_gpu_memory_usage(
                    "After offload actor optimizer during init", logger=logger
                )
        # load from checkpoint
        if self._is_actor:
            OmegaConf.set_struct(self.config.actor, True)
            with open_dict(self.config.actor):
                self.config.actor.use_remove_padding = use_remove_padding
                self.config.actor.use_fused_kernels = use_fused_kernels
                _merge_dllm_config(self.config, self.config.actor)
            actor_type = self.config.actor.get("actor_type", None)
            if actor_type == "distill":
                from verl.workers.actor.distill_actor import DistillDLLMActor
                actor_cls = DistillDLLMActor
            elif actor_type == "rl":
                actor_cls = DLLMDataParallelPPOActor
            else:
                raise ValueError(
                    f"dllm_config.actor_type must be 'rl' or 'distill', got {actor_type!r}."
                )
            self.actor = actor_cls(
                config=self.config.actor,
                actor_module=self.actor_module_fsdp,
                actor_optimizer=self.actor_optimizer,
            )
            self.actor.tokenizer = self.tokenizer

        if self._is_rollout:
            self.rollout, self.rollout_sharding_manager = self._build_rollout(
                trust_remote_code=self.config.model.get("trust_remote_code", False)
            )

        if self._is_ref:
            local_path = copy_to_local(self.config.model.path, use_shm=use_shm)
            self.ref_module_fsdp = self._build_model_optimizer(
                model_path=local_path,
                fsdp_config=self.config.ref.fsdp_config,
                optim_config=None,
                override_model_config=override_model_config,
                use_remove_padding=use_remove_padding,
                use_fused_kernels=use_fused_kernels,
                trust_remote_code=self.config.model.get("trust_remote_code", False),
                use_liger=self.config.model.get("use_liger", False),
                role="ref",
                attn_implementation=self.config.model.get(
                    "attn_implementation", "eager"
                ),
            )[0]
            print(
                f"[Ref] attn_implementation="
                f"{self.config.model.get('attn_implementation', 'eager')}"
            )
            if getattr(self.config.model, "name", None) == "sdar" and _is_b200():
                print(
                    "\033[31m[SDAR] B200 detected — disabling Triton kernels for ref_module_fsdp\033[0m"
                )
                self._disable_triton_kernels(self.ref_module_fsdp)
            elif getattr(self.config.model, "name", None) == "sdar":
                print(
                    "\033[32m[SDAR] non-B200 GPU — Triton kernels kept enabled for ref_module_fsdp ✓\033[0m"
                )
            OmegaConf.set_struct(self.config.ref, True)
            with open_dict(self.config.ref):
                self.config.ref.use_remove_padding = use_remove_padding
                self.config.ref.use_fused_kernels = use_fused_kernels
                # Keep ref log-prob granularity consistent with actor so that
                # KL(actor || ref) is computed at the same (block vs. seq) level.
                self.config.ref.use_block_log_prob = self.config.actor.get(
                    "use_block_log_prob", False
                )
                _merge_dllm_config(self.config, self.config.ref)
            self.ref_policy = DLLMDataParallelPPOActor(
                config=self.config.ref, actor_module=self.ref_module_fsdp
            )

        if self._is_actor:
            self.flops_counter = FlopsCounter(self.actor_model_config)
            self.checkpoint_manager = FSDPCheckpointManager(
                model=self.actor_module_fsdp,
                optimizer=self.actor.actor_optimizer,
                lr_scheduler=self.actor_lr_scheduler,
                processing_class=(
                    self.processor if self.processor is not None else self.tokenizer
                ),
                checkpoint_contents=self.config.actor.checkpoint.contents,
            )

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def update_actor(self, data: DataProto):
        # Support all hardwares
        # data = data.to(get_torch_device().current_device())

        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)
        if self._is_offload_optimizer:
            load_fsdp_optimizer(
                optimizer=self.actor_optimizer,
                device_id=get_torch_device().current_device(),
            )

        with self.ulysses_sharding_manager:
            # data = self.ulysses_sharding_manager.preprocess_data(data=data)
            data = data.to(
                "cpu"
            )  # data will to device with each micro batch on actor.update_policy
            data.meta_info.setdefault("pad_token_id", self.tokenizer.pad_token_id)
            # perform training
            with Timer(name="update_policy", logger=None) as timer:
                metrics = self.actor.update_policy(data=data)
            delta_time = timer.last
            global_num_tokens = data.meta_info["global_token_num"]
            estimated_flops, promised_flops = self.flops_counter.estimate_flops(
                global_num_tokens, delta_time
            )
            metrics["perf/mfu/actor"] = (
                estimated_flops
                * self.config.actor.ppo_epochs
                / promised_flops
                / self.world_size
            )
            metrics["perf/max_memory_allocated_gb"] = (
                get_torch_device().max_memory_allocated() / (1024**3)
            )
            metrics["perf/max_memory_reserved_gb"] = (
                get_torch_device().max_memory_reserved() / (1024**3)
            )
            metrics["perf/cpu_memory_used_gb"] = psutil.virtual_memory().used / (
                1024**3
            )

            lr = self.actor_lr_scheduler.get_last_lr()[0]
            metrics["actor/lr"] = lr
            self.actor_lr_scheduler.step()

            # TODO: here, we should return all metrics
            output = DataProto(meta_info={"metrics": metrics})

            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during update_actor", logger=logger
            )
        if self._is_offload_optimizer:
            offload_fsdp_optimizer(optimizer=self.actor_optimizer)
            log_gpu_memory_usage(
                "After offload actor optimizer during update_actor", logger=logger
            )

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def generate_sequences(self, prompts: DataProto):
        # Support all hardwares
        prompts = prompts.to(get_torch_device().current_device())

        assert self._is_rollout

        _eos_from_gen = (
            self.generation_config.eos_token_id
            if self.generation_config is not None
            else None
        )
        _pad_from_gen = (
            self.generation_config.pad_token_id
            if self.generation_config is not None
            else None
        )
        if _eos_from_gen is None:
            print(
                "\033[31m[WARNING] generation_config.eos_token_id is None, "
                f"falling back to tokenizer.eos_token_id={self.tokenizer.eos_token_id}\033[0m"
            )
        if _pad_from_gen is None:
            print(
                "\033[31m[WARNING] generation_config.pad_token_id is None, "
                f"falling back to tokenizer.pad_token_id={self.tokenizer.pad_token_id}\033[0m"
            )
        meta_info = {
            "eos_token_id": _eos_from_gen or self.tokenizer.eos_token_id,
            "pad_token_id": _pad_from_gen or self.tokenizer.pad_token_id,
        }
        prompts.meta_info.update(meta_info)

        with self.rollout_sharding_manager:
            log_gpu_memory_usage(
                "After entering rollout sharding manager", logger=logger
            )

            prompts = self.rollout_sharding_manager.preprocess_data(prompts)

            output = self.rollout.generate_sequences(prompts=prompts)

            log_gpu_memory_usage("After rollout generation", logger=logger)

            output = self.rollout_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # clear kv cache
        get_torch_device().empty_cache()
        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_rollout_validation_mode(self, enabled: bool):
        """Select student validation or restore the fixed rollout teacher."""
        assert self._is_rollout
        self.rollout_sharding_manager.set_validation_mode(enabled)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def begin_rollout(self):
        """Sync FSDP → inference engine once; keep engine loaded across multi_turn_loop."""
        assert self._is_rollout
        self.rollout_sharding_manager.start_multi_step_rollout()

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def end_rollout(self):
        """Release inference engine memory after multi-step rollout."""
        assert self._is_rollout
        self.rollout_sharding_manager.stop_multi_step_rollout()

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def forward_process(self, prompts: DataProto):
        """
        forward process: add noise to the batch. Only mask the part where attention_mask == 1, padding part and prompt part are not masked

        Args:
            prompts: DataProto containing batch prompts

        Returns:
            DataProto with perturbed sequences, computed log probabilities and related tensors
        """
        batch = prompts.batch
        response_length = prompts.meta_info.get(
            "response_length", self.config.rollout.get("response_length")
        )
        # Extract data from batch
        idx_repeat = batch["prompts"]  # (batch_size * n_rollout, prompt_len)
        responses = batch["responses"]  # (batch_size * n_rollout, response_len)
        input_ids = batch["input_ids"]  # (batch_size * n_rollout, seq_len)
        attention_mask = batch["attention_mask"]  # (batch_size * n_rollout, seq_len)
        position_ids = batch["position_ids"]
        total_batch_size = input_ids.shape[0]

        prompt_len = input_ids.shape[1] - response_length

        # decode_orders_nfe: (batch, seq_len) int64, raw NFE step when each pos was unmasked.
        # Multiple tokens can share the same NFE step (unmasked in the same forward pass).
        # Only present when the sglang rollout engine ran with record_decode_order: true.
        # Required by the "trajectory" noisy backend; ignored by all other backends.
        decode_orders_nfe = batch.get("decode_orders_nfe", None)

        # Curriculum backends combine this trainer-injected current step with their
        # dedicated dllm_config.curriculum_total_steps schedule length.
        _global_step = int(prompts.meta_info.get("global_step", 0))

        _build_kwargs = dict(
            input_ids=input_ids,
            attention_mask=attention_mask,
            prompt_len=prompt_len,
            decode_orders=decode_orders_nfe,
            global_step=_global_step,
        )

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq, mask_indices, p_mask = self.actor.build_noisy_samples(
                **_build_kwargs,
            )  # (batch_size, mc_num, seq_len) each

        batch_dict = {
            "prompts": idx_repeat,
            "responses": responses,
            "input_ids": input_ids,  # Complete prompt + response
            "attention_mask": attention_mask,
            "position_ids": position_ids,  # Complete position_ids, prompt is left-padded, response is right-padded
            "perturbed_seq": perturbed_seq,  # (batch_size * n_rollout, mc_num, seq_len)
            "mask_indices": mask_indices,
            "p_mask": p_mask,
        }
        batch = TensorDict(
            batch_dict,
            batch_size=total_batch_size,
        )

        return DataProto(batch=batch)

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_log_prob(self, data: DataProto):
        # when is_lora is True, we use the actor without lora applied to calculate the log_prob
        # which is mostly used for ref log_prob calculation
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        # Support all hardwares
        from contextlib import nullcontext

        is_lora = data.meta_info.pop("is_lora", False)
        adapter_ctx = (
            self.actor.actor_module.disable_adapter() if is_lora else nullcontext()
        )
        data = data.to(get_torch_device().current_device())
        # we should always recompute old_log_probs when it is HybridEngine
        data.meta_info["micro_batch_size"] = (
            self.config.rollout.log_prob_micro_batch_size_per_gpu
        )
        data.meta_info["max_token_len"] = (
            self.config.rollout.log_prob_max_token_len_per_gpu
        )
        data.meta_info["use_dynamic_bsz"] = self.config.rollout.log_prob_use_dynamic_bsz
        data.meta_info["temperature"] = self.config.rollout.temperature
        # perform recompute log_prob
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            with adapter_ctx:
                entropys, log_probs, loss_per_sample = self.actor.compute_log_prob(
                    data=data, calculate_entropy=True
                )  # (batch_size, steps, seq_length)
            output_tensors = {"old_log_probs": log_probs, "old_entropys": entropys}
            if loss_per_sample is not None:
                output_tensors["old_loss_per_sample"] = loss_per_sample
            output = DataProto.from_dict(
                tensors=output_tensors,
                meta_info={"temperature": self.config.rollout.temperature},
            )
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.actor.actor_module) == 1:
            self.actor.actor_module._handle.reshard(True)

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during compute_log_prob", logger=logger
            )

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_ref_log_prob(self, data: DataProto):
        if self._is_lora:
            # if _is_lora, actor without lora applied is the ref
            data.meta_info["is_lora"] = True
            data = self.compute_log_prob(data)
            # this old_log_probs is in fact ref_log_prob
            data = DataProto.from_dict(
                tensors={"ref_log_probs": data.batch["old_log_probs"]}
            )
            return data
        assert self._is_ref
        # else:
        # otherwise, the class have a standalone ref model
        # Support all hardwares
        data = data.to(get_torch_device().current_device())

        micro_batch_size = self.config.ref.log_prob_micro_batch_size_per_gpu
        data.meta_info["micro_batch_size"] = micro_batch_size
        data.meta_info["temperature"] = self.config.rollout.temperature
        data.meta_info["max_token_len"] = self.config.ref.log_prob_max_token_len_per_gpu
        data.meta_info["use_dynamic_bsz"] = self.config.ref.log_prob_use_dynamic_bsz
        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data)
            entropys, log_probs, loss_per_sample = self.ref_policy.compute_log_prob(
                data=data, calculate_entropy=True
            )
            ref_tensors = {"ref_log_probs": log_probs, "ref_entropys": entropys}
            if loss_per_sample is not None:
                ref_tensors["ref_loss_per_sample"] = loss_per_sample
            output = DataProto.from_dict(tensors=ref_tensors)
            output = self.ulysses_sharding_manager.postprocess_data(output)

        output = output.to("cpu")

        # https://pytorch.org/docs/stable/notes/fsdp.html#fsdp-notes
        # unshard the root FSDP module
        if self.world_size > 1 and fsdp_version(self.ref_policy.actor_module) == 1:
            self.ref_policy.actor_module._handle.reshard(True)

        return output

    @register(dispatch_mode=Dispatch.DP_COMPUTE_PROTO)
    def compute_policy_loss_for_validation(self, data: DataProto):
        assert self._is_actor
        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        data = data.to(get_torch_device().current_device())

        with self.ulysses_sharding_manager:
            data = self.ulysses_sharding_manager.preprocess_data(data=data)
            # perform training
            metrics = self.actor.compute_policy_loss(data=data)

            output = DataProto(meta_info={"metrics": metrics})
            output = self.ulysses_sharding_manager.postprocess_data(data=output)
            output = output.to("cpu")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)
            log_gpu_memory_usage(
                "After offload actor model during update_actor", logger=logger
            )

        return output

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def save_checkpoint(
        self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None
    ):
        # only support save and load ckpt for actor
        assert self._is_actor

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.actor_module_fsdp)

        self.checkpoint_manager.save_checkpoint(
            local_path=local_path,
            hdfs_path=hdfs_path,
            global_step=global_step,
            max_ckpt_to_keep=max_ckpt_to_keep,
        )
        dist.barrier()

        actor_module = getattr(self, "actor_module", self.actor_module_fsdp)
        if self._is_lora and isinstance(actor_module, PeftModel):
            lora_save_path = os.path.join(local_path, "lora_adapter")
            peft_config = {}
            if dist.get_rank() == 0:
                os.makedirs(lora_save_path, exist_ok=True)
                peft_config = asdict(actor_module.peft_config.get("default", {}))
                peft_config["task_type"] = peft_config["task_type"].value
                peft_config["peft_type"] = peft_config["peft_type"].value
                peft_config["target_modules"] = list(peft_config["target_modules"])
            try:
                if isinstance(self.actor_module_fsdp, FSDP):
                    self.actor_module_fsdp = self.actor_module_fsdp.cuda()
                    lora_params = layered_summon_lora_params(self.actor_module_fsdp)
                    if dist.get_rank() == 0:
                        save_file(
                            lora_params,
                            os.path.join(lora_save_path, "adapter_model.safetensors"),
                        )
                        with open(
                            os.path.join(lora_save_path, "adapter_config.json"),
                            "w",
                            encoding="utf-8",
                        ) as f:
                            json.dump(peft_config, f, ensure_ascii=False, indent=4)
            except Exception as e:
                if dist.get_rank() == 0:
                    print(f"[rank-{self.rank}]: Save LoRA Adapter Error ({e})")

            dist.barrier()
            if dist.get_rank() == 0:
                print(f"[rank-{self.rank}]: Saved LoRA adapter to: {lora_save_path}")

        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def set_debug_saver(self, debug_saver, debug_step: int):
        """
        将 DebugStateSaver 注入到 actor，并设置当前需要保存的 step。

        在 Trainer 主循环中，每次判断需要 debug 保存时调用：
            self.actor_rollout_wg.set_debug_saver(self.debug_saver, self.global_steps)

        注意：由于 Ray 序列化的限制，saver 对象需要可被 pickle；
        DebugStateSaver 只含 pathlib.Path 和基本类型，满足要求。
        """
        assert self._is_actor
        self.actor.debug_saver = debug_saver
        self.actor.debug_step = debug_step

    @register(dispatch_mode=Dispatch.ONE_TO_ALL)
    def clear_debug_saver(self):
        """清除 debug saver，恢复正常训练模式（避免连续多步都保存内部数据）。"""
        if self._is_actor and hasattr(self.actor, "debug_saver"):
            self.actor.debug_saver = None
            self.actor.debug_step = None
