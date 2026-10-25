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

# To support different vLLM versions, we add the model into SUPPORTED_MOE_MODELS separately to avoid triggering unsupported issues.
SUPPORTED_MOE_MODELS = []

try:
    from vllm.model_executor.models.deepseek_v2 import (
        DeepseekV2ForCausalLM,
        DeepseekV3ForCausalLM,
    )

    SUPPORTED_MOE_MODELS.append(DeepseekV2ForCausalLM)
    SUPPORTED_MOE_MODELS.append(DeepseekV3ForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.mixtral import MixtralForCausalLM

    SUPPORTED_MOE_MODELS.append(MixtralForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen2_moe import Qwen2MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen2MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_moe import Qwen3MoeForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_vl_moe import Qwen3MoeLLMForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3MoeLLMForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.qwen3_next import Qwen3NextForCausalLM

    SUPPORTED_MOE_MODELS.append(Qwen3NextForCausalLM)
except ImportError:
    pass

try:
    from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration

    SUPPORTED_MOE_MODELS.append(KimiVLForConditionalGeneration)
except ImportError:
    pass

from typing import List

from msgspec import field
from packaging import version as vs
from vllm.lora.request import LoRARequest
from vllm.lora.utils import get_adapter_absolute_path
from vllm.lora.worker_manager import LRUCacheWorkerLoRAManager

try:
    # vLLM >= 0.17.0
    from vllm.lora.model_manager import LoRAModelManager
    from vllm.lora.lora_model import LoRAModel
except ImportError:
    # vLLM < 0.17.0
    from vllm.lora.models import LoRAModelManager, LoRAModel

from verl.third_party.vllm import get_version


def patch_vllm_moe_model_weight_loader(model):
    # this is a work around to load the weight of vllm fused moe model
    # it is from a bug from vllm 0.8.2
    # all the weights are supposed to have a weight_loader, but the moe weights
    # do not have a weight_loader, so we need to patch it
    # (True, 'model.embed_tokens.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.weight')
    # (True, 'model.layers.0.self_attn.qkv_proj.bias')
    # (True, 'model.layers.0.self_attn.o_proj.weight')
    # (True, 'model.layers.0.mlp.gate.weight')
    # (True, 'model.layers.0.mlp.shared_expert.gate_up_proj.weight')
    # (True, 'model.layers.0.mlp.shared_expert.down_proj.weight')
    # (False, 'model.layers.0.mlp.shared_expert_gate.weight')   use default
    # (False, 'model.layers.0.input_layernorm.weight')          use default
    # (False, 'model.layers.0.post_attention_layernorm.weight') use default
    # (False, 'model.layers.0.mlp.experts.w13_weight')          use mlp.experts.weight_loader
    # (False, 'model.layers.0.mlp.experts.w2_weight')          use mlp.experts.weight_loader

    # Early return if no MOE models are supported
    if not SUPPORTED_MOE_MODELS:
        return

    original_model_type = type(model)
    if hasattr(model, "runnable") and "ACLGraphWrapper" in str(original_model_type):
        model = model.runnable
        original_model_type = type(model)

    # Define MLP attribute mapping for different model types
    MLP_ATTR_MAPPING = {}
    try:
        from vllm.model_executor.models.mixtral import MixtralForCausalLM

        MLP_ATTR_MAPPING[MixtralForCausalLM] = "block_sparse_moe"
    except ImportError:
        pass

    DEFAULT_MLP_ATTR = "mlp"

    # Get inner model (either model.model or model.language_model)
    inner_model = getattr(model, "model", None) or getattr(
        model, "language_model", None
    )
    if inner_model is None:
        raise ValueError(
            "The provided model does not have a valid 'model' or 'language_model' attribute."
        )

    if not isinstance(model, tuple(SUPPORTED_MOE_MODELS)) and not isinstance(
        inner_model, tuple(SUPPORTED_MOE_MODELS)
    ):
        return

    # TODO(@leisuzz): class Qwen3MoeLLMForCausalLM is not available if VLLM version < 0.11.0,
    # will update the 'if statement' with 'isinstance' when verl commonly use VLLM version >= 0.11.0
    if type(inner_model).__name__ == "Qwen3MoeLLMForCausalLM":
        inner_model = inner_model.model  # Reassign inner_model in Qwen3-vl

    for layer_idx, layer in enumerate(inner_model.layers):
        mlp_attr = MLP_ATTR_MAPPING.get(original_model_type, DEFAULT_MLP_ATTR)

        mlp = getattr(layer, mlp_attr, None)
        if not mlp:
            continue

        experts = getattr(mlp, "experts", None)
        if not experts or not hasattr(experts, "weight_loader"):
            continue

        # Patch the weight loaders
        for name, param in mlp.named_parameters():
            if "w13_weight" in name or "w2_weight" in name:
                param.weight_loader = experts.weight_loader


class TensorLoRARequest(LoRARequest):
    peft_config: dict = field(default=None)
    lora_tensors: dict = field(default=None)


class VLLMHijack:
    @staticmethod
    def hijack():
        def hijack__load_adapter(self, lora_request: TensorLoRARequest) -> LoRAModel:
            """
            based on vllm.lora.worker_manager.WorkerLoRAManager._load_adapter, support load adapter with lora tensors

            Reason:
            VLLM does not support adding LoRA from tensors directly. It only supports adding LoRA via file paths.
            To synchronize the LoRA tensors of the actor model, we need to find a workaround to enable VLLM to load memory-based LoRA tensors.
            """
            try:
                supported_lora_modules = self._adapter_manager.supported_lora_modules
                packed_modules_mapping = self._adapter_manager.packed_modules_mapping
                expected_lora_modules: List[str] = []
                for module in supported_lora_modules:
                    if module in packed_modules_mapping:
                        expected_lora_modules.extend(packed_modules_mapping[module])
                    else:
                        expected_lora_modules.append(module)

                expected_lora_modules = list(set(expected_lora_modules))

                lora_tensors = None
                from vllm.lora.peft_helper import PEFTHelper

                if isinstance(lora_request, TensorLoRARequest):
                    peft_config = lora_request.peft_config
                    lora_tensors = lora_request.lora_tensors
                    peft_helper = PEFTHelper.from_dict(peft_config)
                else:
                    lora_path = get_adapter_absolute_path(lora_request.lora_path)

                    peft_helper = PEFTHelper.from_local_dir(
                        lora_path, self.max_position_embeddings
                    )

                # Validates the LoRA configuration against requirements before
                # loading weights, throwing an exception if validation fails.
                peft_helper.validate_legal(self.lora_config)

                # For some models like Qwen2VL, we need to use hf_to_vllm_mapper
                # to ensure correct loading of lora weights.
                model = self._adapter_manager.model
                hf_to_vllm_mapper = getattr(model, "hf_to_vllm_mapper", None)
                lora_skip_prefixes = getattr(model, "lora_skip_prefixes", None)

                import inspect

                # 公共基础参数
                lora_kwargs = dict(
                    lora_model_id=lora_request.lora_int_id,
                    peft_helper=peft_helper,
                    device="cpu",
                    dtype=self.lora_config.lora_dtype,
                    weights_mapper=hf_to_vllm_mapper,
                )

                if isinstance(lora_request, TensorLoRARequest):
                    sig = inspect.signature(self._lora_model_cls.from_lora_tensors)
                    if "model_vocab_size" in sig.parameters:
                        lora_kwargs["model_vocab_size"] = self.vocab_size
                    else:
                        lora_kwargs["embeddings"] = None  # 旧版 API 需要显式传 None
                        lora_kwargs["target_embedding_padding"] = (
                            self.vocab_size + getattr(self.lora_config, "lora_extra_vocab_size", 0)
                        )
                        lora_kwargs["embedding_modules"] = self.embedding_modules
                        lora_kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                    if "skip_prefixes" in sig.parameters:
                        lora_kwargs["skip_prefixes"] = lora_skip_prefixes
                    lora_kwargs["tensors"] = lora_tensors
                    lora = self._lora_model_cls.from_lora_tensors(**lora_kwargs)
                else:
                    sig = inspect.signature(self._lora_model_cls.from_local_checkpoint)
                    if "model_vocab_size" in sig.parameters:
                        lora_kwargs["model_vocab_size"] = self.vocab_size
                    else:
                        lora_kwargs["target_embedding_padding"] = (
                            self.vocab_size + getattr(self.lora_config, "lora_extra_vocab_size", 0)
                        )
                        lora_kwargs["embedding_modules"] = self.embedding_modules
                        lora_kwargs["embedding_padding_modules"] = self.embedding_padding_modules
                    if "skip_prefixes" in sig.parameters:
                        lora_kwargs["skip_prefixes"] = lora_skip_prefixes
                    lora_kwargs["lora_dir"] = lora_path
                    lora_kwargs["expected_lora_modules"] = expected_lora_modules
                    if "tensorizer_config_dict" in sig.parameters:
                        lora_kwargs["tensorizer_config_dict"] = getattr(
                            lora_request, "tensorizer_config_dict", None
                        )
                    # from_local_checkpoint 旧版 API 使用位置参数而非关键字参数
                    if "lora_dir" not in sig.parameters:
                        lora = self._lora_model_cls.from_local_checkpoint(
                            lora_kwargs.pop("lora_dir"),
                            lora_kwargs.pop("expected_lora_modules"),
                            **lora_kwargs,
                        )
                    else:
                        lora = self._lora_model_cls.from_local_checkpoint(**lora_kwargs)
            except Exception as e:
                raise e

            lora_extra_vocab_size = getattr(
                self.lora_config, "lora_extra_vocab_size", 0
            )
            extra_vocab_size = getattr(lora, "extra_vocab_size", 0)
            if extra_vocab_size > lora_extra_vocab_size:
                raise ValueError(
                    f"LoRA added vocab size {extra_vocab_size} "
                    f"is greater than lora_extra_vocab_size "
                    f"{lora_extra_vocab_size}."
                )
            return lora

        _orig_create_merged_loras_inplace = LoRAModelManager._create_merged_loras_inplace

        def hijack__create_merged_loras_inplace_safe(self, lora_model: LoRAModel) -> None:
            """
            Guards against StopIteration in vllm/lora/model_manager.py when
            lora_model.loras is empty after packed-module merging.
            Only applied on vLLM >= 0.17.0 where this bug exists.
            """
            try:
                _orig_create_merged_loras_inplace(self, lora_model)
            except StopIteration:
                pass

        def do_hijack(target_cls, target_method_name, hooking_method):
            setattr(target_cls, target_method_name, hooking_method)

        do_hijack(LRUCacheWorkerLoRAManager, "_load_adapter", hijack__load_adapter)
        if is_version_ge(pkg="vllm", minver="0.17.0"):
            do_hijack(LoRAModelManager, "_create_merged_loras_inplace", hijack__create_merged_loras_inplace_safe)


def is_version_ge(pkg: str = "vllm", minver: str = "0.7.3"):
    """check if the package version is greater than or equal to the minimum version"""
    return vs.parse(get_version(pkg)) >= vs.parse(minver)
