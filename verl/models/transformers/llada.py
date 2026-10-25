# Copyright 2025 verl-agent contributors
# Non-invasive addition: LLaDA dLLM support for HuggingFace training in verl.
# Supports LLaDA 2.0/2.1 (MoE) and LLaDA 8B (dense) architectures.

import torch
from transformers import AutoConfig, AutoModelForCausalLM


def _patch_transformers_compat():
    """Patch transformers 5.3.0.dev0 compatibility issues."""
    # 1. Ensure 'default' exists in ROPE_INIT_FUNCTIONS for LLaDA 2.x
    try:
        from transformers.modeling_rope_utils import ROPE_INIT_FUNCTIONS
        if "default" not in ROPE_INIT_FUNCTIONS:
            def _default_rope_init(config, device=None, **kwargs):
                base = config.rope_theta
                partial_rotary_factor = getattr(config, "partial_rotary_factor", 1.0)
                head_dim = getattr(config, "head_dim", None) or config.hidden_size // config.num_attention_heads
                dim = int(head_dim * partial_rotary_factor)
                inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
                return inv_freq, 1.0
            ROPE_INIT_FUNCTIONS["default"] = _default_rope_init
    except ImportError:
        pass

    # 2. Add HybridCache stub if missing (removed in transformers 5.x, needed by peft)
    import transformers
    if not hasattr(transformers, "HybridCache"):
        from transformers import DynamicCache
        transformers.HybridCache = DynamicCache  # stub alias

    # 3. Fix _get_tied_weight_keys for transformers 5.3.0.dev0
    # Some models have _tied_weights_keys as a list but the new code expects dict
    try:
        import transformers.modeling_utils as _mu
        _orig_get_tied = getattr(_mu, "_get_tied_weight_keys", None)
        if _orig_get_tied is not None:
            def _safe_get_tied_weight_keys(model):
                try:
                    return _orig_get_tied(model)
                except AttributeError:
                    # Fallback: return empty list when _tied_weights_keys is a list not dict
                    return []
            _mu._get_tied_weight_keys = _safe_get_tied_weight_keys
    except Exception:
        pass


def _patch_rope_init_functions():
    """Legacy alias for backward compat."""
    _patch_transformers_compat()

_patch_transformers_compat()

# Architecture -> mask_token_id mapping
LLADA_ARCH_MASK_IDS = {
    "LLaDA2MoeModelLM": 156895,  # LLaDA 2.0/2.1 MoE
    "LLaDAModelLM": 126336,       # LLaDA 8B dense
    "SDARForCausalLM": 151669,    # SDAR 1.7B/8B dense
    "SDARMoeForCausalLM": 151669, # SDAR 4B MoE
    "DreamModel": 151666,         # DREAM (mask diffusion, Dream-v0-*)
}

# Known model paths (HuggingFace hub and DARE local paths)
LLADA_MODELS = {
    # HuggingFace hub
    "inclusionAI/LLaDA2.1-mini": {"arch": "LLaDA2MoeModelLM", "mask_id": 156895},
    "inclusionAI/LLaDA2.0-mini": {"arch": "LLaDA2MoeModelLM", "mask_id": 156895},
    "inclusionAI/LLaDA2.1-flash": {"arch": "LLaDA2MoeModelLM", "mask_id": 156895},
    "GSAI-ML/LLaDA-8B-Base": {"arch": "LLaDAModelLM", "mask_id": 126336},
    "GSAI-ML/LLaDA-8B-Instruct": {"arch": "LLaDAModelLM", "mask_id": 126336},
    "JetLM/SDAR-1.7B-Chat": {"arch": "SDARForCausalLM", "mask_id": 151669},
    "JetLM/SDAR-4B-Chat": {"arch": "SDARForCausalLM", "mask_id": 151669},
    "JetLM/SDAR-8B-Chat": {"arch": "SDARForCausalLM", "mask_id": 151669},
}


def get_mask_token_id(model_name_or_config):
    """Get mask token id from model name or config.

    Resolution order:
    1. Fast-path lookup in LLADA_MODELS (HuggingFace hub names).
    2. config.mask_token_id (present on all dLLM models including DREAM/SDAR).
    3. Architecture name lookup in LLADA_ARCH_MASK_IDS.
    4. Fallback to 156895 (LLaDA 2.x default).
    """
    if isinstance(model_name_or_config, str):
        if model_name_or_config in LLADA_MODELS:
            return LLADA_MODELS[model_name_or_config]["mask_id"]
        config = AutoConfig.from_pretrained(model_name_or_config, trust_remote_code=True)
    else:
        config = model_name_or_config
    # All dLLM models (LLaDA, DREAM, SDAR) store mask_token_id directly in config
    if getattr(config, "mask_token_id", None) is not None:
        return config.mask_token_id
    arch = config.architectures[0] if hasattr(config, "architectures") else ""
    return LLADA_ARCH_MASK_IDS.get(arch, 156895)


def is_llada_model(model_name_or_config):
    """Check if model is a dLLM architecture (LLaDA, DREAM, or SDAR)."""
    if isinstance(model_name_or_config, str):
        if model_name_or_config in LLADA_MODELS:
            return True
        try:
            config = AutoConfig.from_pretrained(model_name_or_config, trust_remote_code=True)
        except Exception:
            return False
    else:
        config = model_name_or_config
    # Check model_type (covers local paths not in LLADA_MODELS)
    model_type = getattr(config, "model_type", "")
    if model_type in ("llada", "llada2_moe", "Dream", "sdar", "sdar_moe"):
        return True
    arch = config.architectures[0] if hasattr(config, "architectures") else ""
    return arch in LLADA_ARCH_MASK_IDS


def load_llada_for_training(model_path, torch_dtype=torch.bfloat16, use_lora=False, lora_config=None, device="cuda"):
    """Load a LLaDA model for training with optional LoRA.
    
    Returns (model, config) tuple. The model uses trust_remote_code=True
    since LLaDA has custom HuggingFace implementations.
    
    Note: Model is loaded directly to the target device to avoid NaN issues
    that can occur when loading to CPU first then moving to CUDA.
    """
    _patch_rope_init_functions()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch_dtype, trust_remote_code=True
    ).to(device)
    
    if use_lora and lora_config is not None:
        from peft import LoraConfig as PeftLoraConfig
        from peft import get_peft_model
        if isinstance(lora_config, dict):
            lora_config = PeftLoraConfig(**lora_config)
        model.enable_input_require_grads()
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    
    return model, config
