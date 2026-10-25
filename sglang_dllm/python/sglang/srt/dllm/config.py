from typing import Any

from sglang.srt.configs.model_config import ModelConfig
from sglang.srt.server_args import ServerArgs


class DllmConfig:
    def __init__(
        self,
        algorithm: str,
        algorithm_config: dict[str, Any],
        block_size: int,
        mask_id: int,
        max_running_requests: int,
        needs_full_prefill: bool = False,
        pad_full_generation: bool = False,
        mask_invalid_token: bool = False,
        tokenizer_vocab_size: int = 0,
        eos_id: int = 0,
    ):
        self.algorithm = algorithm
        self.algorithm_config = algorithm_config
        self.block_size = block_size
        self.mask_id = mask_id
        self.max_running_requests = max_running_requests
        self.needs_full_prefill = needs_full_prefill
        self.pad_full_generation = pad_full_generation
        self.mask_invalid_token = mask_invalid_token
        self.tokenizer_vocab_size = tokenizer_vocab_size
        self.eos_id = eos_id

    @staticmethod
    def from_server_args(
        server_args: ServerArgs,
    ):
        if server_args.dllm_algorithm is None:
            return None

        model_config = ModelConfig.from_server_args(
            server_args,
            model_path=server_args.model_path,
            model_revision=server_args.revision,
        )

        needs_full_prefill = False
        if model_config.hf_config.architectures[0] == "LLaDA2MoeModelLM":
            block_size = 32
            mask_id = 156895
        elif model_config.hf_config.architectures[0] == "Qwen3ForCausalLM":
            block_size = 32
            mask_id = 151670  # <|mask|> token id for ReFusion
        elif model_config.hf_config.architectures[0] == "DreamModel":
            block_size = 32
            mask_id = 151666  # 151666
            needs_full_prefill = True
        elif model_config.hf_config.architectures[0] == "LLaDAModelLM":
            block_size = 32
            mask_id = 126336
            needs_full_prefill = True
        elif model_config.hf_config.architectures[0] == "ILLaDAForCausalLM":
            block_size = 32
            mask_id = 5
            needs_full_prefill = True
        elif model_config.hf_config.architectures[0] == "SDARForCausalLM":
            block_size = 4
            mask_id = 151669
        elif model_config.hf_config.architectures[0] == "SDARMoeForCausalLM":
            block_size = 4
            mask_id = 151669
        else:
            raise RuntimeError(
                f"Unknown diffusion LLM: {model_config.hf_config.architectures[0]}"
            )

        max_running_requests = (
            1
            if server_args.max_running_requests is None
            else server_args.max_running_requests
        )

        algorithm_config = {}
        if server_args.dllm_algorithm_config is not None:
            try:
                import yaml
            except ImportError:
                raise ImportError(
                    "Please install PyYAML to use YAML config files. "
                    "`pip install pyyaml`"
                )
            with open(server_args.dllm_algorithm_config, "r") as f:
                algorithm_config = yaml.safe_load(f)

            # Parse common algorithm configurations
            block_size = algorithm_config.get("block_size", block_size)

        # Full-attention algorithms pad ALL max_new_tokens masks at once
        _FULL_GEN_PAD_ALGORITHMS = {"FullAttnMultiBlock"}
        pad_full_generation = (
            needs_full_prefill
            and server_args.dllm_algorithm in _FULL_GEN_PAD_ALGORITHMS
        )

        # Read EOS token id from model HF config (handles int or list-of-int).
        _eos = model_config.hf_config.eos_token_id
        if isinstance(_eos, (list, tuple)):
            _eos = _eos[0]
        eos_id = int(_eos) if _eos is not None else 0

        return DllmConfig(
            algorithm=server_args.dllm_algorithm,
            algorithm_config=algorithm_config,
            block_size=block_size,
            mask_id=mask_id,
            max_running_requests=max_running_requests,
            needs_full_prefill=needs_full_prefill,
            pad_full_generation=pad_full_generation,
            mask_invalid_token=server_args.dllm_mask_invalid_token,
            tokenizer_vocab_size=server_args.dllm_tokenizer_vocab_size,
            eos_id=eos_id,
        )
