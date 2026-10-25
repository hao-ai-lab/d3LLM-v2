import logging

import torch

from sglang.srt.dllm.algorithm import get_algorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.server_args import ServerArgs

logger = logging.getLogger(__name__)


class DllmAlgorithm:

    def __init__(
        self,
        config: DllmConfig,
    ):
        self.block_size = config.block_size
        self.mask_id = config.mask_id
        # EOS token id for early-stopping (0 = not configured / disabled).
        self.eos_id = getattr(config, "eos_id", 0)

        # Invalid-token masking: read config passed through DllmConfig (from ServerArgs).
        self._mask_invalid_token = getattr(config, "mask_invalid_token", False)
        self._tokenizer_vocab_size = getattr(config, "tokenizer_vocab_size", 0) if self._mask_invalid_token else None
        if self._mask_invalid_token and not self._tokenizer_vocab_size:
            raise RuntimeError(
                "mask_invalid_token=True but tokenizer_vocab_size is 0 or missing in DllmConfig."
            )
        logger.info(
            "DllmAlgorithm initialized: mask_invalid_token=%s tokenizer_vocab_size=%s",
            self._mask_invalid_token,
            self._tokenizer_vocab_size,
        )

    def _apply_invalid_token_mask(self, logits: torch.Tensor) -> torch.Tensor:
        """Set logits for token ids >= tokenizer_vocab_size to -inf (in-place).

        Args:
            logits: Tensor of shape [..., model_vocab_size].
        """
        if not self._mask_invalid_token:
            return logits
        model_vocab_size = logits.size(-1)
        logger.debug(
            "Applying invalid-token mask: logits_shape=%s tokenizer_vocab_size=%s "
            "model_vocab_size=%s",
            tuple(logits.shape),
            self._tokenizer_vocab_size,
            model_vocab_size,
        )
        if self._tokenizer_vocab_size < model_vocab_size:
            logits[..., self._tokenizer_vocab_size:] = float("-inf")
            logger.debug(
                "Masked %s invalid token ids in [%s, %s)",
                model_vocab_size - self._tokenizer_vocab_size,
                self._tokenizer_vocab_size,
                model_vocab_size,
            )
        else:
            logger.debug(
                "Invalid-token masking not needed: tokenizer_vocab_size >= model_vocab_size"
            )
        return logits

    @staticmethod
    def from_server_args(server_args: ServerArgs):
        config = DllmConfig.from_server_args(server_args)
        return get_algorithm(config)
