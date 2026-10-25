"""SGLang iLLaDA model using the full-attention LLaDA execution path."""

from collections.abc import Iterable

import torch

from sglang.srt.models.d3llm_llada import LLaDAModelLM
from sglang.srt.models.llama import LlamaForCausalLM


class ILLaDAForCausalLM(LLaDAModelLM):
    """iLLaDA has LLaMA-compatible parameter names and full attention."""

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]):
        return LlamaForCausalLM.load_weights(self, weights)


EntryClass = ILLaDAForCausalLM
