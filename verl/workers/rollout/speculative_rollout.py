"""Speculative Planning rollout: fast approximation model (API) + target model (vLLM).

Used for non-SGLang backends when we want per-step speculative planning (approx + target
inside rollout). For SGLang, speculative planning is handled purely at multi-turn level
in `rollout_loop`, so this wrapper is not used there.
"""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional

from openai import OpenAI

from verl import DataProto
from .base import BaseRollout

logger = logging.getLogger(__name__)


def _extract_action(text: str) -> Optional[str]:
    """Extract content between <action>...</action> tags, matching webshop projection logic."""
    t = text.lower()
    s, e = t.find("<action>"), t.find("</action>")
    if s == -1 or e == -1:
        return None
    return t[s + len("<action>"):e].strip()


def webshop_action_judge(text_a: str, text_b: str) -> bool:
    """Return True if <action>...</action> content is identical in both texts."""
    a, b = _extract_action(text_a), _extract_action(text_b)
    return a is not None and b is not None and a == b


class SpeculativeRollout(BaseRollout):
    """Wraps a target rollout with speculative planning via a fast approximation model API.

    Target model = framework's vLLM engine (e.g. Qwen3-4B).
    Approximation model = external vLLM API server (e.g. Qwen3-0.6B).
    judge_fn is swappable per environment (only webshop implemented for now).
    """

    def __init__(self, target_rollout: BaseRollout, tokenizer, config,
                 api_base: str = "http://localhost:8000/v1",
                 api_model: str = "Qwen/Qwen3-0.6B",
                 judge_fn: Callable[[str, str], bool] = webshop_action_judge):
        super().__init__()
        self.target = target_rollout
        self.tokenizer = tokenizer
        self.config = config
        self.client = OpenAI(base_url=api_base, api_key="EMPTY")
        self.api_model = api_model
        self.judge_fn = judge_fn
        self.accepted = 0
        self.total = 0

    @property
    def inference_engine(self):
        """Delegate to target so FSDPVLLMShardingManager works transparently."""
        return self.target.inference_engine

    def _decode_prompts(self, prompts: DataProto) -> List[str]:
        """Decode prompt tokens to text for API call, handling left-padding."""
        raw = prompts.non_tensor_batch.get("raw_prompt_ids")
        if raw is not None:
            return [self.tokenizer.decode(
                ids.tolist() if hasattr(ids, 'tolist') else list(ids),
                skip_special_tokens=False,
            ) for ids in raw]
        ids, mask = prompts.batch["input_ids"], prompts.batch["attention_mask"]
        return [self.tokenizer.decode(ids[i][mask[i].bool()].tolist(),
                                      skip_special_tokens=False)
                for i in range(ids.size(0))]

    def _call_approx(self, prompt_texts: List[str],
                     temperature: float, max_tokens: int) -> List[str]:
        """Call approximation model via OpenAI-compatible completions API (batched via threads)."""
        def _one(text):
            return self.client.completions.create(
                model=self.api_model, prompt=text,
                temperature=temperature, max_tokens=max_tokens,
            ).choices[0].text

        with ThreadPoolExecutor(max_workers=min(len(prompt_texts), 32)) as pool:
            return list(pool.map(_one, prompt_texts))

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        # Decode prompts to text BEFORE target rollout consumes raw_prompt_ids
        prompt_texts = self._decode_prompts(prompts)

        is_validate = prompts.meta_info.get("validate", False)
        temperature = (self.config.val_kwargs.temperature if is_validate
                       else self.config.temperature)
        max_tokens = self.config.response_length

        # Target model (vLLM) generates canonical output
        target_output = self.target.generate_sequences(prompts)

        # Approximation model generates via API
        approx_texts = self._call_approx(prompt_texts, temperature, max_tokens)

        # Decode target responses and judge
        target_texts = self.tokenizer.batch_decode(
            target_output.batch['responses'], skip_special_tokens=True)

        for at, tt in zip(approx_texts, target_texts):
            self.total += 1
            if self.judge_fn(at, tt):
                self.accepted += 1

        rate = self.accepted / self.total if self.total > 0 else 0.0
        logger.info("[Speculative] accept=%d/%d (%.1f%%)", self.accepted, self.total, rate * 100)

        return target_output
