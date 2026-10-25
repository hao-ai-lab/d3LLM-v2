"""
API Rollout (eval-only, rollout-only).

OpenAI-compatible completions API:
- POST {api_base}/completions
- payload: {model, prompt, temperature, max_tokens}

This rollout converts token prompts -> text, calls API, then encodes text back to tokens.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import List

import torch
from tensordict import TensorDict
from openai import OpenAI

from verl import DataProto
from verl.utils.torch_functional import get_response_mask

from .base import BaseRollout

__all__ = ["APIRollout"]


class APIRollout(BaseRollout):
    def __init__(self, tokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.client = OpenAI(base_url=config.get("api_base", "http://localhost:8000/v1"), api_key="EMPTY")
        self.api_model = config.get("api_model", "")

    def _decode_prompts(self, prompts: DataProto) -> List[str]:
        raw = prompts.non_tensor_batch.get("raw_prompt_ids")
        if raw is None:
            ids, mask = prompts.batch["input_ids"], prompts.batch["attention_mask"]
            raw = [ids[i][mask[i].bool()].tolist() for i in range(ids.size(0))]
        return [
            self.tokenizer.decode(
                ids.tolist() if hasattr(ids, "tolist") else list(ids),
                skip_special_tokens=False,
            )
            for ids in raw
        ]

    def _call_one(self, prompt: str, temperature: float, max_tokens: int) -> str:
        try:
            return (
                self.client.completions.create(
                    model=self.api_model,
                    prompt=prompt,
                    temperature=temperature,
                    max_tokens=max_tokens,
                )
                .choices[0]
                .text
            )
        except Exception:
            return ""

    def generate_sequences(self, prompts: DataProto) -> DataProto:
        batch_size = prompts.batch.batch_size[0]

        is_validate = prompts.meta_info.get("validate", False)
        do_sample = prompts.meta_info.get("do_sample", self.config.get("do_sample", True))
        temperature = prompts.meta_info.get(
            "temperature",
            (self.config.val_kwargs.temperature if is_validate else self.config.get("temperature", 1.0)),
        )
        response_length = int(prompts.meta_info.get("response_length", self.config.response_length))

        eos_token_id = prompts.meta_info["eos_token_id"]
        if isinstance(eos_token_id, (list, tuple)):
            eos_token_id = eos_token_id[0]
        eos_token_id = int(eos_token_id)
        pad_token_id = prompts.meta_info["pad_token_id"]
        if isinstance(pad_token_id, (list, tuple)):
            pad_token_id = pad_token_id[0]
        pad_token_id = int(pad_token_id)

        # API: greedy decoding if do_sample is False
        if not do_sample:
            temperature = 0.0

        prompt_texts = self._decode_prompts(prompts)
        with ThreadPoolExecutor(max_workers=min(batch_size, 16)) as pool:
            out_texts = list(pool.map(lambda t: self._call_one(t, temperature, response_length), prompt_texts))

        device = prompts.batch["input_ids"].device

        # Encode responses and pad to fixed length.
        responses = torch.full(
            (batch_size, response_length),
            pad_token_id,
            dtype=torch.long,
            device=device,
        )
        for i, txt in enumerate(out_texts):
            token_ids = self.tokenizer.encode(txt, add_special_tokens=False)[:response_length]
            if token_ids:
                responses[i, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long, device=device)

        prompt_ids = prompts.batch["input_ids"]
        prompt_len = prompt_ids.size(1)
        seq = torch.cat([prompt_ids, responses], dim=1)

        position_ids = prompts.batch["position_ids"]
        delta_pos = torch.arange(1, response_length + 1, device=position_ids.device).unsqueeze(0).repeat(batch_size, 1)
        response_position_ids = position_ids[:, -1:] + delta_pos
        position_ids = torch.cat([position_ids, response_position_ids], dim=-1)

        attention_mask = prompts.batch["attention_mask"]
        response_attention_mask = get_response_mask(response_id=responses, eos_token=eos_token_id, dtype=attention_mask.dtype)
        attention_mask = torch.cat((attention_mask, response_attention_mask), dim=-1)

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": responses,
                "input_ids": seq,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
            },
            batch_size=batch_size,
        )
        return DataProto(batch=batch)

