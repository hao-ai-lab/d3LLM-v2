# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
#
# Licensed under the Apache License, Version 2.0 (the "License");

"""Length penalty helpers (same placement pattern as invalid_action_penalty)."""

from __future__ import annotations

import numpy as np
import torch

from verl import DataProto


def compute_capped_linear_penalty(
    response_length: int,
    threshold: int,
    coef: float,
    max_penalty: float,
) -> float:
    """Capped linear penalty: min(max_penalty, coef * max(0, length - threshold) / 100)."""
    if threshold <= 0 or response_length <= threshold:
        return 0.0
    excess = response_length - threshold
    return min(max_penalty, coef * excess / 100.0)


def resolve_effective_response_length(data_item) -> int:
    """
    Effective generated response length for penalties.

    When ``dllm_full_response_mask=True``, attention_mask sums the full padded
    response window; prefer rollout ``response_first_eos_pos``, ``loss_mask``,
    or ``decode_orders`` instead.
    """
    batch = data_item.batch
    prompt_length = batch["prompts"].shape[-1]

    if "response_first_eos_pos" in batch:
        pos = batch["response_first_eos_pos"]
        if isinstance(pos, torch.Tensor):
            return max(int(pos.reshape(-1)[0].item()), 0)
        return max(int(pos), 0)

    if "loss_mask" in batch:
        n = int(batch["loss_mask"][prompt_length:].sum().item())
        if n > 0:
            return n

    if "decode_orders" in batch:
        n = int((batch["decode_orders"][prompt_length:] > 0).sum().item())
        if n > 0:
            return n

    return int(batch["attention_mask"][prompt_length:].sum().item())


def resolve_reward_last_token_index(data_item, reward_row: torch.Tensor) -> int:
    """Index of the reward token in the response slice (matches EpisodeRewardManager)."""
    nz = (reward_row.abs() > 1e-9).nonzero(as_tuple=False).flatten()
    if nz.numel() > 0:
        return int(nz[-1].item())
    eff = resolve_effective_response_length(data_item)
    resp_len = reward_row.shape[-1]
    return max(min(eff - 1, resp_len - 1), 0)


def apply_length_penalty(
    data: DataProto,
    length_penalty_coef: float = 0.1,
    length_penalty_threshold: int = 0,
    length_penalty_max: float = 1.0,
    length_penalty_mode: str = "relative",
) -> tuple[DataProto, dict]:
    """Apply length penalty on token_level_scores at the reward token, like invalid_action."""
    mode = length_penalty_mode.strip().lower()
    if mode not in ("relative", "absolute"):
        raise ValueError(f"length_penalty_mode must be 'relative' or 'absolute', got {length_penalty_mode!r}")

    reward_tensor = data.batch["token_level_scores"]
    step_rewards = data.batch["step_rewards"] if "step_rewards" in data.batch.keys() else None

    response_lengths = []
    penalty_values = []

    for i in range(len(data)):
        data_item = data[i]
        eff_len = resolve_effective_response_length(data_item)
        response_lengths.append(eff_len)

        penalty = compute_capped_linear_penalty(
            eff_len,
            length_penalty_threshold,
            length_penalty_coef,
            length_penalty_max,
        )
        penalty_values.append(penalty)
        if penalty <= 0 or eff_len <= 0:
            continue

        last_idx = resolve_reward_last_token_index(data_item, reward_tensor[i])
        if mode == "relative":
            scale = 1.0 - min(1.0, penalty)
            reward_tensor[i, last_idx] *= scale
            if step_rewards is not None:
                step_rewards[i] *= scale
        else:
            penalty_t = torch.tensor(penalty, dtype=reward_tensor.dtype, device=reward_tensor.device)
            reward_tensor[i, last_idx] -= penalty_t
            if step_rewards is not None:
                step_rewards[i] -= penalty_t

    metrics = {
        "episode/mean_response_length": float(np.mean(response_lengths)) if response_lengths else 0.0,
        "episode/mean_length_penalty": float(np.mean(penalty_values)) if penalty_values else 0.0,
        "episode/length_penalty_ratio": float(np.mean([p > 0 for p in penalty_values])) if penalty_values else 0.0,
    }
    return data, metrics
