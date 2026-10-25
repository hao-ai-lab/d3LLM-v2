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
build_noisy_samples.py — noisy sample generation functions for diffusion language model training.

Three sampling strategies, distinguished by the `block` parameter:
  block == -1  : full bidirectional attention models (LLaDA / Dream) — uniform mask count sampling
  block  > 0   : block-DLLM models (SDAR) — independent per-token Bernoulli sampling

Function summary:
  build_random_noisy_batch(...)       : random mask rate t ~ U[eps, 1], used by BGPO / Online-VRPO
  build_gaussian_noisy_batch(...)     : fixed mask rate t_fixed (Gauss-Legendre quadrature point), used by GDPO
  build_d1_noisy_batch(...)           : full response mask + partial prompt mask, used by D1
  build_block_random_noisy_batch(...) : one random block partially masked + subsequent blocks fully masked
  build_trajectory_noisy_batch(...)   : uses actual rollout decode_order trajectory; either samples a
                                        discrete step t ~ U[0, T-1] or projects a curriculum mask ratio

All functions return:
  noisy_batch   : (b, seq_len)       original tokens with masked positions replaced by MASK_TOKEN_ID
  mask_indices  : (b, seq_len) bool  True at masked positions
  p_mask        : (b, seq_len) float per-token mask probability (used as ELBO importance weight)
"""

from typing import Optional

import torch

__all__ = [
    "build_random_noisy_batch",
    "build_gaussian_noisy_batch",
    "build_d1_noisy_batch",
    "build_block_random_noisy_batch",
    "build_trajectory_noisy_batch",
    "build_block_trajectory_noisy_batch",
    "build_all_block_trajectory_noisy_batch",
]


def build_random_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    eps: float = 1e-3,
    MASK_TOKEN_ID: int = 126336,
    block: int = -1,
    mask_ratio: float | None = None,
):
    """Sample a random mask rate t and apply noise to the response region.

    For block==-1 (LLaDA path), mask count k is drawn uniformly from [1, response_len] and
    applied with anti-correlated arrangement across batch items to reduce variance.
    For block>0 (SDAR path), each token is masked independently with Bernoulli(t_i) where
    t_i ~ U[eps, 1] is drawn per sequence.
    When ``mask_ratio`` is provided, it replaces the random rate while token positions
    remain random. This optional override lets ``random`` and ``random_curriculum`` share
    the same implementation without changing the original random backend.

    Args
    ----
    batch          : (b, seq_len) int64  – original token ids
    attention_mask : (seq_len,) int64    – 1 for valid tokens, 0 for padding
    prompt_len     : int  – number of prompt tokens; response starts at this position
    eps            : float – lower bound on mask rate to avoid zero masking
    MASK_TOKEN_ID  : int
    block          : int  – -1 for LLaDA full-attn path; >0 for SDAR Bernoulli path
    mask_ratio     : optional float in [0, 1] – fixed rate supplied by curriculum;
                     None preserves the original uniformly-random rate

    Returns
    -------
    noisy_batch, mask_indices, p_mask  (all shape (b, seq_len))
    """
    b, seq_len = batch.shape

    response_mask = attention_mask.bool().clone()
    response_mask[:prompt_len] = False
    all_response_indices = torch.where(response_mask)[0]
    padded_resp_len = all_response_indices.numel()

    # Compute actual_end by tail-scanning for trailing MASK tokens.
    # Trailing MASK tokens (appended by rollout post-processor after first EOS,
    # decode_order=0, attention_mask=1) must be excluded from loss and p_mask
    # denominator, matching build_block_random_noisy_batch behaviour.
    # All rows are expanded from the same sequence → scan row 0.
    actual_len = 0
    for j in range(padded_resp_len - 1, -1, -1):
        if batch[0, all_response_indices[j]] != MASK_TOKEN_ID:
            actual_len = j + 1
            break
    if actual_len == 0:
        return (
            batch.clone(),
            torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device),
            torch.zeros((b, seq_len), device=batch.device),
        )
    actual_end = prompt_len + actual_len
    response_indices = all_response_indices[all_response_indices < actual_end]
    target_len = response_indices.numel()

    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    p_mask = torch.zeros((b, seq_len), device=batch.device)

    fixed_t = (
        max(eps, min(1.0, float(mask_ratio)))
        if mask_ratio is not None
        else None
    )

    if block == -1:
        if fixed_t is None:
            # LLaDA: sample k uniformly, arrange anti-correlated across batch to reduce variance
            k = torch.randint(1, target_len + 1, (), device=batch.device)
            x = torch.round(
                torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)
            ).long()
            x = ((x - 1) % target_len) + 1
        else:
            # Curriculum: all rows use the scheduled rate; only positions are random.
            k = max(1, min(target_len, round(fixed_t * target_len)))
            x = torch.full((b,), k, dtype=torch.long, device=batch.device)
        assert 1 <= x.min() and x.max() <= target_len

        for i in range(b):
            perm = torch.randperm(target_len, device=batch.device)
            mask_indices[i, response_indices[perm[:x[i]]]] = True

        p_mask = (x / target_len).unsqueeze(1).expand(b, seq_len).clone()

    else:
        # SDAR: independent per-token Bernoulli; curriculum optionally fixes t_i.
        for i in range(b):
            t_i = (
                fixed_t
                if fixed_t is not None
                else eps + (1.0 - eps) * torch.rand((), device=batch.device).item()
            )
            rand_vals = torch.rand(target_len, device=batch.device)
            masked_local = (rand_vals < t_i).nonzero(as_tuple=False).squeeze(1)
            if masked_local.numel() == 0:
                masked_local = torch.randint(0, target_len, (1,), device=batch.device)
            masked_global = response_indices[masked_local]
            mask_indices[i, masked_global] = True
            p_mask[i, masked_global] = t_i

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)
    return noisy_batch, mask_indices, p_mask


def build_gaussian_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    t_fixed: float,
    MASK_TOKEN_ID: int = 126336,
    block: int = -1,
    eps: float = 1e-6,
):
    """Apply noise to the response region at a fixed mask rate from a Gauss-Legendre quadrature point.

    For block==-1 (LLaDA path), the number of masked tokens is deterministic: round(t * response_len).
    For block>0 (SDAR path), each token is masked independently with Bernoulli(t_fixed).

    Args
    ----
    batch          : (b, seq_len) int64  – original token ids
    attention_mask : (seq_len,) int64    – 1 for valid tokens, 0 for padding
    prompt_len     : int  – number of prompt tokens
    t_fixed        : float – mask rate fixed at a Gauss-Legendre quadrature point in [0, 1]
    MASK_TOKEN_ID  : int
    block          : int  – -1 for LLaDA full-attn path; >0 for SDAR Bernoulli path
    eps            : float – lower bound applied to t_fixed

    Returns
    -------
    noisy_batch, mask_indices, p_mask  (all shape (b, seq_len))
    """
    b, seq_len = batch.shape

    response_mask = attention_mask.bool().clone()
    response_mask[:prompt_len] = False
    all_response_indices = torch.where(response_mask)[0]
    padded_resp_len = all_response_indices.numel()

    # Exclude trailing MASK tokens (post-first-EOS, decode_order=0, attn=1).
    actual_len = 0
    for j in range(padded_resp_len - 1, -1, -1):
        if batch[0, all_response_indices[j]] != MASK_TOKEN_ID:
            actual_len = j + 1
            break
    if actual_len == 0:
        return (
            batch.clone(),
            torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device),
            torch.zeros((b, seq_len), device=batch.device),
        )
    actual_end = prompt_len + actual_len
    response_indices = all_response_indices[all_response_indices < actual_end]
    target_len = response_indices.numel()

    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    p_mask = torch.zeros((b, seq_len), device=batch.device)

    t = max(float(t_fixed), eps)

    if block == -1:
        # LLaDA: deterministic mask count = round(t * response_len)
        num_mask = max(1, round(t * target_len))
        p_mask_val = num_mask / target_len
        for i in range(b):
            perm = torch.randperm(target_len, device=batch.device)
            mask_indices[i, response_indices[perm[:num_mask]]] = True
        p_mask[mask_indices] = p_mask_val

    else:
        # SDAR: independent per-token Bernoulli at fixed rate t
        for i in range(b):
            rand_vals = torch.rand(target_len, device=batch.device)
            masked_local = (rand_vals < t).nonzero(as_tuple=False).squeeze(1)
            if masked_local.numel() == 0:
                masked_local = torch.randint(0, target_len, (1,), device=batch.device)
            masked_global = response_indices[masked_local]
            mask_indices[i, masked_global] = True
            p_mask[i, masked_global] = t

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)
    return noisy_batch, mask_indices, p_mask


def build_block_random_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    block_size: int,
    eps: float = 1e-3,
    MASK_TOKEN_ID: int = 126336,
):
    """Block-causal noisy sampling aligned with temp/d3LLM ``forward_process_with_trajectory``.

    Actual response length is determined by scanning from the tail of the valid response
    region and skipping trailing ``<|mdm_mask|>`` tokens.  This correctly includes
    multiple ``<|endoftext|>`` tokens after the first EOS (they are real content, not
    padding) while excluding unresolved mask tokens at the end.

    Block selection (matches d3LLM LLaDA trainer)::

        max_blocks  = actual_len // block_size
        num_blocks  ~ Uniform{0, ..., max_blocks}
        mask_start  = prompt_len + num_blocks * block_size
        mask_end    = seq_len              if num_blocks == max_blocks
                    = mask_start + block_size  otherwise

    Within the target block, ``k ~ U[1, block_len]`` positions are randomly masked
    and contribute to the loss.  Positions in ``[mask_end, seq_len)`` are replaced
    with ``MASK_TOKEN_ID`` as future context (no loss), matching d3LLM.

    Args
    ----
    batch          : (b, seq_len) int64
    attention_mask : (seq_len,) int64  – 1 for valid tokens, 0 for padding
    prompt_len     : int               – start index of the response region
    block_size     : int > 0           – tokens per block

    Returns
    -------
    noisy_batch   : (b, seq_len) int64
    mask_indices  : (b, seq_len) bool  – True ONLY at randomly masked target-block positions
    p_mask        : (b, seq_len) float – k/block_len at masked target positions, 0 elsewhere
    """
    del eps  # API symmetry with other builders
    b, seq_len = batch.shape
    device = batch.device
    pos = torch.arange(seq_len, device=device)
    response_positions = torch.where(attention_mask.bool() & (pos >= prompt_len))[0]
    padded_resp_len = response_positions.numel()
    if padded_resp_len == 0:
        return (
            batch.clone(),
            torch.zeros((b, seq_len), dtype=torch.bool, device=device),
            torch.zeros((b, seq_len), device=device),
        )

    noisy_batch = batch.clone()
    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=device)
    p_mask = torch.zeros((b, seq_len), device=device)

    for i in range(b):
        # Scan from tail: skip trailing MASK tokens to find actual content length.
        actual_len = 0
        for j in range(padded_resp_len - 1, -1, -1):
            if batch[i, response_positions[j]] != MASK_TOKEN_ID:
                actual_len = j + 1
                break

        if actual_len == 0:
            continue

        max_blocks = actual_len // block_size
        num_blocks_to_mask = int(torch.randint(0, max_blocks + 1, (1,), device=device).item())
        mask_start = prompt_len + num_blocks_to_mask * block_size
        if num_blocks_to_mask == max_blocks:
            mask_end = seq_len
        else:
            mask_end = min(mask_start + block_size, seq_len)

        if mask_start >= mask_end:
            continue

        actual_end = prompt_len + actual_len
        target_positions = torch.arange(mask_start, mask_end, device=device)
        eligible = target_positions[
            attention_mask[target_positions].bool() & (target_positions < actual_end)
        ]
        if eligible.numel() == 0:
            eligible = target_positions[attention_mask[target_positions].bool()]
        if eligible.numel() == 0:
            continue

        k = int(torch.randint(1, eligible.numel() + 1, (1,), device=device).item())
        perm = torch.randperm(eligible.numel(), device=device)
        target_masked = eligible[perm[:k]]

        noisy_batch[i, target_masked] = MASK_TOKEN_ID
        mask_indices[i, target_masked] = True
        p_mask[i, target_masked] = k / eligible.numel()

        if mask_end < seq_len:
            future_positions = torch.arange(mask_end, seq_len, device=device)
            future_valid = future_positions[attention_mask[future_positions].bool()]
            if future_valid.numel() > 0:
                noisy_batch[i, future_valid] = MASK_TOKEN_ID

    return noisy_batch, mask_indices, p_mask


def build_d1_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    p: float = 0.15,
    MASK_TOKEN_ID: int = 126336,
):
    """D1 sampling: mask the entire response and randomly mask prompt tokens with probability p.

    Model-agnostic (no block branching); the same logic is used for both LLaDA and SDAR.

    Args
    ----
    batch          : (b, seq_len) int64  – original token ids
    attention_mask : (seq_len,) int64    – 1 for valid tokens, 0 for padding
    prompt_len     : int  – number of prompt tokens
    p              : float – probability of masking each prompt token
    MASK_TOKEN_ID  : int

    Returns
    -------
    noisy_batch, mask_indices, p_mask  (all shape (b, seq_len))
    """
    b, seq_len = batch.shape

    # Compute actual_end: exclude trailing MASK tokens (post-first-EOS,
    # decode_order=0, attn=1) that carry no real training signal.
    resp_region = attention_mask[prompt_len:].bool()
    resp_indices = resp_region.nonzero(as_tuple=False).squeeze(1)
    padded_resp_len = resp_indices.numel()
    actual_len = 0
    for j in range(padded_resp_len - 1, -1, -1):
        if batch[0, prompt_len + resp_indices[j]] != MASK_TOKEN_ID:
            actual_len = j + 1
            break
    # actual_resp_mask: only valid response positions up to actual_end
    actual_resp_mask = torch.zeros(seq_len, dtype=torch.bool, device=batch.device)
    if actual_len > 0:
        actual_end = prompt_len + actual_len
        actual_resp_mask[prompt_len:actual_end] = attention_mask[prompt_len:actual_end].bool()

    prompt_mask = attention_mask[:prompt_len].bool()
    random_mask = torch.rand((b, prompt_len), device=batch.device) < p

    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    mask_indices[:, :prompt_len] = prompt_mask & random_mask
    mask_indices[:, prompt_len:] = actual_resp_mask[prompt_len:]

    noisy_batch = torch.where(mask_indices, MASK_TOKEN_ID, batch)

    p_mask = torch.zeros((b, seq_len), device=batch.device)
    p_mask[:, :prompt_len] = p * prompt_mask.float()
    p_mask[:, prompt_len:] = actual_resp_mask[prompt_len:].float()

    return noisy_batch, mask_indices, p_mask


def build_trajectory_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    decode_orders: torch.Tensor,
    mask_ratio: float | None = None,
    eps: float = 1e-3,
    MASK_TOKEN_ID: int = 126336,
):
    """Trajectory-based noisy sampling using the actual rollout denoising path.

    Requires ``decode_orders_nfe`` from the sglang dLLM rollout engine
    (``record_decode_order: true`` in rollout config, supported by
    ``FullAttnMultiBlock``, ``LowConfidence``, ``EntropyThreshold``).

    ``decode_orders[j]`` is the **raw NFE step** at which position j was first
    unmasked during generation (0 for prompt tokens and padding).  Multiple
    positions may share the same NFE step value because the model can unmask
    several tokens in a single forward pass.  T = max(decode_orders) is the
    total number of NFE steps executed for this sequence.

    Without ``mask_ratio``, each row (independent MC sample) samples a discrete
    NFE step t uniformly from {0, …, T−1}.  With ``mask_ratio``, the target ratio
    is projected onto the resolved response tokens and the cutoff is moved to the
    beginning of the corresponding tied NFE group.  Tokens unmasked by the same
    forward therefore always remain masked or revealed together.

    The corresponding denoising state is:

        positions with decode_orders[j] ≤ t  : already revealed (keep original token)
        positions with decode_orders[j] > t  : still masked (replace with MASK_TOKEN_ID)
        t = 0                                 : all response tokens masked
        t = T−1                               : tokens unmasked at the last NFE step
                                                are still masked (can be > 1 token)

    Loss is computed on all still-masked response positions.

    ``p_mask[j]`` is the true marginal masking probability for position j under
    the uniform sampling of t::

        p_mask[j] = P(j is masked) = P(t < decode_orders[j])
                  = decode_orders[j] / T

    This is a per-token value (tokens revealed later have higher masking prob).

    Args
    ----
    batch          : (b, seq_len) int64  – original token ids
    attention_mask : (seq_len,) int64    – 1 for valid tokens; used to identify unresolved
                                           response positions when dllm_full_response_mask=True
                                           (where attention_mask=1 for the whole response window).
                                           With dllm_full_response_mask=False, positions after
                                           the first EOS have attention_mask=0 and are excluded.
    prompt_len     : int                 – number of prompt tokens (unused; prompt detection
                                           is done via decode_orders == 0)
    decode_orders  : (seq_len,) int64    – raw NFE step when each pos was unmasked;
                                           0 for prompt/padding AND for unresolved response
                                           positions (NFE budget exhausted).
                                           ``decode_orders_nfe`` field from the rollout batch
                                           (NOT the sequentialized ``decode_orders``).
    mask_ratio     : optional float in [0, 1] – curriculum target ratio; None keeps
                                           uniform trajectory-step sampling
    eps            : float               – unused (API symmetry)
    MASK_TOKEN_ID  : int

    Returns
    -------
    noisy_batch   : (b, seq_len) int64
    mask_indices  : (b, seq_len) bool   – True at still-masked response positions
    p_mask        : (b, seq_len) float  – decode_orders[j] / T for resolved positions,
                                          1.0 for unresolved response positions, 0 elsewhere
    """
    b, seq_len = batch.shape

    decode_orders = decode_orders.to(batch.device)
    attention_mask = attention_mask.to(batch.device)

    pos = torch.arange(seq_len, device=batch.device)
    # Candidate response region: attention_mask=1 and beyond prompt.
    # Per-sequence actual_end is computed below to exclude trailing MASK tokens
    # that were injected by the rollout post-processor (decode_order==0 but no
    # real training signal). Without this, dllm_full_response_mask=True would
    # include those positions in the loss, the same bug as build_block_trajectory.
    response_region_base = attention_mask.bool() & (pos >= prompt_len)
    response_positions = response_region_base.nonzero(as_tuple=False).squeeze(1)
    padded_resp_len = response_positions.numel()

    T = int(decode_orders.max().item())

    noisy_batch = batch.clone()
    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=batch.device)
    p_mask = torch.zeros((b, seq_len), device=batch.device)

    if T == 0 or padded_resp_len == 0:
        return noisy_batch, mask_indices, p_mask

    if mask_ratio is not None:
        mask_ratio = max(0.0, min(1.0, mask_ratio))

    for i in range(b):
        # Compute actual_end per-sequence: scan from tail, skip trailing MASK tokens.
        # This correctly excludes the post-first-EOS MASK tokens (decode_order==0,
        # attention_mask==1) that would otherwise look like unresolved positions.
        actual_len = 0
        for j in range(padded_resp_len - 1, -1, -1):
            if batch[i, response_positions[j]] != MASK_TOKEN_ID:
                actual_len = j + 1
                break
        if actual_len == 0:
            continue
        actual_end = prompt_len + actual_len
        # Per-sequence response region clipped to actual content.
        response_region = response_region_base & (pos < actual_end)

        if mask_ratio is not None:
            resolved_do = decode_orders[response_region & (decode_orders > 0)]
            t = (
                _curriculum_mask_threshold(resolved_do, mask_ratio)
                if resolved_do.numel() > 0
                else 0
            )
        elif T > 0:
            t = int(torch.randint(0, T, (), device=batch.device).item())
        else:
            t = 0

        # still_masked: response positions that are not yet revealed at step t.
        # decode_orders == 0 in response_region means unresolved (always masked).
        # decode_orders > t  means resolved but not yet revealed at this step.
        still_masked = response_region & ((decode_orders == 0) | (decode_orders > t))

        noisy_batch[i, still_masked] = MASK_TOKEN_ID
        mask_indices[i, still_masked] = True
        # p=1.0 for all masked positions: unbiased ELBO estimator (d3LLM-aligned).
        # Dividing by per-token decode_order/T (ELBO "correction") is mathematically
        # incorrect here — the sum-over-masked estimator is already unbiased without it.
        p_mask[i, still_masked] = 1.0

    return noisy_batch, mask_indices, p_mask


def _curriculum_mask_threshold(resolved_do: torch.Tensor, mask_ratio: float) -> int:
    """Threshold ``t`` so ``do > t`` masks whole tied NFE groups and cannot be empty.

    Keep the first ``k = int((1 - mask_ratio) * n)`` tokens by decode order. The
    first token past that prefix belongs to a tied group whose shared rank is
    ``g`` (the group's first 位次). Setting ``t = g - 1`` makes ``do > t`` mask
    that entire group and every later group — so a cutoff landing inside the
    last parallel step still contributes loss.
    """
    n = int(resolved_do.numel())
    k = int((1.0 - mask_ratio) * n)
    if k <= 0:
        return 0
    sorted_do, _ = torch.sort(resolved_do)
    if k >= n:
        return int(sorted_do[-1].item())
    g = int(sorted_do[k].item())
    return g - 1


def build_block_trajectory_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    block_size: int,
    decode_orders: torch.Tensor,
    mask_ratio: Optional[float] = None,
    eps: float = 1e-3,
    MASK_TOKEN_ID: int = 126336,
):
    """Block-trajectory noisy sampling: exact alignment with d3LLM ``forward_process_with_trajectory``.

    Combines block_random selection with trajectory-based masking within the chosen block:

    1. Compute actual response length by tail-scanning for the last non-MASK token.
    2. Uniformly select one of the ceil(actual_len / block_size) non-empty blocks.
    3. Within the selected block, sample NFE step t ~ Uniform{prefix_k, ..., prefix_k+T_k-1}
       where prefix_k = min(block decode_orders) - 1 (global-monotonic offset of this block)
       and T_k = block-local NFE count; positions with decode_orders > t are masked (using the
       rollout decode trajectory). Sampling from the block's OWN global NFE sub-range (not from
       0) gives every block ~uniform masking coverage and avoids the "later blocks are almost
       always fully masked" skew. Unresolved positions (decode_orders == 0, attention_mask == 1)
       are always masked.
    4. Future blocks [mask_end, seq_len) are filled with MASK_TOKEN_ID (no loss).
    5. Loss is computed ONLY on masked positions in the selected block.

    This matches d3LLM's ``forward_process_with_trajectory`` where per-block trajectory
    masking replaces a global mask-ratio heuristic.

    Args
    ----
    batch          : (b, seq_len) int64
    attention_mask : (seq_len,) int64  – controls which positions participate
    prompt_len     : int
    block_size     : int > 0
    decode_orders  : (seq_len,) int64  – ``decode_orders_nfe`` from rollout
    mask_ratio     : optional float in [0, 1] – curriculum ratio; None keeps
                     uniform block-local trajectory sampling

    Returns
    -------
    noisy_batch   : (b, seq_len) int64
    mask_indices  : (b, seq_len) bool  – True ONLY in selected block at masked positions
    p_mask        : (b, seq_len) float – decode_orders[j] / T in block, 1.0 for unresolved
    """
    del eps
    b, seq_len = batch.shape
    device = batch.device

    decode_orders = decode_orders.to(device)
    attention_mask = attention_mask.to(device)

    pos = torch.arange(seq_len, device=device)
    response_positions = torch.where(attention_mask.bool() & (pos >= prompt_len))[0]
    padded_resp_len = response_positions.numel()

    T = int(decode_orders.max().item())

    noisy_batch = batch.clone()
    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=device)
    p_mask = torch.zeros((b, seq_len), device=device)

    if padded_resp_len == 0 or T == 0:
        return noisy_batch, mask_indices, p_mask

    if mask_ratio is not None:
        mask_ratio = max(0.0, min(1.0, mask_ratio))

    for i in range(b):
        # Actual response length: scan from tail, skip trailing MASK tokens.
        actual_len = 0
        for j in range(padded_resp_len - 1, -1, -1):
            if batch[i, response_positions[j]] != MASK_TOKEN_ID:
                actual_len = j + 1
                break
        if actual_len == 0:
            continue

        num_blocks = (actual_len + block_size - 1) // block_size
        block_idx = int(torch.randint(0, num_blocks, (1,), device=device).item())
        mask_start = prompt_len + block_idx * block_size
        mask_end = min(mask_start + block_size, seq_len)

        if mask_start >= mask_end:
            continue

        # Block positions: valid (attention_mask=1) and within actual content
        # (< actual_end) within [mask_start, mask_end).
        # The actual_end filter is essential: trailing MASK tokens appended by
        # the rollout post-processor (decode_order==0, attention_mask==1) must
        # NOT enter the loss, matching build_block_random_noisy_batch behaviour.
        actual_end = prompt_len + actual_len
        block_positions = torch.arange(mask_start, mask_end, device=device)
        block_valid = block_positions[
            attention_mask[block_positions].bool() & (block_positions < actual_end)
        ]
        if block_valid.numel() == 0:
            continue

        block_do = decode_orders[block_valid]

        # Sample NFE step t from the BLOCK-LOCAL time axis, matching d3LLM's
        # "select_trajectory_by_ratio" which projects the denoising step onto the
        # current block rather than the whole sequence.
        #
        # decode_orders_nfe is GLOBALLY monotonic (see rollout _make_global_monotonic_nfe):
        # this block's resolved values live in [prefix_k+1, prefix_k+T_k], where prefix_k =
        # sum of previous windows' NFE counts. Sampling t from [0, T_block-1] would put the
        # whole [0, prefix_k] sub-range into "everything masked", so later blocks (large
        # prefix_k) would almost always be fully masked. Raise the lower bound to prefix_k =
        # min_resolved - 1 so t only spans this block's own NFE range; this is exactly
        # equivalent to block-local re-sequentialization and to d3LLM's non-curriculum
        # (uniform) select_trajectory_by_ratio (target_idx = block_start + num_unmasked).
        _block_resolved = block_do[block_do > 0]
        if _block_resolved.numel() > 0:
            if mask_ratio is None:
                T_block = int(_block_resolved.max().item())         # block's last global NFE step
                T_block_lo = int(_block_resolved.min().item()) - 1  # prefix_k ⇒ t=T_block_lo means full mask
                t = int(torch.randint(T_block_lo, T_block, (), device=device).item())  # U[T_block_lo, T_block-1]
            else:
                t = _curriculum_mask_threshold(_block_resolved, mask_ratio)
        else:
            t = 0  # all positions unresolved → will all be masked

        # Within the block: mask positions still unresolved at step t, plus unresolved ones.
        block_still_masked = (block_do == 0) | (block_do > t)
        target_masked = block_valid[block_still_masked]

        if target_masked.numel() == 0:
            continue

        noisy_batch[i, target_masked] = MASK_TOKEN_ID
        mask_indices[i, target_masked] = True
        # p=1.0 for all masked positions: unbiased ELBO estimator (d3LLM-aligned).
        p_mask[i, target_masked] = 1.0

        # Future blocks: fill with MASK (no loss), matching d3LLM noisy_batch[i, mask_end:l].
        if mask_end < seq_len:
            future_positions = torch.arange(mask_end, seq_len, device=device)
            future_valid = future_positions[attention_mask[future_positions].bool()]
            if future_valid.numel() > 0:
                noisy_batch[i, future_valid] = MASK_TOKEN_ID

    return noisy_batch, mask_indices, p_mask


def build_all_block_trajectory_noisy_batch(
    batch: torch.Tensor,
    attention_mask: torch.Tensor,
    prompt_len: int,
    block_size: int,
    decode_orders: torch.Tensor,
    mask_ratio: Optional[float] = None,
    eps: float = 1e-3,
    MASK_TOKEN_ID: int = 126336,
):
    """All-block trajectory noisy sampling for SDAR with LCMB rollout.

    Unlike ``build_block_trajectory_noisy_batch`` which selects a single block and
    masks future blocks (no loss), this variant applies trajectory masking to EVERY
    block simultaneously. By default each block independently samples ``t``; when
    ``mask_ratio`` is provided, each block instead projects that ratio onto its
    trajectory (tied NFE groups are kept atomic: if the group's first 位次 is
    past the keep prefix, the whole group is masked), enabling curriculum
    learning without a separate backend.

    This exploits SDAR's block-causal forward (BD trick), which computes logits for
    ALL blocks in a single pass.  With ``block_size = LCMB_BLOCK_SIZE`` (= NUM_BLOCKS
    x BASE_BLOCK_SIZE), each "block" corresponds to one LCMB decode window, and ALL
    windows contribute to the distillation loss in one actor forward.

    Efficiency vs ``block_trajectory`` (response=768, block_size=8):
      - mc_num=1 here -> gradient signal from 96 windows per forward
      - block_trajectory mc_num=k -> signal from k individual windows per forward

    Block NFE ranges: LCMB decodes each window (``block_size`` tokens) with an
    independent NFE counter from 1 to T_window <= block_size.  Sampling
    ``t ~ Uniform[0, T_window - 1]`` per window and masking ``decode_order > t``
    reproduces the block-causal noisy trajectory for that window.

    Args
    ----
    batch          : (b, seq_len) int64
    attention_mask : (seq_len,) int64  - 1 for real tokens, 0 for padding
    prompt_len     : int
    block_size     : int > 0  - LCMB window size (= NUM_BLOCKS x BASE_BLOCK_SIZE)
    decode_orders  : (seq_len,) int64  - per-token NFE values from LCMB rollout
    mask_ratio     : optional float in [0, 1] - curriculum mask ratio shared by all
                     windows in this draw; None keeps uniform trajectory sampling

    Returns
    -------
    noisy_batch   : (b, seq_len) int64
    mask_indices  : (b, seq_len) bool  - True at positions included in the loss
    p_mask        : (b, seq_len) float - 1.0 at every loss-masked position
    """
    del eps
    b, seq_len = batch.shape
    device = batch.device

    decode_orders = decode_orders.to(device)
    attention_mask = attention_mask.to(device)

    pos = torch.arange(seq_len, device=device)
    response_positions = torch.where(attention_mask.bool() & (pos >= prompt_len))[0]
    padded_resp_len = response_positions.numel()

    noisy_batch = batch.clone()
    mask_indices = torch.zeros((b, seq_len), dtype=torch.bool, device=device)
    p_mask = torch.zeros((b, seq_len), device=device)

    if padded_resp_len == 0:
        return noisy_batch, mask_indices, p_mask

    if mask_ratio is not None:
        mask_ratio = max(0.0, min(1.0, mask_ratio))

    for i in range(b):
        # Actual response length: scan from tail, skip trailing MASK tokens.
        actual_len = 0
        for j in range(padded_resp_len - 1, -1, -1):
            if batch[i, response_positions[j]] != MASK_TOKEN_ID:
                actual_len = j + 1
                break
        if actual_len == 0:
            continue

        actual_end = prompt_len + actual_len
        num_windows = actual_len // block_size

        for w in range(num_windows):
            win_start = prompt_len + w * block_size
            win_end = win_start + block_size

            win_positions = torch.arange(win_start, win_end, device=device)
            win_valid = win_positions[
                attention_mask[win_positions].bool() & (win_positions < actual_end)
            ]
            if win_valid.numel() == 0:
                continue

            win_do_raw = decode_orders[win_valid]

            # Remap global-monotonic NFE → window-local ranks 1..T_win.
            # Rollout stores decode_orders_nfe with per-window cumulative offsets, so
            # window-47 may look like [369,369,370,…] instead of [1,1,2,…].  Remapping
            # by UNIQUE positive NFE values (not per-token position rank) restores a
            # compact local axis while preserving same-step ties: tokens unmasked in
            # the same decode forward keep the same local rank and are masked/revealed
            # together.  T_win = #distinct positive NFE steps in this window.
            _resolved = win_do_raw[win_do_raw > 0]
            if _resolved.numel() > 0:
                unique_nfe = torch.unique(_resolved, sorted=True)  # ascending distinct steps
                # searchsorted: unique_nfe[i] → local rank i+1; unresolved (0) stay 0.
                win_do = torch.zeros_like(win_do_raw)
                pos_mask = win_do_raw > 0
                win_do[pos_mask] = (
                    torch.searchsorted(unique_nfe, win_do_raw[pos_mask]) + 1
                )
                T_win = int(unique_nfe.numel())
                if mask_ratio is None:
                    t = int(torch.randint(0, T_win, (), device=device).item())
                else:
                    t = _curriculum_mask_threshold(win_do[pos_mask], mask_ratio)
            else:
                win_do = win_do_raw
                t = 0  # all unresolved -> mask all

            win_still_masked = (win_do == 0) | (win_do > t)
            target_masked = win_valid[win_still_masked]

            if target_masked.numel() == 0:
                continue

            noisy_batch[i, target_masked] = MASK_TOKEN_ID
            mask_indices[i, target_masked] = True
            p_mask[i, target_masked] = 1.0

    return noisy_batch, mask_indices, p_mask
