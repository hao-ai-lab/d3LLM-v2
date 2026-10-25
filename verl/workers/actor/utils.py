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
"""Common utility functions for actor workers."""

from typing import Tuple, Union
import importlib.util
import os

import torch
from torch.nn.attention.flex_attention import (
    BlockMask,
    create_block_mask,
    flex_attention as _flex_attention,
)


def _compact_sequences(
    *tensors: torch.Tensor,
    attention_mask: torch.Tensor,
    pad_values,
    block_size: int,
) -> Tuple[torch.Tensor, ...]:
    """Move leading padding to the tail for every sample in the batch (vectorised, no Python loops).

    Supports 2D (bs, seq_len) and 3D (bs, N, seq_len) tensors; all are reordered along the last
    dimension.  The sort index is computed once from attention_mask and reused for 3D tensors via
    unsqueeze+expand.

    Strategy: argsort each row by attention_mask descending (1s first, 0s last, stable), then
    gather all tensors with the same index, and masked_fill padding positions with pad_value.

    Args
    ----
    tensors        : tensors to compact in sync, each shape (bs, L) or (bs, N, L)
    attention_mask : (bs, L) int64  – 1 for valid tokens, 0 for padding
    pad_values     : fill value for each tensor's padding positions, in the same order as tensors
    block_size     : passed through for API compatibility; caller must ensure seq_len % block_size == 0

    Returns
    -------
    Tuple of compacted tensors in the same order as the inputs, with leading padding moved to tail
    """
    # sort index: within each row, 1s (valid) come before 0s (padding); stable preserves relative order
    idx_2d = attention_mask.long().argsort(
        dim=-1, descending=True, stable=True
    )  # (bs, L)
    new_attn = attention_mask.gather(1, idx_2d)  # (bs, L)
    pad_2d = ~new_attn.bool()  # (bs, L)

    results = []
    for tensor, pad_val in zip(tensors, pad_values):
        if tensor.dim() == 2:  # (bs, L)
            idx, pad_mask = idx_2d, pad_2d
        else:  # (bs, N, L)
            N = tensor.shape[1]
            idx = idx_2d.unsqueeze(1).expand(-1, N, -1)  # (bs, N, L)
            pad_mask = pad_2d.unsqueeze(1).expand(-1, N, -1)  # (bs, N, L)

        if tensor.dtype == torch.bool:
            gathered = tensor.long().gather(-1, idx).bool()
            gathered = gathered.masked_fill(pad_mask, bool(pad_val))
        else:
            gathered = tensor.gather(-1, idx)
            if pad_val != 0:
                gathered = gathered.masked_fill(pad_mask, pad_val)
        results.append(gathered)

    return tuple(results)


def project_compact_to_response(
    compact_tensor: torch.Tensor,
    prompt_valid: torch.Tensor,
    r_valid: torch.Tensor,
    response_length: int,
) -> torch.Tensor:
    """Project a tensor from compact space back to the original response slot.

    After _compact_sequences moves leading padding to the tail, each sequence has layout:
        compact space : [prompt_tokens | response_tokens | PAD ...]

    This function gathers, for each sample b, the response tokens at positions
    [prompt_valid[b], prompt_valid[b] + r_valid[b]) in compact space and places them at
    positions [0, r_valid[b]) in the output; padding positions (j >= r_valid[b]) are zeroed.

    Args
    ----
    compact_tensor : (bs, seq_len) or (bs, D, seq_len) – tensor in compact space
    prompt_valid   : (bs,) int  – number of prompt tokens per sample in compact space
    r_valid        : (bs,) int  – number of valid response tokens per sample
    response_length: int  – length of the response slice in the original space (includes response padding)

    Returns
    -------
    projected : last dimension replaced by response_length, zeroed at response padding positions
                shape: (bs, response_length)     when input is 2D
                       (bs, D, response_length)  when input is 3D
    """
    device = compact_tensor.device
    seq_length = compact_tensor.shape[-1]

    # position of response token j in compact space is prompt_valid[b] + j
    gather_idx = (
        prompt_valid.unsqueeze(1)  # (bs, 1)
        + torch.arange(response_length, device=device).unsqueeze(0)  # (1, resp_len)
    ).clamp(max=seq_length - 1)  # (bs, response_length)

    valid_resp_mask = torch.arange(response_length, device=device).unsqueeze(
        0
    ) < r_valid.unsqueeze(1)  # (bs, response_length)

    if compact_tensor.dim() == 2:
        projected = compact_tensor.gather(1, gather_idx)
        return projected * valid_resp_mask.to(projected.dtype)
    else:
        D = compact_tensor.shape[1]
        gather_idx_d = gather_idx.unsqueeze(1).expand(-1, D, -1)
        projected = compact_tensor.gather(2, gather_idx_d)
        return projected * valid_resp_mask.unsqueeze(1).to(projected.dtype)


def _triton_available() -> bool:
    if os.environ.get("TORCHDYNAMO_DISABLE", "0") not in ("0", ""):
        return False
    if importlib.util.find_spec("triton") is None:
        return False
    return True


_USE_FLEX_ATTN: bool = _triton_available()


def build_block_causal_attn_mask(
    seq_len: int,
    padding_mask: torch.BoolTensor,
    block_size: int,
    history_window_size: int = None,
) -> Union[BlockMask, torch.Tensor]:
    """Build the attention mask for a Block-Diffusion (BD) doubled sequence.

    The input sequence is constructed by prepare_block_inputs and has layout:
        [xt_b0(block_size), xt_b1(block_size), ..., x0_b0(block_size), x0_b1(block_size), ...]
    i.e. all noisy blocks first, then all clean blocks.

    Default (``history_window_size=None``) — standard SDAR block-causal rules:
      M_BD  : noisy query (xt_q) attends to noisy keys (xt_k) in the same original block
               (bidirectional within block)
      M_OBC : noisy query (xt_q) attends to clean keys (x0_k) from strictly earlier blocks
               (full clean history)
      M_BC  : clean query (x0_q) attends to clean keys (x0_k) in the same or earlier blocks

    LCMB-window-aware (``history_window_size`` set, e.g. = NUM_BLOCKS * block_size) —
    matches the LowConfidenceMultiBlock rollout, where several base blocks are decoded
    jointly inside one decode window.  At rollout time a later base block in the SAME
    window sees its predecessors' partially-decoded (noisy) state, not their clean
    ground truth.  To remove this train/inference mismatch the clean-history rule is
    split by window (``window = block_idx // blocks_per_window``):
      M_BD       : noisy query attends to noisy keys in the same base block (unchanged)
      M_WIN_HIST : noisy query attends to NOISY keys of earlier base blocks in the SAME
                   window  (in-window history is noisy, mirroring rollout)
      M_OUT_HIST : noisy query attends to CLEAN keys of strictly-earlier windows
                   (cross-window history stays clean, as in block diffusion)
      M_BC       : clean query attends to clean keys in the same/earlier block (unchanged)
    With ``history_window_size == block_size`` (one block per window) this reduces
    exactly to the default rules.

    Args
    ----
    seq_len             : total length of the doubled sequence (= 2 * n_blocks * block_size)
    padding_mask        : (bs, seq_len) BoolTensor – True for valid tokens
    block_size          : original (base) block size (> 0)
    history_window_size : LCMB decode-window size (multiple of block_size).  None or
                          <= block_size ⇒ default SDAR rules (no behavior change).

    Returns
    -------
    BlockMask when Triton is available (TORCHDYNAMO_DISABLE=0); otherwise (bs, 1, seq_len, seq_len) float additive mask
    """
    bs = padding_mask.shape[0]
    device = padding_mask.device
    half_len = seq_len // 2  # = n_blocks * block_size

    # blocks_per_window = 1 disables the window split → identical to the default rules.
    if history_window_size and history_window_size > block_size:
        blocks_per_window = history_window_size // block_size
    else:
        blocks_per_window = 1

    if _USE_FLEX_ATTN:

        def block_causal_mask_mod(b, h, q, k):
            is_xt_q = q < half_len
            is_xt_k = k < half_len
            q_block = torch.where(
                is_xt_q,
                q // block_size,
                (q - half_len) // block_size,
            )
            kv_block = torch.where(
                is_xt_k,
                k // block_size,
                (k - half_len) // block_size,
            )
            q_win = q_block // blocks_per_window
            kv_win = kv_block // blocks_per_window
            m_bd = is_xt_q & is_xt_k & (q_block == kv_block)
            # in-window history: attend earlier NOISY blocks of the same window
            m_win_hist = (
                is_xt_q & is_xt_k & (q_win == kv_win) & (q_block > kv_block)
            )
            # cross-window history: attend CLEAN blocks of strictly-earlier windows
            m_out_hist = is_xt_q & ~is_xt_k & (q_win > kv_win)
            m_bc = ~is_xt_q & ~is_xt_k & (q_block >= kv_block)
            return (m_bd | m_win_hist | m_out_hist | m_bc) & padding_mask[b, q] & padding_mask[b, k]

        return create_block_mask(
            block_causal_mask_mod,
            B=bs,
            H=None,
            Q_LEN=seq_len,
            KV_LEN=seq_len,
            device=device,
            _compile=False,
        )

    positions = torch.arange(seq_len, device=device)
    # First half: noisy tokens (xt); second half: clean tokens (x0)
    is_xt = positions < half_len
    orig_block = torch.where(
        is_xt,
        positions // block_size,
        (positions - half_len) // block_size,
    )
    orig_win = orig_block // blocks_per_window

    q_block = orig_block.view(1, seq_len, 1)
    kv_block = orig_block.view(1, 1, seq_len)
    q_win = orig_win.view(1, seq_len, 1)
    kv_win = orig_win.view(1, 1, seq_len)
    q_is_xt = is_xt.view(1, seq_len, 1)
    kv_is_xt = is_xt.view(1, 1, seq_len)

    m_bd = q_is_xt & kv_is_xt & (q_block == kv_block)
    m_win_hist = q_is_xt & kv_is_xt & (q_win == kv_win) & (q_block > kv_block)
    m_out_hist = q_is_xt & ~kv_is_xt & (q_win > kv_win)
    m_bc = ~q_is_xt & ~kv_is_xt & (q_block >= kv_block)

    base_mask = (
        (m_bd | m_win_hist | m_out_hist | m_bc)
        & padding_mask.unsqueeze(1)
        & padding_mask.unsqueeze(2)
    )
    additive = torch.zeros(
        bs, 1, seq_len, seq_len, dtype=torch.float32, device=device
    )
    additive.masked_fill_(~base_mask.unsqueeze(1), float("-inf"))
    return additive


def prepare_block_inputs(
    noisy_all: torch.Tensor,
    orig_all: torch.Tensor,
    attn_all: torch.Tensor,
    pos_all: torch.Tensor,
    mask_all: torch.Tensor,
    block_size: int,
    seq_length: int,
    history_window_size: int = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Concatenate noisy and clean blocks and build the block-causal attention mask.

    Each forward pass in the block-DLLM path requires the model to see all noisy blocks
    followed by all clean blocks so that a block-causal mask can let each noisy block
    attend to the preceding clean blocks.

    Sequence layout: [xt_b0, xt_b1, ..., xt_b(n-1), x0_b0, x0_b1, ..., x0_b(n-1)]

    Args
    ----
    noisy_all  : (batch*mc, seq_len) – noisy token ids
    orig_all   : (batch*mc, seq_len) – clean ground-truth token ids
    attn_all   : (batch*mc, seq_len) – padding attention mask (1/0)
    pos_all    : (batch*mc, seq_len) – RoPE position ids
    mask_all   : (batch*mc, seq_len) – bool, True at masked (prediction target) positions
    block_size : int                 – number of tokens per block
    seq_length : int                 – original seq_len (before doubling)
    history_window_size : int or None – LCMB decode-window size; forwarded to
                          build_block_causal_attn_mask (None ⇒ default SDAR rules).

    Returns
    -------
    seq_d      : (batch*mc, 2*seq_len) – doubled input ids for the model
    attn_d     : (batch*mc, 2*seq_len) – doubled attention mask
    pos_d      : (batch*mc, 2*seq_len) – doubled position ids
    keep_d     : (batch*mc, 2*seq_len) – True only at noisy-block positions that need log_prob output
    block_mask : (batch*mc, 2*seq_len, 2*seq_len) – block-causal 2D attention bias
    """
    seq_d  = torch.cat([noisy_all, orig_all], dim=1)
    attn_d = torch.cat([attn_all, attn_all], dim=1)
    pos_d  = torch.cat([pos_all, pos_all], dim=1)
    keep_d = torch.cat([mask_all, torch.zeros_like(mask_all)], dim=1).bool()
    block_mask = build_block_causal_attn_mask(
        2 * seq_length, attn_d.bool(), block_size, history_window_size=history_window_size
    )
    return seq_d, attn_d, pos_d, keep_d, block_mask


def flatten_mc_inputs(
    perturbed_seq: torch.Tensor,
    mask_indices: torch.Tensor,
    orig_seq: torch.Tensor,
    attention_mask: torch.Tensor,
    p_mask: torch.Tensor,
    pos_all: torch.Tensor,
    bmc: int,
    seq_length: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Flatten (batch, mc_num, seq_len) inputs into (batch*mc, seq_len) for model forward.

    Args
    ----
    perturbed_seq  : (batch, mc_num, seq_len) – noisy token ids
    mask_indices   : (batch, mc_num, seq_len) – True at masked positions
    orig_seq       : (batch, seq_len)          – original clean token ids
    attention_mask : (batch, seq_len)          – 1 for real tokens, 0 for padding
    p_mask         : (batch, mc_num, seq_len)  – per-token mask probabilities
    pos_all        : (batch*mc, seq_len)       – position ids (caller computes; differs between
                                                 block path and full-attention path)
    bmc            : int                       – batch * mc_num
    seq_length     : int                       – seq_len

    Returns (all shape (batch*mc, seq_len))
    -------
    noisy_all, mask_all, orig_all, attn_all, pos_all, pmask_all, labels_all
    """
    mc_num     = perturbed_seq.size(1)
    noisy_all  = perturbed_seq.reshape(bmc, seq_length)
    mask_all   = mask_indices.reshape(bmc, seq_length).bool()
    orig_all   = orig_seq.unsqueeze(1).expand(-1, mc_num, -1).reshape(bmc, seq_length)
    attn_all   = attention_mask.unsqueeze(1).expand(-1, mc_num, -1).reshape(bmc, seq_length)
    pmask_all  = p_mask.reshape(bmc, seq_length)
    labels_all = orig_all.masked_fill(~mask_all, -100)
    return noisy_all, mask_all, orig_all, attn_all, pos_all, pmask_all, labels_all
