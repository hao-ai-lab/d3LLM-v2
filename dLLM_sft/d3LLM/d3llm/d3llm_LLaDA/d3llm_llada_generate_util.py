# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA

import torch
import numpy as np
import torch.nn.functional as F
from transformers import AutoTokenizer
from transformers.cache_utils import DynamicCache
import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
from utils.utils_LLaDA.model.modeling_llada import LLaDAModelLM


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def handle_early_stop(x, block_states, eos_token_id, prompt_length, mask_token_id=None, debug=False):
    """
    Early stop: find first EOS, set all tokens after it to EOS (not mask), mark subsequent blocks as complete.
    Returns: (has_eos, first_eos_abs_pos)
    """
    if eos_token_id is None:
        return False, None
    
    # Only check non-mask tokens in generation region
    gen_region = x[:, prompt_length:]
    eos_mask = (gen_region == eos_token_id)
    
    if not eos_mask.any():
        return False, None
    
    # Find first EOS position (only among decoded, non-mask tokens)
    pos = torch.arange(gen_region.shape[1], device=x.device).unsqueeze(0)
    first_eos_rel = torch.where(eos_mask, pos, gen_region.shape[1]).amin(dim=1)
    first_eos_abs = prompt_length + first_eos_rel[0].item()
    
    if debug:
        print(f"[EarlyStop] Found first EOS at position {first_eos_abs} (relative: {first_eos_rel[0].item()})")
        # Count non-mask tokens before EOS
        if mask_token_id is not None:
            decoded_before_eos = ((x[:, prompt_length:first_eos_abs+1] != mask_token_id).sum().item())
            print(f"[EarlyStop] Decoded tokens before (and including) EOS: {decoded_before_eos}")
    
    # Set all tokens after first EOS to EOS (NOT mask!)
    x[:, first_eos_abs+1:] = eos_token_id
    
    # Update block states based on EOS position
    blocks_marked = 0
    blocks_updated = 0
    for bid in sorted(block_states.keys()):
        if bid > 0:
            start, end = block_states[bid]["start"], block_states[bid]["end"]
            
            if start > first_eos_abs:
                # Block entirely after EOS - mark as complete with no masks
                if block_states[bid]["mask_count"] != 0:
                    block_states[bid]["mask_count"] = 0
                    block_states[bid]["is_complete"] = True
                    blocks_marked += 1
            elif end > first_eos_abs and start <= first_eos_abs:
                # Block CONTAINS EOS - recalculate mask_count for portion before EOS
                if mask_token_id is not None:
                    # Only count masks from block start to EOS position (inclusive)
                    masks_before_eos = (x[:, start:first_eos_abs+1] == mask_token_id).sum().item()
                    old_count = block_states[bid]["mask_count"]
                    if masks_before_eos != old_count:
                        block_states[bid]["mask_count"] = masks_before_eos
                        blocks_updated += 1
                        if masks_before_eos == 0:
                            block_states[bid]["is_complete"] = True
    
    if debug:
        if blocks_marked > 0:
            print(f"[EarlyStop] Marked {blocks_marked} blocks after EOS as complete")
        if blocks_updated > 0:
            print(f"[EarlyStop] Updated {blocks_updated} blocks containing EOS")
    
    return True, first_eos_abs


def get_transfer_index(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


@torch.no_grad()
def generate_multi_block(
    model,
    prompt,
    steps=128,
    max_new_tokens=512,
    block_size=32,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=0.5,
    block_add_threshold=0.5,
    decoded_token_threshold=0.5,
    eos_token_id=None,
    early_stop=False,
):
    """
    Pipelined parallel decoding without cache.

    Args:
        block_add_threshold: Add new block when last block progress >= this threshold. 
                            Set to 1.0 for fully sequential processing (like generate()).
        decoded_token_threshold: Block becomes fully activated when previous block progress >= this threshold.
                                Set to 1.0 for fully sequential processing (like generate()).
        threshold: Entropy threshold for decoding (lower entropy = higher confidence). 
                   Tokens with entropy > threshold are skipped.
                   Same semantics as in generate() method. Typical value: 0.5
        early_stop: If True, stop generation when EOS token is encountered.
    
    When block_add_threshold=1.0 and decoded_token_threshold=1.0, this method behaves 
    identically to generate() with sequential block processing.
    """
    x = torch.full((1, prompt.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    prompt_length = prompt.shape[1]
    
    # Only use eos_token_id if early_stop is enabled
    if not early_stop:
        eos_token_id = None

    # Track block states: {block_id: {start, end, mask_count, total_masks, is_complete}}
    # Initialize with prompt block
    block_states = {
        0: {
            "start": 0,
            "end": prompt.shape[1],
            "mask_count": 0,
            "total_masks": prompt.shape[1],
            "is_complete": True,
        }
    }
    
    # Create first generation block
    num_blocks = max_new_tokens // block_size
    next_block_id = 1
    if next_block_id <= num_blocks:
        block_start = prompt.shape[1] + (next_block_id - 1) * block_size
        block_end = min(block_start + block_size, prompt.shape[1] + max_new_tokens)
        # First block should be immediately activated since prompt (block 0) is already complete
        should_activate = 1.0 >= decoded_token_threshold  # prompt progress is always 1.0
        block_states[next_block_id] = {
            "start": block_start,
            "end": block_end,
            "mask_count": block_end - block_start,
            "total_masks": block_end - block_start,
            "is_complete": should_activate,
        }
        next_block_id += 1

    nfe = 0
    has_eos = False
    
    while True:
        # Check if all blocks are exhausted AND no more blocks to create
        mask_index = x == mask_id
        total_masks = mask_index[:, prompt_length:].sum()
        
        if total_masks == 0 and next_block_id > num_blocks:
            break
        
        nfe += 1

        # Early stop: handle EOS tokens
        if early_stop and eos_token_id is not None:
            has_eos, first_eos_abs = handle_early_stop(x, block_states, eos_token_id, prompt_length, 
                                                        mask_token_id=mask_id, debug=False)
            if has_eos:
                # Create all missing blocks after EOS and mark them as complete
                while next_block_id <= num_blocks:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_start > first_eos_abs:
                        block_states[next_block_id] = {
                            "start": block_start, "end": block_end, "mask_count": 0,
                            "total_masks": block_end - block_start, "is_complete": True,
                        }
                        next_block_id += 1
                    else:
                        break
                # Recalculate total_masks
                total_masks = (x == mask_id)[:, prompt_length:].sum()
                if total_masks == 0:
                    break

        # Update block completion states
        def update_block_activation_states():
            """Update which blocks should be fully activated based on previous block progress."""
            for bid in sorted(block_states.keys()):
                if bid > 0 and not block_states[bid]["is_complete"]:
                    prev_progress = (
                        1
                        - block_states[bid - 1]["mask_count"]
                        / block_states[bid - 1]["total_masks"]
                    )
                    if prev_progress >= decoded_token_threshold:
                        block_states[bid]["is_complete"] = True
        
        update_block_activation_states()
        
        # Add new block dynamically based on last block's progress (skip if EOS detected)
        if next_block_id <= num_blocks and not has_eos:
            last_bid = max(block_states.keys())
            if last_bid > 0:  # Not just prompt
                last_progress = (
                    1
                    - block_states[last_bid]["mask_count"]
                    / block_states[last_bid]["total_masks"]
                )
                # Create next block when:
                # 1. Last block progress >= block_add_threshold (for parallel processing), OR
                # 2. Last block is complete (mask_count == 0) for sequential processing
                should_add_block = (last_progress >= block_add_threshold) or (block_states[last_bid]["mask_count"] == 0)
                
                if should_add_block:
                    # Add next block
                    block_start = prompt.shape[1] + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt.shape[1] + max_new_tokens)
                    if block_end > block_start:
                        # Check how many positions in this block are actually masked
                        actual_mask_count = (x[:, block_start:block_end] == mask_id).sum().item()
                        
                        # Determine if this block should be immediately activated
                        # Check if previous block is complete enough
                        prev_bid = next_block_id - 1
                        prev_progress = (
                            1 - block_states[prev_bid]["mask_count"] / block_states[prev_bid]["total_masks"]
                        )
                        should_activate = prev_progress >= decoded_token_threshold
                        
                        block_states[next_block_id] = {
                            "start": block_start,
                            "end": block_end,
                            "mask_count": actual_mask_count,
                            "total_masks": block_end - block_start,
                            "is_complete": should_activate,
                        }
                        next_block_id += 1

        # Forward pass: only process up to the last complete or semi-activated block
        # Find the rightmost block that should be processed
        rightmost_active_bid = 0
        for bid in sorted(block_states.keys()):
            if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                rightmost_active_bid = bid
        
        if rightmost_active_bid == 0:
            break
            
        active_end = block_states[rightmost_active_bid]["end"]
        
        # Always do forward pass on entire sequence (like generate() does)
        logits = model(x).logits
        
        # Mask out future blocks (positions after active_end) to prevent them from being decoded
        # This is the same as: mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0 in generate()
        mask_index_for_decode = mask_index.clone()
        mask_index_for_decode[:, active_end:] = 0
        
        # Decode using full-sequence approach (like generate())
        x0, transfer_index = get_transfer_index_entropy(
            logits,
            temperature,
            remasking,
            mask_index_for_decode,
            x,
            None,
            threshold if threshold is not None else 999.0,
        )
        
        # For fully activated blocks, ensure at least one token is decoded (guaranteed progress)
        # Find the first fully activated block with masks
        first_fully_activated_bid = None
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                first_fully_activated_bid = bid
                break
        
        if first_fully_activated_bid is not None:
            # Check if any token was decoded in this block
            start, end = block_states[first_fully_activated_bid]["start"], block_states[first_fully_activated_bid]["end"]
            block_transfer = transfer_index[:, start:end]
            
            if not block_transfer.any():
                # Force decode the lowest entropy token in this fully activated block
                p = F.softmax(logits[:, start:end].to(torch.float64), dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
                block_mask = mask_index_for_decode[:, start:end]
                entropy = torch.where(block_mask, entropy, torch.inf)
                best_idx = entropy[0].argmin()
                transfer_index[0, start + best_idx] = True
                x0_resample = torch.argmax(logits[0, start + best_idx], dim=-1)
                x0[0, start + best_idx] = x0_resample
        
        # Apply the decoded tokens
        x[transfer_index] = x0[transfer_index]
        
        # Update block states based on which positions were decoded
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["mask_count"] > 0:
                start, end = block_states[bid]["start"], block_states[bid]["end"]
                block_decoded = transfer_index[:, start:end].sum().item()
                if block_decoded > 0:
                    block_states[bid]["mask_count"] -= block_decoded

        if nfe > 10000:
            break

    return x, nfe


@torch.no_grad()
def generate_multi_block_kv_cache(
    model,
    prompt,
    steps=128,
    max_new_tokens=512,
    block_size=32,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=0.5,
    block_add_threshold=0.5,
    decoded_token_threshold=0.5,
    cache_delay_iter=10000,
    refresh_interval=10000,
    lazy_cache_refresh_num=0,
    eos_token_id=None,
    early_stop=False,
):
    """
    Pipelined parallel decoding with Delayed KV-Cache.
    
    Strategy: Wait cache_delay_iter steps after a block is fully decoded before caching it.
    This allows token representations to stabilize in the diffusion process.
    """
    # Only use eos_token_id if early_stop is enabled
    if not early_stop:
        eos_token_id = None
    x = torch.full((1, prompt.shape[1] + max_new_tokens), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()
    
    prompt_length = prompt.shape[1]
    num_blocks = max_new_tokens // block_size
    
    # Track block states with delayed caching support
    block_states = {
        0: {
            "start": 0,
            "end": prompt_length,
            "mask_count": 0,
            "total_masks": prompt_length,
            "is_complete": True,
            "completed_at_nfe": 0,
            "is_cached": False,
        }
    }
    
    # Create first generation block
    next_block_id = 1
    if next_block_id <= num_blocks:
        block_start = prompt_length + (next_block_id - 1) * block_size
        block_end = min(block_start + block_size, prompt_length + max_new_tokens)
        should_activate = 1.0 >= decoded_token_threshold
        block_states[next_block_id] = {
            "start": block_start,
            "end": block_end,
            "mask_count": block_end - block_start,
            "total_masks": block_end - block_start,
            "is_complete": should_activate,
            "completed_at_nfe": None,
            "is_cached": False,
        }
        next_block_id += 1
    
    past_key_values = None
    manual_cache_length = 0
    nfe = 0
    has_eos = False
    
    def update_block_activation_states():
        """Update which blocks should be fully activated based on previous block progress."""
        for bid in sorted(block_states.keys()):
            if bid > 0 and not block_states[bid]["is_complete"]:
                prev_progress = 1 - block_states[bid - 1]["mask_count"] / block_states[bid - 1]["total_masks"]
                if prev_progress >= decoded_token_threshold:
                    block_states[bid]["is_complete"] = True
    
    while True:
        # Check termination
        mask_index = x == mask_id
        total_masks = mask_index[:, prompt_length:].sum()
        
        if total_masks == 0 and next_block_id > num_blocks:
            break
        
        nfe += 1
        
        # Early stop: handle EOS tokens
        if early_stop and eos_token_id is not None:
            has_eos, first_eos_abs = handle_early_stop(x, block_states, eos_token_id, prompt_length,
                                                        mask_token_id=mask_id, debug=False)
            if has_eos:
                # Create all missing blocks after EOS and mark them as complete
                while next_block_id <= num_blocks:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_start > first_eos_abs:
                        block_states[next_block_id] = {
                            "start": block_start, "end": block_end, "mask_count": 0,
                            "total_masks": block_end - block_start, "is_complete": True,
                            "completed_at_nfe": nfe, "is_cached": True,
                        }
                        next_block_id += 1
                    else:
                        break
                # Recalculate total_masks
                total_masks = (x == mask_id)[:, prompt_length:].sum()
                if total_masks == 0:
                    break
        
        # Update block activation states
        update_block_activation_states()
        
        # Add new block dynamically based on last block's progress (skip if EOS detected)
        if next_block_id <= num_blocks and not has_eos:
            last_bid = max(block_states.keys())
            if last_bid > 0:
                last_progress = 1 - block_states[last_bid]["mask_count"] / block_states[last_bid]["total_masks"]
                should_add_block = (last_progress >= block_add_threshold) or (block_states[last_bid]["mask_count"] == 0)
                
                if should_add_block:
                    block_start = prompt_length + (next_block_id - 1) * block_size
                    block_end = min(block_start + block_size, prompt_length + max_new_tokens)
                    if block_end > block_start:
                        actual_mask_count = (x[:, block_start:block_end] == mask_id).sum().item()
                        prev_progress = 1 - block_states[next_block_id - 1]["mask_count"] / block_states[next_block_id - 1]["total_masks"]
                        should_activate = prev_progress >= decoded_token_threshold
                        
                        block_states[next_block_id] = {
                            "start": block_start,
                            "end": block_end,
                            "mask_count": actual_mask_count,
                            "total_masks": block_end - block_start,
                            "is_complete": should_activate,
                            "completed_at_nfe": None,
                            "is_cached": False,
                        }
                        next_block_id += 1
        
        # Find rightmost active block
        rightmost_active_bid = 0
        for bid in sorted(block_states.keys()):
            if block_states[bid]["is_complete"] or block_states[bid]["mask_count"] > 0:
                rightmost_active_bid = bid
        
        if rightmost_active_bid == 0:
            break
        
        active_end = block_states[rightmost_active_bid]["end"]
        
        refresh_output = None

        cache_length = manual_cache_length
        
        # Update completion timestamps and find blocks ready to cache
        blocks_to_cache = []
        for bid in sorted(block_states.keys()):
            if block_states[bid]["end"] <= cache_length:
                continue  # Already cached
            
            if block_states[bid]["mask_count"] == 0:  # Fully decoded
                if block_states[bid]["completed_at_nfe"] is None:
                    block_states[bid]["completed_at_nfe"] = nfe
                
                delay = nfe - block_states[bid]["completed_at_nfe"]
                if delay >= cache_delay_iter and not block_states[bid]["is_cached"]:
                    blocks_to_cache.append(bid)
                    block_states[bid]["is_cached"] = True
                elif delay < cache_delay_iter:
                    # This block is stabilizing, cannot cache subsequent blocks
                    break
            else:
                break  # Stop at first incomplete block
        
        # Determine update_kvcache: how many new tokens to add to cache
        update_kvcache = 0
        if blocks_to_cache:
            latest_to_cache = max(blocks_to_cache)
            update_kvcache = block_states[latest_to_cache]["end"] - cache_length
        
        # Determine forward strategy
        # Find the earliest position that needs to be forwarded (not cached)
        forward_start_pos = cache_length  # Start from end of cached region
        
        # Check if there are completed but not cached blocks (in stabilization period)
        has_stabilizing_blocks = False
        for bid in sorted(block_states.keys()):
            if (block_states[bid]["mask_count"] == 0 and 
                block_states[bid]["completed_at_nfe"] is not None and
                not block_states[bid]["is_cached"]):
                # This block is completed but not cached (stabilizing)
                has_stabilizing_blocks = True
                forward_start_pos = min(forward_start_pos, block_states[bid]["start"])
        
        # Also include active blocks
        for bid in sorted(block_states.keys()):
            if block_states[bid]["mask_count"] > 0:
                forward_start_pos = min(forward_start_pos, block_states[bid]["start"])
        
        # If forward_start_pos >= active_end, no blocks need forwarding
        if forward_start_pos >= active_end:
            break
        
        # Determine forward strategy
        if update_kvcache > 0:
            # Need to cache new blocks: forward from cache_length to end
            input_seq = x[:, cache_length:]
            process_start_pos = cache_length
            use_kv_cache = True
        elif past_key_values is not None and not has_stabilizing_blocks and nfe % refresh_interval != 0:
            # Have cache and no stabilizing blocks: can use cache
            input_seq = x[:, forward_start_pos:]
            process_start_pos = forward_start_pos
            use_kv_cache = True
        elif past_key_values is not None and (has_stabilizing_blocks or nfe % refresh_interval == 0):
            # Refresh cache when: 1) has stabilizing blocks, or 2) periodic refresh interval
            # print(f"[DEBUG-KV] has_stabilizing_blocks={has_stabilizing_blocks}, nfe={nfe}, refresh_interval={refresh_interval}")
            if has_stabilizing_blocks:
                latest_stabilizing_bid = max([bid for bid in block_states.keys()
                                             if block_states[bid]["mask_count"] == 0 and 
                                                block_states[bid]["completed_at_nfe"] is not None and
                                                not block_states[bid]["is_cached"]])
                refresh_pos = block_states[latest_stabilizing_bid]["start"]
            else:
                # Periodic refresh: refresh up to cache_length
                refresh_pos = cache_length
            
            # Full forward to get fresh cache (only once for all completed blocks)
            refresh_output = model(x, use_cache=True)
            refresh_cache = refresh_output.past_key_values
            
            # Truncate cache to refresh_pos (mimics prefix_cache strategy)
            past_key_values = DynamicCache()
            for layer_idx in range(len(refresh_cache)):
                k, v = refresh_cache[layer_idx]
                past_key_values.update(k[:, :, :refresh_pos, :], v[:, :, :refresh_pos, :], layer_idx)
            
            input_seq = x
            process_start_pos = 0
            use_kv_cache = False
        else:
            # No cache OR has stabilizing blocks: forward full sequence without cache
            input_seq = x
            process_start_pos = 0
            use_kv_cache = False
        
        stabilizing_blocks = []
        for bid in sorted(block_states.keys()):
            if block_states[bid]["mask_count"] == 0 and block_states[bid]["completed_at_nfe"] is not None:
                if not block_states[bid]["is_cached"]:
                    stabilizing_blocks.append(bid)
                    
        # Forward pass: reuse refresh_output if possible to avoid redundant forward
        if refresh_output is not None and process_start_pos == 0:
            # Refresh forward already computed full sequence, reuse it
            outputs = refresh_output
            logits = refresh_output.logits
        elif use_kv_cache:
            # Pass cache_position for correct positional encoding
            cache_position = torch.arange(
                manual_cache_length, 
                manual_cache_length + input_seq.shape[1], 
                device=model.device
            )
            outputs = model(
                input_seq,
                past_key_values=past_key_values,
                use_cache=True,
                update_kvcache=update_kvcache if update_kvcache > 0 else 0,
                cache_position=cache_position
            )
            logits = outputs.logits
        else:
            outputs = model(input_seq)
            logits = outputs.logits
        
        # Update cache and manual cache length only when we have blocks to cache
        if update_kvcache > 0:
            past_key_values = outputs.past_key_values
            manual_cache_length += update_kvcache
        
        # Map logits to full sequence
        logits_full = torch.full((1, x.shape[1], logits.shape[-1]), 
                                 float('-inf'), dtype=logits.dtype, device=logits.device)
        logits_end = process_start_pos + logits.shape[1]
        logits_full[:, process_start_pos:logits_end, :] = logits
        
        # Mask out future blocks
        mask_index_for_decode = mask_index.clone()
        mask_index_for_decode[:, active_end:] = 0
        
        # Decode with entropy threshold
        x0, transfer_index = get_transfer_index_entropy(
            logits_full,
            temperature,
            remasking,
            mask_index_for_decode,
            x,
            None,
            threshold if threshold is not None else 999.0,
        )
        
        # Ensure progress in fully activated blocks
        first_fully_activated_bid = None
        for bid in sorted(block_states.keys()):
            if bid > 0 and block_states[bid]["is_complete"] and block_states[bid]["mask_count"] > 0:
                first_fully_activated_bid = bid
                break
        
        if first_fully_activated_bid is not None:
            start, end = block_states[first_fully_activated_bid]["start"], block_states[first_fully_activated_bid]["end"]
            block_transfer = transfer_index[:, start:end]
            
            if not block_transfer.any():
                # Force decode the lowest entropy token
                p = F.softmax(logits_full[:, start:end].to(torch.float64), dim=-1)
                entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)
                block_mask = mask_index_for_decode[:, start:end]
                entropy = torch.where(block_mask, entropy, torch.inf)
                best_idx = entropy[0].argmin()
                transfer_index[0, start + best_idx] = True
                x0_resample = torch.argmax(logits_full[0, start + best_idx], dim=-1)
                x0[0, start + best_idx] = x0_resample
        
        # Apply decoded tokens
        x[transfer_index] = x0[transfer_index]
        
        # Update block mask counts
        for bid in sorted(block_states.keys()):
            if bid > 0:
                start, end = block_states[bid]["start"], block_states[bid]["end"]
                block_decoded = transfer_index[:, start:end].sum().item()
                if block_decoded > 0:
                    block_states[bid]["mask_count"] = max(0, block_states[bid]["mask_count"] - block_decoded)
        
        if nfe > 10000:
            break
    
    
    return x, nfe

@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        cfg_scale: Unsupervised classifier-free guidance scale.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0

            if threshold is not None:
                x0, transfer_index = get_transfer_index_entropy(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )
            else:
                x0, transfer_index = get_transfer_index(
                    logits,
                    temperature,
                    remasking,
                    mask_index,
                    x,
                    num_transfer_tokens[:, i] if threshold is None else None,
                    threshold,
                )

            x[transfer_index] = x0[transfer_index]
            i += 1
            if (
                x[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break
    return x, nfe


def get_transfer_index_entropy(
    logits,
    temperature,
    remasking,
    mask_index,
    x,
    num_transfer_tokens,
    entropy_threshold=None,
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    # Calculate entropy instead of confidence
    p = F.softmax(logits.to(torch.float64), dim=-1)

    if remasking == "low_confidence":
        # Calculate entropy: -sum(p * log(p))
        entropy = -torch.sum(p * torch.log(p + 1e-12), dim=-1)  # b, l
    elif remasking == "random":
        # For random remasking, use random entropy values
        entropy = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)

    # Use entropy instead of confidence (note: higher entropy means lower confidence)
    # So we set entropy to +inf for non-masked positions (to exclude them from selection)
    entropy_for_selection = torch.where(mask_index, entropy, torch.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)

    if entropy_threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)

    for j in range(entropy_for_selection.shape[0]):
        # Select tokens with lowest entropy (highest confidence)
        # topk with largest=False gives us the smallest values (lowest entropy)
        _, select_index = torch.topk(
            entropy_for_selection[j], k=num_transfer_tokens[j], largest=False
        )
        transfer_index[j, select_index] = True

        if entropy_threshold is not None:
            # Only keep tokens with entropy below threshold (high confidence)
            # If entropy > threshold, set transfer_index to False
            for k in range(
                1, num_transfer_tokens[j]
            ):  # Skip the first one (lowest entropy)
                if entropy[j, select_index[k]] > entropy_threshold:
                    transfer_index[j, select_index[k]] = False

    return x0, transfer_index
    


def main():
    device = "cuda"

    model = (
        LLaDAModelLM.from_pretrained(
            "Zigeng/dParallel-LLaDA-8B-instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "Zigeng/dParallel-LLaDA-8B-instruct", trust_remote_code=True
    )

    prompt = "Natalia sold clips to 48 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May? Please reason step by step, and put your final answer within \\boxed{}."

    # Add special tokens for the Instruct model. The Base model does not require the following two lines.
    m = [
        {"role": "user", "content": prompt},
    ]
    prompt = tokenizer.apply_chat_template(
        m, add_generation_prompt=True, tokenize=False
    )

    input_ids = tokenizer(prompt)["input_ids"]
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)

    out = generate(
        model,
        input_ids,
        steps=256,
        gen_length=256,
        block_length=32,
        temperature=0.0,
        threshold=0.5,
        remasking="low_confidence",
    )
    print(
        tokenizer.batch_decode(
            out[0][:, input_ids.shape[1] :], skip_special_tokens=True
        )[0]
    )
    print("Decoding Steps:", out[1])


if __name__ == "__main__":
    main()
