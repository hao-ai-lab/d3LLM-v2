"""LowConfidence multi-block decoding for causal block-diffusion LLMs (SDAR, LLaDA 2.x).

Architecture note (SDAR / causal block-diffusion):
  - Causal block attention: block-k tokens attend causally to blocks 0..k-1 and
    bidirectionally within block-k only. Block-0 therefore sees exactly the same
    context as LowConfidence(block_size=base_block_size) — its decoding is identical.
  - save_kv_cache = False by default; the mandatory final forward writes KV so
    the next block's prefix is correct.

Multi-block strategy (block_size = num_blocks * base_block_size):
  - Joint convergence loop over the full extend window (all blocks simultaneously).
  - Causal inter-block mask: block-k attends causally to blocks 0..k-1 and
    bidirectionally within block-k → block 0 decodes identically to baseline.
  - block_add_threshold gates each block: block k only accepts tokens once block k-1
    has reached block_add_threshold fraction decoded (default 0.99 → block 1 only
    activates when block 0 is essentially done).
  - ONE shared mandatory final at the very end (not per-block).
  - Expected NFE per call: sum_k(conv_k) + 1.  With bat=0.99 and 2 blocks:
    ≈ 0.97 + 0.97 + 1 = 2.94 vs baseline 2×1.97=3.94 → ~25% TPF improvement.

Causal block mask fix:
  - The custom_mask for flashinfer's ragged wrapper must have exactly
    sum(extend_len_i^2) elements for the batch. We compute it per-sequence based
    on each sequence's actual generation length (not a fixed block_size^2 per item).
    This prevents attention errors for sequences shorter than block_size (last block).

Ablation (block_add_threshold ≥ 1.0 → block_size == base_block_size in sglang_dllm.py):
  - Delegates to _run_single_block(), a byte-for-byte copy of LowConfidence.run().
  - Identical NFE / TPF / accuracy to LowConfidence.
"""

from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner

_D = False  # debug flag
_D_Num = 128  # print debug info for the first _D_Num samples only

def _is_rank0():
    import torch.distributed as dist
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


class LowConfidenceMultiBlock(DllmAlgorithm):
    """Multi-block parallel decoding with confidence threshold for causal dLLMs.

    YAML config keys (algorithm_config):
      threshold               - confidence cutoff for token acceptance  (default: 0.95)
      base_block_size         - model's native block width in tokens    (default: 4)
      block_add_threshold     - prev-block progress fraction to activate next block (default: 0.1)
      decoded_token_threshold - prev-block progress fraction to give next block its
                                conservative floor                      (default: 0.95)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.base_block_size = config.algorithm_config.get("base_block_size", 4)
        self.block_add_threshold = config.algorithm_config.get("block_add_threshold", 0.1)
        self.decoded_token_threshold = config.algorithm_config.get("decoded_token_threshold", 0.95)
        self.record_decode_order = config.algorithm_config.get("record_decode_order", False)
        self._d_cnt = 0  # counts run() calls for _D_Num gating
        print(
            "\033[91m",
            f"LowConfidenceMultiBlock: threshold={self.threshold}, "
            f"window(block_size)={self.block_size}, base_block_size={self.base_block_size}, "
            f"block_add_threshold={self.block_add_threshold}, "
            f"decoded_token_threshold={self.decoded_token_threshold}, "
            f"record_decode_order={self.record_decode_order}",
            "\033[0m",
            flush=True,
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        # _d: whether to print debug for this sample (run() call), rank-0 only
        _d = _D and _is_rank0() and self._d_cnt < _D_Num
        _sid = self._d_cnt  # sample index for logging
        self._d_cnt += 1
        # Ablation: single-block window → delegate to exact LowConfidence logic.
        if self.block_size == self.base_block_size:
            return self._run_single_block(model_runner, forward_batch, _d, _sid)
        return self._run_multi_block(model_runner, forward_batch, _d, _sid)

    # ── Ablation path: byte-for-byte identical to LowConfidence.run() ───────────────────────
    def _run_single_block(self, model_runner, forward_batch, _d, _sid):
        """Exact LowConfidence logic including mandatory final forward → identical NFE/TPF."""
        batch_size = forward_batch.batch_size
        item_lens = forward_batch.extend_seq_lens_cpu
        offsets = [0]
        for l in item_lens:
            offsets.append(offsets[-1] + l)
        start_list = []
        mask_index = forward_batch.input_ids == self.mask_id

        if _d:
            _prefix_lens = list(forward_batch.extend_prefix_lens_cpu) if getattr(forward_batch, 'extend_prefix_lens_cpu', None) is not None else []
            _block_offsets = getattr(forward_batch, 'dllm_block_offsets', None) or []
            print(
                f"\033[95m[MB_SINGLE RUN] sample={_sid}, bs={batch_size}, extend_lens={list(item_lens)}, "
                f"prefix_lens={_prefix_lens}, block_offsets={_block_offsets}, "
                f"total_masks={torch.sum(mask_index).item()}, "
                f"input_ids_len={len(forward_batch.input_ids)}\033[0m", flush=True,
            )

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            next_token_ids = []
            if _d:
                print(f"\033[95m[MB_SINGLE FAST_PATH] sample={_sid}, no masks, nfe=0, returning 0 tokens\033[0m", flush=True)
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each batch item
        for i in range(batch_size):
            block_input_ids = forward_batch.input_ids[offsets[i]:offsets[i + 1]]
            block_mask_index = block_input_ids == self.mask_id
            start = item_lens[i] - torch.sum(block_mask_index).item()
            start_list.append(start)

        # Decode-order tracking (same semantics as LowConfidence)
        decode_orders = None
        if self.record_decode_order:
            decode_orders = [
                torch.zeros(item_lens[i], dtype=torch.long, device=forward_batch.input_ids.device)
                for i in range(batch_size)
            ]

        if _d:
            for i in range(batch_size):
                s, e = offsets[i], offsets[i + 1]
                ids = forward_batch.input_ids[s:e]
                mask_pos = (ids == self.mask_id).nonzero(as_tuple=True)[0].tolist()
                non_mask_ids = ids[ids != self.mask_id][-5:].tolist() if (ids != self.mask_id).any() else []
                print(
                    f"\033[95m[MB_SINGLE INIT] sample={_sid}, i={i}, extend_len={item_lens[i]}, start={start_list[i]}, "
                    f"n_masks={len(mask_pos)}, mask_positions={mask_pos}, "
                    f"last_5_non_mask_ids={non_mask_ids}\033[0m", flush=True,
                )

        nfe = 0
        logits_output = None
        can_run_cuda_graph = False
        for iter_idx in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                if _d:
                    print(f"\033[95m[MB_SINGLE LOOP] sample={_sid}, iter={iter_idx}, all masks done, breaking\033[0m", flush=True)
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            nfe += 1
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph

            # Apply invalid-token mask (token_id >= tokenizer_vocab_size → -inf).
            self._apply_invalid_token_mask(logits_output.full_logits)

            for batch_id in range(batch_size):
                s, e = offsets[batch_id], offsets[batch_id + 1]
                block_input_ids = forward_batch.input_ids[s:e]
                block_mask_index = block_input_ids == self.mask_id
                if torch.sum(block_mask_index).item() == 0:
                    continue
                curr_logits = logits_output.full_logits[s:e]

                # Temperature-based sampling: use temperature from sampling_info.
                # NOTE: SGLang SamplingParams converts temperature=0 → temperature=1.0 + top_k=1
                # for greedy decoding. top_k==1 is the authoritative greedy signal.
                temp = 0.0
                if forward_batch.sampling_info is not None:
                    temp = forward_batch.sampling_info.temperatures[batch_id].item()
                    if forward_batch.sampling_info.top_ks[batch_id].item() == 1:
                        temp = 0.0
                if temp > 1e-6:
                    # Stochastic: sample from softmax(logits / temperature)
                    probs = F.softmax(curr_logits / temp, dim=-1)
                    x = torch.multinomial(probs, num_samples=1).squeeze(-1)
                else:
                    # Greedy: argmax (original behavior)
                    x = torch.argmax(curr_logits, dim=-1)
                p = torch.squeeze(
                    torch.gather(
                        F.softmax(curr_logits, dim=-1),
                        dim=-1,
                        index=torch.unsqueeze(x, -1),
                    ),
                    -1,
                )
                x = torch.where(block_mask_index, x, block_input_ids)
                confidence = torch.where(block_mask_index, p, -np.inf)

                transfer_index = confidence > self.threshold

                floor_forced = False
                if transfer_index.sum().item() == 0:
                    _, select_index = torch.topk(confidence, k=1)
                    transfer_index[select_index] = True
                    floor_forced = True

                if _d:
                    mask_positions = block_mask_index.nonzero(as_tuple=True)[0].tolist()
                    accepted_positions = transfer_index.nonzero(as_tuple=True)[0].tolist()
                    accepted_ids = x[transfer_index].tolist()
                    accepted_confs = confidence[transfer_index].tolist()
                    masks_remaining = block_mask_index.sum().item() - transfer_index.sum().item()
                    print(
                        f"\033[93m[MB_SINGLE ITER] sample={_sid}, nfe={nfe}, iter={iter_idx}, batch={batch_id}, "
                        f"masks_before={len(mask_positions)}, accepted={len(accepted_positions)}, "
                        f"masks_after={masks_remaining}, floor={floor_forced}, "
                        f"positions={accepted_positions}, "
                        f"token_ids={accepted_ids}, "
                        f"confs=[{', '.join(f'{c:.4f}' for c in accepted_confs)}]\033[0m",
                        flush=True,
                    )

                # Record decode order: position unmasked at NFE step `nfe`
                if decode_orders is not None:
                    decode_orders[batch_id][transfer_index] = nfe

                block_input_ids[transfer_index] = x[transfer_index]

        # Mandatory final forward: writes correct KV cache for all decoded positions.
        # SDAR uses KV caching (needs_full_prefill=False); the last loop forward has
        # MASK embeddings for positions decoded in that iteration. This forward corrects
        # those so the next block's prefix is accurate.
        # NOT counted in nfe — it is a KV-cache implementation artifact, not a
        # denoising step. NFE measures only token-producing forwards.
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        self._apply_invalid_token_mask(logits_output.full_logits)
        # Build per-sequence next_token_ids using variable-length offsets
        next_token_ids_list = [
            forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
            for i in range(batch_size)
        ]
        customized = {"nfe": [nfe] * batch_size}
        if decode_orders is not None:
            customized["decode_order"] = [d.cpu().tolist() for d in decode_orders]
        logits_output.customized_info = customized

        if _d:
            for i in range(batch_size):
                toks = next_token_ids_list[i].tolist()
                print(
                    f"\033[96m[MB_SINGLE DONE] sample={_sid}, nfe={nfe}, batch={i}, "
                    f"out_len={len(toks)}, out_ids={toks}\033[0m",
                    flush=True,
                )

        return logits_output, next_token_ids_list, can_run_cuda_graph

    # ── Multi-block sequential decoding with prime-from-mandatory optimization ───────────────
    def _causal_block_mask(self, gen_lengths: List[int], device) -> torch.Tensor:
        """Causal inter-block mask for flashinfer wrapper_ragged (within-extend attention).

        SDAR block-k attends causally to blocks 0..k-1 and bidirectionally within block-k.
        With a flat extend window of block_size tokens, the ragged wrapper uses full
        bidirectional attention by default; this mask restores the correct causal-block pattern.

        Shape: sum(g_i * g_i for g_i in gen_lengths) uint8, 1=attend 0=mask.
        Each sequence has its own g_i x g_i mask based on its actual extend length.
        """
        b = self.base_block_size
        masks = []
        for g in gen_lengths:
            row_blk = torch.arange(g, device=device) // b  # (g,)
            col_blk = torch.arange(g, device=device) // b  # (g,)
            per_item = (row_blk.unsqueeze(1) >= col_blk.unsqueeze(0)).to(torch.uint8)  # (g, g)
            masks.append(per_item.reshape(-1))
        return torch.cat(masks)  # (sum of g_i^2,)

    def _accept_from_logits(self, logits, ids_slice, mask_in_block, temp, conservative_floor, threshold=None):
        """Accept tokens from logits with threshold; apply conservative floor if needed.

        Returns (transfer_mask, x_predictions, p_confidences) for debug.
        Modifies ids_slice in-place.
        """
        if threshold is None:
            threshold = self.threshold
        if temp > 1e-6:
            x = torch.multinomial(F.softmax(logits / temp, dim=-1), 1).squeeze(-1)
        else:
            x = torch.argmax(logits, dim=-1)
        p = F.softmax(logits, dim=-1).gather(-1, x.unsqueeze(-1)).squeeze(-1)
        transfer = (p > threshold) & mask_in_block
        floor_forced = False
        if conservative_floor and transfer.sum().item() == 0 and mask_in_block.any():
            conf = p.clone()
            conf[~mask_in_block] = float("-inf")
            transfer[conf.argmax()] = True
            floor_forced = True
        ids_slice[transfer] = x[transfer]
        return transfer, x, p, floor_forced

    def _run_multi_block(self, model_runner, forward_batch, _d, _sid):
        """Joint convergence for all blocks with causal inter-block mask and bat-gated activation.

        Strategy:
          Single convergence loop over the full extend window (all blocks).
          The causal block mask ensures block-k attends only to blocks 0..k (causally)
          so block 0 decodes identically to the baseline single-block path.
          block_add_threshold gates each block: block k only accepts tokens once block k-1
          has reached block_add_threshold progress — preventing contaminated speculative
          decodes from noisy-context earlier blocks.
          ONE mandatory final at the very end (shared across all blocks).

        NFE per call = conv_iters_until_all_done + 1 mandatory.
        With bat=0.99: block 1 activates only when block 0 is fully decoded.
          conv_0 ≈ 0.97 iters (clean, identical to baseline).
          conv_1 ≈ 0.97 iters (clean context after block 0 done).
          Total ≈ 2.94 NFE for 8 tokens vs baseline 2×1.97=3.94 → ~34% TPF improvement.
        """
        batch_size = forward_batch.batch_size
        item_lens = forward_batch.extend_seq_lens_cpu
        offsets = [0]
        for l in item_lens:
            offsets.append(offsets[-1] + l)

        if _d:
            _prefix_lens = list(forward_batch.extend_prefix_lens_cpu) if getattr(forward_batch, 'extend_prefix_lens_cpu', None) is not None else []
            _block_offsets = getattr(forward_batch, 'dllm_block_offsets', None) or []
            _total_masks = (forward_batch.input_ids == self.mask_id).sum().item()
            print(
                f"\033[95m[MB_MULTI RUN] sample={_sid}, bs={batch_size}, extend_lens={list(item_lens)}, "
                f"prefix_lens={_prefix_lens}, block_offsets={_block_offsets}, "
                f"total_masks={_total_masks}, block_size={self.block_size}, "
                f"base_block_size={self.base_block_size}, bat={self.block_add_threshold}, "
                f"dtt={self.decoded_token_threshold}\033[0m", flush=True,
            )

        # Fast path: no masks
        if (forward_batch.input_ids == self.mask_id).sum().item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            if _d:
                print(f"\033[95m[MB_MULTI FAST_PATH] sample={_sid}, no masks, nfe=0, returning 0 tokens\033[0m", flush=True)
            return out.logits_output, [], out.can_run_graph

        # Per-item setup
        start_list, gen_lengths = [], []
        for i in range(batch_size):
            ids = forward_batch.input_ids[offsets[i] : offsets[i + 1]]
            n = (ids == self.mask_id).sum().item()
            start_list.append(item_lens[i] - n)
            gen_lengths.append(n)

        # Decode-order tracking: decode_orders[i][j] = NFE step when position j was unmasked
        # (0 = prompt / never masked).  Mirrors LowConfidence semantics across multiple blocks.
        decode_orders = None
        if self.record_decode_order:
            decode_orders = [
                torch.zeros(item_lens[i], dtype=torch.long, device=forward_batch.input_ids.device)
                for i in range(batch_size)
            ]

        # Build flat block table: item_blocks[i][k] = {start, end, total, mask_count}
        # start/end are LOCAL indices within item i's extend window.
        item_blocks = []
        for i in range(batch_size):
            n_blocks = (gen_lengths[i] + self.base_block_size - 1) // self.base_block_size
            blocks = []
            for k in range(n_blocks):
                bs = start_list[i] + k * self.base_block_size
                be = min(bs + self.base_block_size, start_list[i] + gen_lengths[i])
                blocks.append({"start": bs, "end": be, "total": be - bs, "mask_count": be - bs})
            item_blocks.append(blocks)
            if _d:
                print(
                    f"\033[95m[MB_MULTI INIT] sample={_sid}, i={i}, gen={gen_lengths[i]}, n_blocks={n_blocks}, "
                    f"base_bs={self.base_block_size}, blocks=["
                    + ", ".join(f"({b['start']}-{b['end']}, masks={b['mask_count']})" for b in blocks)
                    + "]\033[0m", flush=True,
                )

        # Set causal inter-block mask: block-k attends causally to blocks 0..k-1,
        # bidirectionally within block-k.  Block 0 never sees block 1 (MASK) tokens →
        # block 0 decodes identically to the baseline single-block path.
        _updater = getattr(forward_batch.attn_backend, "indices_updater_prefill", None)
        if _updater is not None and hasattr(_updater, "dllm_causal_block_mask"):
            mask_tensor = self._causal_block_mask(gen_lengths, forward_batch.input_ids.device)
            _updater.dllm_causal_block_mask = mask_tensor
            if _d:
                print(
                    f"\033[95m[MB_MULTI CAUSAL_MASK] sample={_sid}, set mask, shape={tuple(mask_tensor.shape)}, "
                    f"total_elements={mask_tensor.numel()}, gen_lens={gen_lengths}\033[0m",
                    flush=True,
                )
                if _sid == 0:
                    n_show = min(64, mask_tensor.numel())
                    print(
                        f"\033[95m[MB_MULTI CAUSAL_MASK SAMPLE] sample={_sid}, first_{n_show}="
                        f"{mask_tensor[:n_show].tolist()}\033[0m",
                        flush=True,
                    )

        nfe = 0
        logits_output = None
        can_run_cuda_graph = False

        for conv_iter in range(self.block_size):
            # Check if all masks across the entire extend window are done.
            if (forward_batch.input_ids == self.mask_id).sum().item() == 0:
                if _d:
                    print(f"\033[95m[MB_MULTI LOOP] sample={_sid}, iter={conv_iter}, all masks done, breaking\033[0m", flush=True)
                break
            # Snapshot mask counts before acceptance so block-k gating uses predecessor
            # progress from the last completed forward pass (avoids stale-logits gating).
            snapshot_mask_counts = [
                [blk["mask_count"] for blk in item_blocks[i]]
                for i in range(batch_size)
            ]

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            nfe += 1
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph
            self._apply_invalid_token_mask(logits_output.full_logits)

            for i in range(batch_size):
                temp = 0.0
                if forward_batch.sampling_info is not None:
                    temp = forward_batch.sampling_info.temperatures[i].item()
                    if forward_batch.sampling_info.top_ks[i].item() == 1:
                        temp = 0.0
                for k, blk in enumerate(item_blocks[i]):
                    if blk["mask_count"] == 0:
                        continue
                    bs = offsets[i] + blk["start"]
                    be = offsets[i] + blk["end"]
                    ids_slice = forward_batch.input_ids[bs:be]
                    mask_in_block = ids_slice == self.mask_id
                    if not mask_in_block.any():
                        blk["mask_count"] = 0
                        continue
                    block_logits = logits_output.full_logits[bs:be]

                    # Determine acceptance strategy based on block index and predecessor progress
                    if k == 0:
                        # Block 0: always normal threshold + floor (baseline-identical).
                        transfer, x, p, floor_forced = self._accept_from_logits(
                            block_logits, ids_slice, mask_in_block, temp,
                            conservative_floor=True, threshold=None,
                        )
                        if _d and temp <= 1e-6:
                            x_ref = torch.argmax(block_logits, dim=-1)
                            p_ref = F.softmax(block_logits, dim=-1).gather(-1, x_ref.unsqueeze(-1)).squeeze(-1)
                            conf_ref = torch.where(mask_in_block, p_ref, -np.inf)
                            transfer_ref = conf_ref > self.threshold
                            ref_floor_forced = False
                            if transfer_ref.sum().item() == 0 and mask_in_block.any():
                                _, select_index = torch.topk(conf_ref, k=1)
                                transfer_ref[select_index] = True
                                ref_floor_forced = True
                            if _sid < 16:
                                mb_pos = transfer.nonzero(as_tuple=True)[0].tolist()
                                lc_pos = transfer_ref.nonzero(as_tuple=True)[0].tolist()
                                print(
                                    f"\033[92m[MB_MULTI BLOCK0 ALIGN] sample={_sid}, nfe={nfe}, iter={conv_iter}, "
                                    f"batch={i}, match={torch.equal(transfer, transfer_ref)}, "
                                    f"mb_pos={mb_pos}, lc_pos={lc_pos}, "
                                    f"mb_floor={floor_forced}, lc_floor={ref_floor_forced}\033[0m",
                                    flush=True,
                                )
                        strategy = "conservative(block0)"
                    else:
                        prev = item_blocks[i][k - 1]
                        prev_mc = snapshot_mask_counts[i][k - 1]
                        prev_progress = 1.0 - prev_mc / prev["total"]
                        if prev_progress < self.block_add_threshold:
                            # Predecessor not ready: skip acceptance entirely.
                            if _d:
                                print(
                                    f"\033[90m[MB_MULTI SKIP] sample={_sid}, nfe={nfe}, iter={conv_iter}, batch={i}, "
                                    f"block={k}, prev_progress={prev_progress:.2f} < bat={self.block_add_threshold}\033[0m",
                                    flush=True,
                                )
                            continue
                        elif prev_progress < self.decoded_token_threshold:
                            # Predecessor not fully decoded: use normal threshold WITH floor.
                            # This ensures block-k makes progress even when block k-1 is still converging.
                            transfer, x, p, floor_forced = self._accept_from_logits(
                                block_logits, ids_slice, mask_in_block, temp,
                                conservative_floor=True, threshold=None,
                            )
                            strategy = f"speculative(prev={prev_progress:.2f},thr={self.threshold})"
                        else:
                            # Predecessor fully decoded: normal threshold + floor (same as block0).
                            transfer, x, p, floor_forced = self._accept_from_logits(
                                block_logits, ids_slice, mask_in_block, temp,
                                conservative_floor=True, threshold=None,
                            )
                            strategy = f"conservative(prev_done)"

                    if _d:
                        mask_positions = mask_in_block.nonzero(as_tuple=True)[0].tolist()
                        accepted_positions = transfer.nonzero(as_tuple=True)[0].tolist()
                        # Map local block positions to global extend positions
                        global_accepted = [blk["start"] + pos for pos in accepted_positions]
                        accepted_ids = x[transfer].tolist()
                        accepted_confs = p[transfer].tolist()
                        masks_remaining = mask_in_block.sum().item() - transfer.sum().item()
                        print(
                            f"\033[93m[MB_MULTI ITER] sample={_sid}, nfe={nfe}, iter={conv_iter}, batch={i}, "
                            f"block={k}, strategy={strategy}, "
                            f"masks_before={len(mask_positions)}, accepted={len(accepted_positions)}, "
                            f"masks_after={masks_remaining}, floor={floor_forced}, "
                            f"local_pos={accepted_positions}, global_pos={global_accepted}, "
                            f"token_ids={accepted_ids}, "
                            f"confs=[{', '.join(f'{c:.4f}' for c in accepted_confs)}]\033[0m",
                            flush=True,
                        )

                    # Record decode order: positions in this block unmasked at step `nfe`
                    if decode_orders is not None:
                        decode_orders[i][blk["start"]:blk["end"]][transfer] = nfe

                    blk["mask_count"] = ids_slice.eq(self.mask_id).sum().item()

            if _d:
                blk_summary = []
                snap_summary = []
                for i in range(batch_size):
                    blk_summary.append(
                        "[" + ", ".join(f"b{k}:{b['mask_count']}/{b['total']}" for k, b in enumerate(item_blocks[i])) + "]"
                    )
                    snap_summary.append(
                        "[" + ", ".join(
                            f"b{k}:{snapshot_mask_counts[i][k]}/{b['total']}"
                            for k, b in enumerate(item_blocks[i])
                        ) + "]"
                    )
                print(
                    f"\033[94m[MB_MULTI ITER_END] sample={_sid}, nfe={nfe}, iter={conv_iter}, "
                    f"snapshot_masks={snap_summary}, live_masks={blk_summary}\033[0m",
                    flush=True,
                )

        # Single mandatory final: writes correct KV for all decoded positions.
        # NOT counted in nfe — it is a KV-cache implementation artifact, not a
        # denoising step. NFE measures only token-producing (convergence) forwards.
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output = out.logits_output
        can_run_cuda_graph = out.can_run_graph
        self._apply_invalid_token_mask(logits_output.full_logits)

        next_token_ids_list = [
            forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
            for i in range(batch_size)
        ]
        customized = {"nfe": [nfe] * batch_size}
        if decode_orders is not None:
            customized["decode_order"] = [d.cpu().tolist() for d in decode_orders]
        logits_output.customized_info = customized

        if _d:
            for i in range(batch_size):
                toks = next_token_ids_list[i].tolist()
                # Check if any masks remain in output (should be 0)
                n_mask_in_out = sum(1 for t in toks if t == self.mask_id)
                print(
                    f"\033[96m[MB_MULTI DONE] sample={_sid}, nfe={nfe}, batch={i}, "
                    f"out_len={len(toks)}, masks_in_output={n_mask_in_out}, "
                    f"out_ids={toks}\033[0m", flush=True,
                )

        # Reset causal mask so it doesn't affect non-dllm forward passes.
        if _updater is not None and hasattr(_updater, "dllm_causal_block_mask"):
            _updater.dllm_causal_block_mask = None
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidenceMultiBlock
