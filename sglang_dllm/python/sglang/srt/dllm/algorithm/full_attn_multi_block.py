"""Full-attention multi-block decoding for bidirectional dLLMs (e.g. Dream).

Matches d3LLM's _sample_multi_block behavior: all max_new_tokens masks are
present from the start. Multiple blocks are decoded in parallel, with new
blocks activated based on previous block's decoding progress.

Key matching with d3LLM _sample_multi_block (no cache):
  - ALL mask positions up to rightmost active block are eligible for decoding
    (not just fully-activated blocks) — entropy threshold is the only gate
  - Forced progress only applies to the first fully-activated block
  - block_add_threshold=0.1 and decoded_token_threshold=0.95 are the defaults
    matching d3LLM's eval commands for parallel multi-block decoding
"""

import logging
from typing import List, Tuple, Union

import torch

from sglang.srt.dllm.algorithm.base import DllmAlgorithm
from sglang.srt.dllm.config import DllmConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.model_runner import ModelRunner


logger = logging.getLogger(__name__)


class FullAttnMultiBlock(DllmAlgorithm):
    """Multi-block parallel decoding with entropy threshold for full-attention dLLMs.

    Config keys (from algorithm_config YAML):
      threshold               - entropy threshold for token acceptance  (default: 0.4)
      block_add_threshold     - prev block progress to add next block   (default: 0.1)
      decoded_token_threshold - prev progress for full activation       (default: 0.95)
      cache_delay_iter        - NFE delay before "cached" state         (default: 10000)
      refresh_interval        - periodic refresh interval               (default: 10000)
    """

    def __init__(self, config: DllmConfig):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.4)
        self.block_add_threshold = config.algorithm_config.get(
            "block_add_threshold", 0.1
        )
        self.decoded_token_threshold = config.algorithm_config.get(
            "decoded_token_threshold", 0.95
        )
        self.cache_delay_iter = config.algorithm_config.get("cache_delay_iter", 10000)
        self.refresh_interval = config.algorithm_config.get("refresh_interval", 10000)
        self.record_decode_order = config.algorithm_config.get("record_decode_order", False)
        # Early-stop: once all content before the first EOS in each item is fully
        # decoded (no MASK tokens remain before the first EOS), stop denoising.
        # The trailing EOS/MASK tokens after the first EOS carry no training signal
        # and are replaced with MASK_TOKEN_ID in verl's post-processing anyway.
        # Set eos_early_stop=false in algorithm_config YAML to disable.
        self.eos_early_stop = config.algorithm_config.get("eos_early_stop", True)
        print(
            "\033[91m",
            f"FullAttnMultiBlock initialized: threshold={self.threshold}, "
            f"block_add={self.block_add_threshold}, "
            f"decoded_thresh={self.decoded_token_threshold}, "
            f"sub_block_size={self.block_size}, "
            f"cache_delay_iter={self.cache_delay_iter}, "
            f"refresh_interval={self.refresh_interval}, "
            f"eos_early_stop={self.eos_early_stop} (eos_id={self.eos_id})",
            "\033[0m",
        )

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        batch_size = forward_batch.batch_size
        item_lens = forward_batch.extend_seq_lens_cpu
        offsets = [0]
        for l in item_lens:
            offsets.append(offsets[-1] + l)

        # Fast path: no masks
        if (forward_batch.input_ids == self.mask_id).sum().item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            return out.logits_output, [], out.can_run_graph

        # Per-item setup
        start_list = []
        gen_lengths = []
        for i in range(batch_size):
            ids = forward_batch.input_ids[offsets[i] : offsets[i + 1]]
            n_masks = (ids == self.mask_id).sum().item()
            start_list.append(item_lens[i] - n_masks)
            gen_lengths.append(n_masks)

        # Build block states per item (matches d3LLM block_states dict)
        # Block 0 = prompt (always complete). Gen blocks start from 1.
        item_blocks = []
        for i in range(batch_size):
            num_blocks = (gen_lengths[i] + self.block_size - 1) // self.block_size
            blocks = []
            for k in range(num_blocks):
                bs = start_list[i] + k * self.block_size
                be = min(bs + self.block_size, start_list[i] + gen_lengths[i])
                blocks.append({
                    "start": bs,
                    "end": be,
                    "total": be - bs,
                    "mask_count": be - bs,
                    "is_active": k == 0,        # block has been "added"
                    "is_fully_active": k == 0,   # prev block progress >= decoded_token_threshold
                    "completed_at_nfe": None,
                    "is_cached": False,
                })
            item_blocks.append(blocks)

        nfe = 0
        logits_output = None
        can_run_cuda_graph = False
        _eos_early_stopped = False  # set to True when EOS early-stop fires

        # decode_order[pos] = NFE step at which position pos was unmasked (0 = prompt)
        decode_orders = None
        if self.record_decode_order:
            decode_orders = [
                torch.zeros(item_lens[i], dtype=torch.long,
                            device=forward_batch.input_ids.device)
                for i in range(batch_size)
            ]

        while True:
            # Check termination: all masks decoded
            if (forward_batch.input_ids == self.mask_id).sum().item() == 0:
                break

            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            nfe += 1
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph

            # Apply invalid-token mask (token_id >= tokenizer_vocab_size → -inf).
            self._apply_invalid_token_mask(logits_output.full_logits)

            for i in range(batch_size):
                s, e = offsets[i], offsets[i + 1]
                ids = forward_batch.input_ids[s:e]
                logits = logits_output.full_logits[s:e]
                blocks = item_blocks[i]

                # Update block activation based on previous block progress
                for k in range(1, len(blocks)):
                    prev = blocks[k - 1]
                    prev_progress = 1 - prev["mask_count"] / prev["total"]
                    if not blocks[k]["is_active"]:
                        if prev_progress >= self.block_add_threshold or prev["mask_count"] == 0:
                            blocks[k]["is_active"] = True
                    if blocks[k]["is_active"] and not blocks[k]["is_fully_active"]:
                        if prev_progress >= self.decoded_token_threshold:
                            blocks[k]["is_fully_active"] = True

                # Find rightmost active block end (matches d3LLM: any block that
                # is_fully_active OR has mask_count > 0 determines active_end)
                active_end = 0
                for blk in blocks:
                    if blk["is_active"] and (blk["is_fully_active"] or blk["mask_count"] > 0):
                        active_end = blk["end"]

                if active_end == 0:
                    continue

                # Build decode mask: ALL mask positions up to active_end
                # (matches d3LLM: mask_index_for_decode[:, active_end:] = 0)
                decode_mask = ids[:active_end].eq(self.mask_id)
                n_decode = decode_mask.sum().item()
                if n_decode == 0:
                    continue

                # Entropy-based denoising across all decodable positions
                mask_logits = logits[:active_end][decode_mask]
                probs = torch.softmax(mask_logits.float(), dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-12), dim=-1)
                # Read per-request temperature from sampling_info (set by rollout,
                # already reflects train vs val distinction).
                # NOTE: SGLang's SamplingParams converts temperature=0 → temperature=1.0 + top_k=1
                # to express greedy decoding. So we must check top_k==1 as the authoritative
                # greedy signal, not just temperature < eps.
                temp = 0.0
                if forward_batch.sampling_info is not None:
                    temp = forward_batch.sampling_info.temperatures[i].item()
                    top_k = forward_batch.sampling_info.top_ks[i].item()
                    if top_k == 1:
                        temp = 0.0  # greedy mode signalled by top_k=1
                if temp > 1e-6:
                    logits_f = mask_logits.float()
                    noise = torch.rand_like(logits_f)
                    gumbel_noise = (-torch.log(noise + 1e-20)) ** temp
                    x0 = torch.argmax(logits_f.exp() / gumbel_noise, dim=-1)
                else:
                    x0 = torch.argmax(mask_logits, dim=-1)

                full_entropy = torch.full(
                    (active_end,), float("inf"), dtype=entropy.dtype, device=ids.device
                )
                full_x0 = ids[:active_end].clone()
                full_entropy[decode_mask] = entropy
                full_x0[decode_mask] = x0

                # Global entropy threshold (matches d3LLM)
                transfer = (full_entropy < self.threshold) & decode_mask

                # Per-block forced progress for first fully-activated block
                # (matches d3LLM: force even if other blocks accepted tokens)
                first_fa_blk = None
                for blk in blocks:
                    if blk["is_fully_active"] and blk["mask_count"] > 0:
                        first_fa_blk = blk
                        break
                if first_fa_blk is not None:
                    bs_, be_ = first_fa_blk["start"], min(first_fa_blk["end"], active_end)
                    blk_transfer = transfer[bs_:be_]
                    if not blk_transfer.any():
                        blk_mask = ids[bs_:be_].eq(self.mask_id)
                        blk_ent = full_entropy[bs_:be_].clone()
                        blk_ent[~blk_mask] = float("inf")
                        best = blk_ent.argmin()
                        transfer[bs_ + best] = True

                ids[:active_end][transfer] = full_x0[transfer]

                # Record decode order for tokens unmasked at this NFE step
                if decode_orders is not None and transfer.any():
                    accepted = transfer.nonzero(as_tuple=True)[0]
                    decode_orders[i][accepted] = nfe

                # Update block mask counts and completion tracking
                for blk in blocks:
                    if blk["mask_count"] > 0:
                        new_count = ids[blk["start"]:blk["end"]].eq(self.mask_id).sum().item()
                        if new_count == 0 and blk["completed_at_nfe"] is None:
                            blk["completed_at_nfe"] = nfe
                        blk["mask_count"] = new_count

            # ── EOS early stopping ────────────────────────────────────────────
            # Stop as soon as, for EVERY item in the batch:
            #   (a) the first EOS token has been decoded in the response region, AND
            #   (b) every position BEFORE that first EOS has also been decoded
            #       (no MASK tokens remain between response-start and first EOS).
            # Positions after the first EOS are left as MASK; verl's post-processing
            # replaces them with MASK_TOKEN_ID and zeros their decode_orders.
            if self.eos_early_stop and self.eos_id:
                _all_can_stop = True
                for _i in range(batch_size):
                    _s, _e = offsets[_i], offsets[_i + 1]
                    _resp = forward_batch.input_ids[_s + start_list[_i] : _e]
                    _eos_pos = (_resp == self.eos_id).nonzero(as_tuple=True)[0]
                    if _eos_pos.numel() == 0:
                        # No EOS yet in this item's response — keep denoising.
                        _all_can_stop = False
                        break
                    _first_eos = int(_eos_pos[0].item())
                    if _first_eos > 0 and _resp[:_first_eos].eq(self.mask_id).sum().item() > 0:
                        # Content before first EOS still has MASK tokens — keep going.
                        _all_can_stop = False
                        break
                if _all_can_stop:
                    logger.debug(
                        "EOS early-stop at nfe=%s: all %s item(s) have EOS with "
                        "fully-decoded prefix",
                        nfe,
                        batch_size,
                    )
                    _eos_early_stopped = True
                    break

            if nfe > 10000:
                break

        # Final forward for fresh logits (needed by sglang framework).
        # NOT counted in nfe — mandatory finals are KV-cache / framework
        # implementation artifacts, not denoising steps.  NFE measures only
        # token-producing denoising forwards.
        # This forward is executed when:
        #   (a) logits_output is None (should not happen after the while loop), or
        #   (b) masks remain and we did NOT exit via EOS early-stop — i.e. the
        #       nfe>10000 safety break fired with tokens still unresolved.
        # EOS early-stop: trailing MASK tokens after the first EOS are intentional
        # (replaced by verl post-processing); we skip the extra forward to avoid
        # feeding stale MASK embeddings into sglang's KV cache.
        _remaining_masks = (forward_batch.input_ids == self.mask_id).sum().item()
        if logits_output is None or (_remaining_masks > 0 and not _eos_early_stopped):
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            # nfe intentionally NOT incremented: this forward is a framework artifact.
            logits_output = out.logits_output
            can_run_cuda_graph = out.can_run_graph
            self._apply_invalid_token_mask(logits_output.full_logits)

        next_token_ids_list = [
            forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
            for i in range(batch_size)
        ]
        customized: dict = {"nfe": [nfe] * batch_size}
        if decode_orders is not None:
            customized["decode_order"] = [d.cpu().tolist() for d in decode_orders]
        logits_output.customized_info = customized
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = FullAttnMultiBlock
