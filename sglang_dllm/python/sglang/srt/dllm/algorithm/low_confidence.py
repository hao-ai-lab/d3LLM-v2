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
_D_Num = 128  # only print debug for the first _D_Num run() calls (samples)

def _is_rank0():
    import torch.distributed as dist
    return (not dist.is_available()) or (not dist.is_initialized()) or dist.get_rank() == 0


class LowConfidence(DllmAlgorithm):

    def __init__(
        self,
        config: DllmConfig,
    ):
        super().__init__(config)
        self.threshold = config.algorithm_config.get("threshold", 0.95)
        self.record_decode_order = config.algorithm_config.get("record_decode_order", False)
        self._d_cnt = 0  # counts run() calls for _D_Num gating
        print("\033[91m", "LowConfidence algorithm initialized with threshold:", self.threshold,
              "record_decode_order:", self.record_decode_order, "\033[0m")

    def run(
        self,
        model_runner: ModelRunner,
        forward_batch: ForwardBatch,
    ) -> Tuple[Union[LogitsProcessorOutput, torch.Tensor], List[torch.Tensor], bool]:
        # _d: whether to print debug for this sample (run() call), rank-0 only
        _d = _D and _is_rank0() and self._d_cnt < _D_Num
        _sid = self._d_cnt  # sample index for logging
        self._d_cnt += 1

        batch_size = forward_batch.batch_size
        # Per-item token counts: supports variable-length sequences (needs_full_prefill)
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
                f"\033[95m[LC RUN] sample={_sid}, bs={batch_size}, extend_lens={list(item_lens)}, "
                f"prefix_lens={_prefix_lens}, block_offsets={_block_offsets}, "
                f"total_masks={torch.sum(mask_index).item()}, "
                f"input_ids_len={len(forward_batch.input_ids)}\033[0m", flush=True,
            )

        # Fast path: if there is no mask token, forward and save kv cache
        if torch.sum(mask_index).item() == 0:
            out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
            logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
            logits_output.customized_info = {
                "nfe": [0] * batch_size,
                "decode_forward_count": [1] * batch_size,
            }

            next_token_ids = []
            if _d:
                print(f"\033[95m[LC FAST_PATH] sample={_sid}, no masks, nfe=0, returning 0 tokens\033[0m", flush=True)
            return logits_output, next_token_ids, can_run_cuda_graph

        # Calculate start positions for each batch item
        for i in range(batch_size):
            block_input_ids = forward_batch.input_ids[offsets[i]:offsets[i + 1]]
            block_mask_index = block_input_ids == self.mask_id
            start = item_lens[i] - torch.sum(block_mask_index).item()
            start_list.append(start)

        if _d:
            for i in range(batch_size):
                s, e = offsets[i], offsets[i + 1]
                ids = forward_batch.input_ids[s:e]
                mask_pos = (ids == self.mask_id).nonzero(as_tuple=True)[0].tolist()
                non_mask_ids = ids[ids != self.mask_id][-5:].tolist() if (ids != self.mask_id).any() else []
                print(
                    f"\033[95m[LC INIT] sample={_sid}, i={i}, extend_len={item_lens[i]}, start={start_list[i]}, "
                    f"n_masks={len(mask_pos)}, mask_positions={mask_pos}, "
                    f"last_5_non_mask_ids={non_mask_ids}\033[0m", flush=True,
                )

        # Pseudo-trajectory: decode_order[i] = step when position i was unmasked (0 = prompt)
        decode_orders = None
        if self.record_decode_order:
            decode_orders = [
                torch.zeros(item_lens[i], dtype=torch.long, device=forward_batch.input_ids.device)
                for i in range(batch_size)
            ]

        nfe = 0
        for iter_idx in range(self.block_size):
            mask_index = forward_batch.input_ids == self.mask_id
            if torch.sum(mask_index).item() == 0:
                if _d:
                    print(f"\033[95m[LC LOOP] sample={_sid}, iter={iter_idx}, all masks done, breaking\033[0m", flush=True)
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

                # Record decode order for tokens being unmasked at this step
                if decode_orders is not None:
                    decode_orders[batch_id][transfer_index] = nfe

                if _d:
                    mask_positions = block_mask_index.nonzero(as_tuple=True)[0].tolist()
                    accepted_positions = transfer_index.nonzero(as_tuple=True)[0].tolist()
                    accepted_ids = x[transfer_index].tolist()
                    accepted_confs = confidence[transfer_index].tolist()
                    masks_remaining = block_mask_index.sum().item() - transfer_index.sum().item()
                    print(
                        f"\033[93m[LC ITER] sample={_sid}, nfe={nfe}, iter={iter_idx}, batch={batch_id}, "
                        f"masks_before={len(mask_positions)}, accepted={len(accepted_positions)}, "
                        f"masks_after={masks_remaining}, floor={floor_forced}, "
                        f"positions={accepted_positions}, "
                        f"token_ids={accepted_ids}, "
                        f"confs=[{', '.join(f'{c:.4f}' for c in accepted_confs)}]\033[0m",
                        flush=True,
                    )

                block_input_ids[transfer_index] = x[transfer_index]

        # Mandatory final forward: produces fresh logits the sglang framework needs and
        # rewrites KV cache for the last accepted tokens.  It decodes 0 NEW tokens, so it
        # is NOT counted in nfe — nfe measures only token-producing (decode) forwards, so
        # TPF = resp_tokens / nfe is always >= 1.  (The final is an implementation /
        # acceleration artifact, not a denoising step.)
        out = model_runner.forward(forward_batch, pp_proxy_tensors=None)
        logits_output, can_run_cuda_graph = out.logits_output, out.can_run_graph
        self._apply_invalid_token_mask(logits_output.full_logits)
        # Build per-sequence next_token_ids using variable-length offsets
        next_token_ids_list = [
            forward_batch.input_ids[offsets[i] + start_list[i] : offsets[i + 1]]
            for i in range(batch_size)
        ]

        if _d:
            for i in range(batch_size):
                toks = next_token_ids_list[i].tolist()
                print(
                    f"\033[96m[LC DONE] sample={_sid}, nfe={nfe}, batch={i}, "
                    f"out_len={len(toks)}, out_ids={toks}\033[0m",
                    flush=True,
                )

        # Token per forward: attach nfe so it flows via customized_info -> meta_info
        customized = {
            "nfe": [nfe] * batch_size,
            # Include the mandatory final forward so this reflects actual decode
            # model invocations rather than token-producing denoising steps only.
            "decode_forward_count": [nfe + 1] * batch_size,
        }
        if decode_orders is not None:
            customized["decode_order"] = [d.cpu().tolist() for d in decode_orders]
        logits_output.customized_info = customized
        return logits_output, next_token_ids_list, can_run_cuda_graph


Algorithm = LowConfidence
