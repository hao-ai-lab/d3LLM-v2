"""DistillDLLMActor — pure online distillation actor.

Extracted from DLLMDataParallelPPOActor so that the base class stays focused
on the RL objective and distillation logic lives in its own class.

``DistillDLLMActor`` overrides only ``update_policy``; all forward/compute
helpers are inherited unchanged from ``DLLMDataParallelPPOActor``.
"""

from __future__ import annotations

import torch

from verl import DataProto
from verl.protocol import ADJUST_BATCH_GRADIENT_MASK_KEY
from verl.trainer.ppo.dllm_core_algos import compute_distillation_loss
from verl.utils.device import get_torch_device
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import rearrange_micro_batches
from verl.workers.actor.dllm_actor import (
    DLLMDataParallelPPOActor,
    _get_adjust_batch_gradient_mask,
)


class DistillDLLMActor(DLLMDataParallelPPOActor):
    """Online-distillation actor.

    ``update_policy`` minimises trajectory CE loss only (no PPO/GRPO objective).
    The model sees its own partially-masked rollout trajectories and is trained
    to reconstruct the original tokens at masked positions.
    """

    def update_policy(self, data: DataProto):
        """Run distillation update steps.

        Does not require ``old_log_probs`` or ``advantages``; only the noisy
        trajectory fields are needed.

        Returns
        -------
        metrics : dict mapping metric name → list of scalar values
        """
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "perturbed_seq",
            "mask_indices",
            "p_mask",
        ]
        if multi_turn:
            select_keys.append("loss_mask")
        if ADJUST_BATCH_GRADIENT_MASK_KEY in data.batch.keys():
            select_keys.append(ADJUST_BATCH_GRADIENT_MASK_KEY)

        batch = data.select(batch_keys=select_keys).batch

        _total_B = int(batch.batch_size[0])
        _mini_bs = getattr(self.config, "ppo_mini_batch_size", _total_B)
        if _mini_bs is None or _mini_bs <= 0:
            _mini_bs = _total_B
        _mini_bs = int(_mini_bs)
        _micro_bs = getattr(self.config, "ppo_micro_batch_size_per_gpu", _mini_bs)
        if _micro_bs is None or _micro_bs <= 0:
            _micro_bs = _mini_bs
        _micro_bs = int(_micro_bs)

        if _micro_bs and not getattr(self.config, "use_dynamic_bsz", False):
            trimmed_n = (_total_B // _micro_bs) * _micro_bs
            if trimmed_n < _total_B:
                batch = batch[:trimmed_n]

        dataloader = batch.split(_mini_bs)

        metrics = {}
        _is_rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if _is_rank0:
            print(
                f"[DistillDLLMActor.update_policy] total_B={_total_B} mini_bs={_mini_bs} "
                f"micro_bs={_micro_bs} gradient_accumulation={_mini_bs // _micro_bs} "
                f"epochs={self.config.ppo_epochs}"
            )

        for _epoch in range(self.config.ppo_epochs):
            for _mini_idx, mini_batch in enumerate(dataloader):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu
                    micro_batches, _ = rearrange_micro_batches(
                        batch=mini_batch, max_token_len=max_token_len
                    )
                else:
                    micro_batches = mini_batch.split(_micro_bs)
                    self.gradient_accumulation = len(micro_batches)

                if _is_rank0:
                    print(
                        f"[DistillDLLMActor.update_policy] epoch={_epoch} "
                        f"mini_batch={_mini_idx} "
                        f"gradient_accumulation={self.gradient_accumulation}"
                    )
                self.actor_optimizer.zero_grad()

                for mb in micro_batches:
                    if isinstance(mb, DataProto):
                        mb = {
                            **mb.batch.to(get_torch_device().current_device()),
                            **mb.non_tensor_batch,
                        }
                    else:
                        mb = mb.to(get_torch_device().current_device())

                    responses = mb["responses"]
                    response_length = responses.size(1)
                    attention_mask = mb["attention_mask"]
                    response_mask = (
                        mb["loss_mask"][:, -response_length:]
                        if multi_turn
                        else attention_mask[:, -response_length:]
                    )

                    grad_scale = (
                        mb["input_ids"].size(0) / _mini_bs
                        if self.config.use_dynamic_bsz
                        else 1.0 / self.gradient_accumulation
                    )

                    step_metrics = self._backward_micro_batch_distillation(
                        mb, temperature, response_mask, grad_scale=grad_scale
                    )
                    append_to_dict(metrics, step_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics

    def _backward_micro_batch_distillation(
        self,
        mb,
        temperature,
        response_mask,
        grad_scale: float = 1.0,
    ):
        """Run one forward+backward pass with cross-entropy distillation loss.

        Uses the DLLM's own rollout trajectories as supervision signal.
        For each MC sample (one sampled diffusion timestep t), the model sees
        the partially-masked sequence ``perturbed_seq`` and must predict the
        original tokens at all still-masked positions.  The objective is plain
        cross-entropy — no importance-weight ratio or p_mask normalisation:

            L = -E_t [ mean_{j: mask_t(j)=1} log p_θ(x_j | noisy_x_t) ]

        This is in contrast to the PPO/GRPO objective which uses reward-weighted
        log-prob ratios.  The gradient of this loss is equivalent to MLE
        fine-tuning on the model's own (already-generated) responses, which can
        stabilise training and accelerate convergence.

        Requires ``noisy_backend="trajectory"`` (or any backend that provides
        ``perturbed_seq`` / ``mask_indices``) in the rollout config.

        Args
        ----
        mb            : dict – micro-batch tensors on the current device.
                         Required keys: input_ids, responses, attention_mask,
                         perturbed_seq, mask_indices, p_mask.
        temperature   : float – (passed for API consistency; not applied here)
        response_mask : (mbs, response_len) float/bool – valid response positions
                        (used only to guard that we do not compute loss outside the
                        response; in practice the trajectory backend never masks
                        prompt tokens so this is a no-op safety check).
        grad_scale    : float – gradient accumulation scale factor.

        Returns
        -------
        metrics : dict with key ``"actor/distillation_loss"``
        """
        loss_agg_mode = self.config.loss_agg_mode

        batch_size = mb["input_ids"].size(0)
        seq_length = mb["input_ids"].size(-1)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = mb["perturbed_seq"]   # (batch, mc_num, seq_len)
            mask_indices = mb["mask_indices"]      # (batch, mc_num, seq_len) bool
            orig_seq = mb["input_ids"]             # (batch, seq_len)
            attention_mask = mb["attention_mask"]  # (batch, seq_len)
            p_mask = mb["p_mask"]                  # (batch, mc_num, seq_len)

            sample_gradient_mask = _get_adjust_batch_gradient_mask(
                mb, batch_size, mask_indices.device
            )
            if sample_gradient_mask is not None:
                mask_indices = mask_indices & sample_gradient_mask[:, None, None]

            mc_num = perturbed_seq.size(1)
            bmc = batch_size * mc_num

            if self.block_size > 0:
                token_lp_all, _ = self._compute_block_masked_token_logprobs(
                    perturbed_seq,
                    mask_indices,
                    p_mask,
                    orig_seq,
                    attention_mask,
                    bmc,
                    mc_num,
                    seq_length,
                )
            else:
                token_lp_all, _ = self._compute_full_attn_masked_token_logprobs(
                    perturbed_seq,
                    mask_indices,
                    p_mask,
                    orig_seq,
                    attention_mask,
                    bmc,
                    mc_num,
                    seq_length,
                )
            # token_lp_all: (batch*mc, seq_len)
            # log p_θ(x_j | noisy_x_t) at every masked position, 0.0 elsewhere.
            # compute_distillation_loss uses (token_lp_all != 0) as the loss mask,
            # which is exactly the set of trajectory-masked response positions.

            distillation_loss = compute_distillation_loss(
                token_lp_all=token_lp_all,
                loss_agg_mode=loss_agg_mode,
            )

        (distillation_loss * grad_scale).backward()

        return {
            "actor/distillation_loss": distillation_loss.detach().item(),
        }
