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

from verl.trainer.ppo.core_algos import *

import torch


def compute_distillation_loss(
    token_lp_all: torch.Tensor,
    loss_agg_mode: str = "token-mean",
) -> torch.Tensor:
    """Online distillation CE loss on trajectory-masked positions.

    For each masked position j, the model predicts the true token given the
    partially-masked context at sampled timestep t.  The loss is simply
    negative log-likelihood (cross-entropy) over all masked positions:

        L = -E_t [ mean_{j: mask_t(j)=1} log p_θ(x_j | noisy_x_t) ]

    Unlike the PPO/GRPO objective, no importance-weight ratio or p_mask
    normalisation is applied — this is a plain supervised CE signal.

    Args
    ----
    token_lp_all : (batch*mc, seq_len) float
        Log-probability of the original token at each masked position,
        0.0 at unmasked / padding positions (as returned by
        _compute_full_attn_masked_token_logprobs or
        _compute_block_masked_token_logprobs).
    loss_agg_mode : str
        Aggregation mode passed to ``agg_loss``.

    Returns
    -------
    distillation_loss : scalar Tensor
    """
    # Positions with token_lp_all != 0 are exactly the masked positions
    # where the CE loss should be computed.  This is correct for both the
    # full-attention and block-DLLM paths because only response tokens are
    # ever masked in trajectory-based sampling.
    distill_mask = (token_lp_all != 0).float()   # (batch*mc, seq_len)
    loss_mat = -token_lp_all                       # (batch*mc, seq_len)
    return agg_loss(loss_mat=loss_mat, loss_mask=distill_mask, loss_agg_mode=loss_agg_mode)


def compute_policy_loss(
    old_l_theta,
    l_theta,
    advantages,
    response_mask,
    ref_l_theta=None,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
    linearize_ratio: bool = False,
):
    """
    Compute the clipped policy objective and related metrics for PPO (token-level cross-entropy losses).

    Args:
        old_l_theta (Tensor): (batch_size, response_length)
        l_theta (Tensor): (batch_size, response_length)
        advantages (Tensor): (batch_size, response_length)
        response_mask (Tensor): (batch_size, response_length) 1/0 mask for valid tokens
        ref_l_theta (Tensor, optional): reference policy log-prob for KL estimation.
        cliprange (float, optional): Clipping parameter ε for standard PPO.
        cliprange_low (float, optional): Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional): Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional): Lower bound of the ratio for dual-clip PPO. Defaults to 3.0.
        loss_agg_mode (str, optional): Aggregation mode: 'token-mean', 'sentence-mean', etc.
        linearize_ratio (bool): If True (BGPO), use first-order Taylor approximation
            ratio ≈ 1 + (l_θ - l_θ_old) for positive-advantage tokens instead of exp.

    Returns:
        pg_loss: aggregated policy-gradient loss.
        pg_clipfrac: fraction of tokens hit by the standard PPO clip (both signs).
        ppo_kl: mean KL estimate.
        pg_clipfrac_lower: fraction of A<0 tokens hit by the dual-clip (clip_ratio_c).
        pg_clipfrac_pos: fraction of A>0 tokens hit by the standard upper clip.
        pg_clipfrac_neg: fraction of A<0 tokens hit by the standard (non-dual) lower clip.
    """
    assert isinstance(old_l_theta, torch.Tensor), f"old_l_theta must be a tensor, got {type(old_l_theta)}"
    assert isinstance(l_theta, torch.Tensor), f"l_theta must be a tensor, got {type(l_theta)}"
    assert isinstance(advantages, torch.Tensor), f"advantages must be a tensor, got {type(advantages)}"
    assert old_l_theta.shape == l_theta.shape == advantages.shape, \
        f"old_l_theta, l_theta and advantages must have the same shape, but got {old_l_theta.shape}, {l_theta.shape} and {advantages.shape}"
    assert clip_ratio_c > 1.0, \
        f"The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0, but get the value: {clip_ratio_c}."

    # Policy ratio r(θ): exact exp, or linearised for positive-advantage tokens (BGPO)
    if linearize_ratio:
        ratio = torch.where(advantages > 0, 1 + l_theta - old_l_theta, torch.exp(l_theta - old_l_theta))
    else:
        ratio = torch.exp(l_theta - old_l_theta)

    # KL divergence estimator E[exp(old-new) - (old-new) - 1]
    # Clamp the exponent before exp to prevent overflow (exp→inf) when policies diverge significantly.
    kl_base = ref_l_theta if ref_l_theta is not None else old_l_theta
    kl_diff = kl_base - l_theta
    ppo_kl = verl_F.masked_mean(torch.exp(kl_diff.clamp(max=20)) - kl_diff - 1, response_mask)

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    # Standard PPO clip is active whenever the clamped loss dominates the raw loss.
    std_clip_active = torch.gt(pg_losses2, pg_losses1).float()
    pg_clipfrac = verl_F.masked_mean(std_clip_active, response_mask)
    # Split the standard clip fraction by advantage sign:
    #   A > 0 -> upper clip (ratio > 1 + cliprange_high)
    #   A < 0 -> lower clip (ratio < 1 - cliprange_low), i.e. the non-dual lower clip
    pg_clipfrac_pos = verl_F.masked_mean(std_clip_active * (advantages > 0).float(), response_mask)
    pg_clipfrac_neg = verl_F.masked_mean(std_clip_active * (advantages < 0).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask)

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_pos, pg_clipfrac_neg


def compute_dpo_loss(
    chosen_log_prob: torch.Tensor,
    rejected_log_prob: torch.Tensor,
    chosen_ref_log_prob: torch.Tensor,
    rejected_ref_log_prob: torch.Tensor,
    beta: float,
) -> torch.Tensor:
    import torch.nn.functional as F

    pi_logratios = chosen_log_prob - rejected_log_prob
    ref_logratios = chosen_ref_log_prob - rejected_ref_log_prob

    logits = pi_logratios - ref_logratios

    losses = -F.logsigmoid(beta * logits)

    return losses.mean()


def kl_penalty(l_theta: torch.FloatTensor, ref_l_theta: torch.FloatTensor, kl_penalty, advantages: torch.FloatTensor = None, use_log_approx: bool = False) -> torch.FloatTensor:
    """Compute KL divergence given l_theta and ref_l_theta.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        l_theta:
        ref_l_theta:
        kl_penalty:
        advantages: required when use_log_approx=True
        use_log_approx: if True (bgpo only), use advantage-conditioned log approximation:
                          - adv > 0: log(1 + diff)  (concave compression for positive advantage)
                          - adv <= 0: diff  (linear penalty)
                        if False, simply use diff = l_theta - ref_l_theta.

    Returns:

    """
    diff = l_theta - ref_l_theta
    if use_log_approx:
        assert advantages is not None, "advantages must be provided when use_log_approx=True"
        # Clamp to avoid log(<=0) when diff <= -1, which causes NaN in gradients
        kl = torch.where(advantages > 0, torch.log((1 + diff).clamp(min=1e-6)), diff)
    else:
        kl = diff

    if kl_penalty in ("kl", "k1"):
        return kl

    if kl_penalty == "abs":
        return kl.abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * kl.square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = -kl  # Based on the old distribution
        # Clamp before exp to prevent overflow (exp→inf) which causes 0*inf=NaN in backward
        # when the output clamp's gradient is zero. exp(20)≈5e8 is well within float range.
        kl_safe = kl.clamp(max=20)
        ratio = torch.exp(kl_safe)
        kld = (ratio - kl_safe - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here l_theta and ref_l_theta should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError
