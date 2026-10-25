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
DLLMDataParallelPPOActor — attention-agnostic base actor for Diffusion Language Models.

Independent of the underlying attention mechanism (full bidirectional / block causal), provides:
  - Noisy sample generation  (build_noisy_samples)
  - Gauss-Legendre sample weighting  (_gauss_weight_samples)
  - Inference loop      (compute_log_prob)
  - Training loop       (update_policy / _backward_micro_batch_*)

Subclasses only need to implement _forward_micro_batch.

Output granularity is controlled by config.log_prob_level:

  log_prob_level    Algorithm           log_prob shape
  ─────────────────────────────────────────────────────
  "token-level"     D1                  (bs, response_len)
  "block-level"     GDPO block          (bs, response_len)
  "sequence-level"  GDPO/VRP/BGPO       (bs, response_len)

When config.approx_per_mc_update=True, all levels additionally compute loss_per_sample
(bs, mc_num, response_len) and perform per-MC linearised backward passes (BGPO approximation).
"""

import itertools
import logging
import os
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from verl import DataProto
from verl.protocol import ADJUST_BATCH_GRADIENT_MASK_KEY
from verl.trainer.ppo.dllm_core_algos import (
    agg_loss,
    compute_policy_loss,
    kl_penalty,
)
from verl.utils.debug import GPUMemoryLogger
from verl.utils.device import get_torch_device
from verl.utils.gaussian_quadrature import compute_gauss_legendre_quadrature
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.workers.actor.utils import (
    _compact_sequences,
    project_compact_to_response,
    flatten_mc_inputs,
    prepare_block_inputs,
)
from verl.workers.actor.build_noisy_samples import (
    build_random_noisy_batch,
    build_gaussian_noisy_batch,
    build_d1_noisy_batch,
    build_block_random_noisy_batch,
    build_trajectory_noisy_batch,
    build_block_trajectory_noisy_batch,
    build_all_block_trajectory_noisy_batch,
)

__all__ = ["DLLMDataParallelPPOActor"]

_NOISY_BACKENDS = frozenset(
    ["d1", "random", "random_curriculum", "gaussian", "block_random", "trajectory",
     "trajectory_curriculum", "block_trajectory",
     "block_trajectory_curriculum", "all_block_trajectory", "all_block_trajectory_curriculum",
     "all_block_trajectory_fixed_t"]
)
_CURRICULUM_BACKENDS = frozenset(
    ["random_curriculum", "trajectory_curriculum", "block_trajectory_curriculum",
     "all_block_trajectory_curriculum"]
)
_LOG_PROB_LEVELS = frozenset(
    [
        "token-level",
        "block-level",
        "sequence-level",
    ]
)


def _get_adjust_batch_gradient_mask(micro_batch, batch_size, device):
    """Return the optional per-row gradient mask added by ``adjust_batch``."""
    sample_mask = micro_batch.get(ADJUST_BATCH_GRADIENT_MASK_KEY)
    if sample_mask is None:
        return None
    if sample_mask.ndim != 1 or sample_mask.numel() != batch_size:
        raise ValueError(
            f"{ADJUST_BATCH_GRADIENT_MASK_KEY} must have shape ({batch_size},), "
            f"got {tuple(sample_mask.shape)}"
        )
    return sample_mask.to(device=device, dtype=torch.bool)


def _apply_adjust_batch_gradient_mask(response_mask, micro_batch):
    """Exclude adjust-batch copies without changing model-forward shapes."""
    sample_mask = _get_adjust_batch_gradient_mask(
        micro_batch, response_mask.size(0), response_mask.device
    )
    if sample_mask is None:
        return response_mask
    return response_mask * sample_mask.unsqueeze(-1).to(response_mask.dtype)

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _rank0_print(*args, **kwargs):
    """Print only on rank-0 to avoid interleaved output from multiple processes."""
    try:
        import torch.distributed as _dist
        if _dist.is_initialized() and _dist.get_rank() != 0:
            return
    except Exception:
        pass
    print(*args, **kwargs)


class DLLMDataParallelPPOActor(DataParallelPPOActor):
    """Attention-agnostic DLLM actor base class.

    Subclasses only need to implement _forward_micro_batch.

    Required config fields:
        log_prob_level (str)       : "token-level" | "block-level" | "sequence-level".
        mc_num         (int)       : number of Monte Carlo samples per sequence.
        n_l            (int)       : number of random mask draws.
        noisy_backend  (str)       : "d1" | "random" | "gaussian".

    Optional config fields:
        block_size            (int,   default -1)   : block size for Block-DLLM; <=0 means full
                                                      bidirectional attention.  Required > 0 when
                                                      log_prob_level='block-level'.
        shift_logits          (bool,  default False): set True for next-token-prediction diffusion
                                                      models (e.g. Dream).  Labels are shifted right
                                                      by one so logit[i] predicts label[i+1].
                                                      Incompatible with block_size > 0.
        normalize_log_prob_by_length (bool, default True)
        p_mask_normalization  (bool, default True): if True, divide per-token log-probs by p_mask
                                                     (ELBO importance-weight correction 1/p_t).
                                                     For token-level this puts 1/p_mask inside the
                                                     PPO ratio; for sequence/block-level it is part
                                                     of the standard ELBO estimator.  Set False to
                                                     use raw log-probs for the ratio (DARE D1 style).
        approx_per_mc_update  (bool, default False): if True, perform one backward per MC sample
                                                      (BGPO linearised approximation, reduces peak
                                                      memory).
        d1_prompt_mask_p      (float, default 0.15): prompt mask probability for D1 backend; with
                                                      mc_num>1 each call draws an independent prompt
                                                      mask pattern to implement MC sampling.

    With ``noisy_backend="gaussian"``, ``mc_num`` is also the number of
    Gauss-Legendre quadrature points.  Nodes and weights are computed once at
    actor startup.  The legacy ``estimator="gauss-N"`` field is accepted for
    backward-compatible configs, but N is derived from ``mc_num``.
    """

    def __init__(self, config, actor_module, actor_optimizer=None):
        """Initialise the actor, validate config, and set up Gauss-Legendre quadrature if needed."""
        super().__init__(config, actor_module, actor_optimizer)
        self.MASK_TOKEN_ID = actor_module.config.mask_token_id
        self.PAD_TOKEN_ID = actor_module.config.pad_token_id
        self.mc_num = config["mc_num"]
        self.n_l = config["n_l"]
        self._noisy_backend = config.get("noisy_backend", "random")
        if self._noisy_backend != "gaussian":
            assert (
                self.mc_num % self.n_l == 0
            ), f"[DLLMActor] mc_num ({self.mc_num}) must be divisible by n_l ({self.n_l})."
        # block_size <= 0 means full bidirectional attention (no block structure)
        self.block_size = int(config.get("block_size", -1))
        # Dream and similar next-token-prediction diffusion models require shifted logits
        self._shift_logits = bool(config.get("shift_logits", False))
        if self._shift_logits and self.block_size > 0:
            raise ValueError(
                "[DLLMActor] shift_logits=True is not supported with block_size > 0 (block-DLLM path). "
                "shift_logits is only implemented for the full-attention path. "
                "Set block_size <= 0 or shift_logits=False."
            )

        self.log_prob_level = config.get("log_prob_level", "sequence-level")
        if self.log_prob_level not in _LOG_PROB_LEVELS:
            raise ValueError(
                f"[DLLMActor] Unknown log_prob_level '{self.log_prob_level}'. "
                f"Choices: {sorted(_LOG_PROB_LEVELS)}"
            )
        if self.log_prob_level == "block-level" and self.block_size <= 0:
            raise ValueError(
                "[DLLMActor] log_prob_level='block-level' requires block_size > 0, "
                f"but got block_size={self.block_size}. "
                "Set block_size > 0 in config, or switch to 'sequence-level'."
            )

        self._normalize_log_prob_by_length = bool(
            config.get("normalize_log_prob_by_length", True)
        )
        self._p_mask_normalization = bool(config.get("p_mask_normalization", True))

        if self._noisy_backend not in _NOISY_BACKENDS:
            raise ValueError(
                f"[DLLMActor] Unknown noisy_backend '{self._noisy_backend}'. "
                f"Choices: {sorted(_NOISY_BACKENDS)}"
            )
        # block_random_block_size: block size used exclusively by the block_random noisy backend.
        # Decoupled from block_size (which controls the forward-pass attention structure) so
        # that full-attention models (block_size=-1) can still use block_random sampling.
        # Falls back to block_size when not explicitly set (covers block-DLLM models where
        # both the forward pass and the noisy sampling share the same block granularity).
        _br_block_size_cfg = config.get("block_random_block_size", None)
        if _br_block_size_cfg is not None:
            self._block_random_block_size = int(_br_block_size_cfg)
        else:
            self._block_random_block_size = self.block_size  # backward-compat default
        if self._noisy_backend == "block_random" and self._block_random_block_size <= 0:
            raise ValueError(
                "[DLLMActor] noisy_backend='block_random' requires block_random_block_size > 0 "
                "(or block_size > 0 as fallback), "
                f"but got block_random_block_size={self._block_random_block_size}. "
                "Set block_random_block_size > 0 in config."
            )
        # D1 prompt mask probability for the prompt region; set to 0.0 in config for a clean forward pass.
        self._d1_prompt_mask_p = float(config.get("d1_prompt_mask_p", 0.15))

        # block_history_window_size: LCMB decode-window size (multiple of block_size).
        # When > block_size, the block-causal training attention is made LCMB-window-aware:
        # a noisy query attends to the NOISY state of earlier base blocks in the SAME
        # window and only to CLEAN history from strictly-earlier windows — matching the
        # LowConfidenceMultiBlock rollout, where later base blocks in one decode window
        # see their predecessors' partially-decoded (noisy) state, not clean ground truth.
        # None / <= block_size ⇒ standard SDAR block-causal rules (no behavior change).
        _bhws_cfg = config.get("block_history_window_size", None)
        self._block_history_window_size = int(_bhws_cfg) if _bhws_cfg else None
        if (
            self._block_history_window_size is not None
            and self.block_size > 0
            and self._block_history_window_size % self.block_size != 0
        ):
            raise ValueError(
                "[DLLMActor] block_history_window_size "
                f"({self._block_history_window_size}) must be a multiple of "
                f"block_size ({self.block_size})."
            )

        # ── noisy-sampling curriculum config ────────────────────────────────
        # The sampling lower bound stays fixed while the upper bound increases linearly:
        # U(min_mask_ratio, current_upper), from [0.1, 0.2] to [0.1, 1.0] by default.
        # curriculum_total_steps is independent of the trainer's total training steps.
        self._curriculum_min_mask_ratio = float(
            config.get("curriculum_min_mask_ratio", 0.1)
        )
        self._curriculum_initial_max_mask_ratio = float(
            config.get("curriculum_initial_max_mask_ratio", 0.2)
        )
        self._curriculum_max_mask_ratio = float(
            config.get("curriculum_max_mask_ratio", 1.0)
        )
        self._curriculum_total_steps = int(config.get("curriculum_total_steps", 300))
        self._fixed_t = float(config.get("fixed_t", 0.5))
        if self._noisy_backend in _CURRICULUM_BACKENDS:
            if not (
                0.0
                <= self._curriculum_min_mask_ratio
                <= self._curriculum_initial_max_mask_ratio
                <= self._curriculum_max_mask_ratio
                <= 1.0
            ):
                raise ValueError(
                    f"[DLLMActor] noisy_backend='{self._noisy_backend}' requires "
                    f"0 <= curriculum_min_mask_ratio ({self._curriculum_min_mask_ratio}) "
                    f"<= curriculum_initial_max_mask_ratio ({self._curriculum_initial_max_mask_ratio}) "
                    f"<= curriculum_max_mask_ratio ({self._curriculum_max_mask_ratio}) <= 1."
                )
            if self._curriculum_total_steps <= 0:
                raise ValueError(
                    f"[DLLMActor] noisy_backend='{self._noisy_backend}' requires "
                    f"curriculum_total_steps > 0, but got {self._curriculum_total_steps}."
                )
            if (
                self._noisy_backend
                in ("block_trajectory_curriculum", "all_block_trajectory_curriculum")
                and self._block_random_block_size <= 0
            ):
                raise ValueError(
                    f"[DLLMActor] noisy_backend='{self._noisy_backend}' requires "
                    "block_random_block_size > 0 (or block_size > 0 as fallback), "
                    f"but got block_random_block_size={self._block_random_block_size}. "
                    "Set block_random_block_size > 0 in config."
                )
        if self._noisy_backend == "all_block_trajectory_fixed_t":
            if not 0.0 < self._fixed_t <= 1.0:
                raise ValueError(
                    "[DLLMActor] noisy_backend='all_block_trajectory_fixed_t' requires "
                    f"0 < fixed_t ({self._fixed_t}) <= 1 to avoid an empty loss."
                )
            if self._block_random_block_size <= 0:
                raise ValueError(
                    "[DLLMActor] noisy_backend='all_block_trajectory_fixed_t' requires "
                    "block_random_block_size > 0 (or block_size > 0 as fallback), "
                    f"but got block_random_block_size={self._block_random_block_size}."
                )

        if self._noisy_backend == "gaussian":
            estimator = str(config.get("estimator", "gauss"))
            legacy_suffix = estimator.removeprefix("gauss-")
            if estimator != "gauss" and not (
                estimator.startswith("gauss-")
                and legacy_suffix.isdigit()
                and int(legacy_suffix) > 0
            ):
                raise ValueError(
                    f"[DLLMActor] Unknown estimator '{estimator}'. "
                    "Expected 'gauss' or the legacy 'gauss-N' form."
                )
            self._gdpo_xi, self._gauss_weights = compute_gauss_legendre_quadrature(
                self.mc_num
            )
            self._gdpo_N = self.mc_num
            self._gauss_N = self._gdpo_N
            self._gdpo_mc_per_point = 1
            # n_l must equal mc_per_point so that each j-iteration in build_noisy_samples
            # corresponds to exactly one quadrature point; override and warn if mismatched.
            if self.n_l != self._gdpo_mc_per_point:
                _rank0_print(
                    f"[DLLMActor] gaussian backend: n_l ({self.n_l}) != mc_per_point "
                    f"({self._gdpo_mc_per_point} = mc_num/gdpo_N). "
                    f"Overriding n_l to {self._gdpo_mc_per_point} to ensure correct quadrature alignment."
                )
                self.n_l = self._gdpo_mc_per_point
            _rank0_print(
                f"[DLLMActor] gaussian backend: n_l={self.n_l}, gdpo_N={self._gdpo_N}, mc_per_point={self._gdpo_mc_per_point}"
            )

        self._approx_per_mc_update = bool(config.get("approx_per_mc_update", False))

        # ── 启动时打印完整配置，方便 debug ──────────────────────────────────
        _sep = "─" * 60
        _lines = [
            "",
            _sep,
            "[DLLMActor] Configuration Summary",
            _sep,
            f"  MASK_TOKEN_ID              : {self.MASK_TOKEN_ID}",
            f"  PAD_TOKEN_ID               : {self.PAD_TOKEN_ID}",
            "  ── Quadrature / MC ──",
            f"  mc_num                     : {self.mc_num}",
            f"  n_l                        : {self.n_l}",
            f"  approx_per_mc_update       : {self._approx_per_mc_update}",
            "  ── Noisy backend ──",
            f"  noisy_backend              : {self._noisy_backend}",
            f"  d1_prompt_mask_p           : {self._d1_prompt_mask_p}",
        ]
        if self._noisy_backend == "gaussian":
            _lines += [
                f"  estimator (requested)      : {config.get('estimator', 'gauss')}",
                f"  quadrature                 : gauss-{self._gdpo_N} (from mc_num)",
                f"  gdpo_N (quadrature points) : {self._gdpo_N}",
                f"  gdpo_xi                    : {[round(x, 6) for x in self._gdpo_xi]}",
                f"  gauss_weights              : {[round(w, 6) for w in self._gauss_weights]}",
                f"  gdpo_mc_per_point          : {self._gdpo_mc_per_point}",
            ]
        if self._noisy_backend == "block_random":
            _lines += [
                f"  block_random_block_size    : {self._block_random_block_size}",
            ]
        if self._noisy_backend in ("trajectory", "trajectory_curriculum"):
            _lines += [
                "  (trajectory backend: decode_orders from rollout; record_decode_order must be true)",
            ]
        if self._noisy_backend in _CURRICULUM_BACKENDS:
            _lines += [
                f"  curriculum_min_mask_ratio  : {self._curriculum_min_mask_ratio}",
                f"  curriculum_initial_max_mask_ratio: {self._curriculum_initial_max_mask_ratio}",
                f"  curriculum_max_mask_ratio  : {self._curriculum_max_mask_ratio}",
                f"  curriculum_total_steps     : {self._curriculum_total_steps}",
            ]
            if self._noisy_backend != "random_curriculum":
                _lines += [
                    f"  block_random_block_size    : {self._block_random_block_size}",
                    "  (requires decode_orders from rollout; record_decode_order must be true)",
                ]
        if self._noisy_backend == "all_block_trajectory_fixed_t":
            _lines += [
                f"  block_random_block_size    : {self._block_random_block_size}",
                f"  fixed_t                    : {self._fixed_t}",
                "  (fixed_t is projected onto each decode-order window)",
                "  (requires decode_orders from rollout; record_decode_order must be true)",
            ]
        _lines += [
            "  ── Model structure ──",
            f"  block_size                 : {self.block_size}",
            f"  block_history_window_size  : {self._block_history_window_size}",
            f"  shift_logits               : {self._shift_logits}",
            "  ── Log-prob ──",
            f"  log_prob_level             : {self.log_prob_level}",
            f"  normalize_log_prob_by_length: {self._normalize_log_prob_by_length}",
            "  ── Training (from parent) ──",
            f"  ppo_mini_batch_size        : {config.get('ppo_mini_batch_size', 'N/A')}",
            f"  ppo_micro_batch_size_per_gpu: {config.get('ppo_micro_batch_size_per_gpu', 'N/A')}",
            f"  use_kl_loss                : {config.get('use_kl_loss', 'N/A')}",
            f"  grad_clip                  : {config.get('grad_clip', 'N/A')}",
            f"  actor_type                 : {config.get('actor_type', 'rl')}",
            f"  param_offload              : {config.get('fsdp_config', {}).get('param_offload', 'N/A')}",
            f"  optimizer_offload          : {config.get('fsdp_config', {}).get('optimizer_offload', 'N/A')}",
            _sep,
            "",
        ]
        _rank0_print("\n".join(_lines), flush=True)

        # ── Debug state saver（由外部注入，None 表示不保存）────────────────
        # 通过 actor.debug_saver = saver; actor.debug_step = step 来激活。
        # 仅在 rank-0 进程保存，其他 rank 不写文件。
        self.debug_saver = None  # DebugStateSaver | None，由外部注入
        self.debug_step: Optional[int] = None
        self._debug_mb_counter: int = 0  # micro-batch 计数，每次 compute_log_prob 重置
        # Counter limiting trajectory-backend trace logs to the first few calls
        self._traj_debug_count: int = 0

    # ── Curriculum learning helpers ──────────────────────────────────────────

    def _get_curriculum_mask_ratio(self, global_step: int) -> float:
        """Compute a per-sample mask ratio for a curriculum noisy backend.

        Uses an easy-to-hard upper-bound schedule:
          1. current_upper = initial_max + progress * (max - initial_max)
          2. mask_ratio ~ U(min, current_upper)

        Early training samples from a narrow low-ratio interval; harder, higher
        mask ratios are introduced gradually while the lower bound stays fixed.

        Args
        ----
        global_step : current training step (0-indexed). The schedule length comes from
                      ``dllm_config.curriculum_total_steps``.

        Returns
        -------
        mask_ratio : float in [min_mask_ratio, current_upper]
        """
        import random as _random

        progress = min(max(global_step, 0) / self._curriculum_total_steps, 1.0)
        current_upper = (
            self._curriculum_initial_max_mask_ratio
            + progress
            * (self._curriculum_max_mask_ratio - self._curriculum_initial_max_mask_ratio)
        )
        return _random.uniform(self._curriculum_min_mask_ratio, current_upper)

    def build_noisy_samples(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        prompt_len: int,
        decode_orders: Optional[torch.Tensor] = None,
        global_step: int = 0,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate noisy (perturbed) sequences for all MC samples.

        For each sequence in the batch, draws mc_num independent noisy copies by calling
        the configured noisy backend n_l times (each producing mc_num//n_l samples).

        Args
        ----
        input_ids      : (batch, seq_len) int64  – original token ids
        attention_mask : (batch, seq_len) int64  – 1 for real tokens, 0 for padding
        prompt_len     : int  – number of prompt tokens; response starts at this position
        decode_orders  : (batch, seq_len) int64 or None – raw NFE step when each position was
                         unmasked during rollout (``decode_orders_nfe`` from the batch).
                         0 = prompt/padding; multiple positions can share the same NFE step.
                         T = max value = total NFE steps for the sequence.
                         Required for all trajectory backends; ignored by all other backends.
                         Produced by sglang dLLM with ``record_decode_order: true``.
                         Also used by ``block_random`` to determine actual response length.
        global_step    : current training step; curriculum backends combine it
                         with ``dllm_config.curriculum_total_steps`` to compute the mask ratio.

        Returns
        -------
        perturbed_seq : (batch, mc_num, seq_len) int64   – noisy token ids
        mask_indices  : (batch, mc_num, seq_len) bool    – True at masked (perturbed) positions
        p_mask        : (batch, mc_num, seq_len) float   – per-token mask probability
        """
        if (
            self._noisy_backend in (
                "trajectory", "trajectory_curriculum",
                "block_trajectory", "block_trajectory_curriculum",
                "all_block_trajectory", "all_block_trajectory_curriculum",
                "all_block_trajectory_fixed_t",
            )
            and decode_orders is None
        ):
            raise ValueError(
                f"[DLLMActor] noisy_backend='{self._noisy_backend}' requires decode_orders to be passed, "
                "but it is None.  Make sure the rollout config has record_decode_order: true "
                "and the rollout engine supports it (FullAttnMultiBlock, LowConfidence, "
                "EntropyThreshold, LowConfidenceMultiBlock with record_decode_order=true)."
            )

        batch_size, _ = input_ids.shape
        n_l = self.n_l
        mc_num = self.mc_num
        n_y_l = mc_num // n_l

        # ── Trajectory backend: batch-level T=0 scan (all sequences, first call only) ──
        if (
            self._noisy_backend in ("trajectory", "trajectory_curriculum")
            and decode_orders is not None
            and self._traj_debug_count == 0
        ):
            _t0_indices = [
                i for i in range(batch_size) if int(decode_orders[i].max().item()) == 0
            ]
            _all_mask_ids = [
                i
                for i in range(batch_size)
                if (input_ids[i] == self.MASK_TOKEN_ID).all().item()
                or (input_ids[i] == self.PAD_TOKEN_ID).all().item()
            ]
            _rank0_print(
                f"[BatchScan] batch_size={batch_size}  T=0 sequences: {len(_t0_indices)}/{batch_size} at indices {_t0_indices}\n"
                f"  all-mask/pad input_ids sequences: {len(_all_mask_ids)}/{batch_size} at indices {_all_mask_ids}"
            )

        all_perturbed, all_mask, all_p = [], [], []

        for i in range(batch_size):
            single = input_ids[i : i + 1].expand(n_l, -1)
            attn_i = attention_mask[i]
            mc_perturbed, mc_mask, mc_p = [], [], []

            for j in range(n_y_l):
                kwargs = dict(
                    batch=single,
                    attention_mask=attn_i,
                    prompt_len=prompt_len,
                    MASK_TOKEN_ID=self.MASK_TOKEN_ID,
                )
                if self._noisy_backend == "d1":
                    # Each call draws an independent prompt mask pattern, naturally implementing MC
                    kwargs["p"] = self._d1_prompt_mask_p
                    ps, mi, pm = build_d1_noisy_batch(**kwargs)
                elif self._noisy_backend in ("random", "random_curriculum"):
                    kwargs["block"] = self.block_size
                    if self._noisy_backend == "random_curriculum":
                        kwargs["mask_ratio"] = self._get_curriculum_mask_ratio(global_step)
                    ps, mi, pm = build_random_noisy_batch(**kwargs)
                elif self._noisy_backend == "block_random":
                    # Each MC call randomly selects one block and masks within it.
                    # Uses block_random_block_size (independent of block_size for full-attn models).
                    kwargs["block_size"] = self._block_random_block_size
                    ps, mi, pm = build_block_random_noisy_batch(**kwargs)
                elif self._noisy_backend in ("trajectory", "trajectory_curriculum"):
                    # Each MC call independently samples a discrete step t from the rollout
                    # trajectory.  The curriculum variant instead projects its scheduled
                    # target mask ratio onto the same raw NFE order.
                    assert (
                        decode_orders is not None
                    )  # guaranteed by the check at method entry
                    kwargs["decode_orders"] = decode_orders[i]  # (seq_len,)
                    if self._noisy_backend == "trajectory_curriculum":
                        kwargs["mask_ratio"] = self._get_curriculum_mask_ratio(global_step)
                    ps, mi, pm = build_trajectory_noisy_batch(**kwargs)

                    # ── Trajectory debug trace (first 5 calls, sample-0 mc-0 only) ─────
                    if i == 0 and j == 0 and self._traj_debug_count < 5:
                        self._traj_debug_count += 1
                        _do   = decode_orders[0]
                        _mi   = mi[0]
                        _pm   = pm[0]
                        _ids  = input_ids[0]
                        _noisy0 = ps[0]
                        _T    = int(_do.max().item())
                        _MASK_ID = self.MASK_TOKEN_ID
                        _EOS_ID  = getattr(getattr(self, "tokenizer", None), "eos_token_id", 126081)

                        def _tok_label_t(tid):
                            if tid == _EOS_ID:   return "EOS"
                            if tid == _MASK_ID:  return "MSK"
                            return str(int(tid))

                        _resp_region = (_do > 0)
                        _n_resp     = int(_resp_region.sum().item())
                        _n_in_loss  = int(_mi[_resp_region].sum().item())
                        _genuine_mask = _resp_region & ~_mi & (_noisy0 != _MASK_ID)
                        _genuine_do   = _do[_genuine_mask]
                        _t_inferred   = int(_genuine_do.max().item()) if _genuine_do.numel() > 0 else 0
                        # trailing = response region token still = MASK in original (post-EOS)
                        _n_trailing = int((_resp_region & (_ids == _MASK_ID)).sum().item())

                        _rows = []
                        _violations = 0
                        for _p in range(int(_ids.shape[0])):
                            if not _resp_region[_p]:
                                continue
                            _d   = int(_do[_p])
                            _m   = bool(_mi[_p])
                            _nid = int(_noisy0[_p])
                            _pk  = round(float(_pm[_p]), 3)
                            _orig_lbl  = _tok_label_t(int(_ids[_p]))
                            _noisy_lbl = _tok_label_t(_nid)
                            _is_trailing = (int(_ids[_p]) == _MASK_ID)
                            _expected = (_d > _t_inferred or _d == 0) and not _is_trailing
                            _ok = _m == _expected
                            if not _ok:
                                _violations += 1
                            _tag = "[M]" if _m else ("(tr)" if _is_trailing else "   ")
                            _rows.append(
                                f"  p{_p:4d} orig={_orig_lbl:>4s} noisy={_noisy_lbl:>4s}"
                                f" do={_d:3d} {_tag} p={_pk:.3f}"
                                f"{' !' if not _ok else ''}"
                            )
                        _rank0_print(
                            f"[trajectory #{self._traj_debug_count}] "
                            f"T(NFE)={_T}  t(inferred)={_t_inferred}  "
                            f"in_loss={_n_in_loss}  trailing(excluded)={_n_trailing}  "
                            f"resp_total={_n_resp}  violations={_violations}\n"
                            f"  legend: [M]=in_loss  (tr)=trailing MASK excluded from loss\n"
                            f"{'  ' + chr(10).join(_rows)}"
                        )

                elif self._noisy_backend == "block_trajectory":
                    # d3LLM-aligned: randomly select one block, then apply trajectory masking
                    # within it (decode_orders > t). Future blocks filled with MASK (no loss).
                    assert decode_orders is not None
                    kwargs["block_size"] = self._block_random_block_size
                    kwargs["decode_orders"] = decode_orders[i]
                    ps, mi, pm = build_block_trajectory_noisy_batch(**kwargs)

                    # ── Trajectory debug trace (first 5 calls, sample-0 mc-0 only) ─────
                    if i == 0 and j == 0 and self._traj_debug_count < 5:
                        self._traj_debug_count += 1
                        _do  = decode_orders[0]   # (seq_len,)
                        _mi  = mi[0]              # (seq_len,) bool
                        _pm  = pm[0]              # (seq_len,) float
                        _ids = input_ids[0]       # (seq_len,)
                        _T   = int(_do.max().item())
                        _MASK_ID = self.MASK_TOKEN_ID
                        _EOS_ID  = getattr(getattr(self, "tokenizer", None),
                                           "eos_token_id", 126081)

                        def _tok_label(tid):
                            if tid == _EOS_ID:   return "EOS"
                            if tid == _MASK_ID:  return "MSK"
                            return str(int(tid))

                        # Categorise each response position (do > 0):
                        #   in_loss     : mask_indices=True
                        #   future_blk  : mask_indices=False but noisy=MASK  (no loss)
                        #   revealed    : mask_indices=False and noisy!=MASK
                        _resp_region = (_do > 0)
                        _resp_pos    = _resp_region.nonzero(as_tuple=False).squeeze(1)
                        _n_resp      = int(_resp_pos.numel())
                        _n_in_loss   = int(_mi[_resp_region].sum().item())

                        # Fix t_inf: use only genuinely revealed positions (exclude future-block)
                        _noisy0 = ps[0]   # noisy tokens for this sample
                        _genuine_mask = _resp_region & ~_mi & (_noisy0 != _MASK_ID)
                        _genuine_do   = _do[_genuine_mask]
                        _t_inferred   = (
                            int(_genuine_do.max().item())
                            if _genuine_do.numel() > 0 else 0
                        )
                        _n_future = int(
                            (_resp_region & ~_mi & (_noisy0 == _MASK_ID)).sum().item()
                        )

                        # Infer selected block index from first [M] position.
                        _masked_pos_idx = _mi.nonzero(as_tuple=False)
                        if _masked_pos_idx.numel() > 0:
                            _first_m = int(_masked_pos_idx[0].item())
                            _blk_idx = (_first_m - prompt_len) // self._block_random_block_size
                            _blk_start = prompt_len + _blk_idx * self._block_random_block_size
                            _blk_end   = _blk_start + self._block_random_block_size
                        else:
                            _blk_idx, _blk_start, _blk_end = -1, -1, -1

                        # Show all response positions (do > 0)
                        _rows = []
                        _violations = 0
                        for _p in range(int(_ids.shape[0])):
                            if not _resp_region[_p]:
                                continue
                            _d   = int(_do[_p])
                            _m   = bool(_mi[_p])
                            _nid = int(_noisy0[_p])
                            _pk  = round(float(_pm[_p]), 3)
                            _orig_lbl  = _tok_label(int(_ids[_p]))
                            _noisy_lbl = _tok_label(_nid)
                            _is_future = (not _m) and (_nid == _MASK_ID)
                            _expected  = (not _is_future) and (_d > _t_inferred or _d == 0)
                            _ok = _m == _expected
                            if not _ok:
                                _violations += 1
                            # block label: "sel" for selected block, "fb" for future, "pre" for previous
                            if _blk_start <= _p < _blk_end:
                                _blk_label = f"b{_blk_idx}"
                            elif _p >= _blk_end:
                                _blk_label = "fb "
                            else:
                                _blk_label = "pre"
                            _status = "[M]" if _m else ("(fb)" if _is_future else "   ")
                            _rows.append(
                                f"  p{_p:4d} blk={_blk_label} orig={_orig_lbl:>4s} noisy={_noisy_lbl:>4s}"
                                f" do={_d:3d} {_status} p={_pk:.3f}"
                                f"{' !' if not _ok else ''}"
                            )
                        _table = "\n".join(_rows)
                        _rank0_print(
                            f"[block_trajectory #{self._traj_debug_count}] "
                            f"T(NFE)={_T}  t(inferred)={_t_inferred}  "
                            f"selected_block=b{_blk_idx}[{_blk_start}:{_blk_end}]  "
                            f"block_size={self._block_random_block_size}  "
                            f"in_loss={_n_in_loss}  future_blk={_n_future}  "
                            f"resp_total={_n_resp}  violations={_violations}\n"
                            f"{_table}"
                        )
                elif self._noisy_backend == "block_trajectory_curriculum":
                    # Curriculum block-trajectory: use the shared block trajectory builder,
                    # passing a scheduled mask ratio instead of sampling t uniformly.
                    assert decode_orders is not None
                    kwargs["block_size"] = self._block_random_block_size
                    kwargs["decode_orders"] = decode_orders[i]
                    kwargs["mask_ratio"] = self._get_curriculum_mask_ratio(global_step)
                    ps, mi, pm = build_block_trajectory_noisy_batch(**kwargs)

                elif self._noisy_backend in (
                    "all_block_trajectory", "all_block_trajectory_curriculum",
                    "all_block_trajectory_fixed_t",
                ):
                    # SDAR + LCMB: apply trajectory masking to ALL blocks simultaneously.
                    # The curriculum backend projects one scheduled mask_ratio onto each LCMB
                    # window; all windows contribute loss in a single block-causal forward.
                    # block_size should be
                    # set to LCMB_BLOCK_SIZE (= NUM_BLOCKS x BASE_BLOCK_SIZE) so each
                    # "block" here is one LCMB decode window.
                    assert decode_orders is not None
                    kwargs["block_size"] = self._block_random_block_size
                    kwargs["decode_orders"] = decode_orders[i]
                    if self._noisy_backend == "all_block_trajectory_curriculum":
                        kwargs["mask_ratio"] = self._get_curriculum_mask_ratio(global_step)
                    elif self._noisy_backend == "all_block_trajectory_fixed_t":
                        kwargs["mask_ratio"] = self._fixed_t
                    ps, mi, pm = build_all_block_trajectory_noisy_batch(**kwargs)

                    # ── all_block_trajectory debug trace (first 5 calls, sample-0 mc-0 only) ─
                    if i == 0 and j == 0 and self._traj_debug_count < 5:
                        self._traj_debug_count += 1
                        _do      = decode_orders[0]   # (seq_len,) raw per-window NFE
                        _mi      = mi[0]              # (seq_len,) bool
                        _noisy0  = ps[0]              # (seq_len,) noisy token ids
                        _ids     = input_ids[0]       # (seq_len,) original token ids
                        _MASK_ID = self.MASK_TOKEN_ID
                        _EOS_ID  = getattr(getattr(self, "tokenizer", None), "eos_token_id", 126081)
                        _blk_sz  = self._block_random_block_size

                        def _tok_label(tid):
                            if tid == _EOS_ID:   return "EOS"
                            if tid == _MASK_ID:  return "MSK"
                            return str(int(tid))

                        _resp_region = (_do > 0)
                        _n_resp    = int(_resp_region.sum().item())
                        _n_in_loss = int(_mi[_resp_region].sum().item())

                        # Per-window breakdown: infer t from revealed vs masked positions.
                        # revealed: mask_indices=False, noisy≠MASK, do>0 → do ≤ t
                        # masked:   mask_indices=True, do>0 → do > t
                        _win_rows = []
                        resp_start = int(_resp_region.nonzero(as_tuple=False)[0].item()) if _n_resp > 0 else prompt_len
                        num_wins = _n_resp // _blk_sz if _n_resp >= _blk_sz else 0
                        for _w in range(num_wins):
                            _ws = resp_start + _w * _blk_sz
                            _we = _ws + _blk_sz
                            if _we > int(_do.shape[0]):
                                break
                            _w_do  = _do[_ws:_we]
                            _w_mi  = _mi[_ws:_we]
                            _w_ids = _noisy0[_ws:_we]
                            # unique-NFE local remap (same as build_all_block_trajectory):
                            # tokens sharing an NFE step keep the same local rank.
                            _vals = [int(v) for v in _w_do.tolist()]
                            _uniq = sorted({v for v in _vals if v > 0})
                            _rank_map = {v: i + 1 for i, v in enumerate(_uniq)}
                            _local_rank = [_rank_map.get(v, 0) for v in _vals]
                            _T_win = len(_uniq)
                            # infer t: max local rank among revealed (not masked) positions
                            _revealed_ranks = [_local_rank[p] for p in range(_blk_sz)
                                               if not bool(_w_mi[p]) and _local_rank[p] > 0
                                               and int(_w_ids[p]) != _MASK_ID]
                            _t_inf = max(_revealed_ranks) if _revealed_ranks else 0
                            _n_masked   = int(_w_mi.sum().item())
                            _n_revealed = _blk_sz - _n_masked
                            _win_rows.append(
                                f"  win{_w:3d}[{_ws}:{_we}] T_win={_T_win:2d}  t~{_t_inf:2d}"
                                f"  masked={_n_masked}/{_blk_sz}  revealed={_n_revealed}/{_blk_sz}"
                            )

                        # Full per-position table for first 2 windows
                        _pos_rows = []
                        for _p in range(int(_ids.shape[0])):
                            if not _resp_region[_p]:
                                continue
                            _win_idx = (_p - resp_start) // _blk_sz
                            if _win_idx >= 2:   # show only first 2 windows
                                continue
                            _d    = int(_do[_p])
                            _m    = bool(_mi[_p])
                            _nid  = int(_noisy0[_p])
                            _orig_lbl  = _tok_label(int(_ids[_p]))
                            _noisy_lbl = _tok_label(_nid)
                            _status = "[M]" if _m else "   "
                            _pos_rows.append(
                                f"  p{_p:4d} win={_win_idx}  orig={_orig_lbl:>4s}"
                                f"  noisy={_noisy_lbl:>4s}  do={_d:3d}  {_status}"
                            )

                        _rank0_print(
                            f"[all_block_trajectory #{self._traj_debug_count}] "
                            f"block_size={_blk_sz}  n_windows={num_wins}"
                            f"  in_loss={_n_in_loss}/{_n_resp}"
                            f"  coverage={100*_n_in_loss/max(_n_resp,1):.1f}%\n"
                            + "\n".join(_win_rows[:16])   # up to 16 windows in summary
                            + ("\n  ...\n" if num_wins > 16 else "\n")
                            + "  [first 2 windows per-position]\n"
                            + "\n".join(_pos_rows)
                        )

                else:  # gaussian
                    kwargs["block"] = self.block_size
                    kwargs["t_fixed"] = self._gdpo_xi[
                        j
                    ]  # j-th iteration → j-th quadrature point
                    ps, mi, pm = build_gaussian_noisy_batch(**kwargs)

                mc_perturbed.append(ps)
                mc_mask.append(mi)
                mc_p.append(pm)

            all_perturbed.append(torch.cat(mc_perturbed, dim=0))
            all_mask.append(torch.cat(mc_mask, dim=0))
            all_p.append(torch.cat(mc_p, dim=0))

        return (
            torch.stack(all_perturbed, dim=0),
            torch.stack(all_mask, dim=0),
            torch.stack(all_p, dim=0),
        )

    def _gauss_weight_samples(
        self,
        values: torch.Tensor,
        batch_size: int,
        mc_num: int,
        device: torch.device,
        extra_dim: Optional[int] = None,
    ) -> torch.Tensor:
        """Scale each MC sample by its Gauss-Legendre quadrature coefficient so that
        a subsequent .mean(dim=mc_axis) yields the correct GL-weighted integral estimate.

        MC samples are grouped by quadrature point: group i contains indices
        [i*mc_per_point, (i+1)*mc_per_point).  All samples in group i are multiplied
        by (w_i * gauss_N), where gauss_N = number of quadrature points.  This ensures:

            mean over mc_num samples = sum_i [ w_i * mean_j(f_{i,j}) ]  (GL estimate)

        because .mean(dim=1) = (1/mc_num) * sum_{i,j} coef_i * f_{i,j}
                              = (1/mc_num) * sum_i coef_i * mc_per_point * mean_j(f_{i,j})
        setting coef_i = w_i * gauss_N gives: sum_i w_i * mean_j(f_{i,j}).

        Use this when full aggregation (collapsing the MC dimension) is not required.

        Args
        ----
        values     : (batch*mc,) or (batch*mc, D)  – per-MC values to scale
        batch_size : int  – number of sequences in the batch
        mc_num     : int  – total MC samples per sequence; must be divisible by gauss_N
        device     : torch.device
        extra_dim  : int or None  – trailing dimension D; None if values is 1-D per sample

        Returns
        -------
        (batch*mc,) or (batch*mc, D)  – same shape as input, each sample scaled by its
                                        GL coefficient so that .mean(dim=mc_axis) = GL estimate
        """
        weights = torch.tensor(self._gauss_weights, dtype=torch.bfloat16, device=device)
        mc_per_point = mc_num // self._gauss_N
        # coef_i = w_i * gauss_N (see docstring for derivation)
        coef = (weights * self._gauss_N).repeat_interleave(mc_per_point)  # (mc_num,)
        if extra_dim is not None:
            return values * coef.unsqueeze(-1).unsqueeze(0).expand(
                batch_size, -1, extra_dim
            ).reshape(batch_size * mc_num, extra_dim)
        else:
            return values * coef.unsqueeze(0).expand(batch_size, -1).reshape(
                batch_size * mc_num
            )

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False):
        """Compute log-probabilities (and optionally entropy) for a batch of noisy sequences.

        Runs the model in eval mode with no_grad over all micro-batches and concatenates results.

        Args
        ----
        data              : DataProto containing perturbed_seq, mask_indices, p_mask,
                            input_ids, attention_mask, position_ids, responses.
        calculate_entropy : bool – if True, also compute per-token entropy.

        Returns
        -------
        entropys         : (total_bs, response_len) float or None
        log_probs        : (total_bs, response_len) float
        loss_per_sample  : (total_bs, mc_num, response_len) float or None
        """
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]
        use_dynamic_bsz = self.config.use_dynamic_bsz
        if self.debug_saver is not None:
            self._debug_mb_counter = 0  # 每次 compute_log_prob 重置 micro-batch 计数

        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "perturbed_seq",
            "mask_indices",
            "p_mask",
        ]
        batch = data.select(batch_keys=select_keys).batch

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"]
            micro_batches, indices = rearrange_micro_batches(
                batch=batch, max_token_len=max_token_len
            )
        else:
            micro_batches = batch.split(micro_batch_size)

        log_prob_lst, entropy_lst, loss_per_sample_lst = [], [], []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_prob, loss_per_sample = self._forward_micro_batch(
                    micro_batch,
                    temperature=temperature,
                    n_l=self.n_l,
                    mc_num=self.mc_num,
                    calculate_entropy=calculate_entropy,
                    call_fn_name="compute_log_prob",
                )
            log_prob_lst.append(log_prob)
            loss_per_sample_lst.append(loss_per_sample)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.cat(log_prob_lst, dim=0)
        # log_probs: (total_bs, response_len)
        loss_per_sample = (
            torch.cat(loss_per_sample_lst, dim=0)
            if self._approx_per_mc_update
            else None
        )  # (total_bs, mc_num, response_len) or None
        entropys = torch.cat(entropy_lst, dim=0) if calculate_entropy else None
        # entropys: (total_bs, response_len)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            revert = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert]
            if loss_per_sample is not None:
                loss_per_sample = loss_per_sample[revert]
            if entropys is not None:
                entropys = entropys[revert]

        return entropys, log_probs, loss_per_sample

    def _backward_micro_batch_single_forward(
        self, mb, temperature, response_mask, advantages, grad_scale: float = 1.0
    ):
        """Run a single forward pass over all MC samples and perform one backward.

        All mc_num noisy copies are processed together in a single model call.  The resulting
        log_prob (MC-averaged) is used directly for the PPO surrogate objective.

        Args
        ----
        mb            : dict – micro-batch tensors on the current device
        temperature   : float
        response_mask : (mbs, response_len) bool/float – 1 at valid response positions
        advantages    : (mbs, response_len) float
        grad_scale    : float – gradient accumulation scale factor

        Returns
        -------
        metrics     : dict of scalar floats for logging
        """
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        clip_ratio = self.config.clip_ratio
        clip_ratio_low = (
            self.config.clip_ratio_low
            if self.config.clip_ratio_low is not None
            else clip_ratio
        )
        clip_ratio_high = (
            self.config.clip_ratio_high
            if self.config.clip_ratio_high is not None
            else clip_ratio
        )
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

        mc_num = mb["perturbed_seq"].size(1)  # scalar
        entropy, log_prob, loss_per_sample = self._forward_micro_batch(
            mb,
            temperature=temperature,
            n_l=self.n_l,
            mc_num=mc_num,
            calculate_entropy=(entropy_coeff != 0),
            call_fn_name="update_policy",
        )
        # log_prob: (bs, response_len)

        old_l_theta = mb["old_log_probs"]  # (bs, response_len)
        l_theta_cur = log_prob  # (bs, response_len)
        pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_pos, pg_clipfrac_neg = compute_policy_loss(
            old_l_theta=old_l_theta,
            l_theta=l_theta_cur,
            advantages=advantages,
            response_mask=response_mask,
            cliprange=clip_ratio,
            cliprange_low=clip_ratio_low,
            cliprange_high=clip_ratio_high,
            clip_ratio_c=clip_ratio_c,
            loss_agg_mode=loss_agg_mode,
        )
        ent_mat = entropy  # (bs, response_len) or None
        kl_lp = l_theta_cur
        kl_ref = (
            mb["ref_log_probs"] if self.config.use_kl_loss else None
        )  # (bs, response_len)

        policy_loss = pg_loss
        kl_loss_val = 0.0
        if entropy_coeff != 0 and ent_mat is not None:
            policy_loss = (
                policy_loss
                - agg_loss(ent_mat, response_mask, loss_agg_mode) * entropy_coeff
            )
        if self.config.use_kl_loss and kl_ref is not None:
            kld = kl_penalty(
                l_theta=kl_lp,
                ref_l_theta=kl_ref,
                kl_penalty=self.config.kl_loss_type,
                advantages=advantages,
            )
            kl_agg = agg_loss(kld, response_mask, loss_agg_mode)
            kl_loss_val = kl_agg.detach().item()
            policy_loss = policy_loss + kl_agg * self.config.kl_loss_coef

        (policy_loss * grad_scale).backward()

        return {
            "actor/pg_loss": pg_loss.detach().item(),
            "actor/pg_clipfrac": pg_clipfrac.detach().item(),
            "actor/ppo_kl": ppo_kl.detach().item(),
            "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
            "actor/pg_clipfrac_pos": pg_clipfrac_pos.detach().item(),
            "actor/pg_clipfrac_neg": pg_clipfrac_neg.detach().item(),
            "actor/kl_loss": kl_loss_val,
        }

    def _backward_micro_batch_per_mc(
        self, mb, temperature, response_mask, advantages, grad_scale: float = 1.0
    ):
        """Run one forward+backward per MC sample using the BGPO linearised approximation.

        Each of the mc_num noisy copies is forwarded separately with mc_num=1, and its
        contribution to the policy loss is computed via a linearised importance-weight ratio
        (linearize_ratio=True).  The gradient is accumulated across all mc_num calls.
        This trades more forward passes for lower peak memory compared to the single-forward path.

        Args
        ----
        mb            : dict – micro-batch tensors; must contain old_loss_per_sample
                         (mbs, mc_num, response_len) from the reference forward pass.
        temperature   : float
        response_mask : (mbs, response_len) bool/float
        advantages    : (mbs, response_len) float
        grad_scale    : float – gradient accumulation scale factor

        Returns
        -------
        metrics : dict of scalar floats averaged over mc_num steps
        """
        entropy_coeff = self.config.entropy_coeff
        loss_agg_mode = self.config.loss_agg_mode
        clip_ratio = self.config.clip_ratio
        clip_ratio_low = (
            self.config.clip_ratio_low
            if self.config.clip_ratio_low is not None
            else clip_ratio
        )
        clip_ratio_high = (
            self.config.clip_ratio_high
            if self.config.clip_ratio_high is not None
            else clip_ratio
        )
        clip_ratio_c = self.config.get("clip_ratio_c", 3.0)

        perturbed_seq = mb["perturbed_seq"]  # (bs, mc_num, seq_len)
        mask_indices = mb["mask_indices"]  # (bs, mc_num, seq_len)
        p_mask = mb["p_mask"]  # (bs, mc_num, seq_len)
        mc_num = perturbed_seq.size(1)  # scalar

        acc_pg, acc_clipfrac, acc_kl, acc_clipfrac_low, acc_kl_loss = (
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
        acc_clipfrac_pos, acc_clipfrac_neg = 0.0, 0.0

        for i in range(mc_num):
            mb_i = {
                **mb,
                "perturbed_seq": perturbed_seq[:, i : i + 1],
                "mask_indices": mask_indices[:, i : i + 1],
                "p_mask": p_mask[:, i : i + 1],
            }
            entropy_i, log_prob_i, loss_per_sample_i = self._forward_micro_batch(
                mb_i,
                temperature=temperature,
                n_l=1,
                mc_num=1,
                calculate_entropy=(entropy_coeff != 0),
                call_fn_name="update_policy",
            )
            # log_prob_i:       (bs, response_len) — mc_num=1 already averaged by mean(dim=1)
            # loss_per_sample_i: (bs, 1, response_len)
            assert loss_per_sample_i is not None
            pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower, pg_clipfrac_pos, pg_clipfrac_neg = compute_policy_loss(
                old_l_theta=mb["old_loss_per_sample"][:, i, :],  # (bs, response_len)
                l_theta=loss_per_sample_i[:, 0, :],  # (bs, response_len)
                advantages=advantages,
                response_mask=response_mask,
                cliprange=clip_ratio,
                cliprange_low=clip_ratio_low,
                cliprange_high=clip_ratio_high,
                clip_ratio_c=clip_ratio_c,
                loss_agg_mode=loss_agg_mode,
                linearize_ratio=True,
            )
            ent_mat = entropy_i  # (bs, response_len) or None

            policy_loss = pg_loss
            if entropy_coeff != 0 and ent_mat is not None:
                policy_loss = (
                    policy_loss
                    - agg_loss(ent_mat, response_mask, loss_agg_mode) * entropy_coeff
                )
            if self.config.use_kl_loss:
                kld = kl_penalty(
                    l_theta=log_prob_i,
                    ref_l_theta=mb["ref_log_probs"],
                    kl_penalty=self.config.kl_loss_type,
                    advantages=advantages,
                )
                kl_loss_i = agg_loss(kld, response_mask, loss_agg_mode)
                acc_kl_loss += kl_loss_i.detach().item()
                policy_loss = policy_loss + kl_loss_i * self.config.kl_loss_coef

            (policy_loss / mc_num * grad_scale).backward()

            acc_pg += pg_loss.detach().item()
            acc_clipfrac += pg_clipfrac.detach().item()
            acc_kl += ppo_kl.detach().item()
            acc_clipfrac_low += pg_clipfrac_lower.detach().item()
            acc_clipfrac_pos += pg_clipfrac_pos.detach().item()
            acc_clipfrac_neg += pg_clipfrac_neg.detach().item()

        return {
            "actor/pg_loss": acc_pg / mc_num,
            "actor/pg_clipfrac": acc_clipfrac / mc_num,
            "actor/ppo_kl": acc_kl / mc_num,
            "actor/pg_clipfrac_lower": acc_clipfrac_low / mc_num,
            "actor/pg_clipfrac_pos": acc_clipfrac_pos / mc_num,
            "actor/pg_clipfrac_neg": acc_clipfrac_neg / mc_num,
            "actor/kl_loss": acc_kl_loss / mc_num,
        }

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        """Run RL policy update for one data batch (multiple epochs × mini-batches).

        PPO/GRPO surrogate objective.  Requires ``old_log_probs``,
        ``advantages`` (and optionally ``ref_log_probs``,
        ``old_loss_per_sample``) in the batch.  Dispatches to
        ``_backward_micro_batch_single_forward`` or
        ``_backward_micro_batch_per_mc`` depending on
        ``config.approx_per_mc_update``.

        Args
        ----
        data : DataProto
            RL mode   : must contain old_log_probs, advantages, perturbed_seq,
                        mask_indices, p_mask; optionally ref_log_probs,
                        old_loss_per_sample, loss_mask.
            Distill   : must contain perturbed_seq, mask_indices, p_mask,
                        input_ids, attention_mask, position_ids, responses;
                        optionally loss_mask (multi-turn).

        Returns
        -------
        metrics : dict mapping metric name → list of scalar values
        """
        self.actor_module.train()

        temperature = data.meta_info["temperature"]
        multi_turn = data.meta_info.get("multi_turn", False)
        approx_per_mc_update = self._approx_per_mc_update

        # ── Select batch keys ────────────────────────────────────────────────
        select_keys = [
            "responses",
            "input_ids",
            "attention_mask",
            "position_ids",
            "perturbed_seq",
            "mask_indices",
            "p_mask",
            "old_log_probs",
            "advantages",
        ]
        if approx_per_mc_update:
            select_keys.append("old_loss_per_sample")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_probs")
        if multi_turn:
            select_keys.append("loss_mask")
        if ADJUST_BATCH_GRADIENT_MASK_KEY in data.batch.keys():
            select_keys.append(ADJUST_BATCH_GRADIENT_MASK_KEY)

        batch = data.select(batch_keys=select_keys).batch

        # Resolve -1 to full batch size so callers can pass -1 to mean "no splitting".
        _total_B = batch.batch_size[0]
        _mini_bs = getattr(self.config, "ppo_mini_batch_size", _total_B)
        if _mini_bs <= 0:
            _mini_bs = _total_B
        _micro_bs = getattr(self.config, "ppo_micro_batch_size_per_gpu", _mini_bs)
        if _micro_bs <= 0:
            _micro_bs = _mini_bs

        # Trim to nearest multiple of ppo_micro_batch_size_per_gpu so every
        # micro-batch has the same size and gradient_accumulation = len(micro_batches)
        # gives accurate loss normalization.  Discards at most (micro_bsz-1) tail
        # samples per update — negligible compared to total turn-level batch size.
        if _micro_bs and not getattr(self.config, "use_dynamic_bsz", False):
            trimmed_n = (_total_B // _micro_bs) * _micro_bs
            if trimmed_n < _total_B:
                batch = batch[:trimmed_n]

        dataloader = batch.split(_mini_bs)

        # ── Choose backward function ─────────────────────────────────────────
        if approx_per_mc_update:
            _backward_mb = self._backward_micro_batch_per_mc
        else:
            _backward_mb = self._backward_micro_batch_single_forward

        metrics = {}
        _is_rank0 = not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
        if _is_rank0:
            print(f"[update_policy] total_B={_total_B} mini_bs={_mini_bs} micro_bs={_micro_bs} "
                  f"gradient_accumulation={_mini_bs // _micro_bs} epochs={self.config.ppo_epochs}")
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
                    print(f"[update_policy] epoch={_epoch} mini_batch={_mini_idx} "
                          f"gradient_accumulation={self.gradient_accumulation}")
                self.actor_optimizer.zero_grad()

                for mb in micro_batches:
                    if isinstance(mb, DataProto):
                        mb = {
                            **mb.batch.to(get_torch_device().current_device()),
                            **mb.non_tensor_batch,
                        }
                    else:
                        mb = mb.to(get_torch_device().current_device())

                    responses = mb["responses"]  # (mbs, response_len)
                    response_length = responses.size(1)
                    attention_mask = mb["attention_mask"]  # (mbs, seq_len)
                    response_mask = (
                        mb["loss_mask"][:, -response_length:]
                        if multi_turn
                        else attention_mask[:, -response_length:]
                    )  # (mbs, response_len)
                    response_mask = _apply_adjust_batch_gradient_mask(response_mask, mb)
                    # advantages is None in distillation mode (ignored by the wrapper)
                    advantages = mb.get("advantages", None)

                    grad_scale = (
                        mb["input_ids"].size(0) / _mini_bs
                        if self.config.use_dynamic_bsz
                        else 1.0 / self.gradient_accumulation
                    )
                    step_metrics = _backward_mb(
                        mb,
                        temperature,
                        response_mask,
                        advantages,
                        grad_scale=grad_scale,
                    )
                    append_to_dict(metrics, step_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {"actor/grad_norm": grad_norm.detach().item()})

        self.actor_optimizer.zero_grad()
        return metrics


    def _forward_micro_batch(
        self,
        micro_batch,
        temperature,
        n_l,
        mc_num,
        calculate_entropy: bool = False,
        call_fn_name: str = "",
    ) -> Tuple[Optional[torch.Tensor], torch.Tensor, Optional[torch.Tensor]]:
        """Run one forward pass on a micro-batch and return log-probabilities.

        Layout convention
        -----------------
        Every sequence in the batch has layout:
            [ prompt tokens (prompt_len) | response tokens (response_len) | right-padding ]
        padded to a fixed ``seq_len = prompt_len + response_len`` (the micro-batch maximum).
        ``mc_num`` independent noisy copies of each sequence are stacked along dim-1, giving
        a (batch, mc_num, seq_len) representation for perturbed inputs.

        Example (batch=1, mc_num=2, prompt_len=3, response_len=4, pad_len=2):
            # seq_len = 3 + 4 + 2 = 9; PAD = pad token id, MSK = MASK_TOKEN_ID
            input_ids       = [[P0, P1, P2,  R0,  R1,  R2,  R3, PAD, PAD]]  # (1, 9)
            attention_mask  = [[ 1,  1,  1,   1,   1,   1,   1,   0,   0]]  # (1, 9)
            position_ids    = [[ 0,  1,  2,   3,   4,   5,   6,   0,   0]]  # (1, 9); 0 at pad is arbitrary
            responses       = [[R0, R1, R2, R3]]                             # (1, 4)

            # MC sample 0: R0 and R2 masked
            perturbed[0,0]  = [P0, P1, P2, MSK,  R1, MSK,  R3, PAD, PAD]
            mask_idx[0,0]   = [ F,  F,  F,   T,   F,   T,   F,   F,   F]
            p_mask[0,0]     = [.0, .0, .0,  .5,  .0,  .5,  .0,  .0,  .0]
            # MC sample 1: R1 and R3 masked
            perturbed[0,1]  = [P0, P1, P2,  R0, MSK,  R2, MSK, PAD, PAD]
            mask_idx[0,1]   = [ F,  F,  F,   F,   T,   F,   T,   F,   F]
            p_mask[0,1]     = [.0, .0, .0,  .0,  .5,  .0,  .5,  .0,  .0]

        Args
        ----
        micro_batch : dict with the following keys
            "input_ids"      – (batch, seq_len) int64
                               Original clean token ids, layout [prompt | response | pad].
                               Example: [[P0, P1, P2, R0, R1, R2, R3, PAD, PAD]] (see above).
            "responses"      – (batch, response_len) int64
                               Response portion only (no prompt, no padding); used solely to
                               read response_len.
                               Example: [[R0, R1, R2, R3]].
            "attention_mask" – (batch, seq_len) int64
                               1 at every non-padding position (prompt + response), 0 at
                               right-padding positions.
                               Example: [[1, 1, 1, 1, 1, 1, 1, 0, 0]].
            "position_ids"   – (batch, seq_len) int64 (block path only)
                               RoPE position index for each token, typically 0-based contiguous
                               counting over non-padding positions; value at padding is arbitrary.
                               Overridden (recomputed from attention_mask) in full-attention path.
                               Example: [[0, 1, 2, 3, 4, 5, 6, 0, 0]].
            "perturbed_seq"  – (batch, mc_num, seq_len) int64
                               mc_num independent noisy versions of input_ids.  Response
                               tokens are randomly replaced with MASK_TOKEN_ID according to
                               the noisy backend; prompt tokens are copied as-is (D1 may also
                               mask a fraction of prompt tokens controlled by d1_prompt_mask_p);
                               padding positions keep PAD.  See example above.
            "mask_indices"   – (batch, mc_num, seq_len) bool
                               True at every position that was replaced with MASK_TOKEN_ID in
                               perturbed_seq; False at prompt / unmasked response / padding.
                               Defines the prediction targets and the positions where log-probs
                               are computed.  Exactly mirrors the MASK_TOKEN_ID positions in
                               perturbed_seq (see example above).
            "p_mask"         – (batch, mc_num, seq_len) float32
                               The per-token mask probability p_t used when drawing the mask for
                               each position in perturbed_seq.  Applied as importance-weight
                               denominator: token_lp / p_t (ELBO correction for non-uniform
                               masking).  Non-zero only where mask_indices is True; 0.0 at
                               prompt and padding positions.
                               Example (gaussian backend, p_t=0.5 for all response tokens):
                                   p_mask[0,0] = [0, 0, 0, 0.5, 0, 0.5, 0, 0, 0]
        temperature : float
            Softmax temperature applied during the forward pass.
        n_l : int
            Number of distinct noise levels (unused directly here; passed to distinguish
            call sites that need it, e.g. build_noisy_samples).
        mc_num : int
            Number of Monte Carlo samples per sequence (= perturbed_seq.size(1)).
            Must equal config.mc_num, or 1 when called per-sample in _update_micro_batch_per_mc.
        calculate_entropy : bool
            If True, compute and return per-token entropy alongside log-probs.
        call_fn_name : str
            Identifier passed for debugging / dispatch purposes; not used in computation.

        Returns
        -------
        entropy : Tensor or None, shape (batch, response_len)
            Per-token entropy H = -p·log(p) at each response position, MC-averaged.
            None when calculate_entropy=False.
            Example (response_len=4): [[0.23, 0.31, 0.18, 0.27]]

        log_prob : Tensor, shape (batch, response_len)
            Log-probability of the response under the current policy, averaged across mc_num
            samples.  Semantics differ by log_prob_level:
            • "token-level"    – log p(x_t | context) at each response token position t,
                                 MC-averaged. E.g. [[-0.4, -0.7, -0.3, -0.5]].
            • "block-level"    – sum of token log-probs within the block containing t, the
                                 same value broadcast to every token in that block.
                                 E.g. with block_size=2: [[-1.1, -1.1, -0.8, -0.8]]
                                 where -1.1 = log p(R0)+log p(R1), -0.8 = log p(R2)+log p(R3).
            • "sequence-level" – total sequence ELBO broadcast uniformly to all response
                                 positions (optionally divided by r_valid).
                                 E.g.: [[-1.9, -1.9, -1.9, -1.9]].

        loss_per_sample : Tensor or None, shape (batch, mc_num, response_len)
            Same quantity as log_prob but *before* averaging over mc_num; needed by
            _update_micro_batch_per_mc for the per-MC linearised importance-weight update.
            None when config.approx_per_mc_update=False.
            Example (mc_num=2): [[[-0.3, -0.8, -0.2, -0.6],
                                   [-0.5, -0.6, -0.4, -0.4]]]
        """
        batch_size = micro_batch["input_ids"].size(0)  # scalar
        approx_per_mc_update = self._approx_per_mc_update
        seq_length = micro_batch["input_ids"].size(-1)  # scalar
        response_length = micro_batch["responses"].size(-1)  # scalar

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            perturbed_seq = micro_batch["perturbed_seq"]  # (batch, mc_num, seq_len)
            mask_indices = micro_batch["mask_indices"]  # (batch, mc_num, seq_len) bool
            orig_seq = micro_batch["input_ids"]  # (batch, seq_len)
            attention_mask = micro_batch["attention_mask"]  # (batch, seq_len)
            p_mask = micro_batch["p_mask"]  # (batch, mc_num, seq_len)

            r_valid = (
                attention_mask[:, -response_length:].sum(dim=-1).long()
            )  # (batch,): actual non-padded response length
            prompt_valid = (
                attention_mask.sum(dim=-1) - r_valid
            )  # (batch,): actual non-padded prompt length

            bmc = (
                batch_size * mc_num
            )  # scalar: flattened batch×MC dimension for model forward

            if self.block_size > 0:
                # ---- Block DLLM path: compact + interleave + block causal mask ---- #
                token_lp_all, pmask_all = self._compute_block_masked_token_logprobs(
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
                # ---- Full Attention path: forward noisy sequence directly ---- #
                token_lp_all, pmask_all = self._compute_full_attn_masked_token_logprobs(
                    perturbed_seq,
                    mask_indices,
                    p_mask,
                    orig_seq,
                    attention_mask,
                    bmc,
                    mc_num,
                    seq_length,
                )

            # ---- Aggregate token log-probs into sequence/block/token level (shared by both paths) ---- #

            # Importance-weight correction: divide by mask probability p (D1: p=1.0 for response, so no-op).
            # Controlled by use_importance_weight; set False to use raw log-probs (DARE D1 style).
            if self._p_mask_normalization:
                token_lp_weighted = token_lp_all / pmask_all.clamp(
                    min=1e-6
                )  # (batch*mc, seq_len): importance-weighted per-token log-prob
            else:
                token_lp_weighted = (
                    token_lp_all  # (batch*mc, seq_len): raw per-token log-prob
                )

            # For gaussian backend, pre-scale each MC sample by its GL quadrature coefficient so that
            # all downstream .mean(dim=mc_axis) calls yield the correct GL-weighted integral estimate.
            # This is done once here so every level (token/block/sequence) and elbo_per_mc are consistent.
            if self._noisy_backend == "gaussian":
                token_lp_weighted = self._gauss_weight_samples(
                    token_lp_weighted,
                    batch_size,
                    mc_num,
                    token_lp_weighted.device,
                    extra_dim=seq_length,
                )  # (batch*mc, seq_len): each sample scaled by w_i * gauss_N

            # set per log_prob_level branch below when approx_per_mc_update=True, else stays None
            loss_per_sample = None

            if self.log_prob_level == "token-level":
                token_lp = token_lp_weighted.view(
                    batch_size, mc_num, seq_length
                )  # (batch, mc_samples, seq_len): GL-weighted (gaussian) or uniform (other backends)
                if self.block_size > 0:
                    # Block path: sequences were compacted by _compact_sequences, so layout is
                    # [prompt|response|tail_pad]. prompt_valid is the correct absolute offset.
                    log_prob_mc = project_compact_to_response(
                        token_lp, prompt_valid, r_valid, response_length
                    )
                else:
                    # Full-attn path: sequences are NOT compacted; layout is
                    # [left_pad|prompt|response|right_pad]. Response is always in the last
                    # response_length positions. shift_logits moves log-probs one slot earlier.
                    if self._shift_logits:
                        log_prob_mc = token_lp[:, :, -(response_length + 1) : -1]
                    else:
                        log_prob_mc = token_lp[:, :, -response_length:]
                    _valid_resp = attention_mask[:, -response_length:].unsqueeze(1)
                    log_prob_mc = log_prob_mc * _valid_resp.to(log_prob_mc.dtype)
                loss_per_sample = (
                    log_prob_mc if approx_per_mc_update else None
                )  # (batch, mc_samples, response_len): kept for per-MC approx update; None otherwise
                log_prob = log_prob_mc.mean(
                    dim=1
                )  # (batch, response_len): MC-averaged token-level log-prob

            elif self.log_prob_level == "block-level":
                block_size = self.block_size
                num_blocks = (seq_length + block_size - 1) // block_size
                pad_len = num_blocks * block_size - seq_length
                tl = (
                    F.pad(token_lp_weighted, (0, pad_len))
                    if pad_len > 0
                    else token_lp_weighted
                )  # (batch*mc, num_blocks*block_size): padded so length is a multiple of block_size
                block_elbo = tl.view(bmc, num_blocks, block_size).sum(
                    dim=-1
                )  # (batch*mc, num_blocks): sum of log-probs within each block
                if self._normalize_log_prob_by_length:
                    block_elbo = block_elbo / block_size
                block_elbo_per_mc = block_elbo.view(
                    batch_size, mc_num, num_blocks
                )  # (batch, mc_num, num_blocks): per-MC block ELBO
                block_ids = (
                    torch.arange(seq_length, device=block_elbo_per_mc.device)
                    // block_size
                )  # (seq_len,): maps each token position to its block index
                log_prob_mc_block = project_compact_to_response(
                    block_elbo_per_mc[:, :, block_ids],  # (batch, mc_num, seq_len)
                    prompt_valid,
                    r_valid,
                    response_length,
                )  # (batch, mc_num, response_len): per-MC block log-prob at each response position
                loss_per_sample = log_prob_mc_block if approx_per_mc_update else None
                log_prob = log_prob_mc_block.mean(
                    dim=1
                )  # (batch, response_len): GL-weighted (gaussian) or uniform mean over MC samples

            elif self.log_prob_level == "sequence-level":
                elbo_all = token_lp_weighted.sum(
                    dim=-1
                )  # (batch*mc,): sum of all token log-probs (full-sequence ELBO) per MC sample
                elbo_mc = elbo_all.view(batch_size, mc_num)  # (batch, mc_num)
                if self._normalize_log_prob_by_length:
                    elbo_mc = elbo_mc / r_valid.clamp(min=1).unsqueeze(-1)
                log_prob_mc_seq = elbo_mc.unsqueeze(-1).expand(
                    -1, -1, response_length
                )  # (batch, mc_num, response_len): sequence-level ELBO broadcast to every response position
                loss_per_sample = (
                    log_prob_mc_seq.contiguous() if approx_per_mc_update else None
                )  # (batch, mc_num, response_len) or None
                log_prob = log_prob_mc_seq.mean(
                    dim=1
                )  # (batch, response_len): MC-averaged sequence-level log-prob

            else:
                raise ValueError(
                    f"[DLLMActor] Unknown log_prob_level '{self.log_prob_level}'."
                )

        entropy = None
        if calculate_entropy:
            token_lp_per_mc = token_lp_all.view(batch_size, mc_num, seq_length)
            if self.block_size > 0:
                # Block path: sequences are compacted by _compact_sequences, layout is
                # [prompt | response | tail-PAD]. Response is NOT at the tail, so tail-slicing
                # would land in the padding region (all zeros) → entropy = 0 always.
                # Use project_compact_to_response to extract the correct response region.
                token_lp_resp = project_compact_to_response(
                    token_lp_per_mc, prompt_valid, r_valid, response_length
                )  # (batch, mc_num, response_len)
            else:
                # Full-attention path: sequences are NOT compacted; layout is
                # [left_pad | prompt | response | right_pad]. Response is at the tail.
                # project_compact_to_response would use prompt_valid (= token count from left
                # including left-pad) as the start offset → reads into left-padding → wrong.
                # Fix: directly slice from the tail.
                if self._shift_logits:
                    token_lp_resp = token_lp_per_mc[:, :, -(response_length + 1) : -1]
                else:
                    token_lp_resp = token_lp_per_mc[:, :, -response_length:]
                # Zero out right-padded response positions
                valid_resp_mask = attention_mask[:, -response_length:].unsqueeze(
                    1
                )  # (batch, 1, resp_len)
                token_lp_resp = token_lp_resp * valid_resp_mask.to(token_lp_resp.dtype)
            prob = token_lp_resp.exp().clamp(
                max=1.0
            )  # (batch, mc_samples, response_len)
            entropy = (-prob * token_lp_resp).mean(
                dim=1
            )  # (batch, response_len): H = -p*log(p) averaged over MC samples

        # ── Debug: 保存 actor-internal 中间数据（仅 rank-0，仅 compute_log_prob 阶段）────
        if (
            self.debug_saver is not None
            and self.debug_step is not None
            and call_fn_name == "compute_log_prob"
        ):
            try:
                import torch.distributed as _dist

                _rank = _dist.get_rank() if _dist.is_initialized() else 0
            except Exception:
                _rank = 0

            if _rank == 0:
                # ── GL 系数 ──────────────────────────────────────────────────
                _gauss_coefs = None
                _gauss_xi = None
                if self._noisy_backend == "gaussian":
                    _weights = torch.tensor(self._gauss_weights, dtype=torch.float32)
                    _mc_per_point = mc_num // self._gauss_N
                    _gauss_coefs = (_weights * self._gauss_N).repeat_interleave(
                        _mc_per_point
                    )
                    # 原始积分点（[0,1] 范围），每条 MC 副本对应一个 xi
                    _gauss_xi = (
                        torch.tensor(
                            self._gdpo_xi, dtype=torch.float32
                        ).repeat_interleave(_mc_per_point)
                        if hasattr(self, "_gdpo_xi")
                        else None
                    )

                _rank0_sample_offset = self._debug_mb_counter * batch_size

                # ── 算法正确性诊断量（纯统计，无额外前向）────────────────────
                _diag: dict = {}
                with torch.no_grad():
                    # 所有诊断量还原到 3D 视图 (bs, mc, seq_len)
                    _tl3 = token_lp_all.reshape(batch_size, mc_num, seq_length)
                    _pm3 = pmask_all.reshape(batch_size, mc_num, seq_length)
                    _tlw3 = token_lp_weighted.reshape(batch_size, mc_num, seq_length)

                    # response 区域 mask（全注意力路径：response 在尾部 response_length 列）
                    # shift_logits=True 时 logit[j] 预测 token[j+1]，有效预测区域向左偏移 1 位：
                    #   [-(r_len+1):-1] 而非 [-r_len:]。
                    _r_len = response_length
                    _resp_attn = attention_mask[:, -_r_len:].float()  # (bs, resp_len)
                    _r_valid_f = _resp_attn.sum(-1).clamp(min=1)  # (bs,)

                    if self._shift_logits:
                        _tl_resp = _tl3[:, :, -(_r_len + 1) : -1]  # (bs, mc, resp_len)
                        _pm_resp = _pm3[:, :, -(_r_len + 1) : -1]
                        _tlw_resp = _tlw3[:, :, -(_r_len + 1) : -1]
                    else:
                        _tl_resp = _tl3[:, :, -_r_len:]  # (bs, mc, resp_len)
                        _pm_resp = _pm3[:, :, -_r_len:]
                        _tlw_resp = _tlw3[:, :, -_r_len:]

                    # ── 诊断 1: 实际 mask 率 vs 期望 GL 积分点 xi_j ──────────
                    # masked 位置：token_lp_all != 0（也等价于 mask_indices 在 response 区域内）
                    _is_masked = (_tl_resp != 0).float()  # (bs, mc, resp_len)
                    _actual_mask_rate = (_is_masked * _resp_attn.unsqueeze(1)).sum(
                        -1
                    ) / _r_valid_f.unsqueeze(
                        1
                    )  # (bs, mc)
                    _diag["actual_mask_rate_per_mc"] = _actual_mask_rate.cpu().float()
                    # 期望积分点（重复为 per-sample 方便对比）
                    if _gauss_xi is not None:
                        _diag["expected_mask_rate_per_mc"] = _gauss_xi.unsqueeze(
                            0
                        ).expand(batch_size, -1)
                        # 偏差：|actual - expected|，若 > 0.05 说明 mask 生成有问题
                        _diag["mask_rate_error"] = (
                            _actual_mask_rate.cpu() - _diag["expected_mask_rate_per_mc"]
                        ).abs()

                    # ── 诊断 2: per-MC ELBO（GL 加权前/后）──────────────────
                    # 加权前：token_lp_all / p_mask 之和（原始 ELBO 估计，未乘 GL 系数）
                    _pm_safe = _pm_resp.clamp(min=1e-6)
                    _elbo_raw = (
                        (_tl_resp / _pm_safe) * _is_masked * _resp_attn.unsqueeze(1)
                    ).sum(
                        -1
                    )  # (bs, mc)
                    _diag["elbo_per_mc_raw"] = _elbo_raw.cpu().float()
                    # 加权后（已乘 w_i * gauss_N）：等价于 token_lp_weighted 之和
                    _elbo_gl = (_tlw_resp * _resp_attn.unsqueeze(1)).sum(-1)  # (bs, mc)
                    _diag["elbo_per_mc_gl_weighted"] = _elbo_gl.cpu().float()
                    # 最终 GL 估计 = GL 加权 ELBO 的均值
                    _diag["elbo_gl_estimate"] = _elbo_gl.mean(-1).cpu().float()  # (bs,)
                    # per-MC 方差（越小说明 GL 积分越稳定）
                    _diag["elbo_mc_variance"] = _elbo_gl.var(-1).cpu().float()  # (bs,)

                    # ── 诊断 3: p_mask 在 masked 位置的均值（gaussian 应 ≈ t_j = xi_j）──
                    _masked_count = _is_masked.sum(-1).clamp(min=1)  # (bs, mc)
                    _pmask_at_masked_mean = (_pm_resp * _is_masked).sum(
                        -1
                    ) / _masked_count  # (bs, mc)
                    _diag["pmask_mean_at_masked_per_mc"] = (
                        _pmask_at_masked_mean.cpu().float()
                    )

                    # ── 诊断 4: token_lp_all 在非 masked 位置的泄漏（应 = 0）──────
                    _non_masked = (1.0 - _is_masked) * _resp_attn.unsqueeze(1)
                    _leakage_max = (_tl_resp.abs() * _non_masked).max()
                    _diag["token_lp_leakage_max"] = float(_leakage_max.item())

                    # ── 诊断 5: sequence-level log prob 内部一致性（broadcast 验证）──
                    # sequence-level 下 log_prob[i,:] 应为常数，std 应 ≈ 0
                    if self.log_prob_level == "sequence-level":
                        _lp_resp = log_prob[:, -_r_len:] * _resp_attn  # (bs, resp_len)
                        _lp_std = _lp_resp.std(-1)  # (bs,)
                        _diag["seq_level_lp_within_resp_std"] = _lp_std.cpu().float()

                    # ── 诊断 6: 数值稳定性检查 ─────────────────────────────────
                    _diag["pmask_near_zero_frac"] = float(
                        (_pm_resp * _resp_attn.unsqueeze(1) < 1e-3)
                        .float()
                        .mean()
                        .item()
                    )
                    _diag["token_lp_all_has_nan"] = bool(
                        token_lp_all.isnan().any().item()
                    )
                    _diag["token_lp_all_has_inf"] = bool(
                        token_lp_all.isinf().any().item()
                    )
                    _diag["token_lp_all_min"] = float(
                        token_lp_all[token_lp_all != 0].min().item()
                        if (token_lp_all != 0).any()
                        else 0.0
                    )
                    _diag["token_lp_all_max"] = float(token_lp_all.max().item())
                    # advantages 在 compute_log_prob 阶段尚未计算，此字段不记录
                    # （如需检查，应在 save_post_advantage 的 trainer_diagnostics 中查看）

                self.debug_saver.save_actor_inner(
                    step=self.debug_step,
                    rank=_rank,
                    mb_idx=self._debug_mb_counter,
                    input_ids=(
                        micro_batch.get("input_ids")
                        if isinstance(micro_batch, dict)
                        else None
                    ),
                    token_lp_all=token_lp_all,
                    pmask_all=pmask_all,
                    token_lp_weighted=token_lp_weighted,
                    gauss_coefs=_gauss_coefs,
                    batch_size=batch_size,
                    mc_num=mc_num,
                    call_fn_name=call_fn_name,
                    rank0_sample_offset=_rank0_sample_offset,
                    diagnostics=_diag,
                )
            self._debug_mb_counter += 1

        return entropy, log_prob, loss_per_sample

    def _compute_block_masked_token_logprobs(
        self,
        perturbed_seq,
        mask_indices,
        p_mask,
        orig_seq,
        attention_mask,
        bmc,
        mc_num,
        seq_length,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log-probs under the Block-DLLM (block-causal attention) path.

        For each noisy sample, the model sees interleaved (noisy block, clean block) pairs
        and attends only within a block-causal structure.  Leading padding is removed before
        the forward pass so that block boundaries align with real token positions.
        The model returns sparse per-token log-probs directly (no full logit tensor).

        Args
        ----
        perturbed_seq  : (batch, mc_num, seq_len) int64  – noisy input token ids
        mask_indices   : (batch, mc_num, seq_len) bool   – True at positions that were masked
        p_mask         : (batch, mc_num, seq_len) float  – per-token mask probability used
                         as ELBO importance weight; returned as pmask_all aligned with
                         token_lp_all after any internal token reordering.
        orig_seq       : (batch, seq_len) int64          – clean ground-truth token ids
        attention_mask : (batch, seq_len) int64          – 1 for real tokens, 0 for padding
        bmc            : int  – batch * mc_num
        mc_num         : int  – number of MC samples per sequence
        seq_length     : int  – seq_len

        Returns
        -------
        token_lp_all : (batch*mc, seq_len) float – log-prob of the original token at each
                       masked position; 0 at unmasked / padding positions.
        pmask_all    : (batch*mc, seq_len) float – p_mask reshaped to match token_lp_all;
                       must be returned here (not recomputed by the caller) because internal
                       sequence compaction may reorder tokens, so pmask_all must be derived
                       after compaction to remain aligned with token_lp_all.
        """
        block_size = self.block_size
        assert seq_length % block_size == 0, (
            f"[DLLMActor] seq_length ({seq_length}) must be divisible by block_size ({block_size}). "
            "Pad your sequences to a multiple of block_size before calling the actor."
        )
        orig_attn = attention_mask
        (perturbed_seq, mask_indices, p_mask, orig_seq, attention_mask) = (
            _compact_sequences(
                perturbed_seq,
                mask_indices,
                p_mask,
                orig_seq,
                attention_mask,
                attention_mask=orig_attn,
                pad_values=[self.MASK_TOKEN_ID, False, 0.0, self.PAD_TOKEN_ID, 0],
                block_size=block_size,
            )
        )
        position_ids = (
            attention_mask.cumsum(dim=-1).sub(1).clamp(min=0)
        )  # (batch, seq_len)
        pos_all = (
            position_ids.unsqueeze(1).expand(-1, mc_num, -1).reshape(bmc, seq_length)
        )  # (batch*mc, seq_len)

        noisy_all, mask_all, orig_all, attn_all, pos_all, pmask_all, labels_all = (
            flatten_mc_inputs(
                perturbed_seq,
                mask_indices,
                orig_seq,
                attention_mask,
                p_mask,
                pos_all,
                bmc,
                seq_length,
            )
        )
        seq_d, attn_d, pos_d, keep_d, block_mask = prepare_block_inputs(
            noisy_all, orig_all, attn_all, pos_all, mask_all, block_size, seq_length,
            history_window_size=self._block_history_window_size,
        )

        token_lp_all = self.actor_module(
            input_ids=seq_d,
            attention_mask=block_mask,
            position_ids=pos_d,
            logits_to_keep=keep_d,
            orig_keep_mask=mask_all,
            orig_seq_len=seq_length,
            labels=labels_all,
            use_cache=False,
        ).logits  # (batch*mc, seq_len): sparse log-probs returned by model when labels are passed
        return token_lp_all, pmask_all

    def _compute_full_attn_masked_token_logprobs(
        self,
        perturbed_seq,
        mask_indices,
        p_mask,
        orig_seq,
        attention_mask,
        bmc,
        mc_num,
        seq_length,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute per-token log-probs under the full bidirectional attention path.

        The noisy sequence is forwarded through the model with full bidirectional attention.
        For next-token-prediction models (self._shift_logits=True, e.g. Dream), labels are
        shifted right by one so that logit[i] is scored against token[i+1].

        Args
        ----
        perturbed_seq  : (batch, mc_num, seq_len) int64  – noisy input token ids
        mask_indices   : (batch, mc_num, seq_len) bool   – True at positions that were masked
        p_mask         : (batch, mc_num, seq_len) float  – per-token mask probability used
                         as ELBO importance weight; returned as pmask_all shape-aligned
                         with token_lp_all.
        orig_seq       : (batch, seq_len) int64          – clean ground-truth token ids
        attention_mask : (batch, seq_len) int64          – 1 for real tokens, 0 for padding
        bmc            : int  – batch * mc_num
        mc_num         : int  – number of MC samples per sequence
        seq_length     : int  – seq_len

        Returns
        -------
        token_lp_all : (batch*mc, seq_len) float – log-prob of the original token at each
                       masked position; 0 at unmasked / padding positions.
        pmask_all    : (batch*mc, seq_len) float – p_mask reshaped to match token_lp_all;
                       returned here to keep reshape logic in one place and guarantee
                       shape compatibility with token_lp_all.
        """
        attn_flat = (
            attention_mask.unsqueeze(1).expand(-1, mc_num, -1).reshape(bmc, seq_length)
        )
        pos_all = attn_flat.cumsum(dim=-1).sub(1).clamp(min=0)  # (batch*mc, seq_len)

        noisy_all, mask_all, orig_all, attn_all, pos_all, pmask_all, labels_all = (
            flatten_mc_inputs(
                perturbed_seq,
                mask_indices,
                orig_seq,
                attention_mask,
                p_mask,
                pos_all,
                bmc,
                seq_length,
            )
        )

        # Dream: next-token prediction model — logit[i] predicts token[i+1].
        # To align loss CE(logit[i], target[i]), we shift the *labels* left by 1:
        #   labels_all[j] = orig_all[j+1]  if mask_all[j+1]=True, else -100.
        if self._shift_logits:
            # torch.roll(shifts=-1) 将内容向左滚动一位（即 labels_all[j] ← labels_all[j+1]）。
            labels_all = torch.roll(labels_all, shifts=-1, dims=-1)
            labels_all[:, -1] = -100
            # Also shift pmask_all left by 1 to stay aligned with the shifted labels:
            # token_lp_all[j] now predicts position j+1, so the importance weight must be
            # p_mask[j+1] (the masking probability of the predicted position), NOT p_mask[j].
            # Without this shift, position j = prompt_len-1 would have token_lp_all[j] != 0
            # (predicting first response token) but pmask_all[j] = 0 (prompt never masked),
            # causing token_lp_weighted[j] = token_lp_all[j] / 1e-6 → huge negative value →
            # ELBO ≈ -inf → NaN gradient norm.
            pmask_all = torch.roll(pmask_all, shifts=-1, dims=-1)
            pmask_all[:, -1] = 0.0

        # Derive loss mask from non-(-100) labels so the shift is correctly accounted for:
        # - no-shift: valid positions are exactly mask_all (labels set to -100 at unmasked positions)
        # - shift_logits: valid positions are mask_all shifted left by 1 (captured by labels_all != -100)
        loss_token_mask = (labels_all != -100).float()  # (bmc, seq_len)

        # Pack into flash-attention varlen format: remove padding, pass cu_seqlens instead of attn mask.
        device = noisy_all.device
        valid_mask = attn_all.bool()  # (bmc, seq_len)
        seqlens = attn_all.sum(dim=-1).int()  # (bmc,)
        cu_seqlens = torch.zeros(bmc + 1, dtype=torch.int32, device=device)
        cu_seqlens[1:] = seqlens.cumsum(dim=0)
        max_seqlen = int(seqlens.max().item())

        # (1, total_valid_tokens): all valid tokens concatenated across the bmc sequences
        packed_input = noisy_all[valid_mask].unsqueeze(0)

        # Compute position_ids for packed format: each sequence gets [0, 1, ..., seqlen-1]
        position_ids_packed = []
        for seq_len in seqlens:
            position_ids_packed.append(
                torch.arange(seq_len, device=device, dtype=torch.long)
            )
        position_ids_packed = torch.cat(position_ids_packed).unsqueeze(
            0
        )  # (1, total_valid_tokens)

        # logits_to_keep: only compute logits at masked positions (sparse optimization)
        logits_to_keep_packed = (
            loss_token_mask[valid_mask].bool().unsqueeze(0)
        )  # (1, total_valid_tokens)

        # Use the sparse cross-entropy path: pass labels + orig_keep_mask so the model
        # computes log-probs directly in a 2D (batch, seq_len) format, avoiding the
        # materialisation of the full (B, L, V) logit tensor.
        # Both labels and orig_keep_mask must be in packed format to match packed_input.
        labels_packed = labels_all[valid_mask].unsqueeze(0)  # (1, total_valid_tokens)
        orig_keep_mask_packed = (
            loss_token_mask[valid_mask].bool().unsqueeze(0)
        )  # (1, total_valid_tokens)
        lm_output = self.actor_module(
            input_ids=packed_input,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            position_ids=position_ids_packed,
            logits_to_keep=logits_to_keep_packed,
            labels=labels_packed,
            orig_keep_mask=orig_keep_mask_packed,
            use_cache=False,
        )
        output_logits = (
            lm_output.logits
        )  # (batch, seq_len, vocab_size) or (batch, seq_len)

        if output_logits.dim() == 2:
            # Sparse path: model returned (1, total_valid_tokens) packed log-probs.
            # Scatter back to (bmc, seq_len) padded layout.
            token_lp_all = torch.zeros(
                bmc, seq_length, device=device, dtype=output_logits.dtype
            )
            token_lp_all[valid_mask] = output_logits[0]
        else:
            # Full logits path: scatter back and compute CE in actor (legacy fallback).
            logits_all = torch.zeros(
                bmc,
                seq_length,
                output_logits.size(-1),
                device=device,
                dtype=output_logits.dtype,
            )
            logits_all[valid_mask] = output_logits[0]

            token_lp_all = (
                -F.cross_entropy(
                    logits_all.view(-1, logits_all.size(-1)),
                    labels_all.clamp(min=0).view(-1),
                    reduction="none",
                ).view(bmc, seq_length)
                * loss_token_mask
            )  # (batch*mc, seq_len): 0 at unmasked positions
        return token_lp_all, pmask_all
