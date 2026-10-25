"""
debug_state_saver.py — DLLM 训练流水线中间状态调试保存工具

使用方式：
    在 ray_trainer_dllm_new.py 里每 N 步触发，分阶段保存：
        Stage 0: rollout batch（token id + 文本解码）
        Stage 1: build_noisy_samples 输出（perturbed_seq / mask_indices / p_mask）
        Stage 2: actor _forward_micro_batch 内部数据（token_lp_all / gauss 权重等）
        Stage 3: ref log probs
        Stage 4: reward（token 级 + episode 级）
        Stage 5: advantage / returns

    最终输出 samples_aligned.jsonl：每行对应一条轨迹样本，汇总所有阶段的标量指标。

目录结构：
    {root}/step_{step:05d}/
        00_rollout.pt
        01_noisy_samples.pt
        02_actor_inner/          ← 来自 actor rank-0，按 micro-batch 分文件
            mb000_rank0.pt
            ...
        03_ref_logprob.pt
        04_reward.pt
        05_advantage.pt
        samples_aligned.jsonl    ← 每行 = 一条 rollout 样本，指标一一对应
"""

import json
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch


# ─────────────────────────────────────────────────────────────────────────────
# 内部工具
# ─────────────────────────────────────────────────────────────────────────────

def _safe_item(x) -> Any:
    """将 tensor/ndarray 标量安全转为 Python 原生类型。"""
    if isinstance(x, torch.Tensor):
        return x.item() if x.numel() == 1 else x.cpu().tolist()
    if isinstance(x, np.ndarray):
        return x.item() if x.ndim == 0 else x.tolist()
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    return x


def _cpu_float(t: Optional[torch.Tensor]) -> Optional[torch.Tensor]:
    if t is None:
        return None
    return t.detach().cpu().float()


def _masked_stats(t: torch.Tensor, mask: torch.Tensor) -> Dict[str, float]:
    """给定 (resp_len,) tensor 和对应 bool mask，返回 mean/sum/min/max。"""
    valid = t[mask.bool()]
    if valid.numel() == 0:
        return {"mean": 0.0, "sum": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": float(valid.mean().item()),
        "sum":  float(valid.sum().item()),
        "min":  float(valid.min().item()),
        "max":  float(valid.max().item()),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 主类
# ─────────────────────────────────────────────────────────────────────────────

class DebugStateSaver:
    """
    DLLM 训练流水线分阶段状态保存器。

    每个保存步骤产生一个独立目录，目录内的文件可单独加载也可通过
    ``load_step(step)`` 一次性读回所有阶段数据。

    Parameters
    ----------
    root : str
        保存根目录。
    save_every : int
        每隔多少步保存一次（0 表示不保存）。
    max_noisy_decode : int
        在 samples_aligned.jsonl 中最多解码多少条 noisy 文本（per sample）；
        设为 0 关闭 noisy 文本解码（节省空间）。
    """

    def __init__(self, root: str, save_every: int = 10, max_noisy_decode: int = 2):
        self.root = Path(root)
        self.save_every = save_every
        self.max_noisy_decode = max_noisy_decode
        self.root.mkdir(parents=True, exist_ok=True)

    # ─── 公共 API ─────────────────────────────────────────────────────────────

    def should_save(self, step: int) -> bool:
        return self.save_every > 0 and (step % self.save_every == 0)

    def step_dir(self, step: int, create: bool = True) -> Path:
        d = self.root / f"step_{step:05d}"
        if create:
            d.mkdir(parents=True, exist_ok=True)
        return d

    # ─── Stage 0: rollout batch ───────────────────────────────────────────────

    def save_rollout(self, step: int, batch, tokenizer=None):
        """
        保存 rollout 输出的完整 batch。

        包含内容：
          - input_ids       (bs, seq_len)          prompt+response 完整 token id
          - responses       (bs, resp_len)          response 部分 token id
          - attention_mask  (bs, seq_len)
          - prompts         (bs, prompt_len)
          - rollout_log_probs (bs, resp_len)        若 rollout 输出了 log prob
          - _prompt_texts   List[str]               解码后的 prompt（含特殊 token）
          - _response_texts List[str]               解码后的 response（含特殊 token）
          - traj_uid / episode_rewards / episode_lengths / is_action_valid
        """
        d = self.step_dir(step)
        b, nb = batch.batch, batch.non_tensor_batch
        save_dict: Dict[str, Any] = {"_stage": "rollout", "_step": step}

        for k in ["input_ids", "responses", "attention_mask", "prompts",
                  "response_first_eos_pos", "response_eos_count"]:
            if k in b:
                save_dict[k] = b[k].cpu()
        if "rollout_log_probs" in b:
            save_dict["rollout_log_probs"] = b["rollout_log_probs"].cpu().float()

        for k in ["traj_uid", "episode_rewards", "episode_lengths",
                  "is_action_valid", "tool_callings"]:
            if k in nb:
                save_dict[k] = nb[k]

        # 解码文本（保留 EOS/PAD 等特殊 token 以便观察完整结构）
        if tokenizer is not None and "input_ids" in b and "responses" in b:
            resp_len = b["responses"].shape[-1]
            prompt_ids = b["input_ids"][:, :-resp_len]
            save_dict["_prompt_texts"] = tokenizer.batch_decode(
                prompt_ids.cpu(), skip_special_tokens=False
            )
            save_dict["_response_texts"] = tokenizer.batch_decode(
                b["responses"].cpu(), skip_special_tokens=False
            )

        torch.save(save_dict, d / "00_rollout.pt")

    # ─── Stage 1: noisy samples ───────────────────────────────────────────────

    def save_noisy_samples(self, step: int, batch, tokenizer=None):
        """
        保存 forward_process（build_noisy_samples）的输出。

        包含内容（01_noisy_samples.pt）：
          - perturbed_seq       (bs, mc_num, seq_len)  加噪后的 token id
          - mask_indices        (bs, mc_num, seq_len)  bool，True = 被 mask 的位置
          - p_mask              (bs, mc_num, seq_len)  每个 token 被 mask 的概率
          - input_ids           (bs, seq_len)          原始 token id
          - attention_mask      (bs, seq_len)
          - decode_orders_nfe   (bs, seq_len)          trajectory 后端专用，NFE 解码顺序
          - _prompt_len         int                    prompt 长度（由 responses 形状推算）
          - _noisy_mc0_texts    List[str]              第 0 个 MC 副本的解码文本
          - _orig_response_texts List[str]             原始 response 解码文本

        额外输出（仅当 decode_orders_nfe 存在）：
          01_noisy_trajectory_detail.jsonl  每行 = 一条样本的完整 trajectory noisy 分析
        """
        d = self.step_dir(step)
        b = batch.batch
        save_dict: Dict[str, Any] = {"_stage": "noisy_samples", "_step": step}

        for k in ["perturbed_seq", "mask_indices", "p_mask",
                  "input_ids", "attention_mask", "decode_orders_nfe"]:
            if k in b:
                save_dict[k] = b[k].cpu()

        # 计算 prompt_len（从 input_ids 和 responses 的长度差推算）
        if "input_ids" in b and "responses" in b:
            _prompt_len = b["input_ids"].shape[1] - b["responses"].shape[1]
            save_dict["_prompt_len"] = _prompt_len
        else:
            _prompt_len = None

        if tokenizer is not None:
            if "perturbed_seq" in b:
                ps = b["perturbed_seq"].cpu()           # (bs, mc_num, seq_len)
                save_dict["_noisy_mc0_texts"] = tokenizer.batch_decode(
                    ps[:, 0, :], skip_special_tokens=False
                )
            if "responses" in b:
                save_dict["_orig_response_texts"] = tokenizer.batch_decode(
                    b["responses"].cpu(), skip_special_tokens=False
                )

        torch.save(save_dict, d / "01_noisy_samples.pt")

        # ── 详细 trajectory 分析（仅当 decode_orders_nfe 存在时生成） ────────────
        if "decode_orders_nfe" in b and "perturbed_seq" in b and "mask_indices" in b:
            self._save_trajectory_detail(
                step=step,
                d=d,
                b=b,
                prompt_len=_prompt_len,
                tokenizer=tokenizer,
            )

    def _save_trajectory_detail(self, step, d, b, prompt_len, tokenizer):
        """
        生成 01_noisy_trajectory_detail.jsonl。

        每行 JSON 对应 batch 里的一条样本，包含：
          sample_idx       : int
          prompt_len       : int
          seq_len          : int
          T                : int          max NFE 步数
          n_unresolved     : int          decode_orders==0 但在 response 区域的位置数
          prompt_text      : str
          orig_response_text : str
          decode_orders_resp : List[int]  response 区域每个位置的 NFE 步（长度=response_len）
          mc_copies        : List[dict]   每个 MC 副本的详细信息，含：
            mc_idx         : int
            t_inferred     : int          从 mask_indices 反推的采样 t
            n_masked       : int          被 mask 的 response 位置数
            n_loss_tokens  : int          参与 loss 的位置（mask_indices=True）
            mask_ratio     : float        n_masked / response_len
            noisy_resp_text : str         noisy response 解码文本
            violations     : int          do>t_inferred 但未被 mask 的位置数（一致性检查）
            token_table    : List[dict]   response 区每个位置，含：
              pos          : int          全局序列位置
              resp_pos     : int          response 内偏移（0-based）
              token_id     : int
              token_label  : str          "EOS"/"MSK"/"MASK_orig"/<id> / 解码单字符
              decode_order : int
              is_masked    : bool
              p_mask       : float
              in_loss      : bool         是否参与 loss（mask_indices=True）
        """
        do_nfe   = b["decode_orders_nfe"].cpu()       # (bs, seq_len)
        ps       = b["perturbed_seq"].cpu()           # (bs, mc_num, seq_len)
        mi       = b["mask_indices"].cpu()            # (bs, mc_num, seq_len)
        pm_all   = b["p_mask"].cpu()                  # (bs, mc_num, seq_len)
        attn     = b["attention_mask"].cpu()          # (bs, seq_len)
        orig_ids = b["input_ids"].cpu()               # (bs, seq_len)

        bs, mc_num, seq_len = ps.shape
        pl = prompt_len if prompt_len is not None else (
            orig_ids.shape[1] - b["responses"].shape[1] if "responses" in b else 0
        )

        MASK_TOKEN_ID = 126336
        EOS_TOKEN_ID  = tokenizer.eos_token_id if tokenizer is not None else 126081

        def _label(tid):
            if tid == EOS_TOKEN_ID:  return "EOS"
            if tid == MASK_TOKEN_ID: return "MSK"
            if tokenizer is not None:
                try:
                    s = tokenizer.decode([tid], skip_special_tokens=False)
                    s = s.strip()
                    return s[:8] if s else f"#{tid}"
                except Exception:
                    pass
            return f"#{tid}"

        out_path = d / "01_noisy_trajectory_detail.jsonl"
        with out_path.open("w", encoding="utf-8") as fout:
            for si in range(bs):
                do_i   = do_nfe[si]           # (seq_len,)
                attn_i = attn[si]             # (seq_len,)
                ids_i  = orig_ids[si]         # (seq_len,)

                # response 区域：attention_mask=1 且 pos >= prompt_len
                resp_mask = attn_i.bool().clone()
                resp_mask[:pl] = False
                resp_pos_global = resp_mask.nonzero(as_tuple=False).squeeze(1)  # 全局下标
                resp_len = resp_pos_global.numel()

                T = int(do_i.max().item())
                # unresolved: response 区域 decode_order==0
                n_unresolved = int((do_i[resp_pos_global] == 0).sum().item())

                # 文本解码
                if tokenizer is not None:
                    try:
                        prompt_text = tokenizer.decode(
                            ids_i[:pl].tolist(), skip_special_tokens=False)
                    except Exception:
                        prompt_text = ""
                    try:
                        resp_ids = ids_i[resp_pos_global]
                        orig_resp_text = tokenizer.decode(
                            resp_ids.tolist(), skip_special_tokens=False)
                    except Exception:
                        orig_resp_text = ""
                else:
                    prompt_text = ""
                    orig_resp_text = ""

                # decode_orders for response region only
                do_resp = do_i[resp_pos_global].tolist()

                mc_records = []
                for mc in range(mc_num):
                    noisy_i  = ps[si, mc]       # (seq_len,)
                    mi_i     = mi[si, mc]       # (seq_len,) bool
                    pm_i     = pm_all[si, mc]   # (seq_len,) float

                    # infer t: max decode_order among GENUINELY revealed positions.
                    # "Genuinely revealed" = mask_indices=False AND noisy token != MASK_TOKEN_ID.
                    # This correctly handles block_trajectory where future-block positions have
                    # mask_indices=False but noisy_tok=MASK (not in loss), which would otherwise
                    # inflate t_inf and produce false-positive ok=False violations.
                    genuine_mask = (
                        (~mi_i[resp_pos_global].bool()) &
                        (noisy_i[resp_pos_global] != MASK_TOKEN_ID)
                    )
                    revealed_do = do_i[resp_pos_global][genuine_mask]
                    revealed_do = revealed_do[revealed_do > 0]
                    t_inf = int(revealed_do.max().item()) if revealed_do.numel() > 0 else 0

                    n_masked    = int(mi_i[resp_pos_global].sum().item())
                    n_loss      = n_masked  # mask_indices=True ↔ contributes to loss
                    mask_ratio  = n_masked / max(resp_len, 1)

                    # noisy response text
                    if tokenizer is not None:
                        try:
                            noisy_resp_text = tokenizer.decode(
                                noisy_i[resp_pos_global].tolist(), skip_special_tokens=False)
                        except Exception:
                            noisy_resp_text = ""
                    else:
                        noisy_resp_text = ""

                    # consistency violations: do > t_inf but not masked (genuine reveals only)
                    violations = 0
                    token_table = []
                    for ri, gp in enumerate(resp_pos_global.tolist()):
                        tid   = int(ids_i[gp].item())
                        ntid  = int(noisy_i[gp].item())
                        d_ord = int(do_i[gp].item())
                        masked = bool(mi_i[gp].item())
                        pm_val = round(float(pm_i[gp].item()), 4)
                        is_future_block = (not masked) and (ntid == MASK_TOKEN_ID)
                        # expected_masked: should this position carry loss?
                        # future-block positions are never expected to carry loss.
                        expected_masked = (not is_future_block) and (d_ord > t_inf or d_ord == 0)
                        if expected_masked and not masked:
                            violations += 1
                        token_table.append({
                            "pos":            gp,
                            "resp_pos":       ri,
                            "token_id":       tid,
                            "noisy_tok_id":   ntid,
                            "token_label":    _label(tid),
                            "noisy_label":    _label(ntid),
                            "decode_order":   d_ord,
                            "is_masked":      masked,
                            "is_future_block": is_future_block,
                            "p_mask":         pm_val,
                            "in_loss":        masked,
                            "expect_mask":    expected_masked,
                            "ok":             (masked == expected_masked),
                        })

                    mc_records.append({
                        "mc_idx":         mc,
                        "t_inferred":     t_inf,
                        "n_masked":       n_masked,
                        "n_loss_tokens":  n_loss,
                        "mask_ratio":     round(mask_ratio, 4),
                        "violations":     violations,
                        "noisy_resp_text": noisy_resp_text,
                        "token_table":    token_table,
                    })

                record = {
                    "sample_idx":         si,
                    "prompt_len":         pl,
                    "seq_len":            seq_len,
                    "resp_len":           resp_len,
                    "T":                  T,
                    "n_unresolved":       n_unresolved,
                    "prompt_text":        prompt_text,
                    "orig_response_text": orig_resp_text,
                    "decode_orders_resp": do_resp,
                    "mc_copies":          mc_records,
                }
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    # ─── Stage 2: actor-internal (from worker rank-0) ─────────────────────────

    def save_actor_inner(
        self,
        step: int,
        rank: int,
        mb_idx: int,
        *,
        input_ids: Optional[torch.Tensor] = None,
        token_lp_all: torch.Tensor,        # (bmc, seq_len)  raw log-prob at masked pos
        pmask_all: torch.Tensor,            # (bmc, seq_len)  p_mask 重排后
        token_lp_weighted: torch.Tensor,   # (bmc, seq_len)  importance-weighted
        gauss_coefs: Optional[torch.Tensor] = None,   # (mc_num,) GL 系数
        batch_size: Optional[int] = None,
        mc_num: Optional[int] = None,
        call_fn_name: str = "",
        rank0_sample_offset: int = 0,
        diagnostics: Optional[Dict[str, Any]] = None,
    ):
        """
        从 actor worker 的 rank-0 进程中调用，保存 _forward_micro_batch 内部数据。

        对应关系说明
        ─────────────
        trainer-side batch（Stages 0/1/3/4/5）的 sample_idx = 0..bs-1 与此处数据的对应：

          全局 sample_idx = rank0_sample_offset + mb_local_idx
                          （仅对 rank-0 处理的那 bs/n_gpus 个样本有效）

          token_lp_all 的展平规则（flatten_mc_inputs 产生）：
            token_lp_all[i * mc_num + j] ↔ micro-batch 内第 i 条样本的第 j 个 MC 副本
            对应 trainer-side: sample_idx = rank0_sample_offset + i

          所以完整的 3D 视图：
            token_lp_all.reshape(batch_size, mc_num, seq_len)[i, j, :]
              ↔ trainer-side sample (rank0_sample_offset + i), MC copy j

        保存内容（基础）：
          - token_lp_all       (bs*mc, seq_len)  masked 位置的 raw log-prob，其余位置为 0
          - pmask_all          (bs*mc, seq_len)  每个 token 的 mask 概率（重排后）
          - token_lp_weighted  (bs*mc, seq_len)  token_lp_all / pmask（importance-weight 修正）
          - gauss_coefs        (mc_num,)          GL 积分系数 w_i * gauss_N
          - input_ids          (bs, seq_len)      clean token id（可与 trainer-side 比对验证）
          - rank0_sample_offset int              本 micro-batch 首个样本在 rank-0 数据中的偏移
          - batch_size, mc_num                   用于 reshape 还原

        保存内容（算法正确性诊断，由 diagnostics 字典传入）：
          - actual_mask_rate_per_mc    (bs, mc)   每条样本每个 MC 副本的实际 mask 率
          - expected_mask_rate_per_mc  (bs, mc)   对应 GL 积分点 xi_j（gaussian backend）
          - mask_rate_error            (bs, mc)   |actual - expected|，应 < 0.05
          - elbo_per_mc_raw            (bs, mc)   GL 加权前的 ELBO 估计（sum token_lp/p_mask）
          - elbo_per_mc_gl_weighted    (bs, mc)   GL 加权后的 ELBO（已乘 w_i * gauss_N）
          - elbo_gl_estimate           (bs,)      最终 GL 积分估计（= old_log_probs[i,0]）
          - elbo_mc_variance           (bs,)      各 MC 副本 ELBO 的方差（越小越稳定）
          - pmask_mean_at_masked_per_mc (bs, mc)  masked 位置 p_mask 均值（gaussian 应 ≈ xi_j）
          - token_lp_leakage_max       float      非 masked 位置 log-prob 绝对值最大值（应 = 0）
          - seq_level_lp_within_resp_std (bs,)    sequence-level 下 response 内 lp 的 std（应 ≈ 0）
          - pmask_near_zero_frac       float      p_mask < 1e-3 的 token 比例（过大 → 除法不稳定）
          - token_lp_all_has_nan/inf   bool       数值健康性
          - advantages_has_nan         bool       advantage 数值健康性

        注：每个 micro-batch 单独一个文件，调用方用 reconstruct_actor_inner() 拼接对齐。
        """
        if rank != 0:
            return
        inner_dir = self.step_dir(step) / "02_actor_inner"
        inner_dir.mkdir(parents=True, exist_ok=True)

        save_dict: Dict[str, Any] = {
            "_stage": "actor_inner",
            "_step": step,
            "_mb_idx": mb_idx,
            "_call_fn": call_fn_name,
            "batch_size": batch_size,
            "mc_num": mc_num,
            # rank-0 内的样本起始偏移（相对于 trainer-side batch 中 rank-0 的部分）
            # full_sample_idx = rank0_sample_offset + micro_batch_local_idx
            "rank0_sample_offset": rank0_sample_offset,
            # (bmc, seq_len): raw log-prob at masked positions; 0 elsewhere
            # reshape(batch_size, mc_num, seq_len)[i, j] = sample (rank0_sample_offset+i), MC j
            "token_lp_all": _cpu_float(token_lp_all),
            # (bmc, seq_len): mask probability used as ELBO importance weight
            "pmask_all": _cpu_float(pmask_all),
            # (bmc, seq_len): importance-weight corrected: token_lp_all / pmask
            "token_lp_weighted": _cpu_float(token_lp_weighted),
            # (mc_num,): Gauss-Legendre quadrature coefficients w_i * gauss_N
            "gauss_coefs": _cpu_float(gauss_coefs),
            # (bs, seq_len): clean token ids (用于与 trainer-side input_ids 比对验证对应关系)
            "input_ids": input_ids.cpu() if input_ids is not None else None,
        }
        # 算法正确性诊断量（tensor 原样存储，标量直接存）
        if diagnostics:
            for k, v in diagnostics.items():
                save_dict[f"diag_{k}"] = _cpu_float(v) if isinstance(v, torch.Tensor) else v
        torch.save(save_dict, inner_dir / f"mb{mb_idx:03d}_rank{rank}.pt")

    # ─── Stage 3: ref log probs ───────────────────────────────────────────────

    def save_ref_logprob(self, step: int, batch):
        """
        保存 reference model 的 log prob。

          - ref_log_probs  (bs, resp_len)
        """
        d = self.step_dir(step)
        save_dict: Dict[str, Any] = {"_stage": "ref_logprob", "_step": step}
        if "ref_log_probs" in batch.batch:
            save_dict["ref_log_probs"] = batch.batch["ref_log_probs"].cpu().float()
        torch.save(save_dict, d / "03_ref_logprob.pt")

    # ─── Stage 4: reward ──────────────────────────────────────────────────────

    def save_reward(self, step: int, batch):
        """
        保存 reward 相关数据。

          - token_level_scores   (bs, resp_len)  原始 reward（未加 KL penalty）
          - token_level_rewards  (bs, resp_len)  加完 KL/invalid penalty 后的 reward
          - step_rewards         (bs, max_steps) 若存在（GiGPO）
          - episode_rewards      (bs,)           episode 总 reward
          - is_action_valid      (bs, max_steps) 每步动作是否合法
        """
        d = self.step_dir(step)
        b, nb = batch.batch, batch.non_tensor_batch
        save_dict: Dict[str, Any] = {"_stage": "reward", "_step": step}

        for k in ["token_level_scores", "token_level_rewards", "step_rewards"]:
            if k in b:
                save_dict[k] = b[k].cpu().float()
        for k in ["episode_rewards", "is_action_valid", "episode_lengths"]:
            if k in nb:
                save_dict[k] = nb[k]

        torch.save(save_dict, d / "04_reward.pt")

    # ─── Stage 5: advantage ───────────────────────────────────────────────────

    def save_advantage(self, step: int, batch):
        """
        保存 advantage 和 returns。

          - advantages  (bs, resp_len)
          - returns     (bs, resp_len)
          - values      (bs, resp_len)  若使用 critic
        """
        d = self.step_dir(step)
        save_dict: Dict[str, Any] = {"_stage": "advantage", "_step": step}
        for k in ["advantages", "returns", "values"]:
            if k in batch.batch:
                save_dict[k] = batch.batch[k].cpu().float()
        torch.save(save_dict, d / "05_advantage.pt")

    # ─── 对齐汇总：samples_aligned.jsonl ─────────────────────────────────────

    def write_aligned_jsonl(self, step: int, batch, tokenizer=None,
                            trainer_diagnostics: Optional[Dict] = None):
        """
        将所有阶段的指标按样本对齐，写出 JSONL，每行对应一条轨迹。

        字段说明（所有字段均可选，存在时才写）：
          sample_idx       : 在当前 batch 中的位置
          traj_uid         : 轨迹唯一 ID
          prompt_text      : prompt 解码文本（含特殊 token）
          response_text    : response 解码文本（含特殊 token）
          response_len_valid: 有效（非 padding）response token 数
          eos_pos          : response 中第一个 EOS 的位置
          eos_count        : response 中 EOS 出现次数

          mask_count_per_mc   : list[int]  每个 MC 副本中 masked token 数
          p_mask_sum_per_mc   : list[float] 每个 MC 副本的 p_mask 总和（= masked token 数 × t）

          old_log_prob_sum / mean : actor 重算 log prob（sequence-level 即对 response 求和/均值）
          ref_log_prob_sum / mean : ref model log prob

          kl_per_token     : (old - ref) 的 per-token 均值（KL 估计）

          token_score_sum  : reward（加 penalty 前）在 response 上的总和
          token_reward_sum : reward（加 penalty 后）在 response 上的总和
          episode_reward   : episode 总 reward
          episode_length   : episode 步数
          valid_action_ratio: 有效动作比例

          advantage_mean/max/min/std : advantage 统计
          return_mean                : returns 均值
        """
        d = self.step_dir(step)
        b, nb = batch.batch, batch.non_tensor_batch
        bs = b["input_ids"].shape[0] if "input_ids" in b else 0
        if bs == 0:
            return

        resp_len = b["responses"].shape[-1] if "responses" in b else 0

        rows: List[Dict] = []
        for i in range(bs):
            row: Dict[str, Any] = {"sample_idx": i}

            # ── 标识符 ──────────────────────────────────────────────────────
            if "traj_uid" in nb:
                row["traj_uid"] = str(nb["traj_uid"][i])

            # ── 解码文本 ─────────────────────────────────────────────────────
            if tokenizer is not None and resp_len > 0 and "input_ids" in b:
                prompt_ids = b["input_ids"][i, :-resp_len].cpu()
                row["prompt_text"] = tokenizer.decode(prompt_ids, skip_special_tokens=False)
                row["response_text"] = tokenizer.decode(
                    b["responses"][i].cpu(), skip_special_tokens=False
                )

            # ── 有效 response 长度 & EOS 信息 ────────────────────────────────
            if "attention_mask" in b and resp_len > 0:
                row["response_len_valid"] = int(
                    b["attention_mask"][i, -resp_len:].sum().item()
                )
            if "response_first_eos_pos" in b:
                row["eos_pos"] = int(b["response_first_eos_pos"][i].item())
            if "response_eos_count" in b:
                row["eos_count"] = int(b["response_eos_count"][i].item())

            # ── Noisy sample 统计（每个 MC 副本）────────────────────────────
            if "mask_indices" in b and resp_len > 0:
                mi = b["mask_indices"][i].cpu()        # (mc_num, seq_len)
                # 只统计 response 部分（后 resp_len 列）
                mi_resp = mi[:, -resp_len:]
                row["mask_count_per_mc"] = mi_resp.sum(dim=-1).tolist()
            if "p_mask" in b and resp_len > 0:
                pm = b["p_mask"][i].cpu()              # (mc_num, seq_len)
                pm_resp = pm[:, -resp_len:]
                row["p_mask_sum_per_mc"] = pm_resp.sum(dim=-1).tolist()

            # ── response mask（用于后续统计）────────────────────────────────
            if "attention_mask" in b and resp_len > 0:
                rmask = b["attention_mask"][i, -resp_len:].bool().cpu()
            else:
                rmask = torch.ones(resp_len, dtype=torch.bool)

            # ── Actor log prob ────────────────────────────────────────────────
            if "old_log_probs" in b and resp_len > 0:
                lp = b["old_log_probs"][i].cpu()
                stats = _masked_stats(lp, rmask)
                row["old_log_prob_sum"] = stats["sum"]
                row["old_log_prob_mean"] = stats["mean"]

            # ── Ref log prob ──────────────────────────────────────────────────
            if "ref_log_probs" in b and resp_len > 0:
                rlp = b["ref_log_probs"][i].cpu()
                stats = _masked_stats(rlp, rmask)
                row["ref_log_prob_sum"] = stats["sum"]
                row["ref_log_prob_mean"] = stats["mean"]

            # ── Per-token KL 估计（actor vs ref）────────────────────────────
            if "old_log_probs" in b and "ref_log_probs" in b and resp_len > 0:
                kl = (b["old_log_probs"][i] - b["ref_log_probs"][i]).cpu()
                row["kl_per_token_mean"] = float(kl[rmask].mean().item())

            # ── Reward ────────────────────────────────────────────────────────
            if "token_level_scores" in b and resp_len > 0:
                row["token_score_sum"] = float(
                    b["token_level_scores"][i].cpu().sum().item()
                )
            if "token_level_rewards" in b and resp_len > 0:
                row["token_reward_sum"] = float(
                    b["token_level_rewards"][i].cpu().sum().item()
                )
            if "episode_rewards" in nb:
                row["episode_reward"] = float(nb["episode_rewards"][i])
            if "episode_lengths" in nb:
                row["episode_length"] = int(nb["episode_lengths"][i])
            if "is_action_valid" in nb:
                valid_arr = nb["is_action_valid"][i]
                row["valid_action_ratio"] = float(np.mean(valid_arr.astype(float)))
                row["is_action_valid"] = valid_arr.tolist()

            # ── Advantage ────────────────────────────────────────────────────
            if "advantages" in b and resp_len > 0:
                adv = b["advantages"][i].cpu()
                stats = _masked_stats(adv, rmask)
                row["advantage_mean"] = stats["mean"]
                row["advantage_max"] = stats["max"]
                row["advantage_min"] = stats["min"]
                row["advantage_sum"] = stats["sum"]
                valid_adv = adv[rmask]
                row["advantage_std"] = float(valid_adv.std().item()) if valid_adv.numel() > 1 else 0.0

            if "returns" in b and resp_len > 0:
                ret = b["returns"][i].cpu()
                row["return_mean"] = float(ret[rmask].mean().item() if rmask.any() else 0.0)

            # ── 算法正确性诊断（trainer-side，batch 级标量直接写入每行）────────
            if trainer_diagnostics:
                # 组内 reward std（此样本所在 group 的 std）
                _g_std = trainer_diagnostics.get("group_reward_within_std")
                if _g_std is not None and len(_g_std) > 0:
                    _group_idx = i // max(1, bs // len(_g_std))
                    if _group_idx < len(_g_std):
                        row["group_reward_within_std"] = float(_safe_item(_g_std[_group_idx]))
                # batch 级标量诊断（直接 copy 到每行）
                for _k in [
                    "nonzero_adv_group_ratio", "within_group_std_mean",
                    "ratio_mean", "ratio_std", "ratio_min", "ratio_max",
                    "ratio_clipfrac_pos", "ratio_clipfrac_neg",
                    "lp_delta_mean", "lp_delta_std", "lp_delta_abs_mean", "lp_delta_has_nan",
                    "adv_mean", "adv_std", "adv_abs_mean", "adv_has_nan",
                ]:
                    if _k in trainer_diagnostics:
                        row[f"diag_{_k}"] = _safe_item(trainer_diagnostics[_k])

            rows.append(row)

        with open(d / "samples_aligned.jsonl", "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    # ─── 便捷：一次性完成 stage4+5+jsonl ─────────────────────────────────────

    def save_trainer_diagnostics(self, step: int, diag: dict):
        """
        保存 trainer-side 算法正确性诊断量（_compute_trainer_diagnostics 的输出）。

        保存为 06_trainer_diagnostics.pt，内容：
          group_reward_within_std/mean  (n_groups,)  GRPO 组内 reward 分布
          nonzero_adv_group_ratio       float        有有效梯度信号的组比例
          within_group_std_mean         float        组内 std 均值
          ratio_mean/std/min/max        float        PPO ratio = exp(old_lp - rollout_lp) 统计
          ratio_clipfrac_pos/neg        float        正/负 advantage 方向的 clip 触发率
          lp_delta_mean/std/abs_mean    float        per-token (old - ref) KL 信号强度
          adv_mean/std/abs_mean         float        advantage 分布统计
          adv_has_nan / lp_delta_has_nan bool        数值健康性
        """
        d = self.step_dir(step)
        save_dict: Dict[str, Any] = {"_stage": "trainer_diagnostics", "_step": step}
        for k, v in diag.items():
            save_dict[k] = _cpu_float(v) if isinstance(v, torch.Tensor) else v
        torch.save(save_dict, d / "06_trainer_diagnostics.pt")

    def save_post_advantage(self, step: int, batch, tokenizer=None,
                            trainer_diagnostics: Optional[Dict] = None):
        """在 compute_advantage 完成后调用，一次性保存 stage 4、5、jsonl 及诊断量。"""
        self.save_reward(step, batch)
        self.save_advantage(step, batch)
        if trainer_diagnostics:
            self.save_trainer_diagnostics(step, trainer_diagnostics)
        self.write_aligned_jsonl(step, batch, tokenizer=tokenizer,
                                 trainer_diagnostics=trainer_diagnostics)

    # ─── 对齐工具：actor inner → trainer-side sample_idx ────────────────────

    def reconstruct_actor_inner(
        self,
        step: int,
        trainer_input_ids: Optional[torch.Tensor] = None,
        verify: bool = True,
    ) -> Dict[str, Any]:
        """
        将多个 micro-batch 文件拼接，还原为与 trainer-side batch 对齐的 3D 格式。

        对应关系图
        ──────────
        trainer-side sample_idx = 0 .. bs-1
                                          ↕ DP 分卡：rank-0 拿到前 rank0_bs 条
        actor_inner 覆盖 sample_idx = rank0_sample_offset .. rank0_sample_offset + rank0_bs - 1
                                          ↕ micro-batch 拆分 + flatten_mc_inputs 展平
        mb_file["token_lp_all"][i*mc_num + j] ↔ 该 mb 第 i 条样本的第 j 个 MC 副本

        经过本函数还原后：
          token_lp_all_3d[k, j, :] ↔ trainer-side sample_indices[k]，MC 副本 j

        参数
        ────
        trainer_input_ids : (bs, seq_len) tensor，来自 00_rollout.pt["input_ids"]；
                            传入时会与 actor_inner 保存的 input_ids 比对以验证对应关系。
        verify            : 是否开启比对断言（建议首次使用时开启）。

        返回
        ────
        token_lp_all_3d      (rank0_bs, mc_num, seq_len)   raw masked log-prob（3D 视图）
        pmask_all_3d         (rank0_bs, mc_num, seq_len)   mask 概率（3D 视图）
        token_lp_weighted_3d (rank0_bs, mc_num, seq_len)   importance-weighted（3D 视图）
        token_lp_mc_mean     (rank0_bs, seq_len)            MC 维度均值（近似 old_log_probs）
        gauss_coefs          (mc_num,) or None              GL 积分系数
        sample_indices       LongTensor (rank0_bs,)         对应的 trainer-side sample_idx
        rank0_bs, mc_num, verified
        """
        inner_dir = self.step_dir(step, create=False) / "02_actor_inner"
        if not inner_dir.exists():
            raise FileNotFoundError(f"actor_inner 目录不存在: {inner_dir}")

        mb_files = sorted(inner_dir.glob("mb*_rank0.pt"))
        if not mb_files:
            raise FileNotFoundError(f"未找到 rank-0 micro-batch 文件: {inner_dir}")

        mbs = [torch.load(p, map_location="cpu", weights_only=False) for p in mb_files]
        mbs.sort(key=lambda x: x["_mb_idx"])

        mc_num = mbs[0]["mc_num"]
        gauss_coefs = mbs[0]["gauss_coefs"]

        def _cat_reshape(key: str) -> torch.Tensor:
            """拼接各 micro-batch 的 (mbs*mc, seq_len) 并 reshape 为 (rank0_bs, mc_num, seq_len)。"""
            parts = []
            for mb in mbs:
                t = mb[key]                           # (mbs*mc_num, seq_len)
                t3d = t.reshape(mb["batch_size"], mc_num, t.shape[-1])
                parts.append(t3d)
            return torch.cat(parts, dim=0)            # (rank0_bs, mc_num, seq_len)

        token_lp_all_3d      = _cat_reshape("token_lp_all")
        pmask_all_3d         = _cat_reshape("pmask_all")
        token_lp_weighted_3d = _cat_reshape("token_lp_weighted")
        rank0_bs = token_lp_all_3d.shape[0]

        token_lp_mc_mean = token_lp_weighted_3d.mean(dim=1)  # (rank0_bs, seq_len)

        # 构造对应的 trainer-side 样本索引
        first_offset = mbs[0]["rank0_sample_offset"]   # 应为 0（rank-0 从 batch 起始处开始）
        sample_indices = torch.arange(first_offset, first_offset + rank0_bs, dtype=torch.long)

        # 可选验证：比对 input_ids
        verified = False
        if trainer_input_ids is not None and verify:
            actor_ids_parts = [mb["input_ids"] for mb in mbs if mb.get("input_ids") is not None]
            if actor_ids_parts:
                actor_ids = torch.cat(actor_ids_parts, dim=0)         # (rank0_bs, seq_len)
                trainer_slice = trainer_input_ids[sample_indices]     # (rank0_bs, seq_len)
                match = bool((actor_ids == trainer_slice).all().item())
                if not match:
                    warnings.warn(
                        "[DebugStateSaver] actor_inner input_ids 与 trainer-side 不匹配！"
                        " DP 分卡顺序可能不是连续的，请改用 input_ids 逐行匹配。"
                    )
                verified = match
            else:
                warnings.warn("[DebugStateSaver] micro-batch 文件中无 input_ids，跳过验证。")

        # ── 拼接算法诊断量（(bs, mc) 形状）─────────────────────────────────
        diag_2d_keys = [
            "diag_actual_mask_rate_per_mc",
            "diag_expected_mask_rate_per_mc",
            "diag_mask_rate_error",
            "diag_elbo_per_mc_raw",
            "diag_elbo_per_mc_gl_weighted",
            "diag_pmask_mean_at_masked_per_mc",
        ]
        diag_1d_keys = [
            "diag_elbo_gl_estimate",
            "diag_elbo_mc_variance",
            "diag_seq_level_lp_within_resp_std",
        ]
        diag_scalar_keys = [
            "diag_token_lp_leakage_max",
            "diag_pmask_near_zero_frac",
            "diag_token_lp_all_has_nan",
            "diag_token_lp_all_has_inf",
            "diag_token_lp_all_min",
            "diag_token_lp_all_max",
            "diag_advantages_has_nan",
        ]

        diagnostics: Dict[str, Any] = {}
        # (bs, mc) 诊断量：直接 cat
        for key in diag_2d_keys:
            parts = [mb[key] for mb in mbs if key in mb and mb[key] is not None]
            if parts:
                diagnostics[key] = torch.cat(parts, dim=0)   # (rank0_bs, mc_num)
        # (bs,) 诊断量：直接 cat
        for key in diag_1d_keys:
            parts = [mb[key] for mb in mbs if key in mb and mb[key] is not None]
            if parts:
                diagnostics[key] = torch.cat(parts, dim=0)   # (rank0_bs,)
        # 标量：取第一个 micro-batch 的值（整个 micro-batch 级别的量）
        for key in diag_scalar_keys:
            for mb in mbs:
                if key in mb:
                    diagnostics[key] = mb[key]
                    break

        return {
            "token_lp_all_3d":      token_lp_all_3d,       # (rank0_bs, mc_num, seq_len)
            "pmask_all_3d":         pmask_all_3d,           # (rank0_bs, mc_num, seq_len)
            "token_lp_weighted_3d": token_lp_weighted_3d,  # (rank0_bs, mc_num, seq_len)
            "token_lp_mc_mean":     token_lp_mc_mean,      # (rank0_bs, seq_len)
            "gauss_coefs":          gauss_coefs,            # (mc_num,) or None
            "sample_indices":       sample_indices,         # trainer-side 对应索引
            "rank0_bs":             rank0_bs,
            "mc_num":               mc_num,
            "verified":             verified,
            **diagnostics,                                  # diag_* 字段
        }

    def align_sample(
        self,
        sample_idx: int,
        step: int,
        actor_inner: Optional[Dict] = None,
        trainer_data: Optional[Dict] = None,
        tokenizer=None,
    ) -> Dict[str, Any]:
        """
        提取单条样本在所有流水线阶段的完整数据，返回按阶段命名的字典。

        字段命名前缀：
          s0_*  Stage 0  rollout batch（token ids、文本）
          s1_*  Stage 1  noisy samples（perturbed_seq、mask_indices、p_mask）
          s2_*  Stage 2  actor inner（token_lp_all_3d、token_lp_weighted_3d 等）
          s3_*  Stage 3  ref log probs
          s4_*  Stage 4  reward（token-level + episode-level）
          s5_*  Stage 5  advantages / returns

        注：s2_* 字段仅在 sample_idx 落入 rank-0 覆盖范围时才存在；
        s2_available=False 时表示该样本由其他 rank 处理，内部数据未保存。
        """
        if trainer_data is None:
            trainer_data = self.load_step(step)
        if actor_inner is None:
            rollout = trainer_data.get("rollout", {})
            actor_inner = self.reconstruct_actor_inner(
                step, rollout.get("input_ids"), verify=True
            )

        out: Dict[str, Any] = {"sample_idx": sample_idx, "_step": step}

        # Stage 0: rollout
        rollout = trainer_data.get("rollout", {})
        for k in ["input_ids", "responses", "attention_mask"]:
            if k in rollout:
                out[f"s0_{k}"] = rollout[k][sample_idx]
        if "traj_uid" in rollout:
            out["traj_uid"] = str(rollout["traj_uid"][sample_idx])
        if "_prompt_texts" in rollout:
            out["prompt_text"] = rollout["_prompt_texts"][sample_idx]
        if "_response_texts" in rollout:
            out["response_text"] = rollout["_response_texts"][sample_idx]

        # Stage 1: noisy samples
        noisy = trainer_data.get("noisy_samples", {})
        for k in ["perturbed_seq", "mask_indices", "p_mask"]:
            if k in noisy:
                out[f"s1_{k}"] = noisy[k][sample_idx]  # (mc_num, seq_len)
        if tokenizer is not None and "perturbed_seq" in noisy:
            out["s1_noisy_texts"] = tokenizer.batch_decode(
                noisy["perturbed_seq"][sample_idx], skip_special_tokens=False
            )

        # Stage 2: actor inner（仅 rank-0 覆盖的样本）
        idx_in_inner = None
        if actor_inner is not None:
            si = actor_inner["sample_indices"]
            hits = (si == sample_idx).nonzero(as_tuple=True)[0]
            if len(hits) > 0:
                idx_in_inner = int(hits[0].item())
                for k in ["token_lp_all_3d", "pmask_all_3d", "token_lp_weighted_3d"]:
                    out[f"s2_{k}"] = actor_inner[k][idx_in_inner]   # (mc_num, seq_len)
                out["s2_token_lp_mc_mean"] = actor_inner["token_lp_mc_mean"][idx_in_inner]
                out["s2_gauss_coefs"] = actor_inner["gauss_coefs"]  # (mc_num,)
                # 算法诊断量（(bs, mc) → (mc,)；(bs,) → scalar）
                for k, v in actor_inner.items():
                    if not k.startswith("diag_"):
                        continue
                    if isinstance(v, torch.Tensor) and v.dim() >= 1 and v.shape[0] == actor_inner["rank0_bs"]:
                        out[f"s2_{k}"] = v[idx_in_inner]   # (mc,) or scalar
                    elif not isinstance(v, torch.Tensor):
                        out[f"s2_{k}"] = v                 # scalar flag
        out["s2_available"] = idx_in_inner is not None

        # Stage 3: ref log prob
        ref = trainer_data.get("ref_logprob", {})
        if "ref_log_probs" in ref:
            out["s3_ref_log_probs"] = ref["ref_log_probs"][sample_idx]  # (resp_len,)

        # Stage 4: reward
        reward = trainer_data.get("reward", {})
        for k in ["token_level_scores", "token_level_rewards"]:
            if k in reward:
                out[f"s4_{k}"] = reward[k][sample_idx]
        if "episode_rewards" in reward:
            out["s4_episode_reward"] = float(reward["episode_rewards"][sample_idx])
        if "is_action_valid" in reward:
            out["s4_is_action_valid"] = reward["is_action_valid"][sample_idx].tolist()

        # Stage 5: advantage
        adv = trainer_data.get("advantage", {})
        for k in ["advantages", "returns"]:
            if k in adv:
                out[f"s5_{k}"] = adv[k][sample_idx]

        return out

    # ─── 加载工具（供事后分析使用）──────────────────────────────────────────

    def load_step(self, step: int) -> Dict[str, Any]:
        """
        加载指定步骤的所有已保存文件，返回字典：
            {
              "rollout": {...},
              "noisy_samples": {...},
              "actor_inner": [mb0_dict, mb1_dict, ...],
              "ref_logprob": {...},
              "reward": {...},
              "advantage": {...},
              "samples_aligned": [{...}, ...],   # JSONL 解析结果
            }
        """
        d = self.step_dir(step, create=False)
        result: Dict[str, Any] = {}

        stage_files = {
            "rollout": "00_rollout.pt",
            "noisy_samples": "01_noisy_samples.pt",
            "ref_logprob": "03_ref_logprob.pt",
            "reward": "04_reward.pt",
            "advantage": "05_advantage.pt",
        }
        for name, fname in stage_files.items():
            p = d / fname
            if p.exists():
                result[name] = torch.load(p, map_location="cpu", weights_only=False)

        inner_dir = d / "02_actor_inner"
        if inner_dir.exists():
            inner_files = sorted(inner_dir.glob("mb*.pt"))
            result["actor_inner"] = [
                torch.load(p, map_location="cpu", weights_only=False)
                for p in inner_files
            ]

        jsonl_path = d / "samples_aligned.jsonl"
        if jsonl_path.exists():
            with open(jsonl_path, encoding="utf-8") as f:
                result["samples_aligned"] = [json.loads(line) for line in f if line.strip()]

        return result
