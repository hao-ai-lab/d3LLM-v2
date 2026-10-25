# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from collections import defaultdict
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Dict, Optional, Type

import numpy as np
import ray
import torch
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from torch.utils.data import Dataset, Sampler
from torchdata.stateful_dataloader import StatefulDataLoader
from tqdm import tqdm

from verl import DataProto
from verl.protocol import (
    ADJUST_BATCH_GRADIENT_MASK_KEY,
    pad_dataproto_to_divisor,
    unpad_dataproto,
)
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.core_algos import agg_loss
from verl.trainer.ppo.reward_penalties import (
    apply_length_penalty,
    resolve_reward_last_token_index,
)
from verl.trainer.ppo.metric_utils import (
    compute_data_metrics,
    compute_throughout_metrics,
    compute_timing_metrics,
    process_validation_metrics,
)
from verl.trainer.ppo.reward import compute_reward, compute_reward_async
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.metric import (
    reduce_metrics,
)
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.torch_functional import masked_mean
from verl.utils.tracking import ValidationGenerationsLogger
from verl.workers.rollout.async_server import AsyncLLMServerManager
from verl.gigpo import core_gigpo

from agent_system.multi_turn_rollout import TrajectoryCollector, adjust_batch

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    
    GAE = "gae"
    GRPO = "grpo"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    GRPO_PASSK = "grpo_passk"
    GiGPO = 'gigpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    """
    
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)
    
    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1
            # that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes, use_gpu=True, max_colocate_count=1, name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool
        
        self._check_resource_available()
    
    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]
    
    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
    
    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get("GPU", 0) if "GPU" in node_info else node_info.get("NPU", 0) for node, node_info in node_available_resources.items()}
        
        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")
        
        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes}" + "cannot be satisfied in this ray cluster")


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty="kl", multi_turn=False):
    """Apply KL penalty to the token-level rewards.

    This function computes the KL divergence between the reference policy and current policy,
    then applies a penalty to the token-level rewards based on this divergence.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        kl_ctrl (core_algos.AdaptiveKLController): Controller for adaptive KL penalty.
        kl_penalty (str, optional): Type of KL penalty to apply. Defaults to "kl".
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.

    Returns:
        tuple: A tuple containing:
            - The updated data with token-level rewards adjusted by KL penalty
            - A dictionary of metrics related to the KL penalty
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    token_level_scores = data.batch["token_level_scores"]
    batch_size = data.batch.batch_size[0]
    
    if multi_turn:
        loss_mask = data.batch["loss_mask"]
        response_mask = loss_mask[:, -response_length:]
    else:
        attention_mask = data.batch["attention_mask"]
        response_mask = attention_mask[:, -response_length:]
    
    # compute kl between ref_policy and current policy
    # When apply_kl_penalty, algorithm.use_kl_in_reward=True, so the reference model has been enabled.
    kld = core_algos.kl_penalty(data.batch["old_log_probs"], data.batch["ref_log_probs"], kl_penalty=kl_penalty)  # (batch_size, response_length)
    kld = kld * response_mask
    beta = kl_ctrl.value
    
    token_level_rewards = token_level_scores - beta * kld
    
    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()
    
    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch["token_level_rewards"] = token_level_rewards
    
    metrics = {"actor/reward_kl_penalty": current_kl, "actor/reward_kl_penalty_coeff": beta}
    
    return data, metrics


def apply_invalid_action_penalty(data: DataProto, invalid_action_penalty_coef=float, invalid_action_penalty_bias=0.0):
    reward_tensor = data.batch['token_level_scores']
    if 'step_rewards' in data.batch.keys():
        step_rewards = data.batch['step_rewards']
    for i in range(len(data)):
        data_item = data[i]  # DataProtoItem
        
        prompt_ids = data_item.batch['prompts']
        
        action_valids = data_item.non_tensor_batch['is_action_valid'].astype(np.float32)
        action_invalids = torch.tensor(1 - action_valids, dtype=torch.float32, device=prompt_ids.device).squeeze(0)
        # invalid action: score * (1 - coef) + bias; valid action unchanged
        last_idx = resolve_reward_last_token_index(data_item, reward_tensor[i])
        penalty_scale = 1.0 - invalid_action_penalty_coef * action_invalids
        reward_tensor[i, last_idx] = reward_tensor[i, last_idx] * penalty_scale + invalid_action_penalty_bias * action_invalids
        
        if 'step_rewards' in data.batch.keys():
            step_rewards[i] = step_rewards[i] * penalty_scale + invalid_action_penalty_bias * action_invalids

    return data


def _compute_trainer_diagnostics(
    batch,
    group_size: int = 1,
    clip_ratio: float = 0.2,
) -> dict:
    """
    在 compute_advantage 完成后、update_actor 之前调用，计算 trainer-side 算法正确性诊断量。

    返回字典包含：

    ── GRPO 组内 reward 结构 ──────────────────────────────────────────────────
      group_reward_within_std   (n_groups,)  每组内 group_size 条 rollout 的 reward std
      group_reward_within_mean  (n_groups,)  每组内 reward 均值
      nonzero_adv_group_ratio   float        组内 std > 1e-4 的比例（= 有有效梯度信号的组）
      within_group_std_mean     float        所有组的平均 within-std

    ── 策略 vs 参考模型 drift ratio ──────────────────────────────────────────
      ratio = exp(old_log_probs - ref_log_probs)，衡量当前策略相对于参考模型的偏移。
      step 0 时两者相同故 ratio ≈ 1，训练后偏离参考则偏离 1。
      （注：DLLM rollout 引擎不输出 GDPO ELBO，rollout_log_probs 恒为 0 故不可用。）
      ratio_mean/std/min/max    float        drift ratio 在有效 token 上的统计
      ratio_clipfrac_pos        float        正 advantage 侧 ratio > 1+ε 的 token 比例
      ratio_clipfrac_neg        float        负 advantage 侧 ratio < 1-ε 的 token 比例

    ── old_log_probs vs ref_log_probs（KL 信号）──────────────────────────────
      lp_delta_mean/std/abs_mean float       per-token (old - ref) 统计，= log(ratio) 均值
      lp_delta_has_nan           bool

    ── advantage 分布 ────────────────────────────────────────────────────────
      adv_mean/std/abs_mean      float       advantage 在有效 response token 上的统计
      adv_has_nan                bool
    """
    b = batch.batch
    nb = batch.non_tensor_batch
    diag: dict = {}

    resp_len = b["responses"].shape[-1] if "responses" in b else 0
    if resp_len == 0:
        return diag

    resp_mask = b["attention_mask"][:, -resp_len:].bool()   # (bs, resp_len)
    bs = b["input_ids"].shape[0]

    # ── GRPO 组内 reward 结构 ──────────────────────────────────────────────
    if "episode_rewards" in nb and group_size > 1:
        ep_rewards = torch.as_tensor(nb["episode_rewards"], dtype=torch.float32)  # (bs,)
        n_groups = bs // group_size
        if n_groups > 0:
            ep_r_grouped = ep_rewards[: n_groups * group_size].view(n_groups, group_size)
            within_std  = ep_r_grouped.std(dim=-1)    # (n_groups,)
            within_mean = ep_r_grouped.mean(dim=-1)   # (n_groups,)
            diag["group_reward_within_std"]  = within_std
            diag["group_reward_within_mean"] = within_mean
            diag["nonzero_adv_group_ratio"]  = float((within_std > 1e-4).float().mean().item())
            diag["within_group_std_mean"]    = float(within_std.mean().item())

    # ── 策略 vs 参考模型 drift ratio ─────────────────────────────────────
    # DLLM rollout 引擎不输出 GDPO ELBO（rollout_log_probs 恒为 0），故改用
    # ref_log_probs 作为基准：ratio = exp(old_lp - ref_lp) 反映策略偏离参考的程度。
    # step 0 时 ratio ≈ 1；训练中 ratio 持续偏离 1 则说明策略与参考已显著分离。
    if "old_log_probs" in b and "ref_log_probs" in b:
        old_lp = b["old_log_probs"]   # (bs, resp_len)
        ref_lp = b["ref_log_probs"]   # (bs, resp_len)
        ratio = torch.exp(old_lp - ref_lp)   # (bs, resp_len)
        ratio_valid = ratio[resp_mask]
        diag["ratio_mean"] = float(ratio_valid.mean().item())
        diag["ratio_std"]  = float(ratio_valid.std().item())
        diag["ratio_min"]  = float(ratio_valid.min().item())
        diag["ratio_max"]  = float(ratio_valid.max().item())

        if "advantages" in b:
            adv = b["advantages"]
            pos_mask = resp_mask & (adv > 0)
            neg_mask = resp_mask & (adv < 0)
            if pos_mask.any():
                diag["ratio_clipfrac_pos"] = float(
                    (ratio[pos_mask] > 1 + clip_ratio).float().mean().item()
                )
            if neg_mask.any():
                diag["ratio_clipfrac_neg"] = float(
                    (ratio[neg_mask] < 1 - clip_ratio).float().mean().item()
                )

    # ── old_log_probs vs ref_log_probs（KL 信号强度）─────────────────────
    # lp_delta = log(ratio)，即上面 ratio 的对数形式，提供更直观的绝对偏移量。
    if "old_log_probs" in b and "ref_log_probs" in b:
        delta = (b["old_log_probs"] - b["ref_log_probs"])[resp_mask]  # (n_valid,)
        diag["lp_delta_mean"]    = float(delta.mean().item())
        diag["lp_delta_std"]     = float(delta.std().item())
        diag["lp_delta_abs_mean"]= float(delta.abs().mean().item())
        diag["lp_delta_has_nan"] = bool(delta.isnan().any().item())

    # ── advantage 分布 ─────────────────────────────────────────────────────
    if "advantages" in b:
        adv_valid = b["advantages"][resp_mask]
        diag["adv_mean"]    = float(adv_valid.mean().item())
        diag["adv_std"]     = float(adv_valid.std().item())
        diag["adv_abs_mean"]= float(adv_valid.abs().mean().item())
        diag["adv_has_nan"] = bool(adv_valid.isnan().any().item())

    return diag


def compute_response_mask(data: DataProto):
    """Compute the attention mask for the response part of the sequence.

    This function extracts the portion of the attention mask that corresponds to the model's response,
    which is used for masking computations that should only apply to response tokens.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.

    Returns:
        torch.Tensor: The attention mask for the response tokens.
    """
    responses = data.batch["responses"]
    response_length = responses.size(1)
    attention_mask = data.batch["attention_mask"]
    return attention_mask[:, -response_length:]


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, multi_turn=False, norm_adv_by_std_in_grpo=True, step_advantage_w=1.0, gigpo_mode="mean_std_norm", gigpo_enable_similarity=False, gigpo_similarity_thresh=0.95, **kwargs):
    """Compute advantage estimates for policy optimization.

    This function computes advantage estimates using various estimators like GAE, GRPO, REINFORCE++, etc.
    The advantage estimates are used to guide policy optimization in RL algorithms.

    Args:
        data (DataProto): The data containing batched model outputs and inputs.
        adv_estimator: The advantage estimator to use (e.g., GAE, GRPO, REINFORCE++).
        gamma (float, optional): Discount factor for future rewards. Defaults to 1.0.
        lam (float, optional): Lambda parameter for GAE. Defaults to 1.0.
        num_repeat (int, optional): Number of times to repeat the computation. Defaults to 1.
        multi_turn (bool, optional): Whether the data is from a multi-turn conversation. Defaults to False.
        norm_adv_by_std_in_grpo (bool, optional): Whether to normalize advantages by standard deviation in GRPO. Defaults to True.

    Returns:
        DataProto: The updated data with computed advantages and returns.
    """
    # Keep adjust_batch copies out of group statistics when the shared switch
    # disables their training contribution.  Compute on real rows using the
    # unchanged estimator, then scatter results back so distributed shapes stay
    # aligned.  Copied rows receive zero advantage/return and are also masked
    # from actor losses downstream.
    adjust_batch_training_mask = data.batch.get(ADJUST_BATCH_GRADIENT_MASK_KEY)
    if adjust_batch_training_mask is not None:
        adjust_batch_training_mask = adjust_batch_training_mask.to(dtype=torch.bool)
        if not bool(adjust_batch_training_mask.all().item()):
            if kwargs.get("use_pf_ppo", False):
                raise ValueError(
                    "adjust_batch_samples_participate_in_gradient=False is not "
                    "compatible with use_pf_ppo=True because PF-PPO resamples rows "
                    "before advantage results can be scattered back safely."
                )
            if not bool(adjust_batch_training_mask.any().item()):
                raise ValueError("adjust_batch excluded every row from advantage computation")

            real_data = data.select_idxs(adjust_batch_training_mask)
            real_data = compute_advantage(
                real_data,
                adv_estimator=adv_estimator,
                gamma=gamma,
                lam=lam,
                num_repeat=num_repeat,
                multi_turn=multi_turn,
                norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                step_advantage_w=step_advantage_w,
                gigpo_mode=gigpo_mode,
                gigpo_enable_similarity=gigpo_enable_similarity,
                gigpo_similarity_thresh=gigpo_similarity_thresh,
                **kwargs,
            )
            for key in ("advantages", "returns"):
                real_tensor = real_data.batch[key]
                full_tensor = real_tensor.new_zeros(
                    (len(data), *real_tensor.shape[1:])
                )
                full_tensor[adjust_batch_training_mask.to(real_tensor.device)] = real_tensor
                data.batch[key] = full_tensor
            return data

    # Back-compatible with trainers that do not compute response mask in fit
    if "response_mask" not in data.batch:
        data.batch["response_mask"] = compute_response_mask(data)
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        advantages, returns = core_algos.compute_gae_advantage_return(
            token_level_rewards=data.batch["token_level_rewards"],
            values=data.batch["values"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
            lam=lam,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
        if kwargs.get("use_pf_ppo", False):
            data = core_algos.compute_pf_ppo_reweight_data(
                data,
                kwargs.get("pf_ppo_reweight_method", "pow"),
                kwargs.get("pf_ppo_weight_pow", 2.0),
            )
    elif adv_estimator == AdvantageEstimator.GRPO:
        # TODO: test on more adv estimator type
        grpo_calculation_mask = data.batch["response_mask"]
        if multi_turn:
            # If multi-turn, replace the mask with the relevant part of loss_mask
            response_length = grpo_calculation_mask.size(1)  # Get length from the initial response mask
            grpo_calculation_mask = data.batch["loss_mask"][:, -response_length:]  # This mask is the one intended for GRPO
        # Call compute_grpo_outcome_advantage with parameters matching its definition
        advantages, returns = core_algos.compute_grpo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=grpo_calculation_mask,
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GRPO_PASSK:
        advantages, returns = core_algos.compute_grpo_passk_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
            norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE:
        advantages, returns = core_algos.compute_reinforce_plus_plus_baseline_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            gamma=gamma,
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        advantages, returns = core_algos.compute_remax_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            reward_baselines=data.batch["reward_baselines"],
            response_mask=data.batch["response_mask"],
        )
        
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        advantages, returns = core_algos.compute_rloo_outcome_advantage(
            token_level_rewards=data.batch["token_level_rewards"],
            response_mask=data.batch["response_mask"],
            index=data.non_tensor_batch["uid"],
            traj_index=data.non_tensor_batch['traj_uid'],
        )
        data.batch["advantages"] = advantages
        data.batch["returns"] = returns
    elif adv_estimator == AdvantageEstimator.GiGPO:
        advantages, returns = core_gigpo.compute_gigpo_outcome_advantage(
            token_level_rewards=data.batch['token_level_rewards'],  # for episode group reward computing
            step_rewards=data.batch['step_rewards'],  # for step group reward computing
            response_mask=data.batch['response_mask'],
            anchor_obs=data.non_tensor_batch['anchor_obs'],
            index=data.non_tensor_batch['uid'],
            traj_index=data.non_tensor_batch['traj_uid'],
            step_advantage_w=step_advantage_w,
            mode=gigpo_mode,
            enable_similarity=gigpo_enable_similarity,
            similarity_thresh=gigpo_similarity_thresh,
        )
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    """Context manager for timing code execution.

    This utility function measures the execution time of code within its context
    and accumulates the timing information in the provided dictionary.

    Args:
        name (str): The name/identifier for this timing measurement.
        timing_raw (Dict[str, float]): Dictionary to store timing information.

    Yields:
        None: This is a context manager that yields control back to the code block.
    """
    with Timer(name=name, logger=None) as timer:
        yield
    if name not in timing_raw:
        timing_raw[name] = 0
    timing_raw[name] += timer.last


class RayPPOTrainer:
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """
    
    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(
        self,
        config,
        tokenizer,
        role_worker_mapping: dict[Role, WorkerType],
        resource_pool_manager: ResourcePoolManager,
        ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
        processor=None,
        reward_fn=None,
        val_reward_fn=None,
        train_dataset: Optional[Dataset] = None,
        val_dataset: Optional[Dataset] = None,
        collate_fn=None,
        train_sampler: Optional[Sampler] = None,
        device_name="cuda",
        traj_collector: TrajectoryCollector = None,
        envs=None,
        val_envs=None,
    ):
        """Initialize distributed PPO trainer with Ray backend."""
        
        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.envs = envs
        self.val_envs = val_envs
        self.traj_collector = traj_collector
        
        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, "Currently, only support hybrid engine"
        
        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f"{role_worker_mapping.keys()=}"
        
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.device_name = device_name
        self.validation_generations_logger = ValidationGenerationsLogger()
        
        # if ref_in_actor is True, the reference policy will be actor without lora applied
        self.ref_in_actor = config.actor_rollout_ref.model.get('lora_rank', 0) > 0
        
        # define in-reward KL control
        # kl loss control currently not suppoorted
        if config.algorithm.use_kl_in_reward:
            self.kl_ctrl_in_reward = core_algos.get_kl_controller(config.algorithm.kl_ctrl)
        
        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
            AdvantageEstimator.GRPO,
            AdvantageEstimator.GRPO_PASSK,
            AdvantageEstimator.REINFORCE_PLUS_PLUS,
            AdvantageEstimator.REMAX,
            AdvantageEstimator.RLOO,
            AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE,
            AdvantageEstimator.GiGPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError
        
        self._validate_config()
        self._create_dataloader(train_dataset, val_dataset, collate_fn, train_sampler)

        # ── Debug state saver：每 N 步保存完整流水线中间状态 ─────────────────
        _debug_every = int(config.trainer.get("debug_save_every", 0))
        if _debug_every > 0:
            from verl.utils.debug_state_saver import DebugStateSaver
            _debug_dir = config.trainer.get(
                "debug_save_dir",
                os.path.join(config.trainer.default_local_dir, "debug_states"),
            )
            self.debug_saver = DebugStateSaver(
                root=_debug_dir,
                save_every=_debug_every,
                max_noisy_decode=int(config.trainer.get("debug_noisy_decode", 2)),
            )
            print(f"[DebugSaver] 启用，保存目录: {_debug_dir}，每 {_debug_every} 步保存一次")
        else:
            self.debug_saver = None

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
        
        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."
        
        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            settings = {
                "actor_rollout_ref.actor"  : "micro_batch_size",
                "critic"                   : "micro_batch_size",
                "reward_model"             : "micro_batch_size",
                "actor_rollout_ref.ref"    : "log_prob_micro_batch_size",
                "actor_rollout_ref.rollout": "log_prob_micro_batch_size",
            }
            
            if name in settings:
                param = settings[name]
                param_per_gpu = f"{param}_per_gpu"
                
                if mbs is None and mbs_per_gpu is None:
                    raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")
                
                if mbs is not None and mbs_per_gpu is not None:
                    raise ValueError(f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove '{name}.{param}' because only '*_{param_per_gpu}'" + "is supported (the former is deprecated).")
        
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.actor.ppo_micro_batch_size,
                config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                "actor_rollout_ref.actor",
            )
            
            if self.use_reference_policy:
                # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
                check_mutually_exclusive(
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                    config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                    "actor_rollout_ref.ref",
                )
            
            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                "actor_rollout_ref.rollout",
            )
        
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu, "critic")
        
        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu, "reward_model")
        
        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus
        
        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        
        if config.algorithm.use_kl_in_reward and config.actor_rollout_ref.actor.use_kl_loss:
            print("NOTICE: You have both enabled in-reward kl and kl loss.")
        
        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get("ulysses_sequence_parallel_size", 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus
        
        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == "fsdp" and (config.actor_rollout_ref.actor.get("ulysses_sequence_parallel_size", 1) > 1 or config.actor_rollout_ref.ref.get("ulysses_sequence_parallel_size", 1) > 1):
            assert config.actor_rollout_ref.model.use_remove_padding, "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
        
        if self.use_critic and config.critic.strategy == "fsdp":
            if config.critic.get("ulysses_sequence_parallel_size", 1) > 1:
                assert config.critic.model.use_remove_padding, "When using sequence parallelism for critic, you must enable `use_remove_padding`."
        
        if config.data.get("val_batch_size", None) is not None:
            print("WARNING: val_batch_size is deprecated." + " Validation datasets are sent to inference engines as a whole batch," + " which will schedule the memory themselves.")
        
        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, "validation gen temperature should be greater than 0 when enabling do_sample"
        
        # check multi_turn with tool config
        if config.actor_rollout_ref.rollout.multi_turn.enable:
            assert config.actor_rollout_ref.rollout.multi_turn.tool_config_path is not None, "tool_config_path must be set when enabling multi_turn with tool, due to no role-playing support"
            assert config.algorithm.adv_estimator in [AdvantageEstimator.GRPO], "only GRPO is tested for multi-turn with tool"
        
        print("[validate_config] All configuration checks passed successfully!")
    
    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        """
        Creates the train and validation dataloaders.
        """
        # TODO: we have to make sure the batch size is divisible by the dp size
        from verl.trainer.main_dllm_new import create_rl_dataset, create_rl_sampler
        
        if train_dataset is None:
            train_dataset = create_rl_dataset(self.config.data.train_files, self.config.data, self.tokenizer, self.processor)
        if val_dataset is None:
            val_dataset = create_rl_dataset(self.config.data.val_files, self.config.data, self.tokenizer, self.processor)

        # Optional: limit val dataset size to speed up periodic validation.
        # Set data.num_val_samples=N in config or pass +data.num_val_samples=N on CLI.
        num_val_samples = self.config.data.get("num_val_samples", None)
        if num_val_samples is not None and num_val_samples < len(val_dataset):
            from torch.utils.data import Subset
            val_dataset = Subset(val_dataset, range(num_val_samples))
            print(f"[val] Limiting validation to {num_val_samples} samples (full dataset: {len(val_dataset)} → subset: {num_val_samples})")

        self.train_dataset, self.val_dataset = train_dataset, val_dataset
        
        if train_sampler is None:
            train_sampler = create_rl_sampler(self.config.data, self.train_dataset)
        if collate_fn is None:
            from verl.utils.dataset.rl_dataset import collate_fn as default_collate_fn
            
            collate_fn = default_collate_fn
        
        self.train_dataloader = StatefulDataLoader(
            dataset=self.train_dataset,
            batch_size=self.config.data.get("gen_batch_size", self.config.data.train_batch_size),
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            drop_last=True,
            collate_fn=collate_fn,
            sampler=train_sampler,
        )
        
        val_batch_size = self.config.data.val_batch_size  # Prefer config value if set
        if val_batch_size is None:
            val_batch_size = len(self.val_dataset)
        
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            batch_size=val_batch_size,
            num_workers=self.config.data.get("dataloader_num_workers", 8),
            shuffle=False,
            drop_last=True,
            collate_fn=collate_fn,
        )
        
        assert len(self.train_dataloader) >= 1, "Train dataloader is empty!"
        assert len(self.val_dataloader) >= 1, "Validation dataloader is empty!"
        
        print(f"Size of train dataloader: {len(self.train_dataloader)}, Size of val dataloader: {len(self.val_dataloader)}")
        
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs
        
        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps
        
        self.total_training_steps = total_training_steps
        print(f"Total training steps: {self.total_training_steps}")
        
        try:
            OmegaConf.set_struct(self.config, True)
            with open_dict(self.config):
                if OmegaConf.select(self.config, "actor_rollout_ref.actor.optim"):
                    self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
                if OmegaConf.select(self.config, "critic.optim"):
                    self.config.critic.optim.total_training_steps = total_training_steps
        except Exception as e:
            print(f"Warning: Could not set total_training_steps in config. Structure missing? Error: {e}")
    
    def _dump_generations(self, inputs, outputs, scores, reward_extra_infos_dict, dump_path):
        """Dump rollout/validation samples as JSONL."""
        os.makedirs(dump_path, exist_ok=True)
        filename = os.path.join(dump_path, f"{self.global_steps}.jsonl")
        
        n = len(inputs)
        base_data = {
            "input" : inputs,
            "output": outputs,
            "score" : scores,
            "step"  : [self.global_steps] * n,
        }
        
        for k, v in reward_extra_infos_dict.items():
            if len(v) == n:
                base_data[k] = v
        
        with open(filename, "w") as f:
            for i in range(n):
                entry = {k: v[i] for k, v in base_data.items()}
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        
        print(f"Dumped generations to {filename}")
    
    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""
        
        generations_to_log = self.config.trainer.log_val_generations
        
        if generations_to_log == 0:
            return
        
        import numpy as np
        
        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text
        
        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        
        # Take first N samples after shuffling
        samples = samples[:generations_to_log]
        
        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)
    
    def _validate(self):
        rollout_config = self.config.actor_rollout_ref.rollout
        use_student = (
            rollout_config.get("fixed_teacher", False)
            and rollout_config.name in ("sglang", "sglang_async")
            and self.config.actor_rollout_ref.actor.strategy in ("fsdp", "fsdp2")
        )
        if not use_student:
            return self._validate_impl()
        self.actor_rollout_wg.set_rollout_validation_mode(True)
        try:
            return self._validate_impl()
        finally:
            self.actor_rollout_wg.set_rollout_validation_mode(False)

    def _validate_impl(self):
        import time as time_module
        val_start_time = time_module.perf_counter()
        
        reward_tensor_lst = []
        data_source_lst = []
        tool_calling_list = []
        traj_uid_list = []
        success_rate_dict = {}
        
        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        
        # Time tracking for validation
        total_inference_time_s = 0.0
        total_generated_tokens = 0
        total_num_episodes = 0
        # dLLM-specific: accumulate per-sample TPF and NFE for val
        val_tpf_all: list = []
        val_nfe_all: list = []
        val_resp_len_all: list = []
        valid_action_lst: list = []
        
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)
            
            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n, interleave=True)
            
            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch["reward_model"]["style"] == "model":
                return {}
            
            # Store original inputs
            input_ids = test_batch.batch["input_ids"]
            # TODO: Can we keep special tokens except for padding tokens?
            input_texts = [self.tokenizer.decode(ids.long(), skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            
            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
            if "multi_modal_data" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            if "env_kwargs" in test_batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("env_kwargs")
            test_gen_batch = test_batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )
            
            test_gen_batch.meta_info = {
                "eos_token_id"      : self.tokenizer.eos_token_id,
                "pad_token_id"      : self.tokenizer.pad_token_id,
                "recompute_log_prob": False,
                "do_sample"         : self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                "validate"          : True,
                "global_step"       : self.global_steps,
            }
            print(f"test_gen_batch meta info: {test_gen_batch.meta_info}")
            
            # # pad to be divisible by dp_size
            # test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            
            # # unpad
            # test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            
            ################ agent-environment loop ###############
            speculative = self.config.actor_rollout_ref.rollout.get("speculative", False)
            speculative_k = self.config.actor_rollout_ref.rollout.get("speculative_k", 3)
            if self.val_envs is not None:
                # Environment-based multi-turn rollout (e.g. AlfWorld, WebShop)
                self.actor_rollout_wg.begin_rollout()
                try:
                    test_output_gen_batch = self.traj_collector.multi_turn_loop(
                        gen_batch=test_gen_batch,
                        actor_rollout_wg=self.actor_rollout_wg,
                        envs=self.val_envs,
                        is_train=False,
                        speculative=speculative,
                        speculative_k=speculative_k,
                    )
                finally:
                    self.actor_rollout_wg.end_rollout()
            else:
                # Tool-based multi-turn rollout handled internally by sglang
                # (e.g. MHQA with wiki_search tool via multi_turn.tool_config_path)
                test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
                test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
                test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
                test_output_gen_batch = test_batch.union(test_output_gen_batch)
                # data_source was popped from test_batch into test_gen_batch; restore it after union
                if 'data_source' not in test_output_gen_batch.non_tensor_batch and 'data_source' in test_gen_batch.non_tensor_batch:
                    test_output_gen_batch.non_tensor_batch['data_source'] = test_gen_batch.non_tensor_batch['data_source']
                # Fill in trajectory metadata fields expected by the rest of _validate
                batch_size_here = len(test_output_gen_batch.batch["responses"])
                import uuid as _uuid
                test_output_gen_batch.non_tensor_batch['traj_uid'] = np.array(
                    [str(_uuid.uuid4()) for _ in range(batch_size_here)], dtype=object
                )
                test_output_gen_batch.non_tensor_batch['tool_callings'] = np.zeros(batch_size_here, dtype=np.float32)
            print('validation generation end')
            
            # Collect time and token statistics from meta_info
            if test_output_gen_batch.meta_info:
                total_inference_time_s += test_output_gen_batch.meta_info.get("val_inference_time_s", 0.0)
                total_generated_tokens += test_output_gen_batch.meta_info.get("val_total_generated_tokens", 0)
            # Count unique episodes in this batch
            batch_traj_uids = test_output_gen_batch.non_tensor_batch['traj_uid']
            total_num_episodes += len(np.unique(batch_traj_uids))
            
            del test_batch
            test_batch = test_output_gen_batch
            # Store generated outputs
            output_ids = test_output_gen_batch.batch["responses"]
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            
            # test_batch = test_batch.union(test_output_gen_batch)
            
            # evaluate using reward_function
            result = self.val_reward_fn(test_batch, return_dict=True)
            reward_tensor = result["reward_tensor"]
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            tool_calling_list.append(test_output_gen_batch.non_tensor_batch['tool_callings'])
            traj_uid_list.append(test_output_gen_batch.non_tensor_batch['traj_uid'])
            if 'is_action_valid' in test_batch.non_tensor_batch:
                valid_action_lst.append(test_batch.non_tensor_batch['is_action_valid'])

            # ── dLLM val TPF / NFE collection ──────────────────────────────────────
            if (
                hasattr(test_output_gen_batch, "batch")
                and "decode_orders_nfe" in test_output_gen_batch.batch.keys()
            ):
                _val_do_nfe = test_output_gen_batch.batch["decode_orders_nfe"].float()    # (bs, total_len)
                _val_resp = (_val_do_nfe > 0).sum(dim=-1).float()                         # (bs,) response tokens
                # NFE denominator = nfe_total = number of distinct positive denoising
                # steps in decode_orders_nfe (set by the rollout worker).  Falls back to
                # the same distinct-step count here when nfe_total is absent — NOT
                # max(decode_orders_nfe), which over-counts forwards that accepted 0
                # retained tokens (post-EOS / accept-0 gaps) and pushes TPF below 1.
                if "nfe_total" in test_output_gen_batch.batch.keys():
                    _val_nfe = test_output_gen_batch.batch["nfe_total"].float()           # (bs,)
                else:
                    _val_nfe = torch.tensor(
                        [int(torch.unique(r[r > 0]).numel()) for r in _val_do_nfe],
                        dtype=torch.float, device=_val_do_nfe.device,
                    )                                                                     # (bs,) fallback
                _val_tpf = _val_resp / _val_nfe.clamp(min=1)
                val_tpf_all.append(_val_tpf.cpu())
                val_nfe_all.append(_val_nfe.cpu())
                val_resp_len_all.append(_val_resp.cpu())
            # success rate metrics
            for k in test_batch.non_tensor_batch.keys():
                if 'success_rate' in k:
                    if k not in success_rate_dict:
                        success_rate_dict[k] = []
                    success_rate_dict[k].append(test_batch.non_tensor_batch[k][0])
                    # all values should be the same within a batch (one scalar per episode broadcast to all steps)
                    for i in range(1, len(test_batch.non_tensor_batch[k])):
                        assert test_batch.non_tensor_batch[k][0] == test_batch.non_tensor_batch[k][i], f'not all {k} are the same, 0: {test_batch.non_tensor_batch[k][0]}, {i}: {test_batch.non_tensor_batch[k][i]}'
        
        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)
        
        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)
        tool_callings = np.concatenate(tool_calling_list, axis=0)
        traj_uids = np.concatenate(traj_uid_list, axis=0)
        success_rate = {k: np.mean(v) for k, v in success_rate_dict.items()}
        
        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
        
        # evaluate tool call based on data source
        # the values in tool_callings represent the tool call count for each trajectory; however, since the batch is expanded by step, we only need to take one value for each unique trajectories.
        data_source_tool_calling = {}
        unique_traj_uid, unique_idx = np.unique(traj_uids, return_index=True)
        unique_data_sources = data_sources[unique_idx]
        unique_tool_callings = tool_callings[unique_idx]

        # ================== NEW METRIC: episode-level test score ==================
        # 使用 traj_uids 将 step-level 的 reward 聚合成 episode-level，
        # 然后按 episode 均匀加权计算平均 reward。
        unique_episode_rewards = reward_tensor[unique_idx]
        unique_episode_data_sources = data_sources[unique_idx]

        episode_data_source_reward = {}
        for i in range(unique_episode_rewards.shape[0]):
            ds = unique_episode_data_sources[i]
            if ds not in episode_data_source_reward:
                episode_data_source_reward[ds] = []
            episode_data_source_reward[ds].append(unique_episode_rewards[i].item())
        # ================== END NEW METRIC: episode-level test score ==================
        
        for i in range(unique_tool_callings.shape[0]):
            data_source = unique_data_sources[i]
            if data_source not in data_source_tool_calling:
                data_source_tool_calling[data_source] = []
            data_source_tool_calling[data_source].append(unique_tool_callings[i].item())
        
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/{data_source}/test_score'] = np.mean(rewards)

        # ================== NEW METRIC: log episode-level test score ==================
        for data_source, rewards in episode_data_source_reward.items():
            metric_dict[f'val/{data_source}/episode_test_score'] = np.mean(rewards)
        # ================== END NEW METRIC: log episode-level test score ==================
        
        for data_source, tool_calls in data_source_tool_calling.items():
            metric_dict[f'val/{data_source}/tool_call_count/mean'] = np.mean(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/max'] = np.max(tool_calls)
            # metric_dict[f'val/{data_source}/tool_call_count/min'] = np.min(tool_calls)
        
        for k, v in success_rate.items():
            metric_dict[f'val/{k}'] = v

        if valid_action_lst:
            all_valid_actions = np.concatenate(valid_action_lst, axis=0)
            metric_dict['val/valid_action_ratio'] = np.mean(all_valid_actions.astype(np.float32)).item()
        
        # ================== dLLM val TPF / NFE metrics ==================
        if val_tpf_all:
            _all_tpf  = torch.cat(val_tpf_all)
            _all_nfe  = torch.cat(val_nfe_all)
            _all_resp = torch.cat(val_resp_len_all)
            metric_dict['val/gen/tpf_mean']      = _all_tpf.mean().item()
            metric_dict['val/gen/tpf_min']       = _all_tpf.min().item()
            metric_dict['val/gen/tpf_max']       = _all_tpf.max().item()
            metric_dict['val/gen/nfe_mean']      = _all_nfe.mean().item()
            metric_dict['val/gen/nfe_max']       = _all_nfe.max().item()
            metric_dict['val/gen/resp_len_mean'] = _all_resp.mean().item()
        # ================== END dLLM val TPF / NFE metrics ==================

        # ================== Time metrics ==================
        # time_per_episode_s: wall clock (val_total_time_s) / num_episodes.
        # inference_time_per_episode_s: target-model wall time (from rollout meta_info) / num_episodes.
        val_total_time_s = time_module.perf_counter() - val_start_time
        metric_dict['val/total_time_s'] = val_total_time_s
        metric_dict['val/inference_time_s'] = total_inference_time_s
        metric_dict['val/num_episodes'] = total_num_episodes
        if total_num_episodes > 0:
            metric_dict['val/time_per_episode_s'] = val_total_time_s / total_num_episodes
            metric_dict['val/inference_time_per_episode_s'] = total_inference_time_s / total_num_episodes
        if total_generated_tokens > 0:
            metric_dict['val/total_generated_tokens'] = total_generated_tokens
            metric_dict['val/inference_tps'] = total_generated_tokens / total_inference_time_s if total_inference_time_s > 0 else 0.0
        # ================== END Time metrics ==================
        
        return metric_dict

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.

        Creates:
        1. Ray resource pools from configuration
        2. Worker groups for each role (actor, critic, etc.)
        """
        self.resource_pool_manager.create_resource_pool()
        
        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}
        
        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor_rollout",
            )
            self.resource_pool_to_cls[resource_pool]["actor_rollout"] = actor_rollout_cls
        else:
            raise NotImplementedError
        
        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls
        
        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy], config=self.config.actor_rollout_ref, role="ref")
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls
        
        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls
        
        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs["ray_wait_register_center_timeout"] = self.config.trainer.ray_wait_register_center_timeout
        
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls, device_name=self.device_name, **wg_kwargs)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
        
        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()
        
        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()
        
        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()
        
        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor_rollout"]
        self.actor_rollout_wg.init_model()
        
        # create async rollout manager and request scheduler
        self.async_rollout_mode = False
        if self.config.actor_rollout_ref.rollout.mode == "async":
            self.async_rollout_mode = True
            self.async_rollout_manager = AsyncLLMServerManager(
                config=self.config.actor_rollout_ref,
                worker_group=self.actor_rollout_wg,
            )
    
    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir, f"global_step_{self.global_steps}")
        
        print(f"local_global_step_folder: {local_global_step_folder}")
        actor_local_path = os.path.join(local_global_step_folder, "actor")
        
        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "actor")
        
        remove_previous_ckpt_in_save = self.config.trainer.get("remove_previous_ckpt_in_save", False)
        if remove_previous_ckpt_in_save:
            print("Warning: remove_previous_ckpt_in_save is deprecated," + " set max_actor_ckpt_to_keep=1 and max_critic_ckpt_to_keep=1 instead")
        max_actor_ckpt_to_keep = self.config.trainer.get("max_actor_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        max_critic_ckpt_to_keep = self.config.trainer.get("max_critic_ckpt_to_keep", None) if not remove_previous_ckpt_in_save else 1
        
        self.actor_rollout_wg.save_checkpoint(actor_local_path, actor_remote_path, self.global_steps, max_ckpt_to_keep=max_actor_ckpt_to_keep)
        
        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, "critic")
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(self.config.trainer.default_hdfs_dir, f"global_step_{self.global_steps}", "critic")
            self.critic_wg.save_checkpoint(critic_local_path, critic_remote_path, self.global_steps, max_ckpt_to_keep=max_critic_ckpt_to_keep)
        
        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, "data.pt")
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)
        
        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir, "latest_checkpointed_iteration.txt")
        with open(local_latest_checkpointed_iteration, "w") as f:
            f.write(str(self.global_steps))
    
    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == "disable":
            return 0
        
        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest
        
        # find global_step_folder
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                print("Training from scratch")
                return 0
        else:
            if self.config.trainer.resume_mode == "resume_path":
                assert isinstance(self.config.trainer.resume_from_path, str), "resume ckpt must be str type"
                assert "global_step_" in self.config.trainer.resume_from_path, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_from_path
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        self.global_steps = int(global_step_folder.split("global_step_")[-1])
        
        print(f"Setting global step to {self.global_steps}")
        print(f"Resuming from {global_step_folder}")
        
        actor_path = os.path.join(global_step_folder, "actor")
        critic_path = os.path.join(global_step_folder, "critic")
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path, del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        
        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, "data.pt")
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path, weights_only=False)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

        # Search tasks come from the dataloader and are restored by data.pt.
        # WebShop's dataloader is only a placeholder; rebuild its real goal
        # sampler directly from the completed global step.
        if self.envs is not None and self.envs.advance_data_state(self.global_steps):
            print(f"Rebuilt environment data position from global step {self.global_steps}")
    
    def _balance_batch(self, batch: DataProto, metrics, logging_prefix="global_seqlen"):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch["attention_mask"]
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch["attention_mask"].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst, k_partitions=world_size, equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst, partitions=global_partition_lst, prefix=logging_prefix)
        metrics.update(global_balance_stats)
    
    def _save_rollout_data(self, batch: DataProto, timestep: int):
        """Save all rollout data to JSON for offline unit testing.

        Format: array of N entries, each containing all fields for one sample.
        For AR models, decode_orders are generated as sequential 1,2,3... for response tokens.
        """
        save_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..", "..", "..",
            "scripts", "training", "grpo_trainer", "webshop_qwen3.5_2B_full", "data_collected_ar",
        )
        save_dir = os.path.normpath(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"data_{timestep}.json")

        B = len(batch)
        # Collect tensor keys
        tensor_keys = {}
        for k, v in batch.batch.items():
            tensor_keys[k] = v.cpu() if isinstance(v, torch.Tensor) else v
        # Collect non-tensor keys
        non_tensor_keys = {}
        for k, v in batch.non_tensor_batch.items():
            non_tensor_keys[k] = v.tolist() if isinstance(v, np.ndarray) else v

        # Generate sequential decode_orders for AR (1,2,3... for response tokens)
        resp_len = tensor_keys.get("responses").shape[1] if "responses" in tensor_keys else 0
        seq_len = tensor_keys.get("input_ids").shape[1] if "input_ids" in tensor_keys else 0
        prompt_len = seq_len - resp_len

        # Decode raw_responses for sampled entries
        raw_responses = []
        if "responses" in tensor_keys:
            for i in range(min(B, 8)):  # Only decode first 8 for efficiency
                resp_ids = tensor_keys["responses"][i]
                raw_responses.append(self.tokenizer.decode(resp_ids, skip_special_tokens=True))

        # Build per-sample entries
        entries = []
        for i in range(B):
            entry = {"index": i}
            for k, v in tensor_keys.items():
                entry[k] = v[i].tolist() if isinstance(v, torch.Tensor) else v[i]
            for k, v in non_tensor_keys.items():
                entry[k] = v[i] if isinstance(v, list) and i < len(v) else v

            # Generate sequential decode_orders for AR models
            if "decode_orders" not in entry or entry.get("decode_orders") is None:
                decode_orders = [0] * prompt_len
                attn = tensor_keys["attention_mask"][i] if "attention_mask" in tensor_keys else None
                for j in range(resp_len):
                    if attn is not None and attn[prompt_len + j] > 0:
                        decode_orders.append(j + 1)  # 1-indexed
                    else:
                        decode_orders.append(0)  # padding
                entry["decode_orders"] = decode_orders

            entries.append(entry)

        class _Encoder(json.JSONEncoder):
            def default(self, o):
                if isinstance(o, (np.integer,)):
                    return int(o)
                if isinstance(o, (np.floating,)):
                    return float(o)
                if isinstance(o, (np.bool_,)):
                    return bool(o)
                if isinstance(o, np.ndarray):
                    return o.tolist()
                return super().default(o)

        with open(path, "w") as f:
            json.dump(entries, f, cls=_Encoder, indent=2)
        print(f"[DataCollect] Saved rollout payload ({B} samples) to {path}")

        # Save sampled version (first 8 entries) with raw_responses for inspection
        sampled_entries = []
        for i, entry in enumerate(entries[:8]):
            entry_copy = entry.copy()
            if i < len(raw_responses):
                entry_copy["raw_responses"] = raw_responses[i]
            sampled_entries.append(entry_copy)
        sampled_path = os.path.join(save_dir, f"data_{timestep}_sampled.json")
        with open(sampled_path, "w") as f:
            json.dump(sampled_entries, f, cls=_Encoder, indent=2)
        print(f"[DataCollect] Saved sampled payload (8 samples) to {sampled_path}")
    
    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC
        to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from omegaconf import OmegaConf
        
        from verl.utils.tracking import Tracking
        
        # Check if we're in data collection mode (no training updates)
        is_update = self.config.algorithm.get("is_update", True)
        
        logger = Tracking(
            project_name=self.config.trainer.project_name,
            experiment_name=self.config.trainer.experiment_name,
            default_backend=self.config.trainer.logger,
            config=OmegaConf.to_container(self.config, resolve=True),
        )
        
        self.global_steps = 0
        
        # load checkpoint before doing anything
        if is_update:
            self._load_checkpoint()
        
        # perform validation before training (skip if not updating)
        # currently, we only support validation using the reward_function.
        if is_update and self.val_reward_fn is not None and self.config.trainer.get("val_before_train", True):
            val_metrics = self._validate()
            assert val_metrics, f"{val_metrics=}"
            pprint(f"Initial validation metrics: {val_metrics}")
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get("val_only", False):
                return
        
        # add tqdm
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        
        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}
                _skip_step = False   # set True to suppress log + global_steps increment
                batch: DataProto = DataProto.from_single_dict(batch_dict)
                
                # pop those keys for generation
                batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_batch_keys_to_pop = ["raw_prompt_ids", "data_source"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("multi_modal_data")
                if "raw_prompt" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("raw_prompt")
                if "tools_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("tools_kwargs")
                if "env_kwargs" in batch.non_tensor_batch:
                    non_tensor_batch_keys_to_pop.append("env_kwargs")
                gen_batch = batch.pop(
                    batch_keys=batch_keys_to_pop,
                    non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
                )
                gen_batch.meta_info = gen_batch.meta_info or {}
                gen_batch.meta_info["global_step"] = self.global_steps

                is_last_step = self.global_steps >= self.total_training_steps
                
                with _timer("step", timing_raw):
                    # generate a batch
                    with _timer("gen", timing_raw):
                        # if not self.async_rollout_mode:
                        #     gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                        # else:
                        #     self.async_rollout_manager.wake_up()
                        #     gen_batch_output = self.async_rollout_manager.generate_sequences(gen_batch)
                        #     self.async_rollout_manager.sleep()
                        
                        ################ agent-environment loop ###############
                        self.actor_rollout_wg.begin_rollout()
                        try:
                            gen_batch_output = self.traj_collector.multi_turn_loop(
                                gen_batch=gen_batch,
                                actor_rollout_wg=self.actor_rollout_wg,
                                envs=self.envs,
                                is_train=True,
                            )
                        finally:
                            self.actor_rollout_wg.end_rollout()
                    if gen_batch_output.meta_info:
                        for _k, _v in gen_batch_output.meta_info.items():
                            if isinstance(_k, str) and _k.startswith("rollout/overlong_"):
                                metrics[_k] = _v
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer("gen_max", timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info["do_sample"] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)
                            
                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)
                            
                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))
                            
                            batch.batch["reward_baselines"] = reward_baseline_tensor
                            
                            del gen_baseline_batch, gen_baseline_output
                    
                    # batch.non_tensor_batch["uid"] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))], dtype=object)
                    # # repeat to align with repeated responses in rollout
                    # batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    # batch = batch.union(gen_batch_output)
                    del batch
                    batch = gen_batch_output
                    
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.GiGPO:
                        step_rewards_tensor = core_gigpo.compute_step_discounted_returns(
                            batch=batch,
                            gamma=self.config.algorithm.gamma
                        )
                        batch.batch['step_rewards'] = step_rewards_tensor
                    
                    _batch_size_before_adjust = len(batch)
                    batch = adjust_batch(self.config, batch)
                    _adjust_batch_added = len(batch) - _batch_size_before_adjust
                    _adjust_batch_excluded = int(
                        (~batch.batch[ADJUST_BATCH_GRADIENT_MASK_KEY]).sum().item()
                    )
                    metrics.update({
                        "batch/adjust_batch_added": float(_adjust_batch_added),
                        "batch/adjust_batch_added_ratio": (
                            float(_adjust_batch_added) / max(_batch_size_before_adjust, 1)
                        ),
                        "batch/adjust_batch_excluded_from_gradient": float(
                            _adjust_batch_excluded
                        ),
                        "batch/adjust_batch_excluded_from_advantage": float(
                            _adjust_batch_excluded
                        ),
                    })
                    
                    batch.batch["response_mask"] = compute_response_mask(batch)
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    if self.config.trainer.balance_batch and is_update:
                        self._balance_batch(batch, metrics=metrics)
                    
                    # compute global_valid tokens
                    batch.meta_info["global_token_num"] = torch.sum(batch.batch["attention_mask"], dim=-1).tolist()

                    # ── dLLM: tokens per forward (TPF) ──────────────────────────────
                    # NFE = number of distinct positive denoising steps in
                    # decode_orders_nfe (== nfe_total set by the rollout worker).
                    # Each distinct step unmasked >=1 RETAINED token, so
                    # resp_tokens >= nfe and TPF = resp_tokens / nfe >= 1 by
                    # construction.  Post-EOS filler and accept-0 forwards leave no
                    # positive entry, so they never inflate nfe.
                    # resp_tokens = count of positions with decode_orders_nfe > 0
                    # (content up to first EOS).
                    if "decode_orders_nfe" in batch.batch.keys():
                        _do_nfe      = batch.batch["decode_orders_nfe"].float()        # (bs, total_len)
                        _resp_tokens = (_do_nfe > 0).sum(dim=-1).float()               # (bs,)
                        if "nfe_total" in batch.batch.keys():
                            _nfe = batch.batch["nfe_total"].float()                    # (bs,)
                        else:
                            _nfe = torch.tensor(
                                [int(torch.unique(r[r > 0]).numel()) for r in _do_nfe],
                                dtype=torch.float, device=_do_nfe.device,
                            )                                                          # (bs,) fallback
                        _tpf = _resp_tokens / _nfe.clamp(min=1)
                        metrics.update({
                            "gen/tpf_mean":        _tpf.mean().item(),
                            "gen/tpf_min":         _tpf.min().item(),
                            "gen/tpf_max":         _tpf.max().item(),
                            "gen/nfe_steps_mean":  _nfe.mean().item(),
                            "gen/nfe_steps_max":   _nfe.max().item(),
                        })
                    
                    # Data collection mode: save rollout data and skip training
                    if not is_update:
                        with _timer("reward", timing_raw):
                            reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = reward_tensor
                            batch.batch["advantages"] = torch.zeros_like(reward_tensor)
                            batch.batch["returns"] = reward_tensor.clone()
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                        self._save_rollout_data(batch, self.global_steps)

                        # Debug: print full response for first 2 samples (first 3 iters only)
                        if self.global_steps <= 3:
                            _rlen = batch.batch['responses'].size(1)
                            for _i in range(min(2, len(batch))):
                                _ids = batch.batch['input_ids'][_i]
                                _am = batch.batch['attention_mask'][_i].bool()
                                _full_text = self.tokenizer.decode(_ids[_am], skip_special_tokens=False)
                                print(f"\n{'='*50} [Iter {self.global_steps}] Sample {_i} {'='*50}")
                                print(_full_text)
                    else:
                        # Normal training mode
                        with _timer("reward", timing_raw):
                            # compute reward model score
                            if self.use_rm:
                                reward_tensor = self.rm_wg.compute_rm_score(batch)
                                batch = batch.union(reward_tensor)

                            if self.config.reward_model.launch_reward_fn_async:
                                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
                            else:
                                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                    
                    # ── Debug Stage 0: rollout batch（在 forward_process 前保存，此时只有 rollout 结果）──
                    _do_debug = self.debug_saver is not None and self.debug_saver.should_save(self.global_steps)
                    if _do_debug:
                        self.debug_saver.save_rollout(self.global_steps, batch, tokenizer=self.tokenizer)

                    if is_update:
                        # recompute old_log_probs
                        # Inject the current step. Curriculum backends use their dedicated
                        # dllm_config.curriculum_total_steps instead of total_training_steps.
                        batch.meta_info["global_step"] = self.global_steps
                        with _timer("forward_process", timing_raw):
                            forward_batch_output = self.actor_rollout_wg.forward_process(batch)
                        batch = batch.union(forward_batch_output)

                        # ── Debug Stage 1: noisy samples（forward_process 完成后，distill/非 distill 均保存）──
                        if _do_debug:
                            self.debug_saver.save_noisy_samples(self.global_steps, batch, tokenizer=self.tokenizer)

                        _dllm_cfg = self.config.actor_rollout_ref.get("dllm_config", {})
                        _actor_type = _dllm_cfg.get("actor_type")
                        if _actor_type is None:
                            raise ValueError(
                                "[Trainer] dllm_config.actor_type must be explicitly set to "
                                "'rl' or 'distill'."
                            )
                        _use_distillation = (_actor_type == "distill")

                        if not _use_distillation:
                            if _do_debug:
                                # 注入 saver 到 actor worker，使 _forward_micro_batch 在 compute_log_prob 时保存内部数据
                                self.actor_rollout_wg.set_debug_saver(self.debug_saver, self.global_steps)

                            # recompute old_log_probs with current actor parameters
                            # entropys: (bs, mc_num, response_length) for d1; (bs, 1, response_length) for bgpo/online_vrpo/gdpo
                            # mask_indices: (bs, mc_num, seq_len), slice tail to align with response
                            with _timer("old_log_prob", timing_raw):
                                old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                                entropys = old_log_prob.batch["old_entropys"]  # (bs, response_len)
                                response_len = entropys.shape[-1]
                                response_masks = batch.batch["attention_mask"][:, -response_len:]  # (bs, response_len)
                                loss_agg_mode = self.config.actor_rollout_ref.actor.loss_agg_mode
                                entropy_loss = agg_loss(loss_mat=entropys, loss_mask=response_masks, loss_agg_mode=loss_agg_mode)
                                metrics.update({"actor/entropy_loss": entropy_loss.detach().item()})
                                old_log_prob.batch.pop("old_entropys")
                                batch = batch.union(old_log_prob)

                                # ── Debug Stage 2: actor compute_log_prob 完成，清除 actor-side saver ──
                                if _do_debug:
                                    self.actor_rollout_wg.clear_debug_saver()

                                if "rollout_log_probs" in batch.batch.keys():
                                    rollout_old_log_probs = batch.batch["rollout_log_probs"]
                                    actor_old_log_probs = batch.batch["old_log_probs"]
                                    attention_mask = batch.batch["attention_mask"]
                                    responses = batch.batch["responses"]
                                    response_length = responses.size(1)
                                    response_mask = attention_mask[:, -response_length:]

                                    rollout_probs = torch.exp(rollout_old_log_probs)
                                    actor_probs = torch.exp(actor_old_log_probs)
                                    rollout_probs_diff = torch.abs(rollout_probs - actor_probs)
                                    rollout_probs_diff = torch.masked_select(rollout_probs_diff, response_mask.bool())
                                    rollout_probs_diff_max = torch.max(rollout_probs_diff)
                                    rollout_probs_diff_mean = torch.mean(rollout_probs_diff)
                                    rollout_probs_diff_std = torch.std(rollout_probs_diff)
                                    metrics.update(
                                        {
                                            "training/rollout_probs_diff_max": rollout_probs_diff_max.detach().item(),
                                            "training/rollout_probs_diff_mean": rollout_probs_diff_mean.detach().item(),
                                            "training/rollout_probs_diff_std": rollout_probs_diff_std.detach().item(),
                                        }
                                    )
 
                            if self.use_reference_policy:
                                # compute reference log_prob
                                with _timer("ref", timing_raw):
                                    if not self.ref_in_actor:
                                        ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                                    else:
                                        ref_log_prob = self.actor_rollout_wg.compute_ref_log_prob(batch)
                                    batch = batch.union(ref_log_prob)
                                # ── Debug Stage 3: ref log prob ─────────────────────────────────
                                if _do_debug:
                                    self.debug_saver.save_ref_logprob(self.global_steps, batch)

                            # compute values
                            if self.use_critic:
                                with _timer("values", timing_raw):
                                    values = self.critic_wg.compute_values(batch)
                                    batch = batch.union(values)
                            
                            with _timer("adv", timing_raw):
                                # we combine with rule-based rm
                                reward_extra_infos_dict: dict[str, list]
                                if self.config.reward_model.launch_reward_fn_async:
                                    reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                                batch.batch["token_level_scores"] = reward_tensor
                                
                                print(f"{list(reward_extra_infos_dict.keys())=}")
                                if reward_extra_infos_dict:
                                    batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                                
                                # apply_invalid_action_penalty only mutates rewards; ratio is logged below
                                if self.config.actor_rollout_ref.actor.get('use_invalid_action_penalty', True):
                                    batch = apply_invalid_action_penalty(batch,
                                        invalid_action_penalty_coef=self.config.actor_rollout_ref.actor.invalid_action_penalty_coef,
                                        invalid_action_penalty_bias=self.config.actor_rollout_ref.actor.get('invalid_action_penalty_bias', 0.0),
                                    )

                                if 'is_action_valid' in batch.non_tensor_batch:
                                    metrics['episode/valid_action_ratio'] = np.mean(
                                        batch.non_tensor_batch['is_action_valid'].astype(np.float32)
                                    ).item()

                                if self.config.actor_rollout_ref.actor.get('use_length_penalty', False):
                                    actor_cfg = self.config.actor_rollout_ref.actor
                                    batch, length_metrics = apply_length_penalty(
                                        batch,
                                        length_penalty_coef=actor_cfg.length_penalty_coef,
                                        length_penalty_threshold=actor_cfg.length_penalty_threshold,
                                        length_penalty_max=actor_cfg.length_penalty_max,
                                        length_penalty_mode=actor_cfg.length_penalty_mode,
                                    )
                                    metrics.update(length_metrics)
                                
                                # compute rewards. apply_kl_penalty if available
                                if self.config.algorithm.use_kl_in_reward:
                                    batch, kl_metrics = apply_kl_penalty(batch, kl_ctrl=self.kl_ctrl_in_reward, kl_penalty=self.config.algorithm.kl_penalty)
                                    metrics.update(kl_metrics)
                                else:
                                    batch.batch["token_level_rewards"] = batch.batch["token_level_scores"]
                                
                                # compute advantages, executed on the driver process
                                
                                norm_adv_by_std_in_grpo = self.config.algorithm.get("norm_adv_by_std_in_grpo", True)  # GRPO adv normalization factor
                                
                                batch = compute_advantage(
                                    batch,
                                    adv_estimator=self.config.algorithm.adv_estimator,
                                    gamma=self.config.algorithm.gamma,
                                    lam=self.config.algorithm.lam,
                                    num_repeat=self.config.actor_rollout_ref.rollout.n,
                                    norm_adv_by_std_in_grpo=norm_adv_by_std_in_grpo,
                                    multi_turn=self.config.actor_rollout_ref.rollout.multi_turn.enable,
                                    use_pf_ppo=self.config.algorithm.use_pf_ppo,
                                    pf_ppo_reweight_method=self.config.algorithm.pf_ppo.reweight_method,
                                    pf_ppo_weight_pow=self.config.algorithm.pf_ppo.weight_pow,
                                    step_advantage_w=self.config.algorithm.gigpo.step_advantage_w,
                                    gigpo_mode=self.config.algorithm.gigpo.mode,
                                    gigpo_enable_similarity=self.config.algorithm.gigpo.enable_similarity,
                                    gigpo_similarity_thresh=self.config.algorithm.gigpo.similarity_thresh,
                                )

                                # ── Debug Stage 4+5+jsonl: reward/advantage 全部就绪 ─────────────
                                if _do_debug:
                                    # 计算 trainer-side 算法正确性诊断量，一并传入 saver
                                    _trainer_diag = _compute_trainer_diagnostics(
                                        batch,
                                        group_size=self.config.actor_rollout_ref.rollout.n,
                                        clip_ratio=self.config.actor_rollout_ref.actor.clip_ratio,
                                    )
                                    self.debug_saver.save_post_advantage(
                                        self.global_steps, batch,
                                        tokenizer=self.tokenizer,
                                        trainer_diagnostics=_trainer_diag,
                                    )
                                    print(
                                        f"[DebugSaver] step {self.global_steps} 已保存至: "
                                        f"{self.debug_saver.step_dir(self.global_steps, create=False)}"
                                    )

                        else:
                            # distillation mode: retrieve reward for logging / distill_all=False
                            # filtering (no advantage computation needed)
                            if self.config.reward_model.launch_reward_fn_async:
                                reward_tensor, reward_extra_infos_dict = ray.get(future_reward)
                            # reward_tensor already available from sync path (line above if block)
                            batch.batch["token_level_scores"] = reward_tensor
                            batch.batch["token_level_rewards"] = reward_tensor
                            batch.batch["advantages"] = torch.zeros_like(reward_tensor)
                            batch.batch["returns"] = reward_tensor.clone()
                            if reward_extra_infos_dict:
                                batch.non_tensor_batch.update({k: np.array(v) for k, v in reward_extra_infos_dict.items()})
                            if 'is_action_valid' in batch.non_tensor_batch:
                                metrics['episode/valid_action_ratio'] = np.mean(
                                    batch.non_tensor_batch['is_action_valid'].astype(np.float32)
                                ).item()

                        # update critic
                        if self.use_critic:
                            with _timer("update_critic", timing_raw):
                                critic_output = self.critic_wg.update_critic(batch)
                            critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                            metrics.update(critic_output_metrics)
                        
                        # implement critic warmup
                        if self.config.trainer.critic_warmup <= self.global_steps:
                            # ── Distillation: filter to successful trajectories if distill_all=False ──
                            _update_batch = batch

                            if _use_distillation:
                                _distill_all = bool(self.config.actor_rollout_ref.actor.get("distill_all", True))
                                if not _distill_all and "token_level_scores" in batch.batch.keys():
                                    # distill_success_threshold: environment-specific reward threshold for
                                    # "success". Defaults to 0 (any positive reward = success).
                                    # Set higher for partial-reward envs (e.g. 0.5 for webshop).
                                    _succ_thresh = float(self.config.actor_rollout_ref.actor.get(
                                        "distill_success_threshold", 0.0
                                    ))
                                    step_scores = batch.batch["token_level_scores"].sum(-1).cpu()  # (bs,)
                                    traj_uids = batch.non_tensor_batch.get("traj_uid", None)
                                    if traj_uids is not None:
                                        # trajectory-level: keep ALL steps from successful trajectories
                                        successful_traj_uids = {
                                            str(traj_uids[i])
                                            for i, s in enumerate(step_scores.tolist())
                                            if s >= _succ_thresh
                                        }
                                        success_mask = np.array(
                                            [str(uid) in successful_traj_uids for uid in traj_uids]
                                        )
                                        n_succ_traj = len(successful_traj_uids)
                                        n_total_traj = len(set(str(u) for u in traj_uids.tolist()))
                                        metrics["distill/success_traj_ratio"] = n_succ_traj / max(n_total_traj, 1)
                                    else:
                                        # fallback: per-step filtering
                                        success_mask = (step_scores >= _succ_thresh).numpy()
                                    if (
                                        self.config.actor_rollout_ref.actor.get("distill_filter_invalid_actions", False)
                                        and "is_action_valid" in batch.non_tensor_batch
                                    ):
                                        success_mask &= batch.non_tensor_batch["is_action_valid"].astype(bool)
                                    n_success = int(success_mask.sum())
                                    n_total = len(success_mask)
                                    metrics.update({
                                        "distill/n_success_steps": n_success,
                                        "distill/success_step_ratio": n_success / max(n_total, 1),
                                        "distill/skipped_update": int(n_success == 0),
                                    })
                                    if n_success == 0:
                                        print(
                                            f"[step {self.global_steps}] distill_all=False: "
                                            "no successful trajectories in batch, skipping actor update"
                                        )
                                        _update_batch = None
                                        _skip_step = True
                                        # placeholder so wandb charts have a value at every step
                                        metrics["actor/distillation_loss"] = float("nan")
                                    elif n_success < n_total:
                                        _update_batch = batch[success_mask]

                            # ── Filter out T=0 sequences (no valid decode trajectory) ──
                            # T=0 means decode_orders_nfe is all-zero for that sequence:
                            # build_trajectory_noisy_batch returns all-False masks → 0 loss
                            # contribution but still costs a full forward pass. Remove them.
                            if (
                                _use_distillation
                                and _update_batch is not None
                                and "decode_orders_nfe" in _update_batch.batch.keys()
                            ):
                                _nfe = _update_batch.batch["decode_orders_nfe"]  # (bs, total_len)
                                _valid_traj = (_nfe.max(dim=-1)[0] > 0).cpu().numpy()
                                _n_t0 = int((~_valid_traj).sum())
                                if _n_t0 > 0:
                                    print(
                                        f"[step {self.global_steps}] distillation: "
                                        f"removed {_n_t0} T=0 sequences (no decode trajectory)"
                                    )
                                    metrics["distill/n_t0_removed"] = _n_t0
                                    _n_valid = int(_valid_traj.sum())
                                    if _n_valid == 0:
                                        _update_batch = None
                                        _skip_step = True
                                        metrics["actor/distillation_loss"] = float("nan")
                                    else:
                                        _update_batch = _update_batch[_valid_traj]

                            # update actor
                            if _update_batch is not None:
                                # _update_batch must be divisible by the DP world_size.
                                # After success-filtering the size may be arbitrary, so
                                # pad to the nearest multiple (extra samples are duplicates
                                # of the last row and contribute minimally to the loss).
                                _dp_size = self.actor_rollout_wg.world_size
                                _ub_len = len(_update_batch)
                                if _ub_len % _dp_size != 0:
                                    _update_batch, _ub_pad = pad_dataproto_to_divisor(
                                        _update_batch, _dp_size
                                    )
                                    if not self.config.trainer.get(
                                        "adjust_batch_samples_participate_in_gradient", True
                                    ):
                                        _gradient_mask = _update_batch.batch[
                                            ADJUST_BATCH_GRADIENT_MASK_KEY
                                        ].clone()
                                        _gradient_mask[-_ub_pad:] = False
                                        _update_batch.batch[
                                            ADJUST_BATCH_GRADIENT_MASK_KEY
                                        ] = _gradient_mask
                                else:
                                    _ub_pad = 0
                                with _timer("update_actor", timing_raw):
                                    _update_batch.meta_info["multi_turn"] = self.config.actor_rollout_ref.rollout.multi_turn.enable
                                    # temperature is normally injected by forward_process; set it
                                    # explicitly here so distillation mode (which skips forward_process)
                                    # can also find it
                                    if "temperature" not in _update_batch.meta_info:
                                        _update_batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature
                                    actor_output = self.actor_rollout_wg.update_actor(_update_batch)
                                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                                metrics.update(actor_output_metrics)
                        
                        # Log rollout generations if enabled
                        rollout_data_dir = self.config.trainer.get("rollout_data_dir", None)
                        if rollout_data_dir:
                            with _timer("dump_rollout_generations", timing_raw):
                                print(batch.batch.keys())
                                inputs = self.tokenizer.batch_decode(batch.batch["prompts"], skip_special_tokens=True)
                                outputs = self.tokenizer.batch_decode(batch.batch["responses"], skip_special_tokens=True)
                                scores = batch.batch["token_level_scores"].sum(-1).cpu().tolist()
                                self._dump_generations(
                                    inputs=inputs,
                                    outputs=outputs,
                                    scores=scores,
                                    reward_extra_infos_dict=reward_extra_infos_dict,
                                    dump_path=rollout_data_dir,
                                )
                        
                        # validate
                        if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.test_freq == 0):
                            with _timer("testing", timing_raw):
                                val_metrics: dict = self._validate()
                                if is_last_step:
                                    last_val_metrics = val_metrics
                            metrics.update(val_metrics)
                        
                        if self.config.trainer.save_freq > 0 and (is_last_step or self.global_steps % self.config.trainer.save_freq == 0):
                            with _timer("save_checkpoint", timing_raw):
                                self._save_checkpoint()
                
                # training metrics
                metrics.update(
                    {
                        "training/global_step": self.global_steps,
                        "training/epoch"      : epoch,
                    }
                )
                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic and is_update))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                # TODO: implement actual tflpo and theoretical tflpo
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                
                # TODO: make a canonical logger that supports various backend
                if not _skip_step:
                    logger.log(data=metrics, step=self.global_steps)
                    progress_bar.update(1)
                    self.global_steps += 1
                if is_last_step:
                    pprint(f"Final validation metrics: {last_val_metrics}")
                    progress_bar.close()
                    return
