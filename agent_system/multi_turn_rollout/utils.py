# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

import torch
import numpy as np
import random
from typing import List, Tuple, Dict
import math
from PIL import Image
from verl import DataProto
from verl.protocol import ADJUST_BATCH_GRADIENT_MASK_KEY

def to_list_of_dict(batch: DataProto) -> list[dict]:
    tensors = batch.batch
    non_tensor = batch.non_tensor_batch
    batch_size = len(tensors['input_ids'])
    save_list = []
    for bs in range(batch_size):
        save_dict = dict()
        for key, val in tensors.items():
            save_dict[key] = val[bs]
        for key, val in non_tensor.items():
            save_dict[key] = val[bs]
        save_list.append(save_dict)
    return save_list


def torch_to_numpy(tensor, is_object=False):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        pass
    else:
        raise ValueError(f"Unsupported type: {type(tensor)})")

    if is_object:
        tensor = tensor.astype(object)
    return tensor

def numpy_to_torch(array, device):
    if isinstance(array, np.ndarray):
        array = torch.from_numpy(array).to(device)
    elif isinstance(array, torch.Tensor):
        array = array.to(device)
    else:
        raise ValueError(f"Unsupported type: {type(array)})")
    return array


def process_image(image, max_pixels: int = 2048 * 2048, min_pixels: int = 256 * 256):
    if isinstance(image, torch.Tensor):
        image = torch_to_numpy(image)
    if image.max() < 1:
        image = image * 255.0
    if image.dtype != np.uint8:
        image = image.astype(np.uint8)
    image = Image.fromarray(image)

    if (image.width * image.height) > max_pixels:
        resize_factor = math.sqrt(max_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if (image.width * image.height) < min_pixels:
        resize_factor = math.sqrt(min_pixels / (image.width * image.height))
        width, height = int(image.width * resize_factor), int(image.height * resize_factor)
        image = image.resize((width, height))

    if image.mode != 'RGB':
        image = image.convert('RGB')

    return image


def adjust_batch(config, data: DataProto, mode="copy") -> DataProto:
    """Align batch size while recording whether copied rows may train.

    The per-row gradient mask preserves the historical behaviour by default.
    When ``trainer.adjust_batch_samples_participate_in_gradient`` is false,
    rows appended by ``mode="copy"`` remain available for shape alignment but
    are excluded from downstream advantage statistics and actor losses.
    """
    world_size = config.trainer.n_gpus_per_node * config.trainer.nnodes
    size_divisor_rollout = config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu * world_size
    if config.algorithm.use_kl_in_reward or config.actor_rollout_ref.actor.use_kl_loss:
        size_divisor_ref = config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu * world_size
    else:
        size_divisor_ref = size_divisor_rollout
    if "multi_modal_inputs" in data.non_tensor_batch:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_mini_batch_size
    else:
        size_divisor_actor = config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu * world_size
    size_divisor = np.lcm.reduce(np.array([size_divisor_ref, size_divisor_rollout, size_divisor_actor])).item()

    # check if the batch size is divisible by the dp size, if not, delete the last few samples to make it divisible
    bs = len(data)
    remainder = bs % size_divisor
    if remainder == 0:
        data.batch[ADJUST_BATCH_GRADIENT_MASK_KEY] = torch.ones(
            bs, dtype=torch.bool, device=data.batch["input_ids"].device
        )
        return data
    
    if mode == "delete":
        # Generate indices to remove, rather than indices to keep
        remove_indices = np.random.choice(bs, remainder, replace=False)
        # Sort remove_indices to maintain stability when deleting
        remove_indices = np.sort(remove_indices)
        
        # Create a boolean mask for elements to keep
        keep_mask = np.ones(bs, dtype=bool)
        keep_mask[remove_indices] = False

        keep_mask_tensor = torch.tensor(keep_mask, dtype=torch.bool, device=data.batch['input_ids'].device)
        # Apply the mask to keep elements in their original order
        tensor_data = data.batch[keep_mask_tensor]
        non_tensor_data = {key: val[keep_mask] for key, val in data.non_tensor_batch.items()}
        adjusted_batch = DataProto(batch=tensor_data, non_tensor_batch=non_tensor_data, meta_info=data.meta_info)
        del data
    elif mode == "copy":
        to_add = size_divisor - remainder
        dup_indices = np.random.choice(bs, to_add, replace=False)
        dup_proto = data.select_idxs(dup_indices)

        adjusted_batch = DataProto.concat([data, dup_proto])
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    gradient_mask = torch.ones(
        len(adjusted_batch),
        dtype=torch.bool,
        device=adjusted_batch.batch["input_ids"].device,
    )
    participate_in_gradient = bool(
        config.trainer.get("adjust_batch_samples_participate_in_gradient", True)
    )
    if mode == "copy" and not participate_in_gradient:
        gradient_mask[bs:] = False
    adjusted_batch.batch[ADJUST_BATCH_GRADIENT_MASK_KEY] = gradient_mask

    return adjusted_batch


def filter_group_data(batch_list : List[Dict],
                        episode_rewards: np.ndarray,
                        episode_lengths: np.ndarray,
                        success: Dict[str, np.ndarray],
                        traj_uid: np.ndarray,
                        tool_callings: np.ndarray,
                        config,
                        last_try: bool = False,
                        ):
    """
    Dynamic Sampling:
    Over-sample and filter out episode group in which all episodes have the same rewards.
    Adopted from DAPO (https://arxiv.org/abs/2503.14476)
    """
    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings
    
    batch_size = config.data.train_batch_size
    group_n = config.env.rollout.n
    if group_n <= 1:
        print("Warning: group_n <= 1, no need to adopt dynamic sampling")

    # Handle each group
    keep_indices = np.array([], dtype=np.int64)
    for i in range(batch_size):
        # Get the indices of the current group
        group_indices = np.arange(i * group_n, (i + 1) * group_n)
        group_rewards = episode_rewards[group_indices]

        # check if all group_traj_uid are the same
        for index in group_indices:
            assert batch_list[index][0]['uid'] == batch_list[group_indices[0]][0]['uid']

        # Check if all rewards in the group are the same
        if not np.all(group_rewards == group_rewards[0]):
            # If so, keep the entire group, otherwise, remove it
            keep_indices = np.concatenate((keep_indices, group_indices))
    
    # Filter the batch_list, episode_rewards, episode_lengths, success, and tool_callings based on the keep_indices
    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    # success = {key: value[keep_indices] for key, value in success.items()}
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings


def _scalar_int(value) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.reshape(-1)[0].item())
    if isinstance(value, np.ndarray):
        return int(value.reshape(-1)[0])
    return int(value)


def _step_is_active(step: dict) -> bool:
    active = step.get("active_masks", True)
    if isinstance(active, (np.bool_, bool, np.integer, int)):
        return bool(active)
    if isinstance(active, np.ndarray):
        return bool(active.reshape(-1)[0])
    if isinstance(active, torch.Tensor):
        return bool(active.reshape(-1)[0].item())
    return bool(active)


def resolve_step_effective_response_length(step: dict) -> int:
    """Effective response length for one rollout step dict (from to_list_of_dict)."""
    if "response_first_eos_pos" in step:
        return max(_scalar_int(step["response_first_eos_pos"]), 0)

    responses = step.get("responses")
    attention_mask = step.get("attention_mask")
    if responses is None or attention_mask is None:
        return 0

    if isinstance(responses, torch.Tensor):
        resp_len = int(responses.shape[-1])
    elif isinstance(responses, np.ndarray):
        resp_len = int(responses.shape[-1])
    else:
        return 0

    if isinstance(attention_mask, torch.Tensor):
        return int(attention_mask[-resp_len:].sum().item())
    if isinstance(attention_mask, np.ndarray):
        return int(attention_mask[-resp_len:].sum())
    return 0


def filter_overlong_response_data(
    batch_list: List,
    episode_rewards: np.ndarray,
    episode_lengths: np.ndarray,
    success: Dict[str, np.ndarray],
    traj_uid: np.ndarray,
    tool_callings: np.ndarray,
    config,
    last_try: bool = False,
):
    """
    Drop trajectories that contain any active turn whose effective response length
    reaches the overlong threshold (default: data.max_response_length).

    Traj-level drop keeps GiGPO episode structure intact.
    """
    cfg = config.algorithm.get("filter_overlong_responses", None)
    empty_metrics = {
        "rollout/overlong_filter_ratio": 0.0,
        "rollout/overlong_filter_n_drop": 0,
        "rollout/overlong_filter_n_keep": len(batch_list),
        "rollout/overlong_eff_len_mean": 0.0,
    }
    if cfg is None or not cfg.get("enable", False):
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, empty_metrics

    if last_try:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, empty_metrics

    mode = str(cfg.get("mode", "traj")).strip().lower()
    if mode != "traj":
        raise ValueError(f"filter_overlong_responses.mode only supports 'traj', got {mode!r}")

    threshold = cfg.get("threshold", None)
    if threshold is None:
        threshold = int(config.data.max_response_length)
    else:
        threshold = int(threshold)

    n_total = len(batch_list)
    keep_indices = []
    dropped_eff_lens = []

    for i in range(n_total):
        traj_steps = batch_list[i]
        traj_overlong = False
        traj_max_eff = 0
        for step in traj_steps:
            if not _step_is_active(step):
                continue
            eff_len = resolve_step_effective_response_length(step)
            traj_max_eff = max(traj_max_eff, eff_len)
            if eff_len >= threshold:
                traj_overlong = True
                break
        if traj_overlong:
            dropped_eff_lens.append(float(traj_max_eff))
        else:
            keep_indices.append(i)

    keep_indices = np.asarray(keep_indices, dtype=np.int64)
    n_keep = int(keep_indices.size)
    n_drop = n_total - n_keep
    metrics = {
        "rollout/overlong_filter_ratio": float(n_drop) / float(n_total) if n_total > 0 else 0.0,
        "rollout/overlong_filter_n_drop": n_drop,
        "rollout/overlong_filter_n_keep": n_keep,
        "rollout/overlong_eff_len_mean": float(np.mean(dropped_eff_lens)) if dropped_eff_lens else 0.0,
    }

    if n_drop == 0:
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, metrics

    if n_keep == 0:
        print(
            f"[filter_overlong_responses] WARNING: all {n_total} trajectories overlong "
            f"(threshold={threshold}); keeping original batch to avoid empty train step."
        )
        metrics["rollout/overlong_filter_n_keep"] = n_total
        metrics["rollout/overlong_filter_n_drop"] = 0
        metrics["rollout/overlong_filter_ratio"] = 0.0
        return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, metrics

    print(
        f"[filter_overlong_responses] threshold={threshold} drop={n_drop}/{n_total} "
        f"ratio={metrics['rollout/overlong_filter_ratio']:.3f} "
        f"dropped_eff_len_mean={metrics['rollout/overlong_eff_len_mean']:.1f}"
    )

    success = {
        key: value[keep_indices]
        for key, value in success.items()
        if len(value) == len(batch_list)
    }
    batch_list = [batch_list[i] for i in keep_indices]
    episode_rewards = episode_rewards[keep_indices]
    episode_lengths = episode_lengths[keep_indices]
    traj_uid = traj_uid[keep_indices]
    tool_callings = tool_callings[keep_indices]

    return batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, metrics
