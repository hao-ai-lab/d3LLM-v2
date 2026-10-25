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

import threading
import time
import torch
import numpy as np
from verl import DataProto
from verl.utils.dataset.rl_dataset import collate_fn
from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F
from transformers import PreTrainedTokenizer
import uuid
from agent_system.multi_turn_rollout.utils import (
    process_image,
    to_list_of_dict,
    torch_to_numpy,
    filter_group_data,
    filter_overlong_response_data,
)
from verl.utils.trajectory_data_writer import TrajectoryDataWriter, resolve_trajectory_output_dir
from agent_system.environments import EnvironmentManagerBase
from typing import List, Dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from concurrent.futures import ThreadPoolExecutor, Future
from openai import OpenAI


def _extract_action(text: str):
    """
    提取 WebShop 风格 `<action>...</action>` 中的内容，全部转为小写。
    找不到标签时返回 None。
    """
    s, e = text.find("<action>"), text.find("</action>")
    if s == -1 or e == -1:
        return None
    return text[s + 8:e].strip()


def _webshop_action_judge(draft_action: str | None, target_action: str | None) -> bool:
    """
    比较两个已经提取好的 action 字符串是否完全一致（均可为 None）。
    """
    return draft_action == target_action


class TrajectoryCollector:
    def __init__(self, config, tokenizer: PreTrainedTokenizer, processor=None):
        """
        Initialize the TrajectoryProcessor class.
        
        Parameters:
            config: Configuration object containing data processing settings
            tokenizer (PreTrainedTokenizer): Tokenizer for text encoding and decoding
            processor: Image processor for multimodal inputs
        """
        self.config = config
        self.tokenizer = tokenizer
        self.processor = processor

        ######
        self.not_use_template = config.data.get("not_use_template", False)
        ######

        # LLaDA2.1 defaults to 'detailed thinking off' in its chat template.
        # Override to 'on' when the env requires <think> tags (e.g., WebShop).
        enable_thinking = config.data.get("enable_thinking", True)
        if (enable_thinking and not self.not_use_template
                and hasattr(tokenizer, 'chat_template') and tokenizer.chat_template
                and "thinking_option = 'off'" in tokenizer.chat_template):
            tokenizer.chat_template = tokenizer.chat_template.replace(
                "thinking_option = 'off'", "thinking_option = 'on'")
            print("[TrajectoryProcessor] Enabled thinking in LLaDA chat template")

    def preprocess_single_sample(
        self,
        item: int,
        gen_batch: DataProto,
        obs: Dict,
    ):
        """
        Process a single observation sample, organizing environment observations (text and/or images) 
        into a format processable by the model.
        
        Parameters:
            item (int): Sample index in the batch
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation, may contain 'text', 'image', 'anchor' keys
        
        Returns:
            dict: Contains processed input data such as input_ids, attention_mask, etc.
        """

        raw_prompt = gen_batch.non_tensor_batch['raw_prompt'][item]
        data_source = gen_batch.non_tensor_batch['data_source'][item]
        apply_chat_template_kwargs = self.config.data.get("apply_chat_template_kwargs", {})
        
        # Get observation components
        obs_texts = obs.get('text', None)
        obs_images = obs.get('image', None)
        obs_anchors = obs.get('anchor', None)
        obs_text = obs_texts[item] if obs_texts is not None else None
        obs_image = obs_images[item] if obs_images is not None else None
        obs_anchor = obs_anchors[item] if obs_anchors is not None else None
        is_multi_modal = obs_image is not None

        _obs_anchor = torch_to_numpy(obs_anchor, is_object=True) if isinstance(obs_anchor, torch.Tensor) else obs_anchor

        # Build chat structure
        # obs_content = raw_prompt[0]['content']
        # if '<image>' in obs_content: 
        #     obs_content = obs_content.replace('<image>', '')

        # Build chat structure
        obs_content = ''
        if obs_text is not None:
            obs_content += obs_text
        else:
            print(f"Warning: No text observation found!")

        
        if not self.not_use_template:
            chat = np.array([{
                "content": obs_content,
                "role": "user",
            }])
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat,
                add_generation_prompt=True,
                tokenize=False,
                **apply_chat_template_kwargs
            )
        else:
            #############
            chat = np.array([obs_content])
            prompt_with_chat_template = obs_content
            ############

        assistant_prefix = self.config.data.get("assistant_prefix", "")
        if assistant_prefix:
            # convert literal \n (from shell/yaml) to real newline
            assistant_prefix = assistant_prefix.replace("\\n", "\n")
            prompt_with_chat_template += assistant_prefix
            if item == 0:
                print(f"[assistant_prefix] full prompt:\n{prompt_with_chat_template}")
        
        # Apply chat template
        
        
        # Initialize return dict
        row_dict = {}
        
        # Process multimodal data
        if is_multi_modal:
            # Replace image placeholder with vision tokens
            raw_prompt = prompt_with_chat_template.replace('<image>', '<|vision_start|><|image_pad|><|vision_end|>')
            row_dict['multi_modal_data'] = {'image': [process_image(obs_image)]}
            image_inputs = self.processor.image_processor(row_dict['multi_modal_data']['image'], return_tensors='pt')
            image_grid_thw = image_inputs['image_grid_thw']
            row_dict['multi_modal_inputs'] = {key: val for key, val in image_inputs.items()}
            if image_grid_thw is not None:
                merge_length = self.processor.image_processor.merge_size**2
                index = 0
                while '<image>' in prompt_with_chat_template:
                    prompt_with_chat_template = prompt_with_chat_template.replace(
                        '<image>',
                        '<|vision_start|>' + '<|placeholder|>' * (image_grid_thw[index].prod() // merge_length) +
                        '<|vision_end|>',
                        1,
                    )
                    index += 1

                prompt_with_chat_template = prompt_with_chat_template.replace('<|placeholder|>',
                                                                                self.processor.image_token)

        else:
            raw_prompt = prompt_with_chat_template
        
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=prompt_with_chat_template,
                                                                            tokenizer=self.tokenizer,
                                                                            max_length=self.config.data.max_prompt_length,
                                                                            pad_token_id=self.tokenizer.pad_token_id,
                                                                            left_pad=True,
                                                                            truncation=self.config.data.truncation,)
        
        

        if is_multi_modal:

            if "Qwen3VLProcessor" in self.processor.__class__.__name__:
                from verl.models.transformers.qwen3_vl import get_rope_index
            else:
                from verl.models.transformers.qwen2_vl import get_rope_index

            vision_position_ids = get_rope_index(
                self.processor,
                input_ids=input_ids[0],
                image_grid_thw=image_grid_thw,
                attention_mask=attention_mask[0],
            )  # (3, seq_length)
            valid_mask = attention_mask[0].bool()
            text_position_ids = torch.ones((1, len(input_ids[0])), dtype=torch.long)
            text_position_ids[0, valid_mask] = torch.arange(valid_mask.sum().item())
            position_ids = [torch.cat((text_position_ids, vision_position_ids), dim=0)]  # (1, 4, seq_length)
        else:
            position_ids = compute_position_id_with_mask(attention_mask)

        raw_prompt_ids = self.tokenizer.encode(raw_prompt, add_special_tokens=False)
        if len(raw_prompt_ids) > self.config.data.max_prompt_length:
            if self.config.data.truncation == "left":
                raw_prompt_ids = raw_prompt_ids[-self.config.data.max_prompt_length :]
            elif self.config.data.truncation == "right":
                raw_prompt_ids = raw_prompt_ids[: self.config.data.max_prompt_length]
            elif self.config.data.truncation == "middle":
                left_half = self.config.data.max_prompt_length // 2
                right_half = self.config.data.max_prompt_length - left_half
                raw_prompt_ids = raw_prompt_ids[:left_half] + raw_prompt_ids[-right_half:]
            elif self.config.data.truncation == "error":
                raise RuntimeError(f"Prompt length {len(raw_prompt_ids)} is longer than {self.config.data.max_prompt_length}.")

        # Build final output dict
        row_dict.update({
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt_ids': raw_prompt_ids,
            'built_prompt': prompt_with_chat_template,
            'anchor_obs': _obs_anchor,
            'index': item,
            'data_source': data_source
        })

        if self.config.data.get('return_raw_chat', False):
            row_dict['raw_prompt'] = chat.tolist()
        
        return row_dict

    def preprocess_batch(
        self,
        gen_batch: DataProto, 
        obs: Dict, 
    ) -> DataProto:
        """
        Process a batch of observation samples, converting environment observations into model-processable format.
        
        Parameters:
            gen_batch (DataProto): Batch data containing original prompts
            obs (Dict): Environment observation dictionary
                - 'text' (None or List[str]): Text observation data
                - 'image' (np.ndarray or torch.Tensor): Image observation data
                - 'anchor' (None or Any): Anchor observation without any histories or additional info. (for GiGPO only).
        
        Returns:
            DataProto: Contains processed batch data with preserved metadata
        """
        batch_size = len(gen_batch.batch['input_ids'])
        processed_samples = []
        
        # Process each sample in parallel
        for item in range(batch_size):
            # Extract per-sample observations
            processed = self.preprocess_single_sample(
                item=item,
                gen_batch=gen_batch,
                obs=obs,
            )
            processed_samples.append(processed)
        
        # Aggregate batch data
        batch = collate_fn(processed_samples)
        
        # Create DataProto with preserved metadata
        new_batch = DataProto.from_single_dict(
            data=batch,
            meta_info=gen_batch.meta_info
        )

        return new_batch


    def gather_rollout_data(
            self,
            total_batch_list: List[List[Dict]],
            episode_rewards: np.ndarray,
            episode_lengths: np.ndarray,
            success: Dict[str, np.ndarray],
            traj_uid: np.ndarray,
            tool_callings: np.ndarray,
            ) -> DataProto:
        """
        Collect and organize trajectory data, handling batch size adjustments to meet parallel training requirements.
        
        Parameters:
            total_batch_list (List[List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
            tool_callings (np.ndarray): Number of tool callings for each environment
        Returns:
            DataProto: Collected and organized trajectory data
        """
        batch_size = len(total_batch_list)

        # Log per-episode success values BEFORE averaging (critical for offline analysis)
        for key, value in success.items():
            arr = np.array(value)
            n_won = int(arr.sum())
            print(f"[GATHER] {key}: per_episode={arr.tolist()[:20]}{'...' if len(arr)>20 else ''} "
                  f"n_won={n_won}/{len(arr)} mean={arr.mean():.4f}")

        success_rate = {}
        for key, value in success.items():
            success_rate[key] = np.mean(value)
        
        effective_batch = []
        for bs in range(batch_size):
            # sum the rewards for each data in total_batch_list[bs]
            for data in total_batch_list[bs]:
                assert traj_uid[bs] == data['traj_uid'], "data is not from the same trajectory"
                if data['active_masks']:
                    # episode_rewards
                    data['episode_rewards'] = episode_rewards[bs]
                    # episode_lengths
                    data['episode_lengths'] = episode_lengths[bs]
                    # tool_callings
                    data['tool_callings'] = tool_callings[bs]
                    # success_rate
                    for key, value in success_rate.items():
                        data[key] = value

                    effective_batch.append(data)
            
        # Convert trajectory data to DataProto format
        gen_batch_output = DataProto.from_single_dict(
            data=collate_fn(effective_batch)
        )
        return gen_batch_output

    def vanilla_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Collects trajectories through parallel agent-environment agent_loop.
        Parameters:
            gen_batch (DataProto): Initial batch with prompts to start the agent_loop
            actor_rollout_wg (WorkerGroup): Worker group containing the actor model for policy decisions
            envs (EnvironmentManagerBase): Environment manager containing parallel environment instances
        
        Returns:
            total_batch_list (List[Dict]): List of trajectory data for each environment
            episode_rewards (np.ndarray): Total rewards for each environment
            episode_lengths (np.ndarray): Total steps for each environment
            success (Dict[str, np.ndarray]): Success samples for each environment
            traj_uid (np.ndarray): Trajectory unique identifiers
        """

        batch_size = len(gen_batch.batch)

        # Initial observations from the environment
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.get('env_kwargs', None))

        lenght_obs = len(obs['text']) if obs['text'] is not None else len(obs['image'])
        assert len(gen_batch.batch) == lenght_obs, f"gen_batch size {len(gen_batch.batch)} does not match obs size {lenght_obs}"
        
        if self.config.env.rollout.n > 0: # env grouping
            uid_batch = []
            for i in range(batch_size):
                if i % self.config.env.rollout.n == 0:
                    uid = str(uuid.uuid4())
                uid_batch.append(uid)
            uid_batch = np.array(uid_batch, dtype=object)
        else: # no env grouping, set all to the same uid
            uid = str(uuid.uuid4())
            uid_batch = np.array([uid for _ in range(len(gen_batch.batch))], dtype=object)
        is_done = np.zeros(batch_size, dtype=bool)
        traj_uid = np.array([str(uuid.uuid4()) for _ in range(batch_size)], dtype=object)
        total_batch_list = [[] for _ in range(batch_size)]
        total_infos = [[] for _ in range(batch_size)]
        episode_lengths = np.zeros(batch_size, dtype=np.float32)
        episode_rewards = np.zeros(batch_size, dtype=np.float32)
        tool_callings = np.zeros(batch_size, dtype=np.float32)
        total_generated_tokens = 0
        inference_time_s = 0.0
        # 每个 env/episode 的起始时间与完成标记，用于打印 vanilla 路径下 episode 级 wall time（仅在 eval 时启用）。
        episode_start_times = np.full(batch_size, time.perf_counter(), dtype=np.float64)
        episode_finished = np.zeros(batch_size, dtype=bool)
        # Trajectory collection loop
        for _step in range(self.config.env.max_steps):
            active_masks = np.logical_not(is_done)
            active_indices = np.where(active_masks)[0]  # indices of still-active envs

            batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)

            batch_keys_to_pop = ["input_ids", "attention_mask", "position_ids"]
            non_tensor_batch_keys_to_pop = ["raw_prompt_ids"]
            if "multi_modal_data" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("multi_modal_data")
            if "raw_prompt" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("raw_prompt")
            if "tools_kwargs" in batch.non_tensor_batch:
                non_tensor_batch_keys_to_pop.append("tools_kwargs")
            batch_input = batch.pop(
                batch_keys=batch_keys_to_pop,
                non_tensor_batch_keys=non_tensor_batch_keys_to_pop,
            )

            batch_input.meta_info = gen_batch.meta_info

            # ── Active-only generation: skip done slots to save SGLang compute ──
            # When all envs are still active (first step), take the fast path directly.
            if active_masks.all():
                batch_input_active = batch_input
            else:
                batch_input_active = batch_input[active_indices]

            # pad to be divisible by dp_size
            batch_input_padded, pad_size = pad_dataproto_to_divisor(batch_input_active, actor_rollout_wg.world_size)
            t0 = time.perf_counter()
            batch_output_padded = actor_rollout_wg.generate_sequences(batch_input_padded)
            generation_elapsed_s = time.perf_counter() - t0
            inference_time_s += generation_elapsed_s
            batch_output_active = unpad_dataproto(batch_output_padded, pad_size=pad_size)
            responses = batch_output_active.batch["responses"]
            response_attention_mask = batch_output_active.batch["attention_mask"][:, -responses.size(1):]
            generated_tokens = int(response_attention_mask.sum().item())
            total_generated_tokens += generated_tokens

            # Scatter active outputs back to full-batch positions.
            # Done slots get zero (pad) responses — they won't be used for training
            # (active_masks=False) and env.step handles them gracefully.
            if active_masks.all():
                batch_output = batch_output_active
            else:
                resp_len = batch_output_active.batch["responses"].size(1)
                device = batch_output_active.batch["responses"].device
                # Build a full-size output by creating zero tensors for done slots,
                # then writing active outputs into the corresponding positions.
                full_tensors = {}
                for key in batch_output_active.batch.keys():
                    t = batch_output_active.batch[key]
                    full_t = torch.zeros(
                        batch_size, *t.shape[1:], dtype=t.dtype, device=device
                    )
                    full_t[active_indices] = t
                    full_tensors[key] = full_t
                full_non_tensors = {}
                for key, arr in batch_output_active.non_tensor_batch.items():
                    full_arr = np.empty((batch_size, *arr.shape[1:]), dtype=arr.dtype)
                    full_arr[active_indices] = arr
                    full_non_tensors[key] = full_arr
                batch_output = DataProto.from_dict(
                    tensors=full_tensors,
                    non_tensors=full_non_tensors,
                    meta_info=batch_output_active.meta_info,
                )

            batch.non_tensor_batch['uid'] = uid_batch
            batch.non_tensor_batch['traj_uid'] = traj_uid

            batch = batch.union(batch_output)
            
            text_actions = self.tokenizer.batch_decode(batch.batch['responses'], skip_special_tokens=True)
            batch.non_tensor_batch['raw_response'] = np.array(text_actions, dtype=object)

            next_obs, rewards, dones, infos = envs.step(text_actions)

            # 在 vanilla 路径中检测 episode 结束，并打印该 episode 的 wall time（仅在 eval / validate 时打印）。
            dones_np = np.array(dones, dtype=bool)
            newly_done = np.logical_and(~episode_finished, dones_np)
            if np.any(newly_done) and gen_batch.meta_info.get("validate", False):
                now = time.perf_counter()
                for i in np.where(newly_done)[0]:
                    elapsed = now - episode_start_times[i]
                    print(f"[Episode {i}] finished at step={_step}, wall_time={elapsed:.2f}s")
            episode_finished = np.logical_or(episode_finished, dones_np)

            if len(rewards.shape) == 2:
                rewards = rewards.squeeze(1)
            if len(dones.shape) == 2:
                # dones is numpy, delete a dimension
                dones = dones.squeeze(1)

            if 'is_action_valid' in infos[0]:
                batch.non_tensor_batch['is_action_valid'] = np.array([info['is_action_valid'] for info in infos], dtype=bool)
            else:
                batch.non_tensor_batch['is_action_valid'] = np.ones(batch_size, dtype=bool)

            if 'tool_calling' in infos[0]:
                tool_callings[active_masks] += np.array([info['tool_calling'] for info in infos], dtype=np.float32)[active_masks]
            else:
                # environments without explicit tool_calling field (e.g. alfworld): each active step counts as 1
                tool_callings[active_masks] += 1
            
            # Extract raw_messages and tools from infos for SFT data generation.
            # Always write these keys (defaulting to []) so every step's data has a consistent key set,
            # which prevents length mismatches in collate_fn when only some steps populate these fields.
            batch.non_tensor_batch['raw_messages'] = np.array([info.get('raw_messages', []) for info in infos], dtype=object)
            batch.non_tensor_batch['tools'] = np.array([info.get('tools', []) for info in infos], dtype=object)
            batch.non_tensor_batch['episode_id'] = np.array(
                [info.get("episode_id") for info in infos], dtype=object
            )
            
            # Create reward tensor, only assign rewards for active environments
            # episode_rewards += torch_to_numpy(rewards) * torch_to_numpy(active_masks)
            episode_rewards[active_masks] += torch_to_numpy(rewards)[active_masks]
            episode_lengths[active_masks] += 1

            assert len(rewards) == batch_size, f"env should return rewards for all environments, got {len(rewards)} rewards for {batch_size} environments"
            batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
            batch.non_tensor_batch['active_masks'] = torch_to_numpy(active_masks, is_object=True)
            
            # Update episode lengths for active environments
            batch_list: list[dict] = to_list_of_dict(batch)

            for i in range(batch_size):
                total_batch_list[i].append(batch_list[i])
                total_infos[i].append(infos[i])

            # Update done states
            is_done = np.logical_or(is_done, dones)
                
            # Update observations for next step
            obs = next_obs

            # Break if all environments are done
            if is_done.all():
                break

        # 若在 validate 下某些 episode 因跑满 max_steps 而结束（env 未返回 done），补打 wall_time
        if gen_batch.meta_info.get("validate", False):
            now = time.perf_counter()
            for i in range(batch_size):
                if not episode_finished[i]:
                    elapsed = now - episode_start_times[i]
                    last_step = self.config.env.max_steps - 1
                    print(f"[Episode {i}] finished at step={last_step} (max_steps), wall_time={elapsed:.2f}s")
        
        success: Dict[str, np.ndarray] = envs.success_evaluator(
                    total_infos=total_infos,
                    total_batch_list=total_batch_list,
                    episode_rewards=episode_rewards,
                    episode_lengths=episode_lengths,
                    )

        # Print per-episode summary for easier debugging (mainly used in evaluation).
        is_validate = gen_batch.meta_info.get("validate", False)
        if is_validate:
            for i in range(batch_size):
                metrics = ", ".join(f"{k}={success[k][i]:.4f}" for k in success.keys() if i < len(success[k]))
                print(f"[Episode {i}] reward={episode_rewards[i]:.4f}, length={episode_lengths[i]:.0f}, {metrics}")

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_generated_tokens, inference_time_s

    def speculative_multi_turn_loop(
            self,
            gen_batch: DataProto,
            actor_rollout_wg,
            envs: EnvironmentManagerBase,
            speculative_k: int = 3,
            judge_fn=None,
            ) -> DataProto:
        """
        k-step speculative planning: draft runs k steps ahead, target verifies in batch.
        Pipeline parallelism: draft generates sequentially with env interaction,
        target verifies all k steps in one GPU batch call.

        Why speculative can be slower than vanilla when draft often misses:
        - No true env rollback: save_state/restore_state only restores the manager's
          memory/pre_text_obs; the underlying env (e.g. Ray workers) is not reverted,
          so replay runs from post-draft state (semantics may still be correct if
          we only use the replayed trajectory for logging/reward).
        So even when degenerating to 1 step per round, wall time ≈ target_time + 2 x env.step, vs vanilla's target_time + env.step.
        """
        if judge_fn is None:
            judge_fn = _webshop_action_judge

        rollout_cfg = self.config.actor_rollout_ref.rollout

        # Draft model via API (OpenAI-compatible completions).
        draft_api_base = rollout_cfg.get("speculative_api_base", "http://localhost:8001/v1")
        draft_model = rollout_cfg.get("speculative_api_model", "Qwen/Qwen3.5-4B")
        draft_client = OpenAI(base_url=draft_api_base, api_key="EMPTY")

        # Target via API (OpenAI-compatible completions). Default to local SGLang server.
        target_api_base = rollout_cfg.get("speculative_target_api_base", "http://localhost:6767/v1")
        target_model = rollout_cfg.get("speculative_target_api_model", rollout_cfg.get("api_model", ""))
        target_client = OpenAI(base_url=target_api_base, api_key="EMPTY")

        def call_draft_single(text, temperature, max_tokens):
            try:
                return draft_client.completions.create(
                    model=draft_model, prompt=text, temperature=temperature, max_tokens=max_tokens
                ).choices[0].text
            except Exception:
                return ""

        def call_target_single(text, temperature, max_tokens):
            try:
                return target_client.completions.create(
                    model=target_model, prompt=text, temperature=temperature, max_tokens=max_tokens
                ).choices[0].text
            except Exception:
                return ""

        batch_size = len(gen_batch.batch)
        assert batch_size == 1, (
            "speculative_multi_turn_loop currently assumes batch_size==1 for simplicity; "
            f"got batch_size={batch_size}"
        )
        obs, infos = envs.reset(kwargs=gen_batch.non_tensor_batch.get('env_kwargs', None))

        uid = str(uuid.uuid4())
        uid_batch = np.array([uid], dtype=object)
        is_done = False
        traj_uid = np.array([str(uuid.uuid4())], dtype=object)
        total_batch_list = [[]]
        total_infos = [[]]
        episode_lengths = np.zeros(1, dtype=np.float32)
        episode_rewards = np.zeros(1, dtype=np.float32)
        tool_callings = np.zeros(1, dtype=np.float32)
        # The start time and completion flag for each env/episode, used to print episode-level wall time.
        episode_start_times = np.full(1, time.perf_counter(), dtype=np.float64)
        episode_finished = np.zeros(1, dtype=bool)
        total_generated_tokens = 0
        inference_time_s = 0.0

        # Get config for draft generation
        is_validate = gen_batch.meta_info.get("validate", False)
        temperature = rollout_cfg.val_kwargs.temperature if is_validate else rollout_cfg.get("temperature", 1.0)
        max_tokens = rollout_cfg.response_length

        total_decoded_steps = 0
        total_rounds = 0

        # Terminate by total environment steps (episode_lengths), not by number of SpecRounds.
        _outer_step = 0
        while (not is_done) and (int(episode_lengths[0]) < self.config.env.max_steps):
            remaining_steps = self.config.env.max_steps - int(episode_lengths[0])
            if remaining_steps <= 0:
                break

            print(f"[SpecRound] start, round={_outer_step}")

            checkpoint = envs.save_state()
            init_obs = obs
            draft_actions: list[str] = []
            k_this_round = min(int(speculative_k), int(remaining_steps))
            accepted_step = 0   # number of steps where draft == target (match count)
            correction: str | None = None

            # Two-thread design: 
            # thread 1 = draft+env pipeline (interruptible by abort);
            # thread 2 = listen for target returns, judge as soon as one returns, set abort to stop thread 1.
            target_pool = ThreadPoolExecutor(max_workers=k_this_round)
            draft_pool = ThreadPoolExecutor(max_workers=1)  # one draft request per step
            in_flight: list[tuple[int, Future]] = []  # (step_j, future[str])
            round_lock = threading.Lock()
            abort_round_ref = [False]
            round_done_ref = [False]
            round_inference_time = [0.0]
            round_tokens = [0]
            target_submit_t: dict[int, float] = {}
            target_actions: list[str] = []  # target outputs collected in step order

            def target_listener():
                nonlocal accepted_step, correction
                # Must judge steps in order: skip steps that arrive out-of-order and
                # wait until all preceding steps have been judged first.
                next_judge_step = 0
                while True:
                    with round_lock:
                        if abort_round_ref[0]:
                            in_flight.clear()
                            break
                        if round_done_ref[0] and len(in_flight) == 0:
                            break
                        # Only pick the future for `next_judge_step` if it's done.
                        ready = [(j, f) for j, f in in_flight if j == next_judge_step and f.done()]
                        if ready:
                            for (j, f) in ready:
                                in_flight.remove((j, f))
                        else:
                            ready = []
                    if not ready:
                        time.sleep(0.01)
                        continue
                    # There is exactly one entry (step == next_judge_step).
                    j_step, fut = ready[0]
                    t_recv = time.perf_counter()
                    target_full = fut.result()

                    # Tokens-per-second for this target request.
                    submit_t = target_submit_t.get(j_step)
                    if submit_t is not None:
                        latency_s = max(1e-6, t_recv - submit_t)
                        gen_tokens = len(self.tokenizer.encode(target_full, add_special_tokens=False))
                        tps = gen_tokens / latency_s
                        with round_lock:
                            round_inference_time[0] += latency_s
                            round_tokens[0] += gen_tokens
                        print(
                            f"[TargetTPS] round={_outer_step}, step={j_step}, "
                            f"tokens={gen_tokens}, latency_s={latency_s:.3f}, tps={tps:.2f}"
                        )

                    # Always record target output in order (used by replay phase).
                    # Ordering is guaranteed by next_judge_step: listener only processes
                    # step j after all steps < j have been processed, so target_actions
                    # is always appended in strict step order.
                    target_actions.append(target_full)

                    # Wait until draft for this step exists (draft is usually faster), or round_done/abort.
                    wait_start = time.perf_counter()
                    while len(draft_actions) <= j_step and not abort_round_ref[0] and not round_done_ref[0]:
                        if time.perf_counter() - wait_start > 300:
                            print(f"[TargetListener] round={_outer_step}, step={j_step}, draft wait timeout")
                            break
                        time.sleep(0.01)

                    # Advance to the next step regardless of whether we can judge this one.
                    next_judge_step += 1

                    # Guard: skip judge if draft not available or a mismatch was already found at an earlier step.
                    draft_available = len(draft_actions) > j_step
                    if not draft_available or correction is not None:
                        if not draft_available:
                            print(f"[TargetListener] round={_outer_step}, step={j_step}, skip judge: draft not available")
                        continue
                    draft_full = draft_actions[j_step]
                    draft_a = _extract_action(draft_full)
                    target_a = _extract_action(target_full)
                    print(f"[Target] round={_outer_step}, step={j_step}, action={target_a}")
                    match = judge_fn(draft_a, target_a)
                    print(
                        f"[Judge] round={_outer_step}, step={j_step}, "
                        f"draft={draft_a}, target={target_a}, match={match}"
                    )
                    if match:
                        with round_lock:
                            accepted_step += 1
                    else:
                        with round_lock:
                            correction = target_full
                            abort_round_ref[0] = True

            listener_thread = threading.Thread(target=target_listener, daemon=True)
            listener_thread.start()

            for j in range(k_this_round):
                if is_done or abort_round_ref[0]:
                    break

                # Prepare target prompt for target model
                batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
                batch_keys = ["input_ids", "attention_mask", "position_ids"]
                non_tensor_keys = ["raw_prompt_ids"]
                if "multi_modal_data" in batch.non_tensor_batch:
                    non_tensor_keys.append("multi_modal_data")
                batch_input = batch.pop(batch_keys=batch_keys, non_tensor_batch_keys=non_tensor_keys)
                batch_input.meta_info = gen_batch.meta_info

                # Decode prompt text once; both target & draft are API calls.
                target_submit_t[j] = time.perf_counter()
                prompt_texts = [
                    self.tokenizer.decode(
                        ids.tolist() if hasattr(ids, "tolist") else list(ids),
                        skip_special_tokens=False,
                    )
                    for ids in batch_input.non_tensor_batch.get("raw_prompt_ids", [])
                ]
                prompt0 = prompt_texts[0] if prompt_texts else ""

                # Submit target model generation task (API)
                fut = target_pool.submit(call_target_single, prompt0, temperature, max_tokens)
                with round_lock:
                    in_flight.append((j, fut))

                # Submit draft model generation task (API)
                future_draft = draft_pool.submit(call_draft_single, prompt0, temperature, max_tokens)
                # Wait for draft model generation task to complete
                while not future_draft.done():
                    if abort_round_ref[0]:
                        break
                    time.sleep(0.01)
                if abort_round_ref[0]:
                    break
                # Get draft model generation result
                draft_full = str(future_draft.result(timeout=1))
                draft_actions.append(draft_full)
                act = _extract_action(draft_full)
                print(f"[Draft] round={_outer_step}, step={j}, action={act}")

                # If draft outputs the terminal action (buy now), skip env.step entirely.
                # The target listener will judge this step; if target also outputs buy now (match),
                # replay will execute it cleanly on the restored env (no premature session reset).
                # If target mismatches, restore & replay handles it normally.
                # Either way, no further draft steps are possible after a terminal action.
                if act is not None and act.lower() == "click[buy now]":
                    print(f"[Draft] round={_outer_step}, step={j}, terminal action detected, skipping env.step")
                    break

                # Step the environment with draft actions
                next_obs, rewards, dones, infos = envs.step([draft_full])
                if len(dones.shape) == 2:
                    dones = dones.squeeze(1)
                done0 = bool(np.array(dones, dtype=bool)[0])
                is_done = is_done or done0
                # Update obs so the next step's prompt reflects the new environment state.
                obs = next_obs

            # Signal listener we are done issuing work for this round. If we aborted,
            # listener will exit immediately without waiting for remaining targets.
            round_done_ref[0] = True
            listener_thread.join(timeout=120)
            inference_time_s += round_inference_time[0]
            total_generated_tokens += round_tokens[0]

            restore_start = time.perf_counter()
            # ———— Restore and replay ————
            # effective_steps = number of target outputs collected:
            #   all-match case:  len == k (all steps accepted)
            #   mismatch case:   len == accepted_steps + 1 (correction included)
            #   terminal break:  len == steps actually submitted before break
            # Also capped by remaining_steps to never exceed env.max_steps.
            effective_steps = min(len(target_actions), remaining_steps)

            envs.restore_state(checkpoint)
            obs = init_obs
            is_done_replay = False

            for j in range(effective_steps):
                # Always use target model's output for replay.
                # target_actions is populated in step order by target_listener.
                action_this_step = target_actions[j]

                # Rebuild batch for this step (only for logging / trajectory collection).
                batch = self.preprocess_batch(gen_batch=gen_batch, obs=obs)
                batch.meta_info = gen_batch.meta_info

                # Synthesize prompts / responses from executed actions so that downstream
                # code (e.g. validation logging / reward fn) can decode them.
                resp_max_len = self.config.data.max_response_length
                resp_tensor = torch.full(
                    (batch_size, resp_max_len),
                    self.tokenizer.pad_token_id,
                    dtype=torch.long,
                )
                if action_this_step:
                    token_ids = self.tokenizer.encode(action_this_step, add_special_tokens=False)
                    token_ids = token_ids[:resp_max_len]
                    if token_ids:
                        resp_tensor[0, : len(token_ids)] = torch.tensor(token_ids, dtype=torch.long)
                # For replayed steps we approximate `prompts` with current input_ids.
                batch.batch["prompts"] = batch.batch["input_ids"].clone()
                batch.batch["responses"] = resp_tensor

                batch.non_tensor_batch['uid'] = uid_batch
                batch.non_tensor_batch['traj_uid'] = traj_uid
                batch.non_tensor_batch['raw_response'] = np.array([action_this_step], dtype=object)

                next_obs, rewards, dones, infos = envs.step([action_this_step])
                if len(rewards.shape) == 2:
                    rewards = rewards.squeeze(1)
                if len(dones.shape) == 2:
                    dones = dones.squeeze(1)

                active = not is_done_replay
                if active:
                    episode_rewards[0] += float(torch_to_numpy(rewards)[0])
                    episode_lengths[0] += 1
                    tool_callings[0] += float(infos[0].get('tool_calling', 1))

                batch.non_tensor_batch['rewards'] = torch_to_numpy(rewards, is_object=True)
                batch.non_tensor_batch['active_masks'] = np.array([active], dtype=object)
                batch.non_tensor_batch['is_action_valid'] = np.array([infos[0].get('is_action_valid', True)], dtype=bool)

                batch_list = to_list_of_dict(batch)
                total_batch_list[0].append(batch_list[0])
                total_infos[0].append(infos[0])

                newly_done_replay = (not episode_finished[0]) and bool(np.array(dones, dtype=bool)[0])
                if newly_done_replay:
                    now = time.perf_counter()
                    elapsed = now - episode_start_times[0]
                    print(f"[SpecEpisode] finished at round={_outer_step}, step={j}, wall_time={elapsed:.2f}s")
                    episode_finished[0] = True
                is_done_replay = is_done_replay or bool(np.array(dones, dtype=bool)[0])
                obs = next_obs

            restore_end = time.perf_counter()
            restore_time = restore_end - restore_start
            print(f"[Restore] round={_outer_step}, restore_time={restore_time:.2f}s")
            is_done = is_done_replay

            # One speculative round finishes here.
            # For each env, effective_steps = accepted draft steps (+1 if corrected by target).
            # Minimum decoded steps per round is 1, maximum is speculative_k ([1, k]).
            total_tool_calls = int(tool_callings.sum())
            total_decoded_steps += effective_steps
            total_rounds += 1
            avg_steps = total_decoded_steps / total_rounds
            print(
                f"[SpecRound] end, round={_outer_step}, "
                f"decoded_steps={effective_steps}, avg_decoded_steps={avg_steps:.2f}, "
                f"tool_call_count={total_tool_calls}"
            )
            _outer_step += 1

        # If during validation some episodes end due to reaching max_steps (env does not return done), print wall_time as a supplement
        if gen_batch.meta_info.get("validate", False):
            now = time.perf_counter()
            for i in range(batch_size):
                if not episode_finished[i]:
                    elapsed = now - episode_start_times[i]
                    print(f"[SpecEpisode {i}] finished at round={_outer_step} (max_steps), wall_time={elapsed:.2f}s")

        success = envs.success_evaluator(
            total_infos=total_infos,
            total_batch_list=total_batch_list,
            episode_rewards=episode_rewards,
            episode_lengths=episode_lengths,
        )

        # Print per-episode summary for speculative evaluation.
        is_validate = gen_batch.meta_info.get("validate", False)
        if is_validate:
            for i in range(batch_size):
                metrics = ", ".join(f"{k}={success[k][i]:.4f}" for k in success.keys())
                print(f"[SpecEpisode {i}] reward={episode_rewards[i]:.4f}, length={episode_lengths[i]:.0f}, {metrics}")

        return total_batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, total_generated_tokens, inference_time_s

    def dynamic_multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            ) -> DataProto:
        """
        Conduct dynamic rollouts until a target batch size is met. 
        Keeps sampling until the desired number of effective trajectories is collected.
        Adopted from DAPO (https://arxiv.org/abs/2503.14476)

        Args:
            gen_batch (DataProto): Initial batch for rollout.
            actor_rollout_wg: Actor model workers for generating responses.
            envs (EnvironmentManagerBase): Environment manager instance.

        Returns:
            total_batch_list (List[Dict]): Complete set of rollout steps.
            total_episode_rewards (np.ndarray): Accumulated rewards.
            total_episode_lengths (np.ndarray): Lengths per episode.
            total_success (Dict[str, np.ndarray]): Success metrics.
            total_traj_uid (np.ndarray): Trajectory IDs.
        """
        total_batch_list = []
        total_episode_rewards = []
        total_episode_lengths = []
        total_success = []
        total_traj_uid = []
        total_tool_callings = []
        total_generated_tokens = 0
        total_inference_time_s = 0.0
        try_count: int = 0
        max_try_count = self.config.algorithm.filter_groups.max_num_gen_batches

        while len(total_batch_list) < self.config.data.train_batch_size * self.config.env.rollout.n and try_count < max_try_count:

            if len(total_batch_list) > 0:
                print(f"valid num={len(total_batch_list)} < target num={self.config.data.train_batch_size * self.config.env.rollout.n}. Keep generating... ({try_count}/{max_try_count})")
            try_count += 1

            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings, gen_tokens, inf_time_s = self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            total_generated_tokens += gen_tokens
            total_inference_time_s += inf_time_s
            batch_list, episode_rewards, episode_lengths, success, traj_uid, tool_callings = filter_group_data(batch_list=batch_list, 
                                                                                                episode_rewards=episode_rewards, 
                                                                                                episode_lengths=episode_lengths, 
                                                                                                success=success, 
                                                                                                traj_uid=traj_uid, 
                                                                                                tool_callings=tool_callings, 
                                                                                                config=self.config,
                                                                                                last_try=(try_count == max_try_count),
                                                                                                )
            
            total_batch_list += batch_list
            total_episode_rewards.append(episode_rewards)
            total_episode_lengths.append(episode_lengths)
            total_success.append(success)
            total_traj_uid.append(traj_uid)
            total_tool_callings.append(tool_callings)

        total_episode_rewards = np.concatenate(total_episode_rewards, axis=0)
        total_episode_lengths = np.concatenate(total_episode_lengths, axis=0)
        total_success = {key: np.concatenate([success[key] for success in total_success], axis=0) for key in total_success[0].keys()}
        total_traj_uid = np.concatenate(total_traj_uid, axis=0)
        total_tool_callings = np.concatenate(total_tool_callings, axis=0)

        return total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, total_tool_callings, total_generated_tokens, total_inference_time_s

    def multi_turn_loop(
            self,
            gen_batch: DataProto, 
            actor_rollout_wg, 
            envs: EnvironmentManagerBase,
            is_train: bool = True,
            speculative: bool = False,
            speculative_k: int = 3,
            ) -> DataProto:
        """
        Select and run the appropriate rollout loop (dynamic, vanilla, or speculative).

        Args:
            gen_batch (DataProto): Initial prompt batch.
            actor_rollout_wg: Actor model workers.
            envs (EnvironmentManagerBase): Environment manager for interaction.
            is_train (bool): Whether in training mode (affects dynamic sampling).
            speculative (bool): Whether to use speculative planning.
            speculative_k (int): Number of lookahead steps for speculative planning.

        Returns:
            DataProto: Final collected trajectory data with metadata.
        """
        if is_train:
            gen_batch = gen_batch.repeat(repeat_times=self.config.env.rollout.n, interleave=True)

        overlong_metrics = {
            "rollout/overlong_filter_ratio": 0.0,
            "rollout/overlong_filter_n_drop": 0,
            "rollout/overlong_filter_n_keep": 0,
            "rollout/overlong_eff_len_mean": 0.0,
        }
            
        # Select loop based on mode
        if speculative and not is_train:
            # Speculative Planning (eval only for now)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_generated_tokens, inference_time_s = \
                self.speculative_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
                speculative_k=speculative_k,
            )
        elif self.config.algorithm.filter_groups.enable and is_train:
            # Dynamic Sampling (for DAPO and Dynamic GiGPO)
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, _total_gen_tokens, _total_inf_time_s = \
                self.dynamic_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )
            total_generated_tokens, inference_time_s = 0, 0.0
        else:
            # Vanilla Sampling
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, total_generated_tokens, inference_time_s = \
                self.vanilla_multi_turn_loop(
                gen_batch=gen_batch,
                actor_rollout_wg=actor_rollout_wg,
                envs=envs,
            )

        # Overlong filter on the final collected samples (train only; shared by vanilla/dynamic).
        if is_train:
            total_batch_list, total_episode_rewards, total_episode_lengths, total_success, total_traj_uid, totoal_tool_callings, overlong_metrics = filter_overlong_response_data(
                batch_list=total_batch_list,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
                success=total_success,
                traj_uid=total_traj_uid,
                tool_callings=totoal_tool_callings,
                config=self.config,
                last_try=False,
            )

        assert len(total_batch_list) == len(total_episode_rewards)
        assert len(total_batch_list) == len(total_episode_lengths)
        assert len(total_batch_list) == len(total_traj_uid)
        assert len(total_batch_list) == len(totoal_tool_callings)

        if self.config.trainer.get("save_trajectory_data", False):
            phase = "train" if is_train else "val"
            global_step = None
            if gen_batch.meta_info:
                global_step = gen_batch.meta_info.get("global_step")
            output_dir = resolve_trajectory_output_dir(
                self.config.trainer.default_local_dir,
                phase=phase,
                global_step=global_step,
            )
            TrajectoryDataWriter(str(output_dir), phase=phase).write_episodes(
                total_batch_list=total_batch_list,
                traj_uid=total_traj_uid,
                success=total_success,
                episode_rewards=total_episode_rewards,
                episode_lengths=total_episode_lengths,
            )

        # Create trajectory data
        gen_batch_output: DataProto = self.gather_rollout_data(
            total_batch_list=total_batch_list,
            episode_rewards=total_episode_rewards,
            episode_lengths=total_episode_lengths,
            success=total_success,
            traj_uid=total_traj_uid,
            tool_callings=totoal_tool_callings,
        )
        gen_batch_output.meta_info = gen_batch_output.meta_info or {}
        if is_train:
            gen_batch_output.meta_info.update(overlong_metrics)
        if not is_train:
            gen_batch_output.meta_info["val_total_generated_tokens"] = total_generated_tokens
            gen_batch_output.meta_info["val_inference_time_s"] = inference_time_s
        
        return gen_batch_output
