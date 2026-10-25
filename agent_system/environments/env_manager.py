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

import os
import time
from collections import defaultdict
from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase, to_numpy
from agent_system.environments.prompts import *
from agent_system.memory import SearchMemory, SimpleMemory


def parse_gamefile(infos):
    gamefile = []
    for info in infos:
        if "extra.gamefile" in info:
            gamefile.append(info["extra.gamefile"])
        else:
            gamefile.append(None)
    return gamefile


def set_gamefile(infos, gamefile):
    for i in range(len(infos)):
        if "extra.gamefile" in infos[i]:
            infos[i]["extra.gamefile"] = gamefile[i]
        else:
            infos[i]["extra.gamefile"] = None
    return infos


def attach_episode_ids(infos: List[Dict[str, Any]], episode_ids: List[Optional[str]]) -> List[Dict[str, Any]]:
    for i, info in enumerate(infos):
        if i < len(episode_ids) and episode_ids[i]:
            info["episode_id"] = episode_ids[i]
    return infos


class SearchEnvironmentManager(EnvironmentManagerBase):
    """
    EnvironmentManager for SearchEnv.
    """

    def __init__(self, envs, projection_f, config):
        self.memory = SearchMemory()
        super().__init__(envs, projection_f, config)

        # Thinking instruction selection (mirrors WebshopEnvironmentManager logic).
        # Priority: explicit config > SDAR flag > SDAR auto-detect
        # > enable_thinking auto-detect > default.
        search_config = getattr(config.env, "search", None)
        custom_instruction = getattr(search_config, "thinking_instruction", None)
        use_sdar_prompt = getattr(search_config, "use_sdar_prompt", False)
        if custom_instruction is not None:
            self.thinking_instruction = custom_instruction
        elif use_sdar_prompt:
            self.thinking_instruction = SEARCH_SDAR_THINKING_INSTRUCTION
        else:
            self.thinking_instruction = SEARCH_DEFAULT_THINKING_INSTRUCTION

        # SDAR models use <reason> tags instead of <think> tags.
        model_path = getattr(config.actor_rollout_ref.model, "path", "")
        model_name = getattr(config.actor_rollout_ref.model, "name", "")
        is_sdar = "sdar" in model_name.lower() or "sdar" in model_path.lower()
        if is_sdar and self.thinking_instruction == SEARCH_DEFAULT_THINKING_INSTRUCTION:
            self.thinking_instruction = SEARCH_SDAR_THINKING_INSTRUCTION

        # enable_thinking=True: chat template auto-injects <think> prefix before generation,
        # so omit the explicit tag instruction to avoid duplicate <think>/<reason> blocks.
        enable_thinking = config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking", False)
        if enable_thinking and self.thinking_instruction in (
            SEARCH_DEFAULT_THINKING_INSTRUCTION,
            SEARCH_SDAR_THINKING_INSTRUCTION,
        ):
            self.thinking_instruction = SEARCH_DEFAULT_THINKING_INSTRUCTION_NO_TAGS

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        obs, infos = self.envs.reset(kwargs=kwargs)
        self.tasks = obs

        self.memory.reset(batch_size=len(obs))
        self._is_done = np.zeros(len(obs), dtype=bool)
        self.episode_ids = [info.get("episode_id") for info in infos]

        observations = {
            "text": self.build_text_obs(obs, init=True),
            "image": None,
            "anchor": obs.copy(),
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        raw_text_actions = list(text_actions)
        actions, valids = self.projection_f(text_actions)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store(
            {
                "search": actions,
                "information": next_obs,
            }
        )

        next_observations = {
            "text": self.build_text_obs(next_obs),
            "image": None,
            "anchor": next_obs.copy(),
        }

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        attach_episode_ids(infos, getattr(self, "episode_ids", []))

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        if not hasattr(self, "_is_done"):
            self._is_done = np.zeros(len(next_obs), dtype=bool)
        self._is_done = np.logical_or(self._is_done, dones)

        _printed = 0
        for _i in range(len(actions)):
            if _printed >= 5:
                break
            if not self._is_done[_i]:
                print(f"[DEBUG proj {_i}] valid={valids[_i]} | raw={repr(raw_text_actions[_i])} | extracted={repr(actions[_i])}")
                _printed += 1

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        postprocess_text_obs: List[str] = []

        if not init and self.config.env.history_length > 0:
            memory_ctx, _ = self.memory.fetch(
                self.config.env.history_length,
                obs_key="information",
                action_key="search",
            )

        for i in range(len(text_obs)):
            if init or self.config.env.history_length <= 0:
                obs_i = SEARCH_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    thinking_instruction=self.thinking_instruction,
                )
            else:
                obs_i = SEARCH_TEMPLATE.format(
                    task_description=self.tasks[i],
                    memory_context=memory_ctx[i],
                    step_count=len(self.memory[i]),
                    thinking_instruction=self.thinking_instruction,
                )
            postprocess_text_obs.append(obs_i)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)

                data_source = info.get("data_source")
                success[f"{data_source}_success_rate"].append(won_value)
                return  # Exit after finding the first active mask


class AlfWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)
        # Thinking instruction selection (mirrors WebshopEnvironmentManager / SearchEnvironmentManager logic).
        # Priority: explicit config > SDAR auto-detect > enable_thinking auto-detect > default.
        custom_instruction = None
        if hasattr(config.env, 'alfworld') and config.env.alfworld is not None:
            custom_instruction = getattr(config.env.alfworld, 'thinking_instruction', None)
        self.thinking_instruction = (
            custom_instruction if custom_instruction is not None
            else ALFWORLD_DEFAULT_THINKING_INSTRUCTION
        )

        # SDAR models use <reason> tags instead of <think> tags.
        model_path = getattr(config.actor_rollout_ref.model, "path", "")
        model_name = getattr(config.actor_rollout_ref.model, "name", "")
        is_sdar = "sdar" in model_name.lower() or "sdar" in model_path.lower()
        if is_sdar and self.thinking_instruction == ALFWORLD_DEFAULT_THINKING_INSTRUCTION:
            self.thinking_instruction = ALFWORLD_SDAR_THINKING_INSTRUCTION

        # enable_thinking=True: chat template auto-injects <think> prefix before generation,
        # so omit the explicit tag instruction to avoid duplicate <think>/<reason> blocks.
        enable_thinking = config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking", False)
        if enable_thinking and self.thinking_instruction in (
            ALFWORLD_DEFAULT_THINKING_INSTRUCTION,
            ALFWORLD_SDAR_THINKING_INSTRUCTION,
        ):
            self.thinking_instruction = ALFWORLD_DEFAULT_THINKING_INSTRUCTION_NO_TAGS

    def reset(self, kwargs):
        text_obs, image_obs, infos = self.envs.reset()
        self.gamefile = parse_gamefile(infos)
        self.episode_ids = [info.get("episode_id") for info in infos]
        # initialize the history buffer
        self.memory.reset(batch_size=len(text_obs))
        self._is_done = np.zeros(len(text_obs), dtype=bool)
        self.tasks = []
        self.pre_text_obs = text_obs
        self.extract_task(text_obs)

        full_text_obs = self.build_text_obs(
            text_obs, self.envs.get_admissible_commands, init=True
        )
        return {"text": full_text_obs, "image": image_obs, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        raw_text_actions = list(text_actions)
        actions, valids = self.projection_f(
            text_actions, self.envs.get_admissible_commands
        )
        text_obs, image_obs, rewards, dones, infos = self.envs.step(actions)
        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, self.envs.get_admissible_commands)
        if infos[0].get("extra.gamefile") is None:
            infos = set_gamefile(infos, self.gamefile)
        attach_episode_ids(infos, getattr(self, "episode_ids", []))

        # add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        next_observations = {
            "text": full_text_obs,
            "image": image_obs,
            "anchor": text_obs,
        }
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        if not hasattr(self, "_is_done"):
            self._is_done = np.zeros(len(text_obs), dtype=bool)
        self._is_done = np.logical_or(self._is_done, dones)

        _printed = 0
        for _i in range(len(actions)):
            if _printed >= 5:
                break
            if not self._is_done[_i]:
                print(f"[DEBUG proj {_i}] valid={valids[_i]} | raw={repr(raw_text_actions[_i])} | extracted={repr(actions[_i])}")
                _printed += 1

        return next_observations, rewards, dones, infos

    def extract_task(self, text_obs: List[str]):
        for obs in text_obs:
            task_start = obs.find("Your task is to: ")

            if task_start != -1:
                self.tasks.append(obs[task_start + len("Your task is to: ") :].strip())
            else:
                raise ValueError("Task description not found in text observation.")

    def build_text_obs(
        self,
        text_obs: List[str],
        admissible_actions: List[List[str]],
        init: bool = False,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length, obs_key="text_obs", action_key="action"
            )

        for i in range(len(text_obs)):
            # exclude 'help' in admissible_actions[i]
            reformatted_admissible_actions = "\n ".join(
                f"'{s}'" for s in admissible_actions[i] if s != "help"
            )

            if init or self.config.env.history_length <= 0:
                obs = ALFWORLD_TEMPLATE_NO_HIS.format(
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    thinking_instruction=self.thinking_instruction, # add custom prompt option
                )
            else:
                obs = ALFWORLD_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    admissible_actions=reformatted_admissible_actions,
                    thinking_instruction=self.thinking_instruction, # add custom prompt option
                )

            postprocess_text_obs.append(obs)
        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        # Find the last entry with active masks
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                success["success_rate"].append(won_value)

                # Process game file if it exists
                gamefile = info.get("extra.gamefile")
                if gamefile:
                    self._process_gamefile(gamefile, won_value, success)
                return  # Exit after finding the first active mask

    def _process_gamefile(self, gamefile, won_value, success):
        tasks = [
            "pick_and_place",
            "pick_two_obj_and_place",
            "look_at_obj_in_light",
            "pick_heat_then_place_in_recep",
            "pick_cool_then_place_in_recep",
            "pick_clean_then_place_in_recep",
        ]

        for task in tasks:
            if task in gamefile:
                success[f"{task}_success_rate"].append(won_value)
                break


class SokobanEnvironmentManager(EnvironmentManagerBase):
    ACTION_LOOKUP = {
        0: "Still",
        1: "Up",
        2: "Down",
        3: "Left",
        4: "Right",
    }

    def __init__(self, envs, projection_f, config):
        self.is_multi_modal = envs.mode == "rgb_array"
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        obs, infos = self.envs.reset()
        if self.is_multi_modal:
            obs = np.array(obs, obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode="tiny_rgb_array")
            observations = {
                "text": self.build_text_obs(infos, init=True),
                "image": obs,
                "anchor": obs,
            }
        else:
            self.pre_text_obs = obs
            observations = {
                "text": self.build_text_obs(infos, obs, init=True),
                "image": None,
                "anchor": obs,
            }
        self.memory.reset(batch_size=len(infos))
        return observations, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        next_obs, rewards, dones, infos = self.envs.step(actions)

        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        self.memory.store(
            {
                "text_obs": self.pre_text_obs,
                "action": [self.ACTION_LOOKUP[act] for act in actions],
            }
        )
        if self.is_multi_modal:
            next_obs = np.array(next_obs, next_obs[0].dtype)
            self.pre_text_obs = self.envs.render(mode="tiny_rgb_array")
            next_observations = {
                "text": self.build_text_obs(infos),
                "image": next_obs,
                "anchor": next_obs,
            }
        else:
            self.pre_text_obs = next_obs
            next_observations = {
                "text": self.build_text_obs(infos, next_obs),
                "image": None,
                "anchor": next_obs,
            }

        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(
        self, infos, text_obs: List[str] = None, init: bool = False
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []

        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length, obs_key="text_obs", action_key="action"
            )

        for i in range(len(infos)):
            if init or self.config.env.history_length <= 0:
                obs = (
                    SOKOBAN_VISUAL_TEMPLATE
                    if self.is_multi_modal
                    else SOKOBAN_TEMPLATE_NO_HIS.format(
                        current_observation=text_obs[i],
                    )
                )
            else:
                if self.is_multi_modal:
                    obs = SOKOBAN_VISUAL_TEMPLATE
                else:
                    obs = SOKOBAN_TEMPLATE.format(
                        step_count=len(self.memory[i]),
                        history_length=valid_lens[i],
                        action_history=memory_contexts[i],
                        current_step=len(self.memory[i]) + 1,
                        current_observation=text_obs[i],
                    )
            postprocess_text_obs.append(obs)

        return postprocess_text_obs


class GymCardEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        # infos = [None] * self.envs.num_envs
        observations = {
            "text": self.build_text_obs(infos),
            "image": obs,
            "anchor": obs.copy(),
        }

        return observations, infos

    def step(self, text_actions: List[str]):
        next_observations, rewards, dones, infos = super().step(text_actions)

        # add text observation to next_observations
        next_observations["text"] = self.build_text_obs(infos)
        next_observations["anchor"] = next_observations["image"].copy()

        return next_observations, rewards, dones, infos

    def build_text_obs(self, infos: Tuple[Dict] = None) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        for i in range(len(infos)):
            if "ezpoints" in self.config.env.env_name.lower():
                text_formula = (
                    "".join(str(element) for element in infos[i]["Formula"])
                    if infos[i] is not None
                    else ""
                )
                obs = GYM_CARDS_EZPOINTS_TEMPLATE.format(text_formula=text_formula)
            elif "points24" in self.config.env.env_name.lower():
                text_formula = (
                    "".join(str(element) for element in infos[i]["Formula"])
                    if infos[i] is not None
                    else ""
                )
                obs = GYM_CARDS_POINTS24_TEMPLATE.format(text_formula=text_formula)
            elif "numberline" in self.config.env.env_name.lower():
                obs = GYM_CARDS_NUMBERLINE_TEMPLATE
            elif "blackjack" in self.config.env.env_name.lower():
                obs = GYM_CARDS_BLACKJACK_TEMPLATE
            else:
                raise ValueError(f"Unsupported environment: {self.config.env.env_name}")
            postprocess_text_obs.append(obs)
        return postprocess_text_obs


class WebshopEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        self._goal_batch_idx = 0
        super().__init__(envs, projection_f, config)
        # Priority: explicit config > SDAR flag > SDAR auto-detect
        # > enable_thinking auto-detect > default.
        webshop_config = getattr(config.env, "webshop", None)
        custom_instruction = getattr(webshop_config, "thinking_instruction", None)
        if isinstance(custom_instruction, str) and custom_instruction.strip().lower() in ("", "null", "none", "~"):
            custom_instruction = None
        use_sdar_prompt = getattr(webshop_config, "use_sdar_prompt", False)
        if custom_instruction is not None:
            self.thinking_instruction = custom_instruction
        elif use_sdar_prompt:
            self.thinking_instruction = WEBSHOP_SDAR_THINKING_INSTRUCTION
        else:
            self.thinking_instruction = WEBSHOP_DEFAULT_THINKING_INSTRUCTION
        # SDAR models use <reason> tags instead of <think> tags.
        model_path = getattr(config.actor_rollout_ref.model, "path", "")
        model_name = getattr(config.actor_rollout_ref.model, "name", "")
        is_sdar = "sdar" in model_name.lower() or "sdar" in model_path.lower()
        if is_sdar and self.thinking_instruction == WEBSHOP_DEFAULT_THINKING_INSTRUCTION:
            self.thinking_instruction = WEBSHOP_SDAR_THINKING_INSTRUCTION
        # Switch to the matching instruction unless the user already set a custom one.
        _webshop_cfg = getattr(config.env, "webshop", None)
        if custom_instruction is None and getattr(_webshop_cfg, "thought_prefix_projection", False):
            self.thinking_instruction = WEBSHOP_THOUGHT_PREFIX_INSTRUCTION
        elif custom_instruction is None and getattr(_webshop_cfg, "action_only_projection", False):
            self.thinking_instruction = WEBSHOP_DEFAULT_THINKING_INSTRUCTION_NO_TAGS
        # When enable_thinking=True, chat template auto-injects <think> prefix before generation,
        # causing duplicate <think>/<reason> blocks if the prompt also requires them explicitly.
        enable_thinking = config.data.get("apply_chat_template_kwargs", {}).get("enable_thinking", False)
        if enable_thinking and self.thinking_instruction in (
            WEBSHOP_DEFAULT_THINKING_INSTRUCTION,
            WEBSHOP_SDAR_THINKING_INSTRUCTION,
        ):
            self.thinking_instruction = WEBSHOP_DEFAULT_THINKING_INSTRUCTION_NO_TAGS
        print(
            f"[WEBSHOP_PROMPT] use_sdar_prompt={use_sdar_prompt} "
            f"enable_thinking={enable_thinking} is_sdar={is_sdar} "
            f"thinking_instruction={self.thinking_instruction!r}",
            flush=True,
        )

    def reset(self, kwargs) -> Dict[str, Any]:
        obs, infos = self.envs.reset()
        group_n = getattr(self.envs, "group_n", 1)
        batch_infos = infos[::group_n]
        canonical_goal_ids = [
            info.get("canonical_goal_idx") for info in batch_infos
        ]
        sampled_goal_idxs = [info.get("goal_idx") for info in batch_infos]
        split = "train" if getattr(self.envs, "is_train", True) else "val"
        print(
            f"[WEBSHOP_GOALS] split={split} batch={self._goal_batch_idx} "
            f"canonical_goal_ids={canonical_goal_ids} "
            f"sampled_goal_idxs={sampled_goal_idxs}",
            flush=True,
        )
        self._goal_batch_idx += 1
        self.tasks = self.extract_task(obs)
        obs = self.format_obs(obs)
        self.episode_ids = [info.get("episode_id") for info in infos]
        observations = {
            "text": self.build_text_obs(obs, infos, init=True),
            "image": None,
            "anchor": obs.copy(),
        }
        self.pre_text_obs = obs
        self._is_done = np.zeros(len(infos), dtype=bool)
        self.memory.reset(batch_size=len(infos))
        # Cache available_actions for admissible projection use in the next step()
        self._current_avail_actions = [info.get("available_actions") for info in infos]
        return observations, infos

    def step(self, text_actions: List[str]):
        raw_text_actions = list(text_actions)  # projection 会原地修改 text_actions，提前保存原始输出
        avail = getattr(self, "_current_avail_actions", None)
        if avail is not None:
            actions, valids = self.projection_f(text_actions, avail)
        else:
            actions, valids = self.projection_f(text_actions, None)
        next_obs, rewards, dones, infos = self.envs.step(actions)
        # Update cached available_actions for the next step's projection
        self._current_avail_actions = [info.get("available_actions") for info in infos]

        next_obs = self.format_obs(next_obs)

        self.memory.store({"text_obs": self.pre_text_obs, "action": actions})
        self.pre_text_obs = next_obs

        # Accumulate done states for tracking; done envs are no longer sent to rollout
        # (handled by active-only batch compression in vanilla_multi_turn_loop).
        if not hasattr(self, "_is_done"):
            self._is_done = np.zeros(len(next_obs), dtype=bool)
        self._is_done = np.logical_or(self._is_done, to_numpy(dones))

        # Print after _is_done update so the step where env just finishes is also suppressed.
        # Collect up to 5 non-done indices so we always show 5 valid samples when available.
        _printed = 0
        for _i in range(len(actions)):
            if _printed >= 5:
                break
            if not self._is_done[_i]:
                print(f"[DEBUG proj {_i}] valid={valids[_i]} | raw={repr(raw_text_actions[_i])} | extracted={repr(actions[_i])}")
                _printed += 1

        built_text = self.build_text_obs(next_obs, infos)
        next_observations = {
            "text": built_text,
            "image": None,
            "anchor": next_obs.copy(),
        }
        # add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])
        attach_episode_ids(infos, getattr(self, "episode_ids", []))

        rewards = to_numpy(rewards)
        if os.environ.get("NEW_REWARD", "").strip() == "1":
            rewards = 10 * (rewards / 10) ** 3
        elif os.environ.get("BINARY_REWARD", "").strip() == "1":
            rewards = (rewards == 10).astype(rewards.dtype)

        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def save_state(self) -> Dict[str, Any]:
        """Save agent-side state for speculative rollback.
        Also saves full env state (browser + server session) when the underlying
        env supports it, so that restore_state can fully undo a done episode.
        Only used by the speculative planning path; no impact on normal rollout.
        """
        state = {"memory": self.memory.snapshot(), "pre_text_obs": list(self.pre_text_obs)}
        if hasattr(self.envs, "save_env_state"):
            try:
                state["env_state"] = self.envs.save_env_state()
            except Exception:
                pass  # non-critical: fall back to agent-side-only restore
        return state

    def restore_state(self, state: Dict[str, Any]):
        """Restore agent-side state from checkpoint.
        Also restores full env state when available (see save_state).
        Only used by the speculative planning path; no impact on normal rollout.
        """
        self.memory.restore(state["memory"])
        self.pre_text_obs = list(state["pre_text_obs"])
        if "env_state" in state and hasattr(self.envs, "restore_env_state"):
            try:
                self.envs.restore_env_state(state["env_state"])
            except Exception:
                pass  # non-critical: env may be in a recoverable state already

    def extract_task(self, text_obs: List[str]):
        tasks = []
        for obs in text_obs:
            parts = obs.split(" [SEP] ")
            assert parts[1] == "Instruction:"
            tasks.append(parts[2])
        return tasks

    def format_obs(self, text_obs):
        postprocess_text_obs = []
        for i in range(len(text_obs)):
            parts = text_obs[i].split(" [SEP] ")
            # the index of self.tasks[i] in parts
            try:
                index = parts.index(self.tasks[i])
                reformatted_obs = " [SEP] ".join(f"'{p}'" for p in parts[index + 1 :])
            except:
                reformatted_obs = text_obs[i]

            postprocess_text_obs.append(reformatted_obs)

        return postprocess_text_obs

    def format_avail_actions(self, avail):
        actions = []

        for key in avail.keys():
            if key not in ["has_search_bar", "clickables"]:
                raise ValueError(f"Unknown key in available actions: {key}")

        if avail["has_search_bar"]:
            actions.append("search[<your query>]")

        for txt in avail["clickables"]:
            actions.append(f"click[{txt}]")

        return actions

    def build_text_obs(
        self,
        text_obs: List[str],
        infos: List[List[str]],
        init: bool = False,
    ) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if not init and self.config.env.history_length > 0:
            memory_contexts, valid_lens = self.memory.fetch(
                self.config.env.history_length, obs_key="text_obs", action_key="action"
            )

        for i in range(len(text_obs)):
            available_actions = self.format_avail_actions(infos[i]["available_actions"])
            reformatted_available_actions = "\n".join(
                f"'{s}'," for s in available_actions
            )

            if init or self.config.env.history_length <= 0:
                obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                    task_description=self.tasks[i],
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                    thinking_instruction=self.thinking_instruction,
                )
            else:
                obs = WEBSHOP_TEMPLATE.format(
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_lens[i],
                    action_history=memory_contexts[i],
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                    available_actions=reformatted_available_actions,
                    thinking_instruction=self.thinking_instruction,
                )
                if len(obs) > 13000:
                    print(f"Warning len(obs)={len(obs)} is too long")
                    obs = WEBSHOP_TEMPLATE_NO_HIS.format(
                        task_description=self.tasks[i],
                        current_observation=text_obs[i],
                        available_actions=reformatted_available_actions,
                        thinking_instruction=self.thinking_instruction,
                    )

            postprocess_text_obs.append(obs)

        return postprocess_text_obs

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            batch_item = total_batch_list[batch_idx][i]
            if batch_item["active_masks"]:
                info = total_infos[batch_idx][i]
                won_value = float(info["won"])
                score_value = float(info["task_score"])
                success["success_rate"].append(won_value)
                success["webshop_task_score (not success_rate)"].append(score_value)
                # Log per-episode success details for offline analysis
                if batch_idx < 5 or won_value > 0:
                    print(f"[SUCCESS_EVAL] ep={batch_idx} step={i} won={won_value} "
                          f"task_score={score_value:.4f} done={info.get('done', '?')}")
                return


class AppWorldEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config):
        self.memory = SimpleMemory()
        super().__init__(envs, projection_f, config)

    def reset(self, kwargs):
        text_obs, infos = self.envs.reset()

        self.supervisors = [info["supervisor"] for info in infos]
        self.memory.reset(batch_size=len(text_obs))
        self.tasks = text_obs.copy()
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs, init=True)
        return {"text": full_text_obs, "image": None, "anchor": text_obs}, infos

    def step(self, text_actions: List[str]):
        actions, valids = self.projection_f(text_actions)

        text_obs, rewards, dones, infos = self.envs.step(actions)

        self.memory.store({"text_obs": text_obs, "action": actions})
        self.pre_text_obs = text_obs

        full_text_obs = self.build_text_obs(text_obs)

        # add action_valid to infos
        for i, info in enumerate(infos):
            info["is_action_valid"] = to_numpy(valids[i])

        next_observations = {"text": full_text_obs, "image": None, "anchor": text_obs}
        rewards = to_numpy(rewards)
        dones = to_numpy(dones)

        return next_observations, rewards, dones, infos

    def build_text_obs(self, text_obs: List[str], init: bool = False) -> List[str]:
        """
        This function builds the text observation for the agent.
        """
        postprocess_text_obs = []
        if init and self.supervisors is not None:
            for i in range(len(text_obs)):
                obs = APPWORLD_TEMPLATE_NO_HIS.format(
                    supervisor_first_name=self.supervisors[i]["first_name"],
                    supervisor_last_name=self.supervisors[i]["last_name"],
                    supervisor_email=self.supervisors[i]["email"],
                    supervisor_phone_number=self.supervisors[i]["phone_number"],
                    task_description=self.tasks[i],
                )
                postprocess_text_obs.append(obs)
        else:
            for i in range(len(text_obs)):
                # Get last `history_length` steps
                recent_history = self.memory[i][-self.config.env.history_length :]
                valid_history_length = len(recent_history)
                start_index = len(self.memory[i]) - valid_history_length
                action_history = ""
                for j, record in enumerate(recent_history):
                    step_number = start_index + j + 1
                    action = record["action"]
                    env_obs = record["text_obs"]
                    action_history += f"\nCode {step_number}: \n{action}\n\nResult {step_number}: \n{env_obs}\n"

                if len(action_history) > 10000:
                    action_history = "... " + action_history[-10000:]

                obs = APPWORLD_TEMPLATE.format(
                    supervisor_first_name=self.supervisors[i]["first_name"],
                    supervisor_last_name=self.supervisors[i]["last_name"],
                    supervisor_email=self.supervisors[i]["email"],
                    supervisor_phone_number=self.supervisors[i]["phone_number"],
                    task_description=self.tasks[i],
                    step_count=len(self.memory[i]),
                    history_length=valid_history_length,
                    action_history=action_history.strip(),
                    current_step=len(self.memory[i]) + 1,
                    current_observation=text_obs[i],
                )
                postprocess_text_obs.append(obs)
        return postprocess_text_obs


def make_envs(config, reward_fn=None):
    """
    Create enviroments
    """
    # check if config.env.rollout.n is an integer
    if not isinstance(config.env.rollout.n, int):
        raise ValueError("config.env.rollout.n should be an integer")
    group_n = config.env.rollout.n if config.env.rollout.n > 0 else 1
    resources_per_worker = OmegaConf.to_container(
        config.env.resources_per_worker, resolve=True
    )

    if "search" in config.env.env_name.lower():
        from agent_system.environments.env_package.search import (
            build_search_envs,
            search_projection,
        )

        _envs = build_search_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            env_config=config.env,
        )
        _val_envs = build_search_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            env_config=config.env,
        )

        projection_f = partial(
            search_projection,
            check_reasoning=bool(getattr(config.env.search, "check_reasoning", False)),
        )
        envs = SearchEnvironmentManager(_envs, projection_f, config)
        val_envs = SearchEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "gym_cards" in config.env.env_name.lower():
        from agent_system.environments.env_package.gym_cards import (
            build_gymcards_envs,
            gym_projection,
        )

        _envs = build_gymcards_envs(
            env_name=config.env.env_name,
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_gymcards_envs(
            env_name=config.env.env_name,
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            resources_per_worker=resources_per_worker,
        )

        projection_f = partial(gym_projection, env_name=config.env.env_name)
        envs = GymCardEnvironmentManager(_envs, projection_f, config)
        val_envs = GymCardEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "alfworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.alfworld import (
            alfworld_projection,
            build_alfworld_envs,
        )

        if config.env.env_name == "alfworld/AlfredThorEnv":
            alf_config_path = os.path.join(
                os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml"
            )
        elif config.env.env_name == "alfworld/AlfredTWEnv":
            alf_config_path = os.path.join(
                os.path.dirname(__file__), "env_package/alfworld/configs/config_tw.yaml"
            )
        else:
            raise ValueError(f"Unsupported environment: {config.env.env_name}")

        env_kwargs = {
            "eval_dataset": config.env.alfworld.eval_dataset,  # 'eval_in_distribution' or 'eval_out_of_distribution'
        }
        _envs = build_alfworld_envs(
            alf_config_path,
            config.env.seed,
            config.data.train_batch_size,
            group_n,
            is_train=True,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_alfworld_envs(
            alf_config_path,
            config.env.seed + 1000,
            config.data.val_batch_size,
            1,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )

        projection_f = partial(alfworld_projection)
        envs = AlfWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AlfWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "sokoban" in config.env.env_name.lower():
        from agent_system.environments.env_package.sokoban import (
            build_sokoban_envs,
            sokoban_projection,
        )

        env_kwargs = {
            "dim_room": config.env.sokoban.dim_room,
            "num_boxes": config.env.sokoban.num_boxes,
            "max_steps": config.env.max_steps,
            "search_depth": config.env.sokoban.search_depth,
        }
        _envs = build_sokoban_envs(
            config.env.seed,
            config.data.train_batch_size,
            group_n,
            mode=config.env.sokoban.mode,
            is_train=True,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_sokoban_envs(
            config.env.seed + 1000,
            config.data.val_batch_size,
            1,
            mode=config.env.sokoban.mode,
            is_train=False,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )

        projection_f = partial(sokoban_projection)
        envs = SokobanEnvironmentManager(_envs, projection_f, config)
        val_envs = SokobanEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "webshop" in config.env.env_name.lower():
        from agent_system.environments.env_package.webshop import (
            build_webshop_envs,
            webshop_projection,
            webshop_projection_action_only,
            webshop_projection_thought_prefix,
            webshop_projection_admissible,
        )

        if config.env.webshop.use_small:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_shuffle_1000.json",
            )
            attr_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_ins_v2_1000.json",
            )
        else:
            file_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_shuffle.json",
            )
            attr_path = os.path.join(
                os.path.dirname(__file__),
                "env_package/webshop/webshop/data/items_ins_v2.json",
            )
        env_kwargs = {
            "observation_mode": "text",
            "num_products": None,
            "human_goals": config.env.webshop.human_goals,
            "file_path": file_path,
            "attr_path": attr_path,
        }
        _envs = build_webshop_envs(
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            is_train=True,
            env_kwargs=env_kwargs,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_webshop_envs(
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            is_train=False,
            env_kwargs=env_kwargs,
            validation_size=getattr(config.env.webshop, "validation_size", None),
            resources_per_worker=resources_per_worker,
        )

        action_only_projection = getattr(config.env.webshop, "action_only_projection", False)
        thought_prefix_projection = getattr(config.env.webshop, "thought_prefix_projection", False)
        admissible_projection = getattr(config.env.webshop, "admissible_projection", False)
        if admissible_projection:
            projection_f = webshop_projection_admissible
        elif thought_prefix_projection:
            projection_f = webshop_projection_thought_prefix
        elif action_only_projection:
            projection_f = webshop_projection_action_only
        else:
            strict_projection = getattr(config.env.webshop, "strict_projection", False)
            projection_f = partial(webshop_projection, strict=strict_projection)
        envs = WebshopEnvironmentManager(_envs, projection_f, config)
        val_envs = WebshopEnvironmentManager(_val_envs, projection_f, config)
        import time

        time.sleep(
            (config.data.train_batch_size * group_n + config.data.val_batch_size) * 0.1
        )  # wait for the envs to be ready
        return envs, val_envs
    elif "appworld" in config.env.env_name.lower():
        from agent_system.environments.env_package.appworld import (
            appworld_projection,
            build_appworld_envs,
        )

        _envs = build_appworld_envs(
            dataset_name="train",
            seed=config.env.seed,
            env_num=config.data.train_batch_size,
            group_n=group_n,
            start_server_id=0,
            resources_per_worker=resources_per_worker,
        )
        _val_envs = build_appworld_envs(
            dataset_name="test_normal",
            seed=config.env.seed + 1000,
            env_num=config.data.val_batch_size,
            group_n=1,
            start_server_id=config.data.train_batch_size * group_n,
            resources_per_worker=resources_per_worker,
        )

        projection_f = partial(appworld_projection)
        envs = AppWorldEnvironmentManager(_envs, projection_f, config)
        val_envs = AppWorldEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    elif "single_turn" in config.env.env_name.lower():
        from agent_system.environments.single_turn_env import (
            SingleTurnEnvironmentManager,
            load_single_turn_data,
        )

        train_data = load_single_turn_data(
            config.env.train_data if hasattr(config.env, "train_data") and config.env.train_data else ""
        )
        val_data = load_single_turn_data(
            config.env.val_data if hasattr(config.env, "val_data") and config.env.val_data else ""
        )

        # Priority: 1) caller-supplied callable  2) config dotted-path string  3) verl default
        if reward_fn is None:
            reward_fn_path = getattr(config.env, "reward_fn", None)
            if reward_fn_path and isinstance(reward_fn_path, str):
                import importlib
                module_path, func_name = reward_fn_path.rsplit(".", 1)
                reward_fn = getattr(importlib.import_module(module_path), func_name)
            else:
                from verl.utils.reward_score import default_compute_score
                reward_fn = default_compute_score

        _train_envs = (train_data, reward_fn, True)
        _val_envs = (val_data, reward_fn, False)
        projection_f = None
        envs = SingleTurnEnvironmentManager(_train_envs, projection_f, config)
        val_envs = SingleTurnEnvironmentManager(_val_envs, projection_f, config)
        return envs, val_envs
    else:
        print("Environment not supported")
        exit(1)
