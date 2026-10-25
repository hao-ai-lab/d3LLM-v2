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

import ray
import gym
import numpy as np

# -----------------------------------------------------------------------------
# Episode id ------------------------------------------------------------------
# -----------------------------------------------------------------------------


def make_webshop_episode_id(canonical_goal_idx: int | str | None) -> str | None:
    """Build static id from pre-shuffle canonical index (val 0..499, train 500+)."""
    if canonical_goal_idx is None:
        return None
    return f"webshop:goal:{int(canonical_goal_idx)}"


# -----------------------------------------------------------------------------
# Ray remote worker actor -----------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopWorker:
    """Ray remote actor that replaces the worker function.
    Each actor hosts a *WebAgentTextEnv* instance.
    """
    
    def __init__(self, seed, env_kwargs):
        # Lazy import avoids CUDA initialisation issues
        import sys
        import os
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), 'webshop'))
        sys.path.append(project_root)
        from web_agent_site.envs import WebAgentTextEnv  # noqa: WPS433 (runtime import)
        
        env_kwargs['seed'] = seed
        self.env = gym.make('WebAgentTextEnv-v0', **env_kwargs)
    
    def step(self, action):
        """Execute a step in the environment"""
        obs, reward, done, info = self.env.step(action)
        info = dict(info or {})  # make a *copy* so we can mutate safely
        info['available_actions'] = self.env.get_available_actions()
        info['task_score'] = reward
        info['tool_calling'] = 1  # Each step counts as one tool call for metrics

        # Redefine reward. We only use rule-based reward - win for 10, lose for 0.
        if done and reward == 1.0:
            info['won'] = True
            reward = 10.0
        # added for dense reward for RL training (unless qwen3.5 training would fail)
        # only effect training, the reward is not effected evaluation
        elif done and reward > 0:
            info['won'] = False
            reward = reward * 10.0  # scale task_score (0~1) to (0~10]
        else:
            info['won'] = False
            reward = 0

        # Log reward decisions for offline analysis (only when done to avoid spam)
        if done:
            # Get verbose reward breakdown if available
            breakdown = ""
            try:
                uw = self.env.unwrapped
                vi = uw.server.user_sessions[uw.session].get('verbose_info', {})
                if vi:
                    breakdown = (f" r_type={vi.get('r_type','?')} r_att={vi.get('r_att','?')}"
                                f" r_option={vi.get('r_option','?')} r_price={vi.get('r_price','?')}")
            except Exception:
                pass
            print(f"[ENV] done=True raw_task_score={info['task_score']:.4f} "
                  f"won={info['won']} scaled_reward={reward} action={action[:80]}{breakdown}")

        return obs, reward, done, info
    
    def reset(self, idx):
        """Reset the environment with given session index"""
        obs, info = self.env.reset(session=idx)
        info = dict(info or {})
        info['available_actions'] = self.env.get_available_actions()
        info['goal_idx'] = idx
        env = self.env.unwrapped
        session = env.server.user_sessions.get(env.session, {})
        goal = session.get("goal") if session else None
        if goal is not None and "canonical_goal_idx" in goal:
            info["canonical_goal_idx"] = goal["canonical_goal_idx"]
        info["episode_id"] = make_webshop_episode_id(info.get("canonical_goal_idx"))
        info['won'] = False
        return obs, info
    
    def render(self, mode_for_render):
        """Render the environment"""
        rendered = self.env.render(mode=mode_for_render)
        return rendered
    
    def get_available_actions(self):
        """Get available actions"""
        return self.env.get_available_actions()
    
    def save_env_state(self):
        """Snapshot the env state for speculative rollback.
        Only saves what can change during non-terminal draft steps:
        - browser URL and page_source (updated by every env.step)
        - session fields that change during browsing (keywords, page, asin, etc.)
        NOTE: session['goal'] and instruction_text never change mid-episode so are not snapshotted.
        NOTE: prev_obs/prev_actions/text_to_clickable are not saved because num_prev_obs=0
              (text_to_clickable is rebuilt from page_source on every step).
        Only used by the speculative planning path; no impact on normal rollout.
        """
        import copy
        env = self.env.unwrapped
        session = env.server.user_sessions.get(env.session, {})
        return {
            "session_id": env.session,
            "browser_url": env.browser.current_url,
            "browser_page": env.browser.page_source,
            # Shallow-copy mutable browsing fields only (skip 'goal' which is immutable).
            # Use copy.copy to preserve container types (e.g. defaultdict for 'actions').
            "session_mutable": {
                k: copy.copy(v) for k, v in session.items() if k != "goal"
            },
        }

    def restore_env_state(self, snap: dict):
        """Restore env to a previously saved snapshot.
        Only used by the speculative planning path; no impact on normal rollout.
        """
        env = self.env.unwrapped
        browser = env.browser
        server = env.server
        session_id = snap["session_id"]
        # Restore browser state
        browser.session_id = session_id
        browser.current_url = snap["browser_url"]
        browser.page_source = snap["browser_page"]
        # Restore env pointer
        env.session = session_id
        # Restore only the mutable session fields (goal is immutable, skip deepcopy)
        session = server.user_sessions.get(session_id)
        if session is not None:
            for k, v in snap["session_mutable"].items():
                import copy
                session[k] = copy.copy(v)
    
    def get_goals(self):
        """Get environment goals"""
        return self.env.server.goals
    
    def close(self):
        """Close the environment"""
        self.env.close()


# -----------------------------------------------------------------------------
# Vectorised Ray environment --------------------------------------------------
# -----------------------------------------------------------------------------

class WebshopMultiProcessEnv(gym.Env):
    """A vectorised, Ray-based wrapper around *WebAgentTextEnv*.

    ``info`` dictionaries returned by :py:meth:`step` **and** :py:meth:`reset`
    automatically contain the key ``'available_actions'`` so downstream RL code
    can obtain the *legal* action set without extra IPC overhead.
    """
    def __init__(
        self,
        seed: int,
        env_num: int,
        group_n: int,
        resources_per_worker: dict,
        is_train: bool = True,
        env_kwargs: dict = None,
        validation_size: int | None = None,
    ) -> None:
        super().__init__()

        # Initialize Ray if not already initialized
        if not ray.is_initialized():
            ray.init()

        self.group_n = group_n
        self.env_num = env_num
        self.num_processes = env_num * group_n
        self.is_train = is_train
        if not is_train: assert group_n == 1

        self._rng = np.random.RandomState(seed)
        self._validation_size = validation_size
        self._validation_goal_order = None
        self._validation_goal_cursor = 0

        self._env_kwargs = env_kwargs if env_kwargs is not None else {'observation_mode': 'text', 'num_products': None}

        # Inject split_range so each worker shuffles only its own subset,
        # guaranteeing train and val goal pools are disjoint.
        VAL_CUT = 500
        split_range = (0, VAL_CUT) if not is_train else (VAL_CUT, None)
        self._env_kwargs = {**self._env_kwargs, 'split_range': split_range}

        if self._validation_size is not None:
            if self.is_train:
                raise ValueError("validation_size is only valid for validation environments.")
            assert 0 < self.env_num <= self._validation_size <= VAL_CUT, (
                "WebShop validation requires "
                f"0 < val_batch_size ({self.env_num}) <= "
                f"val_data_size ({self._validation_size}) <= {VAL_CUT}."
            )
            assert self._validation_size % self.env_num == 0, (
                f"WebShop val_data_size ({self._validation_size}) must be "
                f"divisible by val_batch_size ({self.env_num})."
            )

        # -------------------------- Ray actors setup --------------------------
        # Per-worker goal-shuffle seed.
        #   Train: seed + (i // group_n) — each group has its own in-place shuffle σ_g;
        #          group members share σ_g so np.repeat maps them to the SAME goal (GRPO).
        #   Val  : a SINGLE shared constant so every worker shuffles [0:VAL_CUT] identically.
        #          With one common σ, reset()'s distinct positions (_rng.choice replace=False)
        #          map to DISTINCT canonical goals (σ is a bijection) → the full 500-goal val
        #          set is covered exactly once, no worker-to-worker duplicates or misses, and
        #          the aggregate score becomes independent of env.seed (seed only reorders).
        #          Without this, each val worker used a different seed → different σ_i, so
        #          distinct positions collapsed to ~316 unique goals (coupon-collector) with
        #          duplicates. Train behaviour is intentionally left unchanged.
        VAL_SHUFFLE_SEED = 233
        env_worker = ray.remote(**resources_per_worker)(WebshopWorker)
        self._workers = []
        for i in range(self.num_processes):
            worker_seed = (
                seed + (i // self.group_n) if self.is_train else VAL_SHUFFLE_SEED
            )
            worker = env_worker.remote(worker_seed, self._env_kwargs)
            self._workers.append(worker)

        # Get goals from the first worker (full-length list, split only shuffled in-place).
        goals_future = self._workers[0].get_goals.remote()
        goals = ray.get(goals_future)

        # ------- original ----------#
        # if args.num is None:
        #     if split == 'test':
        #         self.goal_idxs = range(500)
        #     elif split == 'eval':
        #         self.goal_idxs = range(500, 1500)
        #     elif split == 'train':
        #         self.goal_idxs = range(1500, len(self.env.server.goals))
        # else:
        #     self.goal_idxs = range(len(self.env.server.goals))

        if not self.is_train:
            self.goal_idxs = range(VAL_CUT)
        else:
            self.goal_idxs = range(VAL_CUT, len(goals))
        print(self.goal_idxs)

    # ------------------------------------------------------------------
    # Base API ----------------------------------------------------------
    # ------------------------------------------------------------------

    def _sample_goal_indices(self) -> list[int]:
        if self._validation_size is None:
            return self._rng.choice(
                self.goal_idxs,
                size=self.env_num,
                replace=False,
            ).tolist()

        if self._validation_goal_order is None:
            self._validation_goal_order = self._rng.choice(
                self.goal_idxs,
                size=self._validation_size,
                replace=False,
            ).tolist()
            self._validation_goal_cursor = 0

        end = self._validation_goal_cursor + self.env_num
        idx = self._validation_goal_order[self._validation_goal_cursor:end]
        self._validation_goal_cursor = end
        if end == self._validation_size:
            self._validation_goal_order = None
            self._validation_goal_cursor = 0
        return idx

    def step(self, actions: list[str]):
        if len(actions) != self.num_processes:
            raise ValueError(
                f'Expected {self.num_processes} actions, got {len(actions)}',
            )

        # Send step commands to all workers
        futures = []
        for worker, action in zip(self._workers, actions):
            future = worker.step.remote(action)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, reward_list, done_list, info_list = [], [], [], []
        for obs, reward, done, info in results:
            obs_list.append(obs)
            reward_list.append(reward)
            done_list.append(done)
            info_list.append(info)

        return obs_list, reward_list, done_list, info_list

    def reset(self):
        idx = self._sample_goal_indices()
        idx = np.repeat(idx, self.group_n).tolist()

        # Send reset commands to all workers
        futures = []
        for worker, i in zip(self._workers, idx):
            future = worker.reset.remote(i)
            futures.append(future)

        # Collect results
        results = ray.get(futures)
        obs_list, info_list = [], []
        for obs, info in results:
            obs_list.append(obs)
            info_list.append(info)

        return obs_list, info_list

    def advance_data_state(self, num_batches: int) -> None:
        """Rebuild the real goal-sampling position from a checkpoint step."""
        if num_batches < 0:
            raise ValueError(f"num_batches must be non-negative, got {num_batches}")
        for _ in range(num_batches):
            self._sample_goal_indices()

    # ------------------------------------------------------------------
    # Convenience helpers ----------------------------------------------
    # ------------------------------------------------------------------

    def save_env_state(self) -> list:
        """Snapshot full env state from all Ray workers for speculative rollback.
        Returns a list of per-worker state dicts. Only used by speculative path.
        Only used by the speculative planning path; no impact on normal rollout.
        """
        futures = [w.save_env_state.remote() for w in self._workers]
        return ray.get(futures)

    def restore_env_state(self, snaps: list):
        """Restore all Ray workers to previously saved snapshots.
        Only used by speculative path.
        Only used by the speculative planning path; no impact on normal rollout.
        """
        futures = [w.restore_env_state.remote(snap) for w, snap in zip(self._workers, snaps)]
        ray.get(futures)

    def render(self, mode: str = 'text', env_idx: int = None):
        if env_idx is not None:
            future = self._workers[env_idx].render.remote(mode)
            return ray.get(future)

        futures = []
        for worker in self._workers:
            future = worker.render.remote(mode)
            futures.append(future)
        
        return ray.get(futures)

    # ------------------------------------------------------------------
    # Clean‑up ----------------------------------------------------------
    # ------------------------------------------------------------------

    def close(self):
        if getattr(self, '_closed', False):
            return

        # Close all workers and kill Ray actors
        close_futures = []
        for worker in self._workers:
            future = worker.close.remote()
            close_futures.append(future)
        
        # Wait for all workers to close
        ray.get(close_futures)
        
        # Kill all Ray actors
        for worker in self._workers:
            ray.kill(worker)
            
        self._closed = True

    def __del__(self):  # noqa: D401
        self.close()


# -----------------------------------------------------------------------------
# Factory helper --------------------------------------------------------------
# -----------------------------------------------------------------------------

def build_webshop_envs(
    seed: int,
    env_num: int,
    group_n: int,
    resources_per_worker: dict,
    is_train: bool = True,
    env_kwargs: dict = None,
    validation_size: int | None = None,
):
    """Mirror *build_sokoban_envs* so higher‑level code can swap seamlessly."""
    return WebshopMultiProcessEnv(
        seed=seed,
        env_num=env_num,
        group_n=group_n,
        resources_per_worker=resources_per_worker,
        is_train=is_train,
        env_kwargs=env_kwargs,
        validation_size=validation_size,
    )