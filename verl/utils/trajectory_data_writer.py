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

"""Environment-agnostic trajectory prompt-response dump, grouped by traj_uid."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _scalar(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        if value.size == 1:
            value = value.item()
        else:
            return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _is_active(step: Dict[str, Any]) -> bool:
    active = step.get("active_masks", True)
    if isinstance(active, np.ndarray):
        return bool(active.item()) if active.size == 1 else bool(active.any())
    return bool(active)


def resolve_trajectory_output_dir(
    default_local_dir: str,
    phase: str,
    global_step: int | None = None,
) -> Path:
    """Return trajectory JSONL directory under the checkpoint root."""
    root = Path(default_local_dir).expanduser()
    if global_step is not None:
        return root / "trajectory_data" / f"global_step_{global_step}" / phase
    return root / "trajectory_data" / phase


class TrajectoryDataWriter:
    """Write one JSONL file per episode, keyed by traj_uid."""

    def __init__(self, output_dir: str, phase: str = "train"):
        self.output_dir = Path(output_dir).expanduser()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.phase = phase

    def write_episodes(
        self,
        total_batch_list: List[List[Dict]],
        traj_uid: np.ndarray,
        success: Dict[str, np.ndarray],
        episode_rewards: np.ndarray,
        episode_lengths: np.ndarray,
    ) -> int:
        """Persist prompt-response steps for each trajectory in the batch."""
        n_written = 0
        for bs, steps in enumerate(total_batch_list):
            if not steps:
                continue

            uid = str(traj_uid[bs])
            ep_success: Dict[str, Any] = {}
            for key, values in success.items():
                arr = np.asarray(values)
                if bs < len(arr):
                    ep_success[key] = _scalar(arr[bs])

            won = bool(ep_success.get("success_rate", 0))

            fpath = self.output_dir / f"{uid}.jsonl"
            with open(fpath, "w", encoding="utf-8") as f:
                step_idx = 0
                for step in steps:
                    if not _is_active(step):
                        continue

                    record: Dict[str, Any] = {
                        "kind": "step",
                        "phase": self.phase,
                        "traj_uid": uid,
                        "step_idx": step_idx,
                        "built_prompt": step.get("built_prompt") or "",
                        "raw_response": step.get("raw_response") or "",
                        "reward": _scalar(step.get("rewards")),
                    }
                    if "index" in step:
                        record["index"] = _scalar(step["index"])
                    if "uid" in step:
                        record["uid"] = _scalar(step["uid"])
                    episode_id = step.get("episode_id")
                    if episode_id:
                        record["episode_id"] = str(_scalar(episode_id))
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    step_idx += 1

                summary: Dict[str, Any] = {
                    "kind": "episode_summary",
                    "phase": self.phase,
                    "traj_uid": uid,
                    "episode_reward": float(episode_rewards[bs]),
                    "episode_length": int(episode_lengths[bs]),
                    "success": ep_success,
                    "won": won,
                }
                if steps and "uid" in steps[0]:
                    summary["uid"] = _scalar(steps[0]["uid"])
                for step in steps:
                    if not _is_active(step):
                        continue
                    episode_id = step.get("episode_id")
                    if episode_id:
                        summary["episode_id"] = str(_scalar(episode_id))
                        break
                f.write(json.dumps(summary, ensure_ascii=False) + "\n")

            n_written += 1

        print(f"[TrajectoryDataWriter] wrote {n_written} episodes to {self.output_dir}")
        return n_written
