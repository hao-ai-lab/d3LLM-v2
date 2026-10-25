# Copyright 2025 the verl-agent team.
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
Single-turn env: loads tokenizer inside the env (copy_to_local + hf_tokenizer).
Optional ``env.model_path`` overrides ``actor_rollout_ref.model.path`` for tokenizer only.
Use ``data.not_use_template=true`` in rollout so chat_template is not applied twice.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, TextIO, Tuple

import numpy as np
from omegaconf import OmegaConf

from agent_system.environments.base import EnvironmentManagerBase
from verl.utils.dataset.rl_dataset import _messages_to_text
from verl.utils.fs import copy_to_local
from verl.utils import hf_tokenizer


def load_single_turn_data(path: str) -> List[Dict]:
    if not path:
        return []
    path = os.path.expanduser(path)
    if path.endswith(".parquet"):
        import datasets

        ds = datasets.load_dataset("parquet", data_files=path)["train"]
        return [ds[i] for i in range(len(ds))]
    if path.endswith(".jsonl") or path.endswith(".json"):
        out: List[Dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
    raise ValueError(f"Unsupported data format: {path}")


def _ground_truth(item: Dict) -> str:
    rm = item.get("reward_model", {})
    if isinstance(rm, dict):
        return str(rm.get("ground_truth", "") or "")
    return str(rm)


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _group_n(config) -> int:
    r = getattr(getattr(config, "env", None), "rollout", None)
    return max(1, int(getattr(r, "n", 1) or 1))


def _messages_for_item(item: Dict, prompt_key: str) -> List[Dict[str, Any]]:
    raw = item.get(prompt_key, item.get("question", ""))
    if isinstance(raw, list):
        return [dict(m) for m in raw if isinstance(m, dict)]
    if isinstance(raw, str):
        return [{"role": "user", "content": raw}]
    return [{"role": "user", "content": str(raw)}]


class SingleTurnEnvironmentManager(EnvironmentManagerBase):
    def __init__(self, envs, projection_f, config) -> None:
        data, reward_fn = envs[0], envs[1]
        for_training = bool(envs[2]) if len(envs) >= 3 else True
        super().__init__(envs, projection_f, config)

        self._for_training = for_training
        self.reward_fn = reward_fn
        self._prompt_key = config.data.get("prompt_key", "prompt")
        self._template_kw: Dict[str, Any] = (
            OmegaConf.to_container(config.data.get("apply_chat_template_kwargs", {}), resolve=True)
            or {}
        )

        env_cfg = config.env
        mp = getattr(env_cfg, "model_path", None)
        model_src = mp or config.actor_rollout_ref.model.path
        local = copy_to_local(
            model_src,
            use_shm=config.actor_rollout_ref.model.get("use_shm", False),
        )
        trust = config.data.get("trust_remote_code", False)
        lfo = config.actor_rollout_ref.model.get("local_files_only", False)
        self._tokenizer = hf_tokenizer(local, trust_remote_code=trust, local_files_only=lfo)

        if not bool(config.data.get("not_use_template", False)):
            print(
                "\033[93m[SingleTurnEnv] set data.not_use_template=true (env already applies chat_template).\033[0m",
                flush=True,
            )

        if for_training:
            dc = config.data
            base = int(getattr(dc, "gen_batch_size", None) or getattr(dc, "train_batch_size", 1) or 1)
            gn = _group_n(config)
            self._base_bs, self._gn = base, gn
            self._bs = base * gn
        else:
            dc = config.data
            vb = int(getattr(dc, "val_batch_size", None) or getattr(dc, "train_batch_size", 1) or 1)
            self._base_bs, self._gn, self._bs = vb, 1, vb

        self.data = self._truncate(data, self._bs)
        self._data_idx = 0
        self._current: Optional[List[Dict]] = None
        self._log_fp: Optional[TextIO] = None
        self._reset_seq = 0
        self._batch_reset_seq = 0

    @staticmethod
    def _truncate(data: List[Dict], bs: int) -> List[Dict]:
        n = len(data)
        if n == 0 or bs <= 0 or n % bs == 0:
            return data
        t = (n // bs) * bs
        print(
            f"\033[91m[SingleTurnEnv] truncate {n}->{t} (batch {bs})\033[0m",
            flush=True,
        )
        return data[:t]

    def _observation(self, item: Dict) -> str:
        messages = _messages_for_item(item, self._prompt_key)
        tok = self._tokenizer
        if getattr(tok, "chat_template", None):
            return tok.apply_chat_template(
                messages,
                add_generation_prompt=True,
                tokenize=False,
                **self._template_kw,
            )
        return _messages_to_text(messages)

    def reset(self, kwargs) -> Tuple[Dict[str, Any], List[Dict]]:
        if not self.data:
            raise RuntimeError("SingleTurnEnvironmentManager: empty data (set env.train_data / env.val_data).")
        n = len(self.data)
        start = self._data_idx % n
        unique = [self.data[(start + i) % n] for i in range(self._base_bs)]
        self._data_idx += self._base_bs
        batch_reset_seq = self._reset_seq
        self._batch_reset_seq = batch_reset_seq
        items: List[Dict] = []
        infos: List[Dict[str, Any]] = []
        for batch_sample_idx, x in enumerate(unique):
            if self._gn > 1:
                for group_inner_id in range(self._gn):
                    items.append(x)
                    infos.append(
                        {
                            "data_source": x.get("data_source", "unknown"),
                            "batch_sample_idx": batch_sample_idx,
                            "group_inner_id": group_inner_id,
                            "sample_id": f"{batch_sample_idx}_{group_inner_id}",
                            "reset_seq": batch_reset_seq,
                        }
                    )
            else:
                items.append(x)
                infos.append(
                    {
                        "data_source": x.get("data_source", "unknown"),
                        "batch_sample_idx": batch_sample_idx,
                        "group_inner_id": 0,
                        "sample_id": f"{batch_sample_idx}_0",
                        "reset_seq": batch_reset_seq,
                    }
                )
        self._reset_seq = batch_reset_seq + 1
        self._current = items
        prompts = [self._observation(it) for it in items]
        obs = {"text": prompts, "image": None, "anchor": list(prompts)}
        return obs, infos

    def step(self, text_actions: List[str]):
        assert self._current is not None
        n = len(self._current)
        if len(text_actions) != n:
            m = min(len(text_actions), n)
            print(f"\033[91m[SingleTurnEnv] action len mismatch, use {m}\033[0m", flush=True)
            text_actions = text_actions[:m]
            self._current = self._current[:m]

        rewards, infos = [], []
        for i, action in enumerate(text_actions):
            it = self._current[i]
            ds = it.get("data_source", "unknown")
            sc = self.reward_fn(
                data_source=ds,
                solution_str=action,
                ground_truth=_ground_truth(it),
                extra_info=it.get("extra_info"),
            )
            r = float(sc.get("score", 0.0)) if isinstance(sc, dict) else float(sc)
            rewards.append(r)
            batch_sample_idx = i // self._gn
            group_inner_id = i % self._gn
            sample_id = f"{batch_sample_idx}_{group_inner_id}"
            infos.append(
                {
                    "won": r >= 1.0,
                    "is_action_valid": True,
                    "reward": r,
                    "batch_sample_idx": batch_sample_idx,
                    "group_inner_id": group_inner_id,
                    "sample_id": sample_id,
                    "reset_seq": self._batch_reset_seq,
                }
            )
            self._log(
                ds,
                self._observation(it),
                action,
                _ground_truth(it),
                r,
                sc,
                it.get("extra_info"),
                batch_sample_idx=batch_sample_idx,
                group_inner_id=group_inner_id,
                sample_id=sample_id,
                reset_seq=self._batch_reset_seq,
            )

        R = np.array(rewards, dtype=np.float32)
        D = np.ones(len(text_actions), dtype=bool)
        return {"text": [""] * len(text_actions), "image": None, "anchor": None}, R, D, infos

    def _log(
        self,
        ds,
        prompt,
        response,
        gt,
        reward,
        score_obj,
        extra,
        *,
        batch_sample_idx: int,
        group_inner_id: int,
        sample_id: str,
        reset_seq: int,
    ):
        try:
            d = _repo_root() / "single_turn_log"
            d.mkdir(parents=True, exist_ok=True)
            if self._log_fp is None:
                self._log_fp = open(d / f"rollout_pid{os.getpid()}.jsonl", "a", encoding="utf-8")
            rec: Dict[str, Any] = {
                "ts": time.time(),
                "data_source": ds,
                "reset_seq": reset_seq,
                "batch_sample_idx": batch_sample_idx,
                "group_inner_id": group_inner_id,
                "sample_id": sample_id,
                "prompt": prompt,
                "response": response,
                "ground_truth": gt,
                "reward": reward,
            }
            if isinstance(score_obj, dict):
                rec["score_detail"] = score_obj
            if extra is not None:
                rec["extra_info"] = extra
            self._log_fp.write(json.dumps(rec, ensure_ascii=False) + "\n")
            self._log_fp.flush()
        except Exception as e:
            print(f"\033[91m[SingleTurnEnv] log failed: {e}\033[0m", flush=True)

    def close(self) -> None:
        if self._log_fp:
            self._log_fp.close()
            self._log_fp = None

    def _process_batch(self, batch_idx, total_batch_list, total_infos, success):
        for i in reversed(range(len(total_batch_list[batch_idx]))):
            b = total_batch_list[batch_idx][i]
            if b["active_masks"]:
                success["success_rate"].append(float(total_infos[batch_idx][i].get("won", False)))
                return
