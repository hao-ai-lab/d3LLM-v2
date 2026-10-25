"""Convert generic trajectory JSONL (grouped by traj_uid) to distillation parquet.

Input JSONL files live under trainer.default_local_dir/trajectory_data/{phase}/,
or trainer.default_local_dir/trajectory_data/global_step_{N}/{phase}/ during training.

Each input JSONL file = one episode (named {traj_uid}.jsonl). Columns written:
  steps  : list[dict]  with keys built_observation (str) and raw_response (str)
  won    : bool        episode succeeded
  episode_id (optional): stable data id when present in JSONL

Usage:
    python scripts/Misc/convert_trajectory_data_to_distill.py \
        --input_dir  data/distill/webshop_qwen2.5_7b_gigpo_step150/trajectory_data/val \
        --output     data/distill/webshop_qwen2.5_7b_gigpo_step150/distill_trajectories.parquet \
        [--success_only]
"""
import argparse
import glob
import json
import os
import re

import pandas as pd

_LLADA_REASONING_PREFIX = "Reasoning:\n"
_OPEN_TAG_RE = re.compile(r"<\s*redacted_thinking\s*>\s*", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"\s*<\s*/\s*redacted_thinking\s*>\s*", re.IGNORECASE)


def qwen_response_to_llada(text: str) -> str:
    """Convert Qwen-style reasoning tags to LLaDA Reasoning: prefix."""
    if not text:
        return text
    out = _OPEN_TAG_RE.sub(_LLADA_REASONING_PREFIX, text)
    out = _CLOSE_TAG_RE.sub("\n", out)
    if _LLADA_REASONING_PREFIX not in out and not out.lstrip().startswith("Reasoning:"):
        if "<action>" in out and not out.lstrip().startswith("<action>"):
            out = _LLADA_REASONING_PREFIX + out.lstrip()
    return out.strip()


def jsonl_to_episode(fpath: str):
    """Return (steps, won, episode_id) from one traj_uid JSONL file."""
    lines = []
    with open(fpath, encoding="utf-8") as f:
        for raw in f:
            raw = raw.strip()
            if raw:
                lines.append(json.loads(raw))

    steps = []
    won = False
    episode_id = None
    for line in lines:
        kind = line.get("kind")
        if kind == "episode_summary":
            won = bool(line.get("won", False))
            episode_id = line.get("episode_id") or episode_id
            if episode_id is None and line.get("gamefile"):
                from agent_system.environments.env_package.alfworld.envs import make_alfworld_episode_id
                episode_id = make_alfworld_episode_id(line.get("gamefile"))
            continue
        if kind != "step":
            continue

        entry = {"built_observation": line.get("built_prompt") or ""}
        raw = line.get("raw_response") or ""
        if raw:
            entry["raw_response"] = qwen_response_to_llada(raw)
        steps.append(entry)
        if episode_id is None and line.get("episode_id"):
            episode_id = line.get("episode_id")
        if episode_id is None and line.get("gamefile"):
            from agent_system.environments.env_package.alfworld.envs import make_alfworld_episode_id
            episode_id = make_alfworld_episode_id(line.get("gamefile"))

    return steps, won, episode_id


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True,
                        help="Directory of {traj_uid}.jsonl files (or parent dir with --recursive)")
    parser.add_argument("--recursive", action="store_true",
                        help="Recursively find *.jsonl under input_dir")
    parser.add_argument("--output", required=True,
                        help="Output .parquet path")
    parser.add_argument("--success_only", action="store_true",
                        help="Only keep episodes where won==True")
    args = parser.parse_args()

    if args.recursive:
        files = sorted(glob.glob(os.path.join(args.input_dir, "**", "*.jsonl"), recursive=True))
    else:
        files = sorted(glob.glob(os.path.join(args.input_dir, "*.jsonl")))
    if not files:
        raise FileNotFoundError(f"No *.jsonl files found in {args.input_dir}")
    print(f"Found {len(files)} JSONL files")

    rows = []
    skipped = 0
    for fpath in files:
        steps, won, episode_id = jsonl_to_episode(fpath)
        if not steps:
            skipped += 1
            continue
        if args.success_only and not won:
            skipped += 1
            continue
        row = {"steps": steps, "won": won}
        if episode_id:
            row["episode_id"] = episode_id
        rows.append(row)

    if not rows:
        raise ValueError(f"No episodes kept from {args.input_dir} (skipped={skipped})")

    print(f"Episodes kept: {len(rows)}  skipped: {skipped}")
    won_count = sum(1 for r in rows if r["won"])
    print(f"  won={won_count} ({100 * won_count / len(rows):.1f}%)")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_parquet(args.output, index=False)
    print(f"Saved → {args.output}  ({os.path.getsize(args.output) // 1024} KB)")


if __name__ == "__main__":
    main()
