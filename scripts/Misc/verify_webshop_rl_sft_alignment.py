#!/usr/bin/env python3
"""Verify WebShop RL train goal sampling vs exclude manifest and SFT data.

Reads trajectory JSONL from a real verl main_ppo rollout (train phase),
extracts per-step train goals, and checks:
  1) alignment with rl150_excluded.manifest.json (Ray simulation ground truth)
  2) zero exact overlap with filtered SFT / distill data
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
TASK_RE = re.compile(r"Your task is to:\s*(.+?)(?:\n|$)", re.DOTALL)
PRICE_RE = re.compile(r", and price lower than [0-9.]+ dollars\.?$")
OBS_TASK_RE = re.compile(r"Instruction: \[SEP\] (.+?) \[SEP\]")


def norm_task(text: str) -> str:
    return PRICE_RE.sub("", (text or "").strip()).strip().rstrip(".")


def task_from_built_text(built: str) -> str | None:
    for ln in (built or "").split("\n"):
        if "Your task is to:" in ln:
            return ln.split("Your task is to:", 1)[1].strip()
    m = TASK_RE.search(built or "")
    if m:
        return m.group(1).strip()
    m = OBS_TASK_RE.search(built or "")
    if m:
        return m.group(1).strip()
    return None


def task_from_raw_obs(raw: str) -> str | None:
    parts = (raw or "").split(" [SEP] ")
    if len(parts) > 2 and parts[1] == "Instruction:":
        return norm_task(parts[2])
    return None


def load_sft_tasks(path: Path) -> set[str]:
    tasks: set[str] = set()
    if not path.exists():
        return tasks
    if path.suffix == ".jsonl":
        with path.open(encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                for msg in rec.get("messages", []):
                    if msg.get("role") != "user":
                        continue
                    m = TASK_RE.search(msg.get("content", ""))
                    if m:
                        tasks.add(norm_task(m.group(1)))
                        break
    elif path.suffix == ".parquet":
        df = pd.read_parquet(path)

        def from_steps(steps):
            if steps is None or len(steps) == 0:
                return None
            built = steps[0].get("built_observation") or steps[0].get("built_text") or ""
            m = TASK_RE.search(built)
            return norm_task(m.group(1)) if m else None

        for t in df["steps"].apply(from_steps):
            if t:
                tasks.add(t)
    else:
        raise SystemExit(f"Unsupported SFT file type: {path}")
    return tasks


def _first_step_record(path: Path) -> dict | None:
    with path.open(encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("kind") == "step":
                return rec
    return None


def _iter_train_step_dirs(history_dir: Path) -> list[tuple[int, Path]]:
    """Return (global_step, train_dir) pairs under a checkpoint root."""
    step_dirs: list[tuple[int, Path]] = []
    for d in sorted(history_dir.glob("trajectory_data/global_step_*/train")):
        m = re.search(r"global_step_(\d+)", str(d))
        step_idx = int(m.group(1)) if m else len(step_dirs)
        step_dirs.append((step_idx, d))
    if step_dirs:
        return step_dirs
    for candidate in (
        history_dir / "trajectory_data" / "train",
        history_dir / "train",
        history_dir,
    ):
        if candidate.is_dir() and list(candidate.glob("*.jsonl")):
            return [(0, candidate)]
    return []


def load_train_steps_from_history(history_dir: Path, group_n: int) -> list[dict]:
    """Return ordered train steps grouped by global training step."""
    del group_n  # kept for CLI compatibility
    steps = []
    for seg_idx, (step_idx, dir_path) in enumerate(_iter_train_step_dirs(history_dir)):
        assignments = []
        for fp in sorted(dir_path.glob("*.jsonl")):
            rec = _first_step_record(fp)
            if not rec:
                continue
            built = rec.get("built_prompt") or rec.get("built_text") or ""
            raw = rec.get("raw_obs") or ""
            task_raw = extract_task(built) or extract_task(raw) or task_from_raw_obs(raw)
            task_norm = norm_task(task_raw) if task_raw else None
            if task_norm is None:
                continue
            sample_idx = rec.get("index", len(assignments))
            assignments.append(
                {
                    "sample_idx": int(sample_idx),
                    "task_norm": task_norm,
                    "task_raw": task_raw,
                }
            )
        assignments.sort(key=lambda x: x["sample_idx"])
        if assignments:
            steps.append(
                {
                    "segment_id": seg_idx + 1,
                    "manifest_step_idx": step_idx,
                    "assignments": assignments,
                }
            )
    return steps


def extract_task(text: str) -> str | None:
    return task_from_built_text(text)


def load_manifest_steps(manifest_path: Path, num_steps: int) -> list[dict]:
    with manifest_path.open(encoding="utf-8") as f:
        manifest = json.load(f)
    return manifest["per_step_assignments"][:num_steps]


def compare_step(manifest_step: dict, live_step: dict, step_idx: int) -> dict:
    m_assign = {a["sample_idx"]: a for a in manifest_step["assignments"]}
    l_assign = {a["sample_idx"]: a for a in live_step["assignments"]}
    norm_match = 0
    mismatches = []
    common = sorted(set(m_assign) & set(l_assign))
    for s in common:
        m, l = m_assign[s], l_assign[s]
        ok_norm = m["task_norm"] == l["task_norm"]
        if ok_norm:
            norm_match += 1
        if not ok_norm:
            mismatches.append(
                f"  sample {s}:\n"
                f"    manifest: {m['task_norm'][:90]}\n"
                f"    live:     {l['task_norm'][:90]}"
            )
    return {
        "step": step_idx,
        "live_segment_id": live_step["segment_id"],
        "samples_compared": len(common),
        "task_norm_match": norm_match,
        "mismatches": mismatches,
    }


def prefix_near_dup_count(task: str, sft_tasks: set[str]) -> int:
    p50 = task[:50]
    return sum(1 for t in sft_tasks if t[:50] == p50 and t != task)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--history-dir", type=Path, required=True)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT
        / "temp/nju_downloads/distill/webshop_qwen3_8b_grpo_step150/rl150_excluded.manifest.json",
    )
    parser.add_argument(
        "--sft-file",
        type=Path,
        default=REPO_ROOT
        / "temp/nju_downloads/distill/webshop_qwen3_8b_grpo_step150/distill_trajectories_rl150_excluded.parquet",
    )
    parser.add_argument("--group-n", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=50)
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print per-step OK lines (default: only mismatches + summary)",
    )
    args = parser.parse_args()

    if not args.history_dir.exists():
        raise SystemExit(f"History dir not found: {args.history_dir}")

    live_steps = load_train_steps_from_history(args.history_dir, args.group_n)
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    manifest_steps_all = manifest["per_step_assignments"]

    live_steps_mapped = list(live_steps)
    if live_steps_mapped and "manifest_step_idx" not in live_steps_mapped[0]:
        for live in live_steps_mapped:
            live["manifest_step_idx"] = int(live["segment_id"]) - 1
    live_steps_mapped = [
        live
        for live in live_steps_mapped
        if 0 <= int(live["manifest_step_idx"]) < args.num_steps
    ]
    live_steps_mapped.sort(key=lambda x: x["manifest_step_idx"])

    sft_tasks = load_sft_tasks(args.sft_file)

    print("=" * 72)
    print("WebShop RL ↔ SFT alignment report")
    print("=" * 72)
    print(f"History dir : {args.history_dir}")
    print(f"Manifest    : {args.manifest}")
    print(f"SFT file    : {args.sft_file} ({len(sft_tasks)} unique tasks)")
    print(f"Live steps  : {len(live_steps)} captured, comparing first {args.num_steps}")
    print()

    if not live_steps_mapped:
        print("ERROR: no train reset records in requested step range")
        sys.exit(1)

    all_live_tasks: set[str] = set()
    total_norm_match = 0
    total_samples = 0
    steps_ok = 0
    steps_fail = 0
    missing_steps = set(range(args.num_steps))

    for live in live_steps_mapped:
        step_idx = live["manifest_step_idx"]
        missing_steps.discard(step_idx)
        if step_idx >= len(manifest_steps_all):
            print(f"[segment {live['segment_id']}] ERROR: no manifest step index {step_idx}")
            steps_fail += 1
            continue
        rep = compare_step(manifest_steps_all[step_idx], live, step_idx)
        total_norm_match += rep["task_norm_match"]
        total_samples += rep["samples_compared"]
        for a in live["assignments"]:
            all_live_tasks.add(a["task_norm"])

        step_ok = rep["task_norm_match"] == rep["samples_compared"]
        if step_ok:
            steps_ok += 1
        else:
            steps_fail += 1

        if args.verbose or not step_ok:
            status = "OK" if step_ok else "MISMATCH"
            print(
                f"[step {step_idx:3d}] segment_id={rep['live_segment_id']:3d}  "
                f"{rep['task_norm_match']:2d}/{rep['samples_compared']:2d}  [{status}]"
            )
            for line in rep["mismatches"][:3]:
                print(line)
            if len(rep["mismatches"]) > 3:
                print(f"  ... +{len(rep['mismatches']) - 3} more mismatches")

    if missing_steps:
        print()
        print(f"WARNING: missing manifest steps in history: {sorted(missing_steps)[:20]}"
              f"{'...' if len(missing_steps) > 20 else ''} ({len(missing_steps)} total)")

    print()
    print(f"Steps summary   : {steps_ok}/{args.num_steps} steps fully matched (16/16 each)")
    if steps_fail:
        print(f"                  : {steps_fail} step(s) with mismatches")
    pct = 100.0 * total_norm_match / total_samples if total_samples else 0.0
    print(f"Manifest alignment: {total_norm_match}/{total_samples} task_norm matches ({pct:.2f}%)")

    exact_overlap = sorted(all_live_tasks & sft_tasks)
    print(f"SFT exact overlap : {len(exact_overlap)} / {len(all_live_tasks)} live unique tasks")
    if exact_overlap:
        print("  OVERLAP examples:")
        for t in exact_overlap[:5]:
            print(f"    - {t[:100]}")

    near_dup = sum(prefix_near_dup_count(t, sft_tasks) for t in all_live_tasks)
    print(f"SFT near-dup (same first 50 chars, diff attrs): {near_dup} pairs (informational)")

    print()
    passed = (
        steps_ok == args.num_steps
        and not missing_steps
        and total_norm_match == total_samples
        and not exact_overlap
    )
    if passed:
        print("RESULT: PASS — live RL goals match manifest and have zero SFT exact overlap.")
        sys.exit(0)
    print("RESULT: FAIL — see details above.")
    sys.exit(1)


if __name__ == "__main__":
    main()
