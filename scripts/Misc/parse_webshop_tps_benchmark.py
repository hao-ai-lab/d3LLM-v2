#!/usr/bin/env python3
"""Summarize timestamped WebShop TPS benchmark events."""

from __future__ import annotations

import argparse
import csv
import fcntl
import json
import os
import re
from pathlib import Path


ENGINE_TPS_RE = re.compile(
    r"\[Engine TPS\]\s+backend=(?P<backend>\S+)\s+bs=(?P<bs>\d+)\s+"
    r"prompt_tokens=(?P<prompt>\d+)\s+response_tokens=(?P<response>\d+).*?"
    r"elapsed=(?P<elapsed>[0-9.]+)s\s+tps=(?P<tps>[0-9.]+)"
)
DLLM_TPS_RE = re.compile(
    r"\[dLLM TPS\]\s+bs=(?P<bs>\d+),\s+prompt=(?P<prompt>\d+).*?"
    r"resp=(?P<response>\d+).*?elapsed=(?P<elapsed>[0-9.]+)s[,\s]+"
    r"tps=(?P<tps>[0-9.]+)"
)
NFE_RE = re.compile(r"\bnfe=(?P<nfe>\d+)\s+tpf=(?P<tpf>[0-9.]+)")
FORWARD_RE = re.compile(
    r"\bdecode_forwards=(?P<forwards>\d+)\s+"
    r"equivalent_tpf=(?P<equivalent_tpf>[0-9.]+)"
)
DECODE_RE = re.compile(
    r"\bprefill=(?P<prefill>[0-9.]+)s\s+decode=(?P<decode>[0-9.]+)s\s+"
    r"decode_tps=(?P<decode_tps>[0-9.]+)"
)
CSV_FIELDS = (
    "model",
    "tokens",
    "prompt_tokens",
    "nfe",
    "tpf",
    "decode_forward_count",
    "equivalent_tpf",
    "prefill_seconds",
    "decode_seconds",
    "decode_tps",
    "generation_seconds",
    "window_seconds",
    "model_tps",
    "e2e_tps",
    "completed_calls",
    "all_bs_one",
    "completed_window",
    "process_returncode",
    "event_log",
)


def summarize(event_log: Path, requested_window_s: float) -> dict[str, object]:
    tokens = 0
    prompt_tokens = 0
    nfe = 0
    decode_forward_count = 0
    prefill_seconds = 0.0
    decode_seconds = 0.0
    generation_seconds = 0.0
    completed_calls = 0
    all_bs_one = True
    run_end: dict[str, object] = {}
    model = event_log.stem

    with event_log.open(encoding="utf-8") as handle:
        for raw_line in handle:
            event = json.loads(raw_line)
            if event.get("type") == "run_start":
                model = str(event["model"])
                continue
            if event.get("type") == "run_end":
                run_end = event
                continue
            if event.get("type") != "output" or event.get("phase") != "measure":
                continue
            line = str(event.get("line", ""))
            match = ENGINE_TPS_RE.search(line) or DLLM_TPS_RE.search(line)
            if match is None:
                continue
            batch_size = int(match.group("bs"))
            all_bs_one = all_bs_one and batch_size == 1
            prompt_tokens += int(match.group("prompt"))
            tokens += int(match.group("response"))
            generation_seconds += float(match.group("elapsed"))
            completed_calls += 1
            nfe_match = NFE_RE.search(line)
            if nfe_match is not None:
                nfe += int(nfe_match.group("nfe"))
            forward_match = FORWARD_RE.search(line)
            if forward_match is not None:
                decode_forward_count += int(forward_match.group("forwards"))
            decode_match = DECODE_RE.search(line)
            if decode_match is not None:
                prefill_seconds += float(decode_match.group("prefill"))
                decode_seconds += float(decode_match.group("decode"))

    measured_window_s = float(run_end.get("measurement_elapsed_s", 0.0))
    completed_window = bool(run_end.get("completed_window", False))
    model_tps = tokens / generation_seconds if generation_seconds > 0 else 0.0
    e2e_tps = tokens / measured_window_s if measured_window_s > 0 else 0.0

    return {
        "model": model,
        "tokens": tokens,
        "prompt_tokens": prompt_tokens,
        "nfe": nfe,
        "tpf": round(tokens / nfe, 6) if nfe > 0 else None,
        "decode_forward_count": decode_forward_count,
        "equivalent_tpf": (
            round(tokens / decode_forward_count, 6) if decode_forward_count > 0 else None
        ),
        "prefill_seconds": round(prefill_seconds, 6),
        "decode_seconds": round(decode_seconds, 6),
        "decode_tps": round(tokens / decode_seconds, 6) if decode_seconds > 0 else None,
        "generation_seconds": round(generation_seconds, 6),
        "window_seconds": round(measured_window_s, 6),
        "model_tps": round(model_tps, 6),
        "e2e_tps": round(e2e_tps, 6),
        "completed_calls": completed_calls,
        "all_bs_one": all_bs_one,
        "completed_window": completed_window,
        "process_returncode": run_end.get("process_returncode"),
        "event_log": str(event_log.resolve()),
    }


def upsert_csv(csv_path: Path, summary: dict[str, object]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path = csv_path.with_suffix(csv_path.suffix + ".lock")
    with lock_path.open("w", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle, fcntl.LOCK_EX)
        rows: list[dict[str, str]] = []
        if csv_path.exists():
            with csv_path.open(newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))
        replacement = {field: str(summary.get(field, "")) for field in CSV_FIELDS}
        rows = [row for row in rows if row.get("model") != replacement["model"]]
        rows.append(replacement)
        temp_path = csv_path.with_suffix(csv_path.suffix + f".tmp.{os.getpid()}")
        with temp_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS)
            writer.writeheader()
            writer.writerows(rows)
        temp_path.replace(csv_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--event-log", type=Path, required=True)
    parser.add_argument("--window-seconds", type=float, default=300.0)
    parser.add_argument("--json-output", type=Path)
    parser.add_argument("--csv-output", type=Path)
    args = parser.parse_args()

    summary = summarize(args.event_log, args.window_seconds)
    rendered = json.dumps(summary, ensure_ascii=False, indent=2)
    print(rendered)
    if args.json_output:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(rendered + "\n", encoding="utf-8")
    if args.csv_output:
        upsert_csv(args.csv_output, summary)
    return 0 if summary["completed_window"] and summary["completed_calls"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
