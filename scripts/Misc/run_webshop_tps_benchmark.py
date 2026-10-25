#!/usr/bin/env python3
"""Run a command and collect a fixed WebShop TPS window after warmup."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import queue
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


EPISODE_RE = re.compile(r"\[Episode \d+\] finished")
GENERATION_RE = re.compile(r"\[(?:Engine|dLLM) TPS\]")


def write_event(handle, event: dict[str, object]) -> None:
    handle.write(json.dumps(event, ensure_ascii=False) + "\n")
    handle.flush()


def stop_process_group(process: subprocess.Popen[str], grace_s: float = 30.0) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--duration-seconds", type=float, default=300.0)
    parser.add_argument("--target-episodes", type=int)
    parser.add_argument("--warmup-generations", type=int, default=1)
    parser.add_argument("--startup-timeout-seconds", type=float, default=1800.0)
    parser.add_argument("--max-overrun-seconds", type=float, default=600.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        parser.error("a benchmark command is required after --")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_log = args.output_dir / f"{args.model}.events.jsonl"
    raw_log = args.output_dir / f"{args.model}.log"
    summary_json = args.output_dir / f"{args.model}.summary.json"
    summary_csv = args.output_dir / "summary.csv"

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        start_new_session=True,
    )
    assert process.stdout is not None
    output_queue: queue.Queue[tuple[float, str] | None] = queue.Queue()

    def read_output() -> None:
        for line in process.stdout:
            output_queue.put((time.monotonic(), line))
        output_queue.put(None)

    reader = threading.Thread(target=read_output, daemon=True)
    reader.start()
    launch_time = time.monotonic()
    measurement_start: float | None = launch_time if args.warmup_generations == 0 else None
    measurement_end: float | None = None
    warmup_generations = 0
    measured_episodes = 0
    completed_window = False

    with (
        event_log.open("w", encoding="utf-8") as event_handle,
        raw_log.open("w", encoding="utf-8") as raw_handle,
    ):
        write_event(
            event_handle,
            {
                "type": "run_start",
                "model": args.model,
                "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                "command": command,
                "duration_s": args.duration_seconds,
                "target_episodes": args.target_episodes,
                "warmup_generations": args.warmup_generations,
            },
        )
        try:
            while True:
                now = time.monotonic()
                if measurement_start is None:
                    remaining = args.startup_timeout_seconds - (now - launch_time)
                    if remaining <= 0:
                        raise TimeoutError("warmup did not finish before startup timeout")
                else:
                    measurement_elapsed = now - measurement_start
                    if args.target_episodes is not None:
                        remaining = args.duration_seconds - measurement_elapsed
                        if remaining <= 0:
                            raise TimeoutError(
                                f"target episode count did not finish within "
                                f"{args.duration_seconds:.0f}s"
                            )
                    elif measurement_elapsed < args.duration_seconds:
                        remaining = args.duration_seconds - measurement_elapsed
                    else:
                        remaining = args.max_overrun_seconds - (
                            measurement_elapsed - args.duration_seconds
                        )
                        if remaining <= 0:
                            raise TimeoutError("no generation completed within the allowed overrun")

                try:
                    item = output_queue.get(timeout=min(1.0, max(remaining, 0.01)))
                except queue.Empty:
                    if process.poll() is not None:
                        break
                    continue
                if item is None:
                    break

                received_at, line = item
                raw_handle.write(line)
                raw_handle.flush()
                print(line, end="", flush=True)

                phase = "warmup" if measurement_start is None else "measure"
                offset_s = None if measurement_start is None else received_at - measurement_start
                write_event(
                    event_handle,
                    {
                        "type": "output",
                        "phase": phase,
                        "offset_s": offset_s,
                        "line": line.rstrip("\n"),
                    },
                )
                if measurement_start is None and GENERATION_RE.search(line):
                    warmup_generations += 1
                    if warmup_generations >= args.warmup_generations:
                        # The generation that establishes the boundary remains warmup.
                        # This avoids cross-actor Episode/TPS log reordering.
                        measurement_start = received_at
                        print(
                            f"[TPS BENCHMARK] measurement started for {args.duration_seconds:.0f}s",
                            flush=True,
                        )
                if phase == "measure" and EPISODE_RE.search(line):
                    measured_episodes += 1
                if (
                    args.target_episodes is None
                    and measurement_start is not None
                    and received_at - measurement_start >= args.duration_seconds
                    and GENERATION_RE.search(line)
                ):
                    measurement_end = received_at
                    completed_window = True
                    break
        except BaseException as exc:
            write_event(event_handle, {"type": "runner_error", "error": repr(exc)})
            raise
        finally:
            fallback_measurement_end = time.monotonic()
            stop_process_group(process)
            if args.target_episodes is not None:
                completed_window = measured_episodes >= args.target_episodes
                measurement_end = fallback_measurement_end
            measurement_elapsed_s = (
                0.0
                if measurement_start is None
                else (measurement_end or fallback_measurement_end) - measurement_start
            )
            write_event(
                event_handle,
                {
                    "type": "run_end",
                    "completed_window": completed_window,
                    "measured_episodes": measured_episodes,
                    "target_episodes": args.target_episodes,
                    "measurement_elapsed_s": measurement_elapsed_s,
                    "process_returncode": process.returncode,
                    "utc": dt.datetime.now(dt.timezone.utc).isoformat(),
                },
            )

    parser_script = Path(__file__).with_name("parse_webshop_tps_benchmark.py")
    parse_result = subprocess.run(
        [
            sys.executable,
            str(parser_script),
            "--event-log",
            str(event_log),
            "--window-seconds",
            str(args.duration_seconds),
            "--json-output",
            str(summary_json),
            "--csv-output",
            str(summary_csv),
        ],
        check=False,
    )
    return parse_result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
