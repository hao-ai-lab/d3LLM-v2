#!/usr/bin/env bash
# Run seven independent single-GPU WebShop TPS benchmarks on two allocations.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
HEAD_JOB_ID=${HEAD_JOB_ID:-5788}
WORKER_JOB_ID=${WORKER_JOB_ID:-5789}
BENCH_SECONDS=${BENCH_SECONDS:-300}
WARMUP_GENERATIONS=${WARMUP_GENERATIONS:-1}
TARGET_EPISODES=${TARGET_EPISODES:-0}
RESULTS_DIR=${RESULTS_DIR:-"${REPO_ROOT}/outputs/webshop_tps/parallel_$(date +%Y%m%d_%H%M%S)"}

run_worker() {
  local models=("$@")
  local index=0
  local pids=()
  local model
  for model in "${models[@]}"; do
    echo "[LAUNCH] host=$(hostname) gpu=${index} model=${model}"
    (
      export CUDA_VISIBLE_DEVICES="${index}"
      # Each local SGLang engine needs a distinct base port. Otherwise engines
      # started together inherit the same random seed and can select one NCCL port.
      export SGLANG_PORT="$((20000 + index * 2000))"
      export RESULTS_DIR BENCH_SECONDS WARMUP_GENERATIONS TARGET_EPISODES
      export RAY_TMPDIR="/tmp/ray_tps_${SLURM_JOB_ID}_${model}"
      export DATA_DIR="${HOME}/data/webshop_tps_${SLURM_JOB_ID}_${model}"
      export RUN_DIR="${REPO_ROOT}/outputs/webshop_tps/runs/${SLURM_JOB_ID}_${model}"
      export MODEL_NAME="${model}"
      bash "${SCRIPT_DIR}/run_tps_suite.sh" one
    ) >"${RESULTS_DIR}/${model}.launcher.log" 2>&1 &
    pids+=("$!")
    index=$((index + 1))
  done

  local failed=0
  local pid
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  return "${failed}"
}

if [[ "${1:-}" == "--worker" ]]; then
  shift
  mkdir -p "${RESULTS_DIR}"
  run_worker "$@"
  exit $?
fi

mkdir -p "${RESULTS_DIR}"
echo "[TPS SUITE] results=${RESULTS_DIR}"
echo "[TPS SUITE] job ${HEAD_JOB_ID}: sdar_base d1 espo gdpo"
echo "[TPS SUITE] job ${WORKER_JOB_ID}: bgpo distill qwen3"

child_pids=()
cleanup() {
  local pid
  for pid in "${child_pids[@]:-}"; do
    kill "${pid}" >/dev/null 2>&1 || true
  done
}
trap cleanup TERM INT

srun --jobid="${HEAD_JOB_ID}" --overlap --nodes=1 --ntasks=1 \
  bash "${BASH_SOURCE[0]}" --worker sdar_base d1 espo gdpo &
child_pids+=("$!")
srun --jobid="${WORKER_JOB_ID}" --overlap --nodes=1 --ntasks=1 \
  bash "${BASH_SOURCE[0]}" --worker bgpo distill qwen3 &
child_pids+=("$!")

failed=0
for pid in "${child_pids[@]}"; do
  if ! wait "${pid}"; then
    failed=1
  fi
done

if [[ "${failed}" -ne 0 ]]; then
  echo "ERROR: at least one benchmark failed; inspect ${RESULTS_DIR}/*.launcher.log" >&2
  exit 1
fi

"${PYTHON}" - "${RESULTS_DIR}" <<'PY'
import csv
import sys
from pathlib import Path

root = Path(sys.argv[1])
expected = {"sdar_base", "d1", "espo", "gdpo", "bgpo", "distill", "qwen3"}
with (root / "summary.csv").open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
found = {row["model"] for row in rows if row.get("completed_window") == "True"}
missing = expected - found
if missing:
    raise SystemExit(f"incomplete benchmark results: {sorted(missing)}")
print(root / "summary.csv")
PY

echo "[DONE] all seven benchmarks completed"
