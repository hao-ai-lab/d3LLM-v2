#!/usr/bin/env bash
# Benchmark the WebShop models serially. Each model is a HuggingFace model directory given by an environment variable:
#   SDAR_BASE_MODEL_PATH  SDAR base model (default: verl/models/SDAR-8B-Chat)
#   DISTILL_MODEL_PATH    distilled model, e.g. checkpoints/released_distill/SDAR-8B-Chat-distill-webshop
#   QWEN3_MODEL_PATH      Qwen3-8B (AR baseline)
# MODELS selects the models to run (default: "sdar_base distill qwen3"); the path of every selected model must exist.

set -euo pipefail

ACTION=${1:-all}
case "${ACTION}" in
  smoke|one|run|all) ;;
  *)
    echo "Usage: run_tps_suite.sh [smoke|one|run|all]" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
RESULTS_DIR=${RESULTS_DIR:-"${REPO_ROOT}/outputs/webshop_tps/$(date +%Y%m%d_%H%M%S)"}
BENCH_SECONDS=${BENCH_SECONDS:-300}
SMOKE_SECONDS=${SMOKE_SECONDS:-30}
WARMUP_GENERATIONS=${WARMUP_GENERATIONS:-1}
TARGET_EPISODES=${TARGET_EPISODES:-0}

declare -A MODEL_DIRS=(
  [sdar_base]="${SDAR_BASE_MODEL_PATH:-${REPO_ROOT}/verl/models/SDAR-8B-Chat}"
  [distill]="${DISTILL_MODEL_PATH:-}"
  [qwen3]="${QWEN3_MODEL_PATH:-}"
)
read -r -a MODELS <<< "${MODELS:-sdar_base distill qwen3}"

has_hf_weights() {
  local model_dir=$1
  "${PYTHON}" "${REPO_ROOT}/scripts/Misc/validate_hf_model_dir.py" "${model_dir}" >/dev/null 2>&1
}

run_one() {
  local name=$1
  local duration=$2
  local model_kind=sdar
  [[ "${name}" == qwen3 ]] && model_kind=qwen
  if [[ -z "${MODEL_DIRS[$name]:-}" ]]; then
    echo "ERROR: model path of ${name} is not set (see the header of this script)" >&2
    return 1
  fi
  if ! has_hf_weights "${MODEL_DIRS[$name]}"; then
    echo "ERROR: HuggingFace model is incomplete for ${name}: ${MODEL_DIRS[$name]}" >&2
    return 1
  fi
  mkdir -p "${RESULTS_DIR}"
  local target_args=()
  local val_data_size=${VAL_DATA_SIZE:-500}
  if [[ "${TARGET_EPISODES}" -gt 0 ]]; then
    target_args=(--target-episodes "${TARGET_EPISODES}")
    val_data_size="${TARGET_EPISODES}"
  fi
  CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} \
    VAL_DATA_SIZE="${val_data_size}" \
    "${PYTHON}" "${REPO_ROOT}/scripts/Misc/run_webshop_tps_benchmark.py" \
      --model "${name}" \
      --duration-seconds "${duration}" \
      "${target_args[@]}" \
      --warmup-generations "${WARMUP_GENERATIONS}" \
      --output-dir "${RESULTS_DIR}" \
      -- \
      bash "${SCRIPT_DIR}/benchmark_tps_single_gpu.sh" \
        "${model_kind}" "${MODEL_DIRS[$name]}" "${name}"
}

run_suite() {
  local name
  for name in "${MODELS[@]}"; do
    echo "[BENCHMARK] ${name} for ${BENCH_SECONDS}s"
    run_one "${name}" "${BENCH_SECONDS}"
  done
}

case "${ACTION}" in
  smoke)
    run_one sdar_base "${SMOKE_SECONDS}"
    ;;
  one)
    run_one "${MODEL_NAME:?MODEL_NAME is required for action=one}" "${BENCH_SECONDS}"
    ;;
  run)
    run_suite
    ;;
  all)
    run_one sdar_base "${SMOKE_SECONDS}"
    run_suite
    ;;
esac

echo "[DONE] results: ${RESULTS_DIR}"
