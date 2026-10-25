#!/usr/bin/env bash
# Merge missing checkpoints and benchmark all WebShop 8B models serially.

set -euo pipefail

ACTION=${1:-all}
case "${ACTION}" in
  merge|smoke|one|run|all) ;;
  *)
    echo "Usage: run_tps_suite.sh [merge|smoke|one|run|all]" >&2
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

declare -A ACTOR_DIRS=(
  [d1]="${REPO_ROOT}/checkpoints/verl_dllm/webshop_8b_d1_no_penalty_gdpo_gauss-3_5742_20260820_035833/global_step_60/actor"
  [espo]="${REPO_ROOT}/checkpoints/verl_dllm/webshop_8b_espo_no_penalty_gdpo_gauss-3_5711_20260820_035649/global_step_150/actor"
  [gdpo]="${REPO_ROOT}/checkpoints/verl_dllm/webshop_8b_gdpo_no_penalty_gdpo_gauss-3_5708_20260820_035458/global_step_150/actor"
  [bgpo]="${REPO_ROOT}/checkpoints/verl_dllm/webshop_8b_bgpo_no_penalty_gdpo_gauss-3_5787_20260820_040320/global_step_150/actor"
  [distill]="${REPO_ROOT}/checkpoints/verl_dllm/dllm_webshop_sdar_8b_distill_lc_all_trajectory_distill_lc_block_traj_mc4_5789_20260820_084256/global_step_300/actor"
)
declare -A MODEL_DIRS=(
  [sdar_base]="${REPO_ROOT}/verl/models/SDAR-8B-Chat"
  [d1]="${ACTOR_DIRS[d1]}/merged_model"
  [espo]="${ACTOR_DIRS[espo]}/merged_model"
  [gdpo]="${ACTOR_DIRS[gdpo]}/merged_model"
  [bgpo]="${ACTOR_DIRS[bgpo]}/merged_model"
  [distill]="${ACTOR_DIRS[distill]}/merged_model"
  [qwen3]="${QWEN3_MODEL_PATH:-${HOME}/.cache/huggingface/hub/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218}"
)
MODELS=(sdar_base d1 espo gdpo bgpo distill qwen3)

has_hf_weights() {
  local model_dir=$1
  "${PYTHON}" "${REPO_ROOT}/scripts/Misc/validate_hf_model_dir.py" "${model_dir}" >/dev/null 2>&1
}

merge_models() {
  local name actor_dir target_dir temp_dir
  for name in d1 espo gdpo bgpo distill; do
    actor_dir=${ACTOR_DIRS[$name]}
    target_dir=${MODEL_DIRS[$name]}
    if has_hf_weights "${target_dir}"; then
      echo "[MERGE] reuse ${name}: ${target_dir}"
      continue
    fi
    if [[ ! -f "${actor_dir}/config.json" ]]; then
      echo "ERROR: checkpoint config missing for ${name}: ${actor_dir}" >&2
      return 1
    fi
    echo "[MERGE] ${name}: ${actor_dir} -> ${target_dir}"
    temp_dir="${target_dir}.partial.$$"
    if [[ -e "${temp_dir}" ]]; then
      echo "ERROR: temporary merge directory already exists: ${temp_dir}" >&2
      return 1
    fi
    # Actor checkpoints do not contain modeling_sdar.py; use the matching
    # base model for remote-code config while retaining checkpoint weights.
    "${PYTHON}" "${REPO_ROOT}/scripts/Misc/model_merger.py" merge \
      --backend fsdp \
      --local_dir "${actor_dir}" \
      --hf_model_path "${REPO_ROOT}/verl/models/SDAR-8B-Chat" \
      --target_dir "${temp_dir}"
    if ! has_hf_weights "${temp_dir}"; then
      echo "ERROR: merged model validation failed: ${temp_dir}" >&2
      return 1
    fi
    if [[ -e "${target_dir}" ]]; then
      mv "${target_dir}" "${target_dir}.incomplete.$(date +%Y%m%d_%H%M%S)"
    fi
    mv "${temp_dir}" "${target_dir}"
  done
}

run_one() {
  local name=$1
  local duration=$2
  local model_kind=sdar
  [[ "${name}" == qwen3 ]] && model_kind=qwen
  if ! has_hf_weights "${MODEL_DIRS[$name]}"; then
    echo "ERROR: merged model is incomplete for ${name}: ${MODEL_DIRS[$name]}" >&2
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
  merge)
    merge_models
    ;;
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
    merge_models
    run_one sdar_base "${SMOKE_SECONDS}"
    run_suite
    ;;
esac

echo "[DONE] results: ${RESULTS_DIR}"
