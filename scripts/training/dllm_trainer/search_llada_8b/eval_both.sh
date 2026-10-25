#!/usr/bin/env bash
# Search d3LLaDA-8B standard / TPF1 evaluation.
#
# Usage:
#   bash eval_both.sh standard
#   bash eval_both.sh tpf1
#   bash eval_both.sh both                 # default
#   EVAL_CKPT_DIR=/path/to/ckpt bash eval_both.sh both
#   origin=True bash eval_both.sh both     # evaluate the base model
#
# Evaluate a checkpoint trained with LoRA (train_distill_lora.sh):
#   LORA=True MODEL_PATH=<base model used for the LoRA training> EVAL_CKPT_DIR=checkpoints/verl_dllm/<experiment> bash eval_both.sh both
#   LORA_RANK / LORA_ALPHA / LORA_DROPOUT / LORA_TARGET_MODULES must match the training.

set -o pipefail
USER_MODEL_PATH=${MODEL_PATH:-}
USER_EVAL_CKPT_DIR=${EVAL_CKPT_DIR:-}

EVAL_MODE=${1:-both}
ENGINE=${2:-sglang}
case "${EVAL_MODE}" in
  standard|tpf1|both) ;;
  *)
    echo "ERROR: EVAL_MODE must be standard, tpf1, or both; got ${EVAL_MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

DEFAULT_CKPT="${REPO_ROOT}/checkpoints/verl_dllm/dllm_search_d3llada8b_distill_full_distill_block_trajectory_curriculum_b32_mc16_5742_20260903_040554"
BASE_EXP_NAME=${EXP_NAME:-dllm_search_d3llada8b_eval}
RUN_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)"
origin=${origin:-${ORIGIN:-False}}

case "${origin}" in
  True|true|TRUE|1)
    RESUME_MODE=disable
    EVAL_TARGET=origin
    EVAL_CKPT_DIR=${EVAL_CKPT_DIR:-"${REPO_ROOT}/temp/eval_scratch/search_d3llada8b_origin"}
    mkdir -p "${EVAL_CKPT_DIR}"
    ;;
  False|false|FALSE|0)
    RESUME_MODE=auto
    EVAL_TARGET=checkpoint
    EVAL_CKPT_DIR=${EVAL_CKPT_DIR:-"${DEFAULT_CKPT}"}
    ;;
  *)
    echo "ERROR: origin must be True or False; got ${origin}" >&2
    exit 2
    ;;
esac

lora=${lora:-${LORA:-False}}
case "${lora}" in
  True|true|TRUE|1) lora=True ;;
  False|false|FALSE|0) lora=False ;;
  *)
    echo "ERROR: LORA must be True or False; got ${lora}" >&2
    exit 2
    ;;
esac
lora_rank=${LORA_RANK:-128}
lora_alpha=${LORA_ALPHA:-256}
lora_dropout=${LORA_DROPOUT:-0.0}
lora_target_modules=${LORA_TARGET_MODULES:-all-linear}
LATEST_STEP=""
if [ "${lora}" = "True" ]; then
  case "${origin}" in
    True|true|TRUE|1)
      echo "ERROR: origin=True and LORA=True cannot be used together" >&2
      exit 2
      ;;
  esac
  EVAL_TARGET=lora_checkpoint
  EVAL_CKPT_DIR=${USER_EVAL_CKPT_DIR:?set EVAL_CKPT_DIR to the LoRA training checkpoint directory}
  MODEL_PATH=${USER_MODEL_PATH:?set MODEL_PATH to the base model used for the LoRA training}
  if [ ! -f "${MODEL_PATH}/config.json" ]; then
    echo "ERROR: invalid base model directory: ${MODEL_PATH}" >&2
    exit 2
  fi
  if [ ! -s "${EVAL_CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: checkpoint missing latest_checkpointed_iteration.txt: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
  LATEST_STEP=$(tr -d '[:space:]' < "${EVAL_CKPT_DIR}/latest_checkpointed_iteration.txt")
  if [ -z "${LATEST_STEP}" ] || [ ! -d "${EVAL_CKPT_DIR}/global_step_${LATEST_STEP}/actor" ]; then
    echo "ERROR: latest actor checkpoint is incomplete: global_step_${LATEST_STEP}" >&2
    exit 2
  fi
fi

MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/verl/models/d3LLM_LLaDA"}
TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/searchR1_processed_direct/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/searchR1_processed_direct/test.parquet"}
STANDARD_CONFIG=${STANDARD_CONFIG:-"${REPO_ROOT}/verl/trainer/config/llada.yaml"}
TPF1_CONFIG=${TPF1_CONFIG:-"${REPO_ROOT}/verl/trainer/config/llada_tpf1.yaml"}
RETRIEVER_INDEX=${RETRIEVER_INDEX:-"$HOME/data/searchR1/e5_Flat.index"}
RETRIEVER_CORPUS=${RETRIEVER_CORPUS:-"$HOME/data/searchR1/wiki-18.jsonl"}
SERVER_URL=${SERVER_URL:-"http://127.0.0.1:8000/retrieve"}
SERVER_PID=""

if [[ "${RESUME_MODE}" == "auto" ]]; then
  if [[ ! -d "${EVAL_CKPT_DIR}" ]]; then
    echo "ERROR: checkpoint directory does not exist: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
  LATEST_FILE="${EVAL_CKPT_DIR}/latest_checkpointed_iteration.txt"
  if [[ ! -s "${LATEST_FILE}" ]]; then
    echo "ERROR: checkpoint is missing latest_checkpointed_iteration.txt: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
  LATEST_STEP=$(tr -d '[:space:]' < "${LATEST_FILE}")
  if [[ -z "${LATEST_STEP}" || ! -d "${EVAL_CKPT_DIR}/global_step_${LATEST_STEP}/actor" ]]; then
    echo "ERROR: latest actor checkpoint is incomplete: global_step_${LATEST_STEP}" >&2
    exit 2
  fi
fi

for required_path in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_DATA}" \
  "${VAL_DATA}" \
  "${STANDARD_CONFIG}" \
  "${TPF1_CONFIG}" \
  "${RETRIEVER_INDEX}" \
  "${RETRIEVER_CORPUS}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path does not exist: ${required_path}" >&2
    exit 2
  fi
done

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_d3llada_eval_${SLURM_JOB_ID:-$$}}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_d3llada_eval_${SLURM_JOB_ID:-$$}}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
unset RAY_ADDRESS

mkdir -p "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"

mc_num=${MC_NUM:-3}
n_l=1

train_data_size=${TRAIN_DATA_SIZE:-64}
if [[ "${lora}" == "True" ]]; then
  val_data_size=${VAL_DATA_SIZE:-256}
else
  val_data_size=${VAL_DATA_SIZE:-512}
fi
num_val_samples=${NUM_VAL_SAMPLES:-null}
group_size=${GROUP_SIZE:-5}
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
max_prompt_length=${MAX_PROMPT_LENGTH:-3580}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
max_model_len=4096
if (( max_prompt_length + max_response_length >= max_model_len )); then
  echo "ERROR: MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH=$((max_prompt_length + max_response_length)) 必须小于 max_model_len=${max_model_len}" >&2
  exit 2
fi

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo ">>> Cleaning Retrieval Server (PID: ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if curl -s -o /dev/null "${SERVER_URL}" --connect-timeout 2; then
  echo ">>> Reusing Retrieval Server: ${SERVER_URL}"
else
  echo ">>> Starting Retrieval Server..."
  "${RETRIEVER_PYTHON}" "${REPO_ROOT}/scripts/training/search/retriever/retrieval_server.py" \
    --index_path "${RETRIEVER_INDEX}" \
    --corpus_path "${RETRIEVER_CORPUS}" \
    --topk "${RETRIEVER_TOPK:-3}" \
    --retriever_name e5 \
    --retriever_model "${RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
    --faiss_gpu \
    --port "${RETRIEVER_PORT:-8000}" \
    > "${REPO_ROOT}/retrieval_server.log" 2>&1 &
  SERVER_PID=$!

  success=false
  for ((i=1; i<=${MAX_RETRIES:-60}; i++)); do
    if curl -s -o /dev/null "${SERVER_URL}" --connect-timeout 2; then
      success=true
      echo ">>> Retrieval Server is ready"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: Retrieval Server exited; see retrieval_server.log" >&2
      exit 1
    fi
    sleep "${RETRY_INTERVAL:-5}"
  done
  if [[ "${success}" != "true" ]]; then
    echo "ERROR: Retrieval Server startup timed out" >&2
    exit 1
  fi
fi

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}

MODE_ARGS=()
if [ "${lora}" = "True" ]; then
  MODE_ARGS=(
    actor_rollout_ref.model.lora_rank="${lora_rank}"
    actor_rollout_ref.model.lora_alpha="${lora_alpha}"
    +actor_rollout_ref.model.lora_dropout="${lora_dropout}"
    actor_rollout_ref.model.target_modules="${lora_target_modules}"
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}"
    trainer.resume_mode=auto
  )
else
  MODE_ARGS=(
    actor_rollout_ref.actor.optim.lr="${LR:-5e-7}"
    trainer.resume_mode="${RESUME_MODE}"
  )
fi

run_eval() {
  local eval_name=$1
  local rollout_config=$2
  local exp_name="${BASE_EXP_NAME}_${EVAL_TARGET}_${eval_name}_${RUN_TAG}"

  echo "================================================================"
  echo ">>> Starting ${eval_name} evaluation"
  echo ">>> target: ${EVAL_TARGET}"
  echo ">>> base model: ${MODEL_PATH}"
  echo ">>> checkpoint: ${EVAL_CKPT_DIR}"
  echo ">>> rollout config: ${rollout_config}"
  echo ">>> samples: ${num_val_samples}"
  echo "================================================================"

  if [ "${lora}" = "True" ]; then
    echo ">>> LoRA: rank=${lora_rank}, alpha=${lora_alpha}, dropout=${lora_dropout}, target=${lora_target_modules}, latest step=${LATEST_STEP}"
  fi

  SEARCH_HISTORY_RUN_ID="${exp_name}" \
  "${PYTHON}" -m verl.trainer.main_dllm_new \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    +data.num_val_samples="${num_val_samples}" \
    data.max_prompt_length="${max_prompt_length}" \
    data.max_response_length="${max_response_length}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation=left \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.not_use_template=False \
    env.search.use_sdar_prompt=True \
    +actor_rollout_ref.model.name=llada \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    +actor_rollout_ref.model.attn_implementation=flash_attention_2 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH:-128}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH:-2}" \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock] \
    actor_rollout_ref.actor.strategy=fsdp \
    actor_rollout_ref.ref.strategy=fsdp \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_dynamic_bsz=False \
    +actor_rollout_ref.dllm_config.mc_num="${mc_num}" \
    +actor_rollout_ref.dllm_config.n_l="${n_l}" \
    +actor_rollout_ref.dllm_config.actor_type=rl \
    +actor_rollout_ref.dllm_config.noisy_backend=random \
    +actor_rollout_ref.dllm_config.log_prob_level=sequence-level \
    +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
    +actor_rollout_ref.dllm_config.p_mask_normalization=True \
    +actor_rollout_ref.dllm_config.approx_per_mc_update=False \
    +actor_rollout_ref.dllm_config.block_size=-1 \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.use_block_log_prob=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    +actor_rollout_ref.rollout.dllm_algorithm=FullAttnMultiBlock \
    +actor_rollout_ref.rollout.dllm_algorithm_config="${rollout_config}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MB:-4}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.25}" \
    actor_rollout_ref.rollout.max_model_len="${max_model_len}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}" \
    +actor_rollout_ref.rollout.mask_invalid_token="${MASK_INVALID_TOKEN:-True}" \
    +actor_rollout_ref.rollout.num_workers="${ROLLOUT_WORKERS:-1}" \
    +actor_rollout_ref.rollout.num_gpus_per_worker="${ROLLOUT_GPUS:-4}" \
    +actor_rollout_ref.rollout.mc_num="${mc_num}" \
    +actor_rollout_ref.rollout.n_l="${n_l}" \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    +actor_rollout_ref.rollout.dllm_full_response_mask=True \
    actor_rollout_ref.rollout.temperature="${EVAL_TEMPERATURE:-0.0}" \
    actor_rollout_ref.rollout.do_sample=False \
    actor_rollout_ref.rollout.val_kwargs.n=1 \
    actor_rollout_ref.rollout.val_kwargs.temperature="${EVAL_TEMPERATURE:-0.0}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=False \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB:-8}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.use_dynamic_bsz=False \
    env.env_name=search \
    env.seed="${ENV_SEED:-0}" \
    env.max_steps="${MAX_STEPS:-4}" \
    env.history_length="${HISTORY_LENGTH:-4}" \
    env.search.search_url="${SERVER_URL}" \
    env.rollout.n="${group_size}" \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_dllm \
    trainer.experiment_name="${exp_name}" \
    trainer.balance_batch=False \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
    trainer.nnodes="${NNODES:-1}" \
    trainer.save_freq=50 \
    trainer.test_freq=5 \
    trainer.total_epochs=300 \
    trainer.default_local_dir="${EVAL_CKPT_DIR}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    +ray_init.address=local \
    "${MODE_ARGS[@]}"
  local exit_code=$?

  stop_ray
  if [[ "${exit_code}" -eq 0 ]]; then
    echo ">>> [OK] ${eval_name} evaluation completed"
  else
    echo ">>> [FAIL] ${eval_name} evaluation failed with code ${exit_code}" >&2
  fi
  return "${exit_code}"
}

fail_count=0
case "${EVAL_MODE}" in
  standard)
    run_eval standard "${STANDARD_CONFIG}" || fail_count=1
    ;;
  tpf1)
    run_eval tpf1 "${TPF1_CONFIG}" || fail_count=1
    ;;
  both)
    run_eval standard "${STANDARD_CONFIG}" || fail_count=$((fail_count + 1))
    run_eval tpf1 "${TPF1_CONFIG}" || fail_count=$((fail_count + 1))
    ;;
esac

echo "================================================================"
echo ">>> Search d3LLaDA-8B evaluation finished; mode=${EVAL_MODE}, failures=${fail_count}"
echo "================================================================"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
