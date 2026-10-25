#!/usr/bin/env bash
# ALFWorld SDAR-8B b4 standard / TPF1 evaluation.
#
# Usage:
#   EVAL_CKPT_DIR=/path/to/training/checkpoint bash eval_both.sh both
#   EVAL_CKPT_DIR=/path/to/training/checkpoint bash eval_both.sh standard
#   EVAL_CKPT_DIR=/path/to/training/checkpoint bash eval_both.sh tpf1
#   origin=True bash eval_both.sh both
#
# Evaluate a checkpoint trained with LoRA (train_distill_lora.sh):
#   LORA=True MODEL_PATH=<base model used for the LoRA training> EVAL_CKPT_DIR=checkpoints/verl_dllm/<experiment> bash eval_both.sh both
#   LORA_RANK / LORA_ALPHA / LORA_DROPOUT / LORA_TARGET_MODULES must match the training.

set -u
set -o pipefail
USER_MODEL_PATH=${MODEL_PATH:-}

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

MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/verl/models/SDAR-8B-Chat"}
BASE_EXP_NAME=${EXP_NAME:-dllm_alfworld_sdar8b_b4_eval}
RUN_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)"
STANDARD_CONFIG=${STANDARD_CONFIG:-"${REPO_ROOT}/verl/trainer/config/sdar_dllm.yaml"}
TPF1_CONFIG=${TPF1_CONFIG:-"${REPO_ROOT}/verl/trainer/config/sdar_dllm_tpf1.yaml"}

origin=${origin:-${ORIGIN:-False}}
case "${origin}" in
  True|true|TRUE|1)
    RESUME_MODE=disable
    EVAL_TARGET=origin
    EVAL_CKPT_DIR=${EVAL_CKPT_DIR:-"${REPO_ROOT}/temp/eval_scratch/alfworld_sdar8b_b4_origin"}
    mkdir -p "${EVAL_CKPT_DIR}"
    ;;
  False|false|FALSE|0)
    RESUME_MODE=auto
    EVAL_TARGET=checkpoint
    EVAL_CKPT_DIR=${EVAL_CKPT_DIR:-${CKPT_DIR:-}}
    if [[ -z "${EVAL_CKPT_DIR}" ]]; then
      echo "ERROR: set EVAL_CKPT_DIR to the ALFWorld training checkpoint directory" >&2
      exit 2
    fi
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
lora_rank=${LORA_RANK:-64}
lora_alpha=${LORA_ALPHA:-128}
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
  export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
fi

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: invalid model directory: ${MODEL_PATH}" >&2
  exit 2
fi
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python is not executable: ${PYTHON}" >&2
  exit 2
fi
for config in "${STANDARD_CONFIG}" "${TPF1_CONFIG}"; do
  if [[ ! -f "${config}" ]]; then
    echo "ERROR: rollout config not found: ${config}" >&2
    exit 2
  fi
done
if [[ "${RESUME_MODE}" == auto ]]; then
  if [[ ! -d "${EVAL_CKPT_DIR}" ]]; then
    echo "ERROR: checkpoint directory not found: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
  if [[ ! -f "${EVAL_CKPT_DIR}/latest_checkpointed_iteration.txt" ]]; then
    echo "ERROR: checkpoint missing latest_checkpointed_iteration.txt: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
fi

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-"/tmp/flashinfer_alfworld_sdar8b_b4_eval_${SLURM_JOB_ID:-$$}"}
export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/r_${SLURM_JOB_ID:-$$}"}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-0}
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
unset RAY_ADDRESS
mkdir -p "${FLASHINFER_CACHE_DIR}" "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"

mc_num=${MC_NUM:-3}
n_l=${N_L:-1}
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}
mode=${GIGPO_MODE:-mean_norm}

DATA_DIR=${DATA_DIR:-"${HOME}/data/verl-agent_alfworld_eval_${RUN_TAG}"}
mkdir -p "${DATA_DIR}/text"
rm -f "${DATA_DIR}/text/train.parquet" "${DATA_DIR}/text/test.parquet"
if ! "${PYTHON}" -m scripts.training.data_preprocess.prepare \
  --mode text \
  --local_dir "${DATA_DIR}" \
  --train_data_size "${train_data_size}" \
  --val_data_size "${val_data_size}"; then
  echo "ERROR: ALFWorld evaluation data preparation failed" >&2
  exit 1
fi

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap stop_ray EXIT

MODE_ARGS=()
if [ "${lora}" = "True" ]; then
  MODE_ARGS=(
    actor_rollout_ref.model.lora_rank="${lora_rank}"
    actor_rollout_ref.model.lora_alpha="${lora_alpha}"
    +actor_rollout_ref.model.lora_dropout="${lora_dropout}"
    actor_rollout_ref.model.target_modules="${lora_target_modules}"
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}"
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer]
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.ref.strategy=fsdp
    actor_rollout_ref.rollout.val_kwargs.n=1
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
  echo ">>> origin=${origin}"
  echo ">>> resume_mode=${RESUME_MODE}"
  echo ">>> model_path=${MODEL_PATH}"
  echo ">>> checkpoint_dir=${EVAL_CKPT_DIR}"
  echo ">>> rollout_config=${rollout_config}"
  echo "================================================================"

  if [ "${lora}" = "True" ]; then
    echo ">>> LoRA: rank=${lora_rank}, alpha=${lora_alpha}, dropout=${lora_dropout}, target=${lora_target_modules}, latest step=${LATEST_STEP}"
  fi

  "${PYTHON}" -m verl.trainer.main_dllm_new \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode="${mode}" \
    data.train_files="${DATA_DIR}/text/train.parquet" \
    data.val_files="${DATA_DIR}/text/test.parquet" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-2565}" \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation=error \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.not_use_template=False \
    +actor_rollout_ref.model.name=sdar \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH:-128}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH:-4}" \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
    actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-True}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-True}" \
    actor_rollout_ref.actor.use_torch_compile=True \
    +actor_rollout_ref.rollout.block_length=4 \
    +actor_rollout_ref.dllm_config.mc_num="${mc_num}" \
    +actor_rollout_ref.dllm_config.n_l="${n_l}" \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.dllm_config.actor_type=rl \
    +actor_rollout_ref.dllm_config.noisy_backend=random \
    +actor_rollout_ref.dllm_config.log_prob_level=sequence-level \
    +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
    +actor_rollout_ref.dllm_config.approx_per_mc_update=False \
    +actor_rollout_ref.dllm_config.block_size=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW:-1.2e-3}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-1.6e-3}" \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP:-1.0}" \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef="${INVALID_PENALTY:-1.0}" \
    actor_rollout_ref.actor.invalid_action_penalty_bias="${INVALID_PENALTY_BIAS:-0.0}" \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
    +actor_rollout_ref.rollout.dllm_algorithm_config="${rollout_config}" \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MB:-8}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.25}" \
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.mask_invalid_token="${MASK_INVALID_TOKEN:-True}" \
    +actor_rollout_ref.rollout.fixed_teacher=False \
    +actor_rollout_ref.rollout.mc_num="${mc_num}" \
    +actor_rollout_ref.rollout.n_l="${n_l}" \
    actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMP:-1.0}" \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB:-8}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.use_dynamic_bsz=False \
    env.env_name=alfworld/AlfredTWEnv \
    env.seed="${ENV_SEED:-0}" \
    env.max_steps="${MAX_STEPS:-50}" \
    env.rollout.n="${group_size}" \
    env.resources_per_worker.num_cpus="${NUM_CPUS_PER_ENV_WORKER:-0.1}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_dllm \
    trainer.experiment_name="${exp_name}" \
    trainer.balance_batch=False \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-30}" \
    trainer.test_freq="${TEST_FREQ:-5}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-150}" \
    trainer.save_trajectory_data=False \
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
    echo ">>> [FAIL] ${eval_name} evaluation failed, exit code: ${exit_code}" >&2
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
echo ">>> ALFWorld SDAR-8B b4 evaluation finished: mode=${EVAL_MODE}, failures=${fail_count}"
echo "================================================================"

if [[ "${fail_count}" -gt 0 ]]; then
  exit 1
fi
