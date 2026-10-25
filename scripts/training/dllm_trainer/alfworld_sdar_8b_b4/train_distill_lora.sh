#!/usr/bin/env bash
# SDAR-8B ALFWorld LoRA online trajectory distillation.
# MODEL_PATH is the merged RL checkpoint (global_step_*/actor_merged_hf) used as teacher and student initialization.

set -euo pipefail
set -x

export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-0}

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the merged RL checkpoint (global_step_*/actor_merged_hf) to distill}
TRAINER_YAML=${TRAINER_YAML:-"${REPO_ROOT}/verl/trainer/config/ppo_trainer.yaml"}
ROLLOUT_YAML=${ROLLOUT_YAML:-"${REPO_ROOT}/verl/trainer/config/sdar_dllm.yaml"}

if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python is not executable: ${PYTHON}" >&2
  exit 1
fi
for required_path in "${TRAINER_YAML}" "${ROLLOUT_YAML}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path not found: ${required_path}" >&2
    exit 1
  fi
done

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-"/tmp/flashinfer_alfworld_sdar8b_distill_lora_${SLURM_JOB_ID:-$$}"}
export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/r_${SLURM_JOB_ID:-$$}"}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
unset RAY_ADDRESS
mkdir -p "${FLASHINFER_CACHE_DIR}" "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi

lora_rank=${LORA_RANK:-64}
lora_alpha=${LORA_ALPHA:-128}
lora_dropout=${LORA_DROPOUT:-0.0}
lora_target_modules=${LORA_TARGET_MODULES:-all-linear}
mc_num=${MC_NUM:-4}
n_l=${N_L:-1}
curriculum_total_steps=${CURRICULUM_TOTAL_STEPS:-300}
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=1

BASE_EXP_NAME=${EXP_NAME:-alfworld_sdar8b_b4_distill_lora}
RUN_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${BASE_EXP_NAME}_r${lora_rank}_a${lora_alpha}_mc${mc_num}_${RUN_TAG}${JOB_TAG:+_${JOB_TAG}}"
CKPT_DIR=${CKPT_DIR:-"${REPO_ROOT}/checkpoints/verl_dllm/${EXP_NAME}"}
DATA_DIR=${DATA_DIR:-"${HOME}/data/verl-agent_${EXP_NAME}"}
mkdir -p "${CKPT_DIR}" "${DATA_DIR}/text"

exec > >(tee -a "${CKPT_DIR}/output.out") 2> >(tee -a "${CKPT_DIR}/error.err" >&2)

rm -f "${DATA_DIR}/text/train.parquet" "${DATA_DIR}/text/test.parquet"
"${PYTHON}" -m scripts.training.data_preprocess.prepare \
  --mode text \
  --local_dir "${DATA_DIR}" \
  --train_data_size "${train_data_size}" \
  --val_data_size "${val_data_size}"

cp "$0" "${CKPT_DIR}/$(basename "$0")"
cp "${TRAINER_YAML}" "${CKPT_DIR}/ppo_trainer.yaml"
cp "${ROLLOUT_YAML}" "${CKPT_DIR}/sdar_dllm.yaml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true
git -C "${REPO_ROOT}" diff HEAD >> "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap stop_ray EXIT

RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
TRAINER_RESUME_MODE=auto
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  TRAINER_RESUME_MODE=resume_path
fi
HYDRA_RESUME_ARGS=("trainer.resume_mode=${TRAINER_RESUME_MODE}")
if [[ "${TRAINER_RESUME_MODE}" == "resume_path" ]]; then
  HYDRA_RESUME_ARGS+=("trainer.resume_from_path=${RESUME_FROM_PATH}")
else
  HYDRA_RESUME_ARGS+=("trainer.resume_from_path=null")
fi

echo "=== SDAR-8B b4 ALFWorld LoRA online trajectory distillation ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== LoRA: rank=${lora_rank}, alpha=${lora_alpha}, dropout=${lora_dropout}, target=${lora_target_modules} ==="
echo "=== train=${train_data_size}, val=${val_data_size}, group=${group_size} ==="
echo "=== mc_num=${mc_num}, block_size=4, curriculum_total_steps=${curriculum_total_steps} ==="
echo "=== EXP_NAME=${EXP_NAME} ==="
echo "=== CKPT_DIR=${CKPT_DIR} ==="

"${PYTHON}" -m verl.trainer.main_dllm_new \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  data.train_files="${DATA_DIR}/text/train.parquet" \
  data.val_files="${DATA_DIR}/text/test.parquet" \
  data.train_batch_size="${train_data_size}" \
  data.val_batch_size="${val_data_size}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-512}" \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=1 \
  data.truncation=error \
  data.return_raw_chat=True \
  data.trust_remote_code=True \
  data.not_use_template=False \
  +actor_rollout_ref.model.name=sdar \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.model.lora_rank="${lora_rank}" \
  actor_rollout_ref.model.lora_alpha="${lora_alpha}" \
  +actor_rollout_ref.model.lora_dropout="${lora_dropout}" \
  actor_rollout_ref.model.target_modules="${lora_target_modules}" \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr="${LR:-1e-5}" \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH:--1}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH:-4}" \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.param_offload="${PARAM_OFFLOAD:-True}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OPTIMIZER_OFFLOAD:-True}" \
  actor_rollout_ref.actor.use_torch_compile=True \
  +actor_rollout_ref.rollout.block_length=4 \
  +actor_rollout_ref.dllm_config.mc_num="${mc_num}" \
  +actor_rollout_ref.dllm_config.n_l="${n_l}" \
  +actor_rollout_ref.actor.cfg_scale=0.0 \
  +actor_rollout_ref.dllm_config.noisy_backend=all_block_trajectory_curriculum \
  +actor_rollout_ref.dllm_config.curriculum_min_mask_ratio="${CURRICULUM_MIN_MASK_RATIO:-0.1}" \
  +actor_rollout_ref.dllm_config.curriculum_initial_max_mask_ratio="${CURRICULUM_INITIAL_MAX_MASK_RATIO:-0.25}" \
  +actor_rollout_ref.dllm_config.curriculum_max_mask_ratio="${CURRICULUM_MAX_MASK_RATIO:-1.0}" \
  +actor_rollout_ref.dllm_config.curriculum_total_steps="${curriculum_total_steps}" \
  +actor_rollout_ref.dllm_config.block_size=4 \
  +actor_rollout_ref.dllm_config.log_prob_level=sequence-level \
  +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
  +actor_rollout_ref.actor.use_block_log_prob=False \
  +actor_rollout_ref.dllm_config.actor_type=distill \
  +actor_rollout_ref.actor.distill_success_threshold="${DISTILL_THRESH:-1.0}" \
  +actor_rollout_ref.actor.distill_all="${DISTILL_ALL:-False}" \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef="${INVALID_PENALTY:-1.0}" \
  actor_rollout_ref.actor.grad_clip="${GRAD_CLIP:-1.0}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
  +actor_rollout_ref.rollout.dllm_algorithm_config="${ROLLOUT_YAML}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MB:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.25}" \
  actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN:-2565}" \
  actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}" \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.mask_invalid_token="${MASK_INVALID_TOKEN:-True}" \
  +actor_rollout_ref.rollout.fixed_teacher=False \
  +actor_rollout_ref.rollout.mc_num="${mc_num}" \
  +actor_rollout_ref.rollout.n_l="${n_l}" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMP:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.cfg_scale=0.0 \
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
  trainer.experiment_name="${EXP_NAME}" \
  trainer.balance_batch=False \
  "${HYDRA_RESUME_ARGS[@]}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.test_freq="${TEST_FREQ:-10}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-300}" \
  trainer.save_trajectory_data=False \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.val_only=False \
  +ray_init.address=local

if [[ "${RUN_MERGE:-0}" == "1" ]]; then
  echo "RUN_MERGE=1 is not supported for LoRA checkpoints."
  echo "Use global_step_*/actor/lora_adapter together with MODEL_PATH."
fi

echo "=== Finished SDAR-8B b4 ALFWorld LoRA online trajectory distillation ==="
