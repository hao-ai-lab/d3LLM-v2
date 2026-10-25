#!/usr/bin/env bash
# SDAR-8B b4 + ESPO: ALFWorld multi-turn RL with Ray, SGLang, and FSDP.

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
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/dLLM_sft/sfted_models/SDAR-8B-Chat-sfted-alfworld"}
TRAINER_YAML=${TRAINER_YAML:-"${REPO_ROOT}/verl/trainer/config/ppo_trainer.yaml"}
ROLLOUT_YAML=${ROLLOUT_YAML:-"${REPO_ROOT}/verl/trainer/config/sdar_dllm.yaml"}

for required_path in "${MODEL_PATH}" "${TRAINER_YAML}" "${ROLLOUT_YAML}"; do
  if [[ ! -e "${required_path}" ]]; then
    echo "ERROR: required path not found: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python is not executable: ${PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-"/tmp/flashinfer_alfworld_sdar8b_b4_${SLURM_JOB_ID:-$$}"}
export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/ray_alfworld_sdar8b_b4_${SLURM_JOB_ID:-$$}"}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
unset RAY_ADDRESS
mkdir -p "${FLASHINFER_CACHE_DIR}" "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"

mc_num=${MC_NUM:-3}
n_l=${N_L:-1}
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}
mode=${GIGPO_MODE:-mean_norm}

BASE_EXP_NAME=${EXP_NAME:-alfworld_sdar8b_b4_espo}
RUN_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)"
EXP_NAME="${BASE_EXP_NAME}_mc${mc_num}_${RUN_TAG}${JOB_TAG:+_${JOB_TAG}}"
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

echo "=== SDAR-8B b4 ALFWorld ESPO ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== train=${train_data_size}, val=${val_data_size}, group=${group_size} ==="
echo "=== mc_num=${mc_num}, block_size=4, mode=${mode} ==="
echo "=== EXP_NAME=${EXP_NAME} ==="
echo "=== CKPT_DIR=${CKPT_DIR} ==="

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
  actor_rollout_ref.actor.optim.lr="${LR:-5e-7}" \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH:-512}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH:-8}" \
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
  actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW:-3e-4}" \
  actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-4e-4}" \
  actor_rollout_ref.actor.grad_clip="${GRAD_CLIP:-1.0}" \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef="${INVALID_PENALTY:-1.0}" \
  actor_rollout_ref.actor.invalid_action_penalty_bias="${INVALID_PENALTY_BIAS:-0.0}" \
  actor_rollout_ref.rollout.name="${ENGINE}" \
  +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
  +actor_rollout_ref.rollout.dllm_algorithm_config="${ROLLOUT_YAML}" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MB:-16}" \
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
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB:-16}" \
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
  trainer.resume_mode="${TRAINER_RESUME_MODE:-auto}" \
  trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
  trainer.nnodes=1 \
  trainer.save_freq="${SAVE_FREQ:-30}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-150}" \
  trainer.save_trajectory_data=False \
  trainer.default_local_dir="${CKPT_DIR}" \
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.val_only=False \
  +ray_init.address=local

echo "=== Finished SDAR-8B b4 ALFWorld ESPO ==="
