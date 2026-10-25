#!/bin/bash
# ALFWorld + GiGPO — Ray + SGLang + FSDP
# 参考：gigpo_trainer/run_alfworld.sh（GiGPO / ALFWorld 超参）
#       dllm_trainer/alfworld_sdar_8b_b4/train_rl.sh（SGLang 环境）
set -x
set -euo pipefail
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

TRAINER_YAML="${REPO_ROOT}/verl/trainer/config/ppo_trainer.yaml"

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_cache_${SLURM_JOB_ID:-$$}}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}
mkdir -p "$RAY_TMPDIR"
mkdir -p "${REPO_ROOT}/temp/invalid"
unset RAY_ADDRESS
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false

MODEL_PATH=${MODEL_PATH:-${REPO_ROOT}/dLLM_sft/sfted_models/Qwen3-8B-sfted-alfworld}
BASE_EXP_NAME=${EXP_NAME:-grpo_alfworld_qwen3_8b_sglang}
VERSION_TAG="${SLURM_JOB_ID:-$$}${JOB_TAG:+_${JOB_TAG}}"
VERSION_DIR="${REPO_ROOT}/temp/a_logs/${BASE_EXP_NAME}_${VERSION_TAG}"
mkdir -p "$VERSION_DIR"

exec > >(tee "${VERSION_DIR}/output.out") 2> >(tee "${VERSION_DIR}/error.err" >&2)

EXP_NAME="${BASE_EXP_NAME}_${VERSION_TAG}"
CKPT_DIR=${CKPT_DIR:-checkpoints/verl_agent_alfworld/${EXP_NAME}}
MERGED_MODEL_DIR=${MERGED_MODEL_DIR:-${REPO_ROOT}/checkpoints/verl_agent_alfworld/${EXP_NAME}_merged}
cp "$0" "$VERSION_DIR/run_alfworld_sglang.sh"
cp "$TRAINER_YAML" "$VERSION_DIR/ppo_trainer.yaml"
git -C "${REPO_ROOT}" rev-parse HEAD > "$VERSION_DIR/git_commit.txt" 2>/dev/null || true
git -C "${REPO_ROOT}" diff HEAD >> "$VERSION_DIR/git_commit.txt" 2>/dev/null || true

num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}
mode=${GIGPO_MODE:-mean_std_norm} # "mean_norm" or "mean_std_norm"

echo "=== Experiment snapshot: ${VERSION_DIR} ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== EXP_NAME=${EXP_NAME} ==="
echo "=== CKPT_DIR=${CKPT_DIR} ==="
echo "=== ENGINE=${ENGINE}  GiGPO mode=${mode} ==="

DATA_DIR=${DATA_DIR:-${HOME}/data/verl-agent_${SLURM_JOB_ID:-$$}}
mkdir -p "$DATA_DIR/text"
rm -f "$DATA_DIR/text/train.parquet" "$DATA_DIR/text/test.parquet"
$PYTHON -m scripts.training.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "$DATA_DIR" \
  --train_data_size "$train_data_size" \
  --val_data_size "$val_data_size"

$PYTHON -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="${DATA_DIR}/text/train.parquet" \
  data.val_files="${DATA_DIR}/text/test.parquet" \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=2048 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation='error' \
  data.return_raw_chat=True \
  +data.apply_chat_template_kwargs.enable_thinking=True \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH:-256} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH:-32} \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.01 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB:-32} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=${TP_SIZE:-1} \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.6} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB:-32} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_PENALTY:-1.0} \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=$mode \
  env.env_name=alfworld/AlfredTWEnv \
  env.seed=0 \
  env.max_steps=50 \
  env.rollout.n=$group_size \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  trainer.critic_warmup=0 \
  trainer.logger=['console','wandb'] \
  trainer.project_name='verl_dllm' \
  trainer.experiment_name=$EXP_NAME \
  trainer.balance_batch=False \
  trainer.resume_mode=auto \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-50} \
  trainer.test_freq=${TEST_FREQ:-10} \
  trainer.total_epochs=${TOTAL_EPOCHS:-150} \
  trainer.default_local_dir=${REPO_ROOT}/checkpoints/verl_agent_alfworld/grpo_alfworld_qwen3_8b_sglang_5789 \
  trainer.val_before_train=True \
  trainer.val_only=True \
  +ray_init.address=local \
  "${@:2}"

if [[ "${RUN_MERGE:-0}" = "1" ]]; then
  LATEST_CKPT=$(ls -td "${CKPT_DIR}"/global_step_*/actor 2>/dev/null | head -1)
  if [[ -n "$LATEST_CKPT" ]]; then
    echo "Merging FSDP checkpoint from: $LATEST_CKPT"
    $PYTHON scripts/Misc/model_merger.py merge \
      --backend fsdp \
      --local_dir "$LATEST_CKPT" \
      --hf_model_path "${MODEL_PATH}" \
      --target_dir "$MERGED_MODEL_DIR"
    echo "Merged model saved to: $MERGED_MODEL_DIR"
  else
    echo "No checkpoint found in $CKPT_DIR"
  fi
fi

echo "=== Finished ALFWorld GiGPO SGLang (${EXP_NAME}) ==="
