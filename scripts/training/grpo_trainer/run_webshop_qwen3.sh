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

MODEL_PATH=${MODEL_PATH:-Qwen/Qwen3-8B}
BASE_EXP_NAME=${EXP_NAME:-gigpo_alfworld_qwen3_8b_sglang}
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


python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/verl-agent/text/train.parquet \
    data.val_files=$HOME/data/verl-agent/text/test.parquet \
    data.train_batch_size=$train_data_size \
    data.val_batch_size=$val_data_size \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.return_raw_chat=True \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.path=Qwen/Qwen3-8B \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.1 \
    algorithm.use_kl_in_reward=False \
    algorithm.gamma=0.95 \
    algorithm.gigpo.step_advantage_w=1.0 \
    algorithm.gigpo.mode=$mode \
    env.env_name=Webshop \
    env.seed=1 \
    env.max_steps=15 \
    env.rollout.n=$group_size \
    env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name='verl_agent_webshop' \
    trainer.experiment_name='grpo_qwen3_8b_thinking' \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=5 \
    trainer.total_epochs=150 \
    trainer.val_before_train=True $@