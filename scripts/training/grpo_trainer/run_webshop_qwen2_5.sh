set -x
ENGINE=${1:-sglang}
export VLLM_ATTENTION_BACKEND=XFORMERS
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO

#!/bin/bash
# ===========================================================================
# Qwen2.5-7B GiGPO checkpoint → WebShop train-split 蒸馏轨迹生成
#
# 使用训练好的 HuggingFace checkpoint 在 train split（goal 500+）上 rollout，
#   - raw_obs / raw_next_obs : 环境原始 observation
#   - built_text             : 经模板/历史处理后的 observation（送入模型的文本）
#   - raw_response           : 模型完整原始输出（projection 前）
#   - action                 : projection 后送入环境的动作
#
# 用法（在 repo 根目录）：
#   bash examples/gigpo_trainer/generate_distill_webshop_qwen2_5.sh
#
# 环境变量：
#   CKPT_PATH          HuggingFace 格式 checkpoint 目录（默认 temp/global_step_150_hf）
#   OUTPUT_DIR         蒸馏数据输出根目录
#   BATCH_SIZE         每批并行 episode 数（默认 64）
#   NUM_BATCHES        批次数（默认 100，约 6400 条轨迹）
#   CUDA_VISIBLE_DEVICES
# ===========================================================================
set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

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

if [[ -z "${PYTHON}" ]]; then
  echo "[Monitor] ❌ PYTHON 未设置"; exit 1
fi


num_cpus_per_env_worker=0.1 # The CPU resource allocated for each environment worker. If you want to use less CPU resources, you can decrease this value.

train_data_size=16
val_data_size=500
group_size=8
mode="mean_norm" # "mean_norm" or "mean_std_norm"

# 轨迹 JSONL 写入 trainer.default_local_dir/trajectory_data/{train|val}/
# 训练结束后：
#   python scripts/Misc/convert_trajectory_data_to_distill.py \
#     --input_dir "${CKPT_DIR}/trajectory_data/val" \
#     --output "${DISTILL_OUTPUT_DIR}/distill_trajectories.parquet"
echo "=== Trajectory data ==="
echo "  CKPT_DIR           = ${CKPT_DIR}"
echo "  DISTILL_OUTPUT_DIR = ${DISTILL_OUTPUT_DIR}"
echo "======================="

# We only use data preparation to indicate the modality and the data size.
python3 -m scripts.training.data_preprocess.prepare \
    --mode 'text' \
    --train_data_size $train_data_size \
    --val_data_size $val_data_size # evaluate 2 × val_data_size tasks during each iteration

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
    actor_rollout_ref.model.path="Qwen/Qwen2.5-7B-Instruct" \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=$ENGINE \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.25 \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    actor_rollout_ref.actor.invalid_action_penalty_coef=0.0 \
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
    trainer.experiment_name='grpo_qwen2.5_7b_thinking' \
    trainer.resume_mode='auto' \
    trainer.default_local_dir=${REPO_ROOT}/checkpoints/verl_agent_webshop \
    trainer.save_trajectory_data=False \
    trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4} \
    trainer.nnodes=1 \
    trainer.save_freq=${SAVE_FREQ:-150} \
    trainer.test_freq=${TEST_FREQ:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-150} \
    trainer.val_only=True \
    trainer.val_before_train=True $@
