#!/bin/bash
# SDAR-4B + ESPO: WebShop 多轮 RL — Ray + SGLang + FSDP
# 算法：ESPO（random Monte Carlo，关闭 BGPO 逐样本近似）
# Actor log-prob: sequence-level.
set -x
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
CONDA_ENV_PREFIX=${CONDA_ENV_PREFIX:-${ENV_PREFIX}}
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/dLLM_sft/sfted_models/SDAR-4B-Chat-sfted-webshop"}
TRAINER_YAML="${REPO_ROOT}/verl/trainer/config/ppo_trainer.yaml"
ROLLOUT_YAML="${REPO_ROOT}/verl/trainer/config/sdar_dllm.yaml"

if [[ ! -x "${CONDA_ENV_PREFIX}/bin/python" ]]; then
  echo "ERROR: conda environment not found: ${CONDA_ENV_PREFIX}" >&2
  exit 1
fi
if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model not found: ${MODEL_PATH}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export JAVA_HOME=${JAVA_HOME:-"${CONDA_ENV_PREFIX}/lib/jvm"}
export PATH="${CONDA_ENV_PREFIX}/bin:${JAVA_HOME}/bin:${PATH}"
export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}
mkdir -p "$RAY_TMPDIR"
mkdir -p "${REPO_ROOT}/temp/invalid"
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
PYTHON=${PYTHON:-"${CONDA_ENV_PREFIX}/bin/python"}

# ── ESPO sampling: mc_num controls the number of noisy samples ──
mc_num=${MC_NUM:-3}

BASE_EXP_NAME=${EXP_NAME:-webshop_4b_espo}
TIME_TAG=$(date +%Y%m%d_%H%M%S)
VERSION_TAG="${SLURM_JOB_ID:-$$}_${TIME_TAG}${JOB_TAG:+_${JOB_TAG}}"
EXP_NAME="${BASE_EXP_NAME}_${VERSION_TAG}"
CKPT_DIR=${CKPT_DIR:-"${REPO_ROOT}/checkpoints/verl_dllm/${EXP_NAME}"}
mkdir -p "${CKPT_DIR}"

# 日志与复现记录都直接保存到 CKPT 目录
exec > >(tee "${CKPT_DIR}/output.out") 2> >(tee "${CKPT_DIR}/error.err" >&2)

# 快照：当前执行的脚本 + 关键 yaml
cp "$0" "${CKPT_DIR}/$(basename "$0")"
cp "$TRAINER_YAML" "${CKPT_DIR}/ppo_trainer.yaml"
cp "$ROLLOUT_YAML" "${CKPT_DIR}/sdar_dllm.yaml"
# 保存 git commit 和未提交的已跟踪改动
git -C "${REPO_ROOT}" rev-parse HEAD > "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true
git -C "${REPO_ROOT}" diff HEAD >> "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true
echo "=== Repro info saved to CKPT dir: ${CKPT_DIR} ==="

train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=${GROUP_SIZE:-8}
num_cpus_per_env_worker=0.1

mode="mean_norm" # "mean_norm" or "mean_std_norm"

# Monte Carlo samples per update (actor & rollout must agree)
n_l=1

echo "=== Experiment snapshot: ${CKPT_DIR} ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== EXP_NAME: ${EXP_NAME} ==="
echo "=== CKPT_DIR: ${CKPT_DIR} ==="
echo "=== mc_num=${mc_num} ==="
echo "=== LR=${LR:-1e-6} GRAD_CLIP=${GRAD_CLIP:-1.0} MINI_BATCH=${MINI_BATCH:-128} MICRO_BATCH=${MICRO_BATCH:-4} ==="

DATA_DIR=${DATA_DIR:-$HOME/data/verl-agent_${SLURM_JOB_ID:-$$}}
export TOKENIZERS_PARALLELISM=false
mkdir -p "$DATA_DIR/text"
rm -f "$DATA_DIR/text/train.parquet" "$DATA_DIR/text/test.parquet"
$PYTHON -m scripts.training.data_preprocess.prepare \
  --mode 'text' \
  --local_dir "$DATA_DIR" \
  --train_data_size "$train_data_size" \
  --val_data_size "$val_data_size"

export TORCHDYNAMO_DISABLE=0
$PYTHON -m verl.trainer.main_dllm_new \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  algorithm.gamma=0.95 \
  algorithm.gigpo.step_advantage_w=1.0 \
  algorithm.gigpo.mode=$mode \
  data.train_files="$DATA_DIR/text/train.parquet" \
  data.val_files="$DATA_DIR/text/test.parquet" \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  data.max_prompt_length=4096 \
  data.max_response_length=768 \
  actor_rollout_ref.rollout.max_model_len=4866 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=1 \
  data.truncation='error' \
  data.return_raw_chat=True \
  data.trust_remote_code=True \
  data.not_use_template=False \
  +actor_rollout_ref.model.name=sdar \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=${LR:-1e-6} \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=128 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH:-4} \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
  +actor_rollout_ref.rollout.block_length=4 \
  +actor_rollout_ref.dllm_config.mc_num=$mc_num \
  +actor_rollout_ref.dllm_config.n_l=$n_l \
  +actor_rollout_ref.actor.cfg_scale=0.0 \
  +actor_rollout_ref.dllm_config.actor_type=rl \
  +actor_rollout_ref.dllm_config.noisy_backend=random \
  +actor_rollout_ref.dllm_config.log_prob_level=sequence-level \
  +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
  +actor_rollout_ref.dllm_config.approx_per_mc_update=False \
  actor_rollout_ref.actor.use_kl_loss=False \
  +actor_rollout_ref.dllm_config.block_size=4 \
  actor_rollout_ref.actor.kl_loss_coef=0.001 \
  actor_rollout_ref.actor.clip_ratio_low=${CLIP_RATIO_LOW:-3e-4} \
  actor_rollout_ref.actor.clip_ratio_high=${CLIP_RATIO_HIGH:-4e-4} \
  actor_rollout_ref.actor.grad_clip=${GRAD_CLIP:-1.0} \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16} \
  actor_rollout_ref.actor.use_torch_compile=True \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB:-8} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.name=$ENGINE \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.25} \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  +actor_rollout_ref.rollout.mask_invalid_token=${MASK_INVALID_TOKEN:-True} \
  +actor_rollout_ref.rollout.fixed_teacher=False \
  +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
  +actor_rollout_ref.rollout.dllm_algorithm_config=verl/trainer/config/sdar_dllm.yaml \
  +actor_rollout_ref.rollout.mc_num=$mc_num \
  +actor_rollout_ref.rollout.n_l=$n_l \
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB:-8} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.use_dynamic_bsz=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_PENALTY:-1.0} \
  env.env_name=Webshop \
  env.seed=1 \
  env.max_steps=15 \
  env.rollout.n=$group_size \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  trainer.logger=['console','wandb'] \
  trainer.project_name='verl_dllm' \
  trainer.experiment_name=$EXP_NAME \
  trainer.balance_batch=False \
  trainer.resume_mode=auto \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-30} \
  trainer.test_freq=${TEST_FREQ:-5} \
  trainer.total_epochs=${TOTAL_EPOCHS:-150} \
  trainer.save_trajectory_data=False \
  trainer.default_local_dir=${CKPT_DIR} \
  trainer.val_before_train=True \
  trainer.val_only=False \
  +ray_init.address=local

if [ "${RUN_MERGE:-0}" = "1" ]; then
  LATEST_CKPT=$(ls -td "${CKPT_DIR}"/global_step_*/actor 2>/dev/null | head -1)
  if [ -n "$LATEST_CKPT" ]; then
    echo "Merging FSDP checkpoint from: $LATEST_CKPT"
    $PYTHON scripts/Misc/model_merger.py merge \
      --backend fsdp \
      --local_dir "$LATEST_CKPT" \
      --target_dir "${CKPT_DIR}/merged_model"
    echo "Merged model saved to: ${CKPT_DIR}/merged_model"
  fi
fi

echo "=== Finished WebShop SDAR-4B ESPO sequence-level (mc_num=${mc_num}) ==="
