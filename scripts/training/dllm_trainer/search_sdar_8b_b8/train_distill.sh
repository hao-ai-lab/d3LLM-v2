#!/bin/bash
# SDAR-8B b8 Search online trajectory distillation.
# LowConfidence rollout + all_block_trajectory_curriculum CE.
set -x
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
export SEARCH_HISTORY_LOG=${SEARCH_HISTORY_LOG:-1}

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/checkpoints/verl_dllm/search_8b_b8_espo_gdpo_mc3_5788_20260909_062202/global_step_200/actor_merged_hf"}
TRAINER_YAML="${REPO_ROOT}/verl/trainer/config/ppo_trainer.yaml"
# LC algorithm config（b8 单 block LowConfidence）
LC_CFG="${REPO_ROOT}/verl/trainer/config/sdar_bs8.yaml"

# ── Search 数据集路径 ──────────────────────────────────────────────────────────
TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/searchR1_processed_direct/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/searchR1_processed_direct/test.parquet"}

# retriever conda 环境的 Python（装有 faiss）

# ==============================================================================
# 1. 启动 Retrieval Server 并等待就绪（使用 retriever 环境）
# ==============================================================================
echo ">>> [1/3] 启动 Retrieval Server (retriever env)..."
$RETRIEVER_PYTHON "${REPO_ROOT}/scripts/training/search/retriever/retrieval_server.py" \
  --index_path "$HOME/data/searchR1/e5_Flat.index" \
  --corpus_path "$HOME/data/searchR1/wiki-18.jsonl" \
  --topk 3 \
  --retriever_name e5 \
  --retriever_model intfloat/e5-base-v2 \
  --faiss_gpu \
  --port 8000 \
  > "${REPO_ROOT}/retrieval_server.log" 2>&1 &
SERVER_PID=$!

SERVER_URL="http://127.0.0.1:8000/retrieve"
MAX_RETRIES=60
RETRY_INTERVAL=5

echo ">>> 等待 Retrieval Server 就绪 ($SERVER_URL)..."
success=false
for ((i=1; i<=MAX_RETRIES; i++)); do
    if curl -s -o /dev/null "$SERVER_URL" --connect-timeout 2; then
        echo ">>> [OK] Retrieval Server 已响应，开始训练。"
        success=true
        break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then
        echo ">>> [ERROR] Retrieval Server 进程意外退出，请查看 retrieval_server.log"
        exit 1
    fi
    echo ">>> [第 $i/$MAX_RETRIES 次尝试] Server 尚未就绪，等待 ${RETRY_INTERVAL}s..."
    sleep $RETRY_INTERVAL
done

if [ "$success" = false ]; then
    echo ">>> [TIMEOUT] Retrieval Server 在规定时间内未启动，任务终止。"
    kill $SERVER_PID
    exit 1
fi

# ==============================================================================
# 2. 环境变量 & Python 路径
# ==============================================================================
export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}
mkdir -p "$RAY_TMPDIR"
mkdir -p "${REPO_ROOT}/temp/invalid"
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false

# ── 蒸馏超参 ──────────────────────────────────────────────────────────────────
mc_num=${MC_NUM:-4}
n_l=1
train_data_size=${TRAIN_DATA_SIZE:-64}
# num_val_samples：限制验证集条数（Search 全量 test 很大；val_batch_size 已废弃，不能控数量）
val_data_size=${VAL_DATA_SIZE:-512}
num_val_samples=${NUM_VAL_SAMPLES:-1024}
# group_size=1：蒸馏不需要 group 对比
group_size=1
num_cpus_per_env_worker=0.1

BASE_EXP_NAME=${EXP_NAME:-dllm_search_sdar_8b_b8_distill_lc}
TIME_TAG=$(date +%Y%m%d_%H%M%S)
VERSION_TAG="${SLURM_JOB_ID:-$$}_${TIME_TAG}${JOB_TAG:+_${JOB_TAG}}"
EXP_NAME="${BASE_EXP_NAME}_${VERSION_TAG}"
CKPT_DIR=${CKPT_DIR:-"${REPO_ROOT}/checkpoints/verl_dllm/${EXP_NAME}"}
mkdir -p "${CKPT_DIR}"

export SEARCH_HISTORY_RUN_ID="${SEARCH_HISTORY_RUN_ID:-${VERSION_TAG}}"

# 日志与复现记录都直接保存到 CKPT 目录
exec > >(tee "${CKPT_DIR}/output.out") 2> >(tee "${CKPT_DIR}/error.err" >&2)

# 快照：当前执行的脚本 + 关键 yaml
cp "$0" "${CKPT_DIR}/$(basename "$0")"
cp "$TRAINER_YAML" "${CKPT_DIR}/ppo_trainer.yaml"
cp "$LC_CFG" "${CKPT_DIR}/sdar_bs8.yaml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true
git -C "${REPO_ROOT}" diff HEAD >> "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true
echo "=== Repro info saved to CKPT dir: ${CKPT_DIR} ==="

# ── Resume 逻辑 ───────────────────────────────────────────────────────────────
RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
TRAINER_RESUME_MODE=auto
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  TRAINER_RESUME_MODE=resume_path
fi
HYDRA_RESUME_ARGS=( "trainer.resume_mode=${TRAINER_RESUME_MODE}" )
if [[ "${TRAINER_RESUME_MODE}" == "resume_path" ]]; then
  HYDRA_RESUME_ARGS+=( "trainer.resume_from_path=${RESUME_FROM_PATH}" )
else
  HYDRA_RESUME_ARGS+=( "trainer.resume_from_path=null" )
fi

echo "=== Experiment snapshot: ${CKPT_DIR} ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== EXP_NAME: ${EXP_NAME} ==="
echo "=== CKPT_DIR: ${CKPT_DIR} ==="
echo "=== ALGO: Online Distillation  noisy_backend=all_block_trajectory_curriculum  block_size=8 ==="
echo "=== ROLLOUT: LowConfidence  mc_num=${mc_num}  group_size=${group_size} ==="
echo "=== LR=${LR:-5e-7}  GRAD_CLIP=${GRAD_CLIP:-1.0} ==="
echo "=== TRAIN_DATA=${TRAIN_DATA} ==="
echo "=== VAL_DATA=${VAL_DATA}  num_val_samples=${num_val_samples} ==="

# ==============================================================================
# 3. 启动 RL 训练
# ==============================================================================
echo ">>> [2/3] 开始运行主训练程序..."

export TORCHDYNAMO_DISABLE=0
$PYTHON -m verl.trainer.main_dllm_new \
  algorithm.adv_estimator=grpo \
  algorithm.norm_adv_by_std_in_grpo=True \
  algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size=$train_data_size \
  data.val_batch_size=$val_data_size \
  +data.num_val_samples=$num_val_samples \
  data.max_prompt_length=4096 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.filter_overlong_prompts_workers=1 \
  data.truncation='left' \
  data.return_raw_chat=True \
  data.trust_remote_code=True \
  data.not_use_template=False \
  +actor_rollout_ref.model.name=sdar \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.trust_remote_code=True \
  actor_rollout_ref.actor.optim.lr=${LR:-5e-7} \
  actor_rollout_ref.actor.optim.weight_decay=0.01 \
  actor_rollout_ref.model.use_remove_padding=False \
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH:-128} \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH:-4} \
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
  actor_rollout_ref.actor.strategy=fsdp \
  actor_rollout_ref.ref.strategy=fsdp \
  actor_rollout_ref.actor.fsdp_config.param_offload=True \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.use_torch_compile=True \
  +actor_rollout_ref.rollout.block_length=8 \
  +actor_rollout_ref.dllm_config.mc_num=$mc_num \
  +actor_rollout_ref.dllm_config.n_l=$n_l \
  +actor_rollout_ref.dllm_config.block_size=8 \
  +actor_rollout_ref.actor.cfg_scale=0.0 \
  +actor_rollout_ref.dllm_config.noisy_backend=all_block_trajectory_curriculum \
  +actor_rollout_ref.dllm_config.log_prob_level=block-level \
  +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
  +actor_rollout_ref.actor.use_block_log_prob=False \
  +actor_rollout_ref.dllm_config.actor_type=distill \
  +actor_rollout_ref.actor.distill_success_threshold=${DISTILL_THRESH:-1.0} \
  +actor_rollout_ref.actor.distill_all=${DISTILL_ALL:-False} \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.use_invalid_action_penalty=True \
  actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_PENALTY:-1.0} \
  actor_rollout_ref.actor.grad_clip=${GRAD_CLIP:-1.0} \
  actor_rollout_ref.rollout.name=$ENGINE \
  +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
  +actor_rollout_ref.rollout.dllm_algorithm_config="${REPO_ROOT}/verl/trainer/config/sdar_bs8.yaml" \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB:-8} \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.25} \
  actor_rollout_ref.rollout.max_model_len=4610 \
  actor_rollout_ref.rollout.enable_chunked_prefill=False \
  actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.free_cache_engine=False \
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16} \
  +actor_rollout_ref.rollout.mask_invalid_token=${MASK_INVALID_TOKEN:-True} \
  +actor_rollout_ref.rollout.fixed_teacher=False \
  +actor_rollout_ref.rollout.mc_num=$mc_num \
  +actor_rollout_ref.rollout.n_l=$n_l \
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
  actor_rollout_ref.rollout.temperature=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=True \
  +actor_rollout_ref.rollout.cfg_scale=0.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB:-8} \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  +actor_rollout_ref.ref.use_dynamic_bsz=False \
  env.env_name=search \
  env.seed=0 \
  env.max_steps=4 \
  env.history_length=4 \
  env.search.search_url='http://127.0.0.1:8000/retrieve' \
  env.rollout.n=$group_size \
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker \
  trainer.logger=['console','wandb'] \
  trainer.project_name='verl_dllm' \
  trainer.experiment_name=$EXP_NAME \
  trainer.balance_batch=False \
  "${HYDRA_RESUME_ARGS[@]}" \
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4} \
  trainer.nnodes=1 \
  trainer.save_freq=${SAVE_FREQ:-50} \
  trainer.test_freq=${TEST_FREQ:-10} \
  trainer.total_epochs=${TOTAL_EPOCHS:-300} \
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-300} \
  trainer.default_local_dir=$CKPT_DIR \
  trainer.val_before_train=True \
  trainer.val_only=False \
  +ray_init.address=local

TRAIN_EXIT=$?
FINAL_EXIT=$TRAIN_EXIT

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

# ==============================================================================
# 4. 根据 checkpoint 完整性决定是否评测，不依赖训练进程退出码
# ==============================================================================
CHECKPOINT_READY=0
LATEST_STEP=""
if [ -s "${CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
  LATEST_STEP=$(tr -d '[:space:]' < "${CKPT_DIR}/latest_checkpointed_iteration.txt")
  if [ -n "${LATEST_STEP}" ] && \
     [ -d "${CKPT_DIR}/global_step_${LATEST_STEP}/actor" ]; then
    CHECKPOINT_READY=1
  fi
fi

if [ "${AUTO_EVAL:-1}" != "1" ]; then
  echo ">>> AUTO_EVAL=0，跳过自动评测。"
elif [ "$CHECKPOINT_READY" != "1" ]; then
  echo ">>> [WARN] 未找到完整 checkpoint，跳过自动评测。"
  echo ">>> 检查文件: ${CKPT_DIR}/latest_checkpointed_iteration.txt"
elif [ "$TRAIN_EXIT" -ne 0 ]; then
  echo ">>> [WARN] 训练进程退出码为 ${TRAIN_EXIT}，但 global_step_${LATEST_STEP} 完整，继续评测。"
fi

if [ "${AUTO_EVAL:-1}" = "1" ] && [ "$CHECKPOINT_READY" = "1" ]; then
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi

  if [ "${AUTO_EVAL_TPF1:-1}" = "1" ]; then
    EVAL_MODE=both
    echo ">>> 开始 standard + tpf1 自动评测: ${CKPT_DIR}"
  else
    EVAL_MODE=standard
    echo ">>> AUTO_EVAL_TPF1=0，仅运行 standard 自动评测: ${CKPT_DIR}"
  fi

  EVAL_CKPT_DIR="${CKPT_DIR}" \
  MODEL_PATH="${MODEL_PATH}" \
  EXP_NAME="${EXP_NAME}_post_train" \
    bash "${SCRIPT_DIR}/eval_both.sh" "${EVAL_MODE}" "${ENGINE}"
  EVAL_EXIT=$?

  if [ "$EVAL_EXIT" -ne 0 ]; then
    echo ">>> [ERROR] 自动评测失败，退出码: ${EVAL_EXIT}"
    if [ "$FINAL_EXIT" -eq 0 ]; then
      FINAL_EXIT=$EVAL_EXIT
    fi
  fi
fi

# ==============================================================================
# 5. 清理 Retrieval Server
# ==============================================================================
echo ">>> 训练和评测结束，清理后台 Retrieval Server (PID: $SERVER_PID)..."
kill "$SERVER_PID"
wait "$SERVER_PID" 2>/dev/null

echo "=== Finished Search SDAR-8B-b8 Online Distillation + LC (all_block_trajectory_curriculum, block_size=8, mc_num=${mc_num}) ==="
exit $FINAL_EXIT
