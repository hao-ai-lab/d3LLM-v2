#!/bin/bash
# Search SDAR-4B 标准配置 / tpf1 配置评测入口。
#
# 用法：
#   bash eval_both.sh standard
#   bash eval_both.sh tpf1
#   bash eval_both.sh both                 # 默认
#   EVAL_CKPT_DIR=/path/to/ckpt bash eval_both.sh both
#   origin=True bash eval_both.sh both     # 不加载 checkpoint，直接评测 base model
#
# Evaluate a checkpoint trained with LoRA (train_distill_lora.sh):
#   LORA=True MODEL_PATH=<base model used for the LoRA training> EVAL_CKPT_DIR=checkpoints/verl_dllm/<experiment> bash eval_both.sh both
#   LORA_RANK / LORA_ALPHA / LORA_DROPOUT / LORA_TARGET_MODULES must match the training.

set -o pipefail
USER_MODEL_PATH=${MODEL_PATH:-}

EVAL_MODE=${1:-both}
ENGINE=${2:-sglang}
case "${EVAL_MODE}" in
  standard|tpf1|both) ;;
  *)
    echo "ERROR: EVAL_MODE 必须是 standard、tpf1 或 both，当前值: ${EVAL_MODE}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

BASE_EXP_NAME=${EXP_NAME:-dllm_search_sdar_4b_eval}
RUN_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)"
origin=${origin:-${ORIGIN:-False}}

case "${origin}" in
  True|true|TRUE|1)
    RESUME_MODE=disable
    EVAL_TARGET=origin
    EVAL_CKPT_DIR=${EVAL_CKPT_DIR:-"${REPO_ROOT}/temp/eval_scratch/search_sdar_4b_origin"}
    mkdir -p "${EVAL_CKPT_DIR}"
    ;;
  False|false|FALSE|0)
    RESUME_MODE=auto
    EVAL_TARGET=checkpoint
    EVAL_CKPT_DIR=${EVAL_CKPT_DIR:?set EVAL_CKPT_DIR to checkpoints/verl_dllm/<experiment>}
    ;;
  *)
    echo "ERROR: origin 必须是 True 或 False，当前值: ${origin}" >&2
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
  export SEARCH_HISTORY_LOG=${SEARCH_HISTORY_LOG:-1}
fi

MODEL_PATH=${MODEL_PATH:-"${REPO_ROOT}/verl/models/SDAR-4B-Chat"}
TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/searchR1_processed_direct/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/searchR1_processed_direct/test.parquet"}
STANDARD_CONFIG="${REPO_ROOT}/verl/trainer/config/sdar_dllm.yaml"
TPF1_CONFIG="${REPO_ROOT}/verl/trainer/config/sdar_dllm_tpf1.yaml"
SERVER_URL=http://127.0.0.1:8000/retrieve
SERVER_PID=""

if [ "${RESUME_MODE}" = "auto" ]; then
  if [ ! -d "${EVAL_CKPT_DIR}" ]; then
    echo "ERROR: checkpoint 目录不存在: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
  if [ ! -f "${EVAL_CKPT_DIR}/latest_checkpointed_iteration.txt" ]; then
    echo "ERROR: checkpoint 缺少 latest_checkpointed_iteration.txt: ${EVAL_CKPT_DIR}" >&2
    exit 2
  fi
fi
if [ ! -d "${MODEL_PATH}" ]; then
  echo "ERROR: 基础模型目录不存在: ${MODEL_PATH}" >&2
  exit 2
fi
for data_file in "${TRAIN_DATA}" "${VAL_DATA}"; do
  if [ ! -f "${data_file}" ]; then
    echo "ERROR: Search 数据文件不存在: ${data_file}" >&2
    exit 2
  fi
done
for config in "${STANDARD_CONFIG}" "${TPF1_CONFIG}"; do
  if [ ! -f "${config}" ]; then
    echo "ERROR: rollout 配置不存在: ${config}" >&2
    exit 2
  fi
done

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=0
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
mkdir -p "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"

if [ "${lora}" = "True" ]; then
  mc_num=${MC_NUM:-4}
else
  mc_num=${MC_NUM:-3}
fi
n_l=1

train_data_size=${TRAIN_DATA_SIZE:-64}
val_data_size=${VAL_DATA_SIZE:-512}
if [ "${lora}" = "True" ]; then
  group_size=${GROUP_SIZE:-1}
else
  group_size=${GROUP_SIZE:-5}
fi
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
mode=${GIGPO_MODE:-mean_std_norm}
enable_similarity=${ENABLE_SIMILARITY:-True}
similarity_thresh=${SIMILARITY_THRESH:-0.9}

cleanup() {
  if [ -n "${SERVER_PID}" ] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo ">>> 清理 Retrieval Server (PID: ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

if curl -s -o /dev/null "${SERVER_URL}" --connect-timeout 2; then
  echo ">>> 复用已有 Retrieval Server: ${SERVER_URL}"
else
  echo ">>> 启动 Retrieval Server..."
  "${RETRIEVER_PYTHON}" "${REPO_ROOT}/scripts/training/search/retriever/retrieval_server.py" \
    --index_path "$HOME/data/searchR1/e5_Flat.index" \
    --corpus_path "$HOME/data/searchR1/wiki-18.jsonl" \
    --topk 3 \
    --retriever_name e5 \
    --retriever_model intfloat/e5-base-v2 \
    --faiss_gpu \
    --port 8000 \
    > "${REPO_ROOT}/retrieval_server.log" 2>&1 &
  SERVER_PID=$!

  success=false
  for ((i=1; i<=60; i++)); do
    if curl -s -o /dev/null "${SERVER_URL}" --connect-timeout 2; then
      success=true
      echo ">>> Retrieval Server 已就绪"
      break
    fi
    if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
      echo "ERROR: Retrieval Server 意外退出，请查看 retrieval_server.log" >&2
      exit 1
    fi
    echo ">>> 等待 Retrieval Server，第 ${i}/60 次尝试..."
    sleep 5
  done
  if [ "${success}" != "true" ]; then
    echo "ERROR: Retrieval Server 启动超时" >&2
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
    actor_rollout_ref.model.target_modules=all-linear
    actor_rollout_ref.actor.optim.lr="${LR:-1e-5}"
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH:--1}"
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer]
    actor_rollout_ref.actor.strategy=fsdp
    actor_rollout_ref.ref.strategy=fsdp
    +actor_rollout_ref.dllm_config.n_l=1
    +actor_rollout_ref.dllm_config.noisy_backend=all_block_trajectory
    +actor_rollout_ref.dllm_config.log_prob_level=block-level
    +actor_rollout_ref.dllm_config.actor_type=distill
    +actor_rollout_ref.actor.distill_success_threshold="${DISTILL_THRESH:-1.0}"
    +actor_rollout_ref.actor.distill_all="${DISTILL_ALL:-False}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.25}"
    actor_rollout_ref.rollout.max_model_len=4610
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}"
    +actor_rollout_ref.rollout.n_l=1
    actor_rollout_ref.rollout.val_kwargs.n=1
    +actor_rollout_ref.rollout.cfg_scale=0.0
    trainer.resume_mode=auto
    trainer.test_freq="${TEST_FREQ:-10}"
  )
else
  MODE_ARGS=(
    algorithm.gamma=0.95
    algorithm.gigpo.step_advantage_w=1.0
    algorithm.gigpo.mode="${mode}"
    algorithm.gigpo.enable_similarity="${enable_similarity}"
    algorithm.gigpo.similarity_thresh="${similarity_thresh}"
    actor_rollout_ref.actor.optim.lr="${LR:-5e-7}"
    actor_rollout_ref.actor.ppo_mini_batch_size=320
    +actor_rollout_ref.dllm_config.n_l="${n_l}"
    +actor_rollout_ref.dllm_config.actor_type=rl
    +actor_rollout_ref.dllm_config.noisy_backend=random
    +actor_rollout_ref.dllm_config.log_prob_level=sequence-level
    actor_rollout_ref.actor.kl_loss_coef=0.001
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW:-3e-4}"
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH:-4e-4}"
    actor_rollout_ref.rollout.max_num_seqs="${ROLLOUT_MAX_NUM_SEQS:-16}"
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEM_UTIL:-0.4}"
    +actor_rollout_ref.rollout.max_model_len=4610
    +actor_rollout_ref.rollout.n_l="${n_l}"
    trainer.resume_mode="${RESUME_MODE}"
    trainer.test_freq="${TEST_FREQ:-5}"
  )
fi

run_eval() {
  local eval_name=$1
  local rollout_config=$2
  local exp_name="${BASE_EXP_NAME}_${EVAL_TARGET}_${eval_name}_${RUN_TAG}"

  echo "================================================================"
  echo ">>> 开始 ${eval_name} 评测"
  echo ">>> origin: ${origin}"
  echo ">>> resume mode: ${RESUME_MODE}"
  echo ">>> model path: ${MODEL_PATH}"
  echo ">>> local dir: ${EVAL_CKPT_DIR}"
  echo ">>> experiment: ${exp_name}"
  echo ">>> rollout config: ${rollout_config}"
  echo "================================================================"

  if [ "${lora}" = "True" ]; then
    echo ">>> LoRA: rank=${lora_rank}, alpha=${lora_alpha}, dropout=${lora_dropout}, target=${lora_target_modules}, latest step=${LATEST_STEP}"
  fi

  SEARCH_HISTORY_RUN_ID="${exp_name}" \
  "${PYTHON}" -m verl.trainer.main_dllm_new \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    data.train_files="${TRAIN_DATA}" \
    data.val_files="${VAL_DATA}" \
    data.train_batch_size="${train_data_size}" \
    data.val_batch_size="${val_data_size}" \
    data.max_prompt_length=4096 \
    data.max_response_length=512 \
    data.filter_overlong_prompts=True \
    data.filter_overlong_prompts_workers=1 \
    data.truncation=left \
    data.return_raw_chat=True \
    data.trust_remote_code=True \
    data.not_use_template=False \
    +actor_rollout_ref.model.name=sdar \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.trust_remote_code=True \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH:-4}" \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[SDARDecoderLayer] \
    +actor_rollout_ref.rollout.block_length=4 \
    +actor_rollout_ref.dllm_config.mc_num="${mc_num}" \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
    +actor_rollout_ref.actor.use_block_log_prob=False \
    actor_rollout_ref.actor.use_kl_loss=False \
    +actor_rollout_ref.dllm_config.block_size=4 \
    actor_rollout_ref.actor.grad_clip="${GRAD_CLIP:-1.0}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.actor.use_torch_compile=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOGPROB_MB:-8}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name="${ENGINE}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=False \
    +actor_rollout_ref.rollout.mask_invalid_token="${MASK_INVALID_TOKEN:-True}" \
    +actor_rollout_ref.rollout.fixed_teacher=False \
    +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
    +actor_rollout_ref.rollout.dllm_algorithm_config="${rollout_config}" \
    +actor_rollout_ref.rollout.mc_num="${mc_num}" \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.temperature=0.0 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${REF_LOGPROB_MB:-8}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    +actor_rollout_ref.ref.use_dynamic_bsz=False \
    actor_rollout_ref.actor.use_invalid_action_penalty=True \
    actor_rollout_ref.actor.invalid_action_penalty_coef="${INVALID_PENALTY:-1.0}" \
    env.env_name=search \
    env.seed="${ENV_SEED:-0}" \
    env.max_steps=4 \
    env.history_length=4 \
    env.search.search_url="${SERVER_URL}" \
    env.rollout.n="${group_size}" \
    env.resources_per_worker.num_cpus="${num_cpus_per_env_worker}" \
    trainer.logger="['console','wandb']" \
    trainer.project_name=verl_dllm \
    trainer.experiment_name="${exp_name}" \
    trainer.balance_batch=False \
    trainer.n_gpus_per_node="${N_GPUS_PER_NODE:-4}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ:-50}" \
    trainer.total_epochs="${TOTAL_EPOCHS:-300}" \
    trainer.default_local_dir="${EVAL_CKPT_DIR}" \
    trainer.val_before_train=True \
    trainer.val_only=True \
    +ray_init.address=local \
    "${MODE_ARGS[@]}"
  local exit_code=$?

  stop_ray
  if [ "${exit_code}" -eq 0 ]; then
    echo ">>> [OK] ${eval_name} 评测完成"
  else
    echo ">>> [FAIL] ${eval_name} 评测失败，退出码: ${exit_code}" >&2
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
echo ">>> Search SDAR-4B 评测结束，模式: ${EVAL_MODE}，失败数: ${fail_count}"
echo "================================================================"

if [ "${fail_count}" -gt 0 ]; then
  exit 1
fi
