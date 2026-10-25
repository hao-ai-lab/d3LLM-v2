#!/bin/bash
# WebShop d3LLM_DREAM full SFT — same data pipeline as LoRA version, trains all weights
#
# Usage:
#   bash temp/d3llm/d3llm/d3llm_DREAM/webshop_training/train_webshop_sft_full_4gpu.sh
#
# Background:
#   setsid bash temp/d3llm/d3llm/d3llm_DREAM/webshop_training/train_webshop_sft_full_4gpu.sh \
#     >> dream_sft_full_log.txt 2>&1 < /dev/null &

set -euo pipefail
set -x
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
DISTILL_DIR="$(cd "${SCRIPT_DIR}/../distill_2_training" && pwd)"

# ── 环境 ──────────────────────────────────────────────────────────────────────
export VERL_AGENT_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-${HOME}/.cache/huggingface/datasets}"
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"

export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_cache_eval_lc_$$}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_eval_lc_$$}
mkdir -p "$RAY_TMPDIR"
unset RAY_ADDRESS
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1

# DeepSpeed Triton autotune 在 NFS 上会 hang，cache 放到本地 /tmp
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_$$}"
mkdir -p "${TRITON_CACHE_DIR}"


TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON}")/torchrun}"
N_GPUS="${N_GPUS:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export CUDA_VISIBLE_DEVICES

# 与 LoRA 版一致用 torchrun 启动；ZeRO-2 由 d3llm_dream_train.py 内 TrainingArguments 配置
# 勿用 python -m deepspeed 启动，NFS 上 Triton autotune 容易卡住
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
export D3LLM_TRAIN_CONFIG="${D3LLM_TRAIN_CONFIG:-d3llm_webshop_train_full.yaml}"

MODEL_PATH="${MODEL_PATH:-d3LLM/d3LLM_Dream}"
DISTILL_RAW="${DISTILL_RAW:-${REPO_ROOT}/temp/nju_downloads/distill/webshop_qwen3_8b_grpo_step150/distill_trajectories_rl150_excluded_half.parquet}"
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/temp/nju_downloads/distill/webshop_qwen3_8b_grpo_step150/d3llm_dream_train_half.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dLLM_sft/sfted_models/ckpt/d3LLM_Dream-sfted-webshop-full-half}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
LR="${LR:-1e-5}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] PYTHON not executable: ${PYTHON}"
  exit 1
fi

if [[ ! -f "${DISTILL_RAW}" ]]; then
  echo "[ERROR] Raw distill parquet not found: ${DISTILL_RAW}"
  exit 1
fi

# ── Step 1: Convert raw distill trajectories → qa parquet ─────────────────
if [[ "${SKIP_DATA_PREP:-0}" != "1" && ! -f "${TRAIN_FILE}" ]]; then
  echo "[INFO] Converting raw distill trajectories to DREAM qa parquet: ${TRAIN_FILE}"
  "${PYTHON}" "${SCRIPT_DIR}/../../d3llm_LLaDA/distill_1_data_prepare/convert_distill_to_llada_parquet.py" \
    --input "${DISTILL_RAW}" \
    --output "${TRAIN_FILE}" \
    --model_name "${MODEL_PATH}" \
    --max_length "${MAX_LENGTH}"
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[ERROR] Preprocessed train parquet not found: ${TRAIN_FILE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo " WebShop d3LLM_DREAM FULL SFT (d3llm_dream_train.py)"
echo "========================================="
echo "REPO_ROOT         = ${REPO_ROOT}"
echo "DISTILL_DIR       = ${DISTILL_DIR}"
echo "D3LLM_TRAIN_CONFIG= ${D3LLM_TRAIN_CONFIG}"
echo "MODEL_PATH        = ${MODEL_PATH}"
echo "TRAIN_FILE        = ${TRAIN_FILE}"
echo "DISTILL_RAW       = ${DISTILL_RAW}"
echo "OUTPUT_DIR        = ${OUTPUT_DIR}"
echo "NUM_EPOCHS        = ${NUM_EPOCHS}"
echo "N_GPUS            = ${N_GPUS}"
echo "MAX_LENGTH        = ${MAX_LENGTH}"
echo "PER_DEVICE_BS     = ${PER_DEVICE_BS}"
echo "GRAD_ACCUM        = ${GRAD_ACCUM}"
echo "LR                = ${LR}"
echo "USE_DEEPSPEED     = ${USE_DEEPSPEED}"
echo "========================================="

cd "${DISTILL_DIR}"

TRAIN_ARGS=(
  model.name="${MODEL_PATH}"
  data.train_files="${TRAIN_FILE}"
  training.output_dir="${OUTPUT_DIR}"
  training.num_train_epochs="${NUM_EPOCHS}"
  training.per_device_train_batch_size="${PER_DEVICE_BS}"
  training.gradient_accumulation_steps="${GRAD_ACCUM}"
  training.learning_rate="${LR}"
  lora.enabled=false
  distillation.max_length="${MAX_LENGTH}"
  distillation.use_naive_random_mask=true
  distillation.trajectory_dataset_path=null
)

export TRAINING_TIMESTAMP="${TRAINING_TIMESTAMP:-$(date +%m%d_%H%M%S)}"

if [[ "${N_GPUS}" -le 1 ]]; then
  echo "[INFO] Single-GPU full SFT"
  "${PYTHON}" d3llm_dream_train.py "${TRAIN_ARGS[@]}"
elif [[ "${USE_DEEPSPEED}" == "1" ]] && "${PYTHON}" -c "import deepspeed" 2>/dev/null; then
  echo "[INFO] Launching full SFT with deepspeed (${N_GPUS} GPUs, ZeRO-2)"
  "${PYTHON}" -m deepspeed --num_gpus="${N_GPUS}" d3llm_dream_train.py "${TRAIN_ARGS[@]}"
elif [[ -x "${TORCHRUN}" ]]; then
  echo "[INFO] Launching full SFT with ${TORCHRUN} (${N_GPUS} GPUs, DDP)"
  "${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
    d3llm_dream_train.py "${TRAIN_ARGS[@]}"
else
  "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
    d3llm_dream_train.py "${TRAIN_ARGS[@]}"
fi

bash "${REPO_ROOT}/scripts/Misc/sync_verl_modeling.sh" "${MODEL_PATH}" "${OUTPUT_DIR}_${SLURM_JOB_ID:-local}_${TRAINING_TIMESTAMP}"
bash "${REPO_ROOT}/scripts/Misc/export_sfted_model.sh" "${OUTPUT_DIR}_${SLURM_JOB_ID:-local}_${TRAINING_TIMESTAMP}" "${REPO_ROOT}/dLLM_sft/sfted_models/$(basename "${OUTPUT_DIR}")"
