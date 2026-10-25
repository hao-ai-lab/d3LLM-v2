#!/bin/bash
# WebShop d3LLM SFT for iLLaDA — uses d3llm_illada_train.py + question/llm_response parquet
#
# Usage:
#   bash temp/d3llm/d3llm/d3llm_iLLaDA/webshop_training/train_webshop_sft.sh
#
# Background:
#   nohup bash temp/d3llm/d3llm/d3llm_iLLaDA/webshop_training/train_webshop_sft.sh >> sft_log_illada.txt 2>&1 &

set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"
DISTILL_DIR="$(cd "${SCRIPT_DIR}/../distill_2_training" && pwd)"

export VERL_AGENT_ROOT="${REPO_ROOT}"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/raid/qyy/HuggingFace/datasets}"
export HF_HOME="${HF_HOME:-/raid/qyy/HuggingFace}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export D3LLM_TRAIN_CONFIG="${D3LLM_TRAIN_CONFIG:-d3llm_webshop_train.yaml}"

PYTHON="${PYTHON:-/home/yif034/miniconda3/envs/verl-agent-new/bin/python3}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON}")/torchrun}"
N_GPUS="${N_GPUS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1,3}"
export CUDA_VISIBLE_DEVICES

MODEL_PATH="${MODEL_PATH:-GSAI-ML/iLLaDA-8B-Instruct}"
# 使用 Qwen3-8B GRPO checkpoint 生成的 WebShop 蒸馏轨迹数据
TRAIN_FILE="${TRAIN_FILE:-${REPO_ROOT}/data/distill/webshop_qwen3_8b_grpo_step150/distill_trajectories.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/dLLM_sft/sfted_models/ckpt/iLLaDA-8B-Instruct-sfted-webshop-qwen3}"
NUM_EPOCHS="${NUM_EPOCHS:-3}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-5}"

if [[ ! -x "${PYTHON}" ]]; then
  echo "[ERROR] PYTHON not executable: ${PYTHON}"
  exit 1
fi

if [[ ! -f "${TRAIN_FILE}" ]]; then
  echo "[ERROR] Train parquet not found: ${TRAIN_FILE}"
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "========================================="
echo " WebShop d3LLM SFT (iLLaDA-8B-Instruct)"
echo "========================================="
echo "REPO_ROOT    = ${REPO_ROOT}"
echo "DISTILL_DIR  = ${DISTILL_DIR}"
echo "MODEL_PATH   = ${MODEL_PATH}"
echo "TRAIN_FILE   = ${TRAIN_FILE} (webshop_distill, auto-detect)"
echo "OUTPUT_DIR   = ${OUTPUT_DIR}"
echo "N_GPUS       = ${N_GPUS}"
echo "MAX_LENGTH   = ${MAX_LENGTH}"
echo "========================================="

cd "${DISTILL_DIR}"

TRAIN_ARGS=(
  model.name="${MODEL_PATH}"
  data.train_files="${TRAIN_FILE}"
  data.format=webshop_distill
  data.success_only=true
  training.output_dir="${OUTPUT_DIR}"
  training.num_train_epochs="${NUM_EPOCHS}"
  training.per_device_train_batch_size="${PER_DEVICE_BS}"
  training.gradient_accumulation_steps="${GRAD_ACCUM}"
  training.learning_rate="${LR}"
  distillation.max_length="${MAX_LENGTH}"
  distillation.use_naive_random_mask=true
  distillation.trajectory_dataset_path=null
)

export TRAINING_TIMESTAMP="${TRAINING_TIMESTAMP:-$(date +%m%d_%H%M%S)}"

if [[ "${N_GPUS}" -le 1 ]]; then
  echo "[INFO] Single-GPU training"
  "${PYTHON}" d3llm_illada_train.py "${TRAIN_ARGS[@]}"
elif [[ "${USE_DEEPSPEED:-0}" == "1" ]] && "${PYTHON}" -c "import deepspeed" 2>/dev/null; then
  echo "[INFO] Launching with deepspeed (${N_GPUS} GPUs)"
  "${PYTHON}" -m deepspeed --num_gpus="${N_GPUS}" d3llm_illada_train.py "${TRAIN_ARGS[@]}"
elif [[ -x "${TORCHRUN}" ]]; then
  echo "[INFO] Launching with ${TORCHRUN} (${N_GPUS} GPUs, DDP)"
  "${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
    d3llm_illada_train.py "${TRAIN_ARGS[@]}"
else
  "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
    d3llm_illada_train.py "${TRAIN_ARGS[@]}"
fi

bash "${REPO_ROOT}/scripts/Misc/sync_verl_modeling.sh" "${MODEL_PATH}" "${OUTPUT_DIR}_${SLURM_JOB_ID:-local}_${TRAINING_TIMESTAMP}"
bash "${REPO_ROOT}/scripts/Misc/export_sfted_model.sh" "${OUTPUT_DIR}_${SLURM_JOB_ID:-local}_${TRAINING_TIMESTAMP}" "${REPO_ROOT}/dLLM_sft/sfted_models/$(basename "${OUTPUT_DIR}")"
