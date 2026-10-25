#!/usr/bin/env bash
# Full-parameter WebShop SFT: iLLaDA uses <think> targets without a tag instruction.
#
# Usage:
#   bash dLLM_sft/d3LLM/run_sft.sh <preset>
#
# Presets (SFT model written to dLLM_sft/sfted_models/):
#   webshop-llada-8b   d3LLM_LLaDA-sfted-webshop
#   webshop-illada-8b  iLLaDA-8B-Instruct-sfted-webshop
#
# Optional environment variables:
#   MODEL_PATH TRAIN_FILE OUTPUT_DIR EXPORT_DIR   paths (defaults below)
#   NUM_EPOCHS                                    epochs (default: LLaDA 3, iLLaDA 6)
#   PER_DEVICE_BS LR MAX_LENGTH                    defaults: 2, 1e-5, 4096
#   GRAD_ACCUM                                    keeps the global batch at 16, i.e. 2 on 4 GPUs
#   N_GPUS CUDA_VISIBLE_DEVICES                   GPUs (default 4 on 0,1,2,3; N_GPUS defaults to the number of
#                                                 devices in CUDA_VISIBLE_DEVICES when that is set)
#   TRAINING_TIMESTAMP                            suffix of the training output directory (default: current time)
#   REPORT_TO                                     none (default), or wandb
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REPO_ROOT="${ROOT}"

PRESET="${1:-}"
case "${PRESET}" in
    webshop-llada-8b)
        MODEL_NAME=d3LLM_LLaDA
        REASONING_TAG=reason
        PROMPT_STYLE="explicit <reason> tags"
        MODEL_FAMILY=d3llm_LLaDA
        TRAIN_SCRIPT=d3llm_llada_train.py
        TRAIN_CONFIG=d3llm_webshop_train_full.yaml
        DEFAULT_LR=1e-5
        DEFAULT_NUM_EPOCHS=3
        ;;
    webshop-illada-8b)
        MODEL_NAME=iLLaDA-8B-Instruct
        REASONING_TAG=think
        PROMPT_STYLE="reasoning without explicit tag instruction"
        MODEL_FAMILY=d3llm_iLLaDA
        TRAIN_SCRIPT=d3llm_illada_train.py
        TRAIN_CONFIG=d3llm_webshop_train_full.yaml
        DEFAULT_LR=1e-5
        DEFAULT_NUM_EPOCHS=6
        ;;
    *)
        sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac
OUT_NAME="${MODEL_NAME}-sfted-webshop"
DISTILL_DIR="${ROOT}/dLLM_sft/d3LLM/d3llm/${MODEL_FAMILY}/distill_2_training"

source "${ROOT}/scripts/Misc/common_env.sh"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" && -z "${N_GPUS:-}" ]]; then
    IFS=, read -r -a visible_devices <<< "${CUDA_VISIBLE_DEVICES}"
    N_GPUS="${#visible_devices[@]}"
fi
N_GPUS="${N_GPUS:-4}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

MODEL_PATH="${MODEL_PATH:-${ROOT}/verl/models/${MODEL_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${ROOT}/dLLM_sft/data/d3llm/webshop_sft_tags.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dLLM_sft/sfted_models/ckpt/${OUT_NAME}}"
NUM_EPOCHS="${NUM_EPOCHS:-${DEFAULT_NUM_EPOCHS}}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
PER_DEVICE_BS="${PER_DEVICE_BS:-2}"
# The released model was trained with a global batch of 16 (4 GPUs x 2 x 2 accumulation steps). The default number of
# accumulation steps keeps the global batch at 16 for other GPU counts.
GLOBAL_BATCH=16
GRAD_ACCUM="${GRAD_ACCUM:-$((GLOBAL_BATCH / (N_GPUS * PER_DEVICE_BS) > 0 ? GLOBAL_BATCH / (N_GPUS * PER_DEVICE_BS) : 1))}"
LR="${LR:-${DEFAULT_LR}}"
# DeepSpeed ZeRO-2 (with CPU optimizer offload) is configured in TrainingArguments of the selected trainer, so torchrun
# (the default) already trains with it. USE_DEEPSPEED=1 starts the same script with the deepspeed launcher instead
# (`python -m deepspeed` has no __main__, so the launcher module is called directly). Triton autotune can hang on NFS
# under this launcher, hence the default is 0.
USE_DEEPSPEED="${USE_DEEPSPEED:-0}"
TORCHRUN="${TORCHRUN:-$(dirname "${PYTHON}")/torchrun}"
TRAINING_TIMESTAMP="${TRAINING_TIMESTAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${OUTPUT_DIR}_${SLURM_JOB_ID:-local}_${TRAINING_TIMESTAMP}"
# Export to a stable model directory for subsequent RL.
EXPORT_DIR="${EXPORT_DIR:-${ROOT}/dLLM_sft/sfted_models/${OUT_NAME}}"

if [[ ! -x "${PYTHON}" ]]; then
    echo "ERROR: Python is not executable: ${PYTHON}" >&2
    exit 1
fi
for p in "${MODEL_PATH}/config.json" "${TRAIN_FILE}"; do
    if [[ ! -e "${p}" ]]; then
        echo "ERROR: not found: ${p}" >&2
        echo "       the training data can be regenerated with dLLM_sft/data/README.md" >&2
        exit 1
    fi
done

export PYTHONUNBUFFERED=1
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=1
export NCCL_TIMEOUT="${NCCL_TIMEOUT:-10800000}"
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC="${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}"
export GLOO_TIMEOUT_MS="${GLOO_TIMEOUT_MS:-10800000}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/triton_cache_${USER:-user}_$$}"
export D3LLM_TRAIN_CONFIG="${D3LLM_TRAIN_CONFIG:-${TRAIN_CONFIG}}"
export TRAINING_TIMESTAMP
mkdir -p "${TRITON_CACHE_DIR}" "${OUTPUT_DIR%/*}"

echo "======================================================"
echo " d3LLM SFT: ${PRESET}  -  $(date)"
echo " base model : ${MODEL_PATH}"
echo " data       : ${TRAIN_FILE}"
echo " targets    : <${REASONING_TAG}>...</${REASONING_TAG}>"
echo " prompt     : ${PROMPT_STYLE}"
echo " config     : ${D3LLM_TRAIN_CONFIG}"
echo " output     : ${RUN_DIR}"
echo " GPUs       : ${N_GPUS} (CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES})"
echo " epochs=${NUM_EPOCHS} max_length=${MAX_LENGTH} per_device_bs=${PER_DEVICE_BS} grad_accum=${GRAD_ACCUM} lr=${LR}"
echo "======================================================"

TRAIN_ARGS=(
    model.name="${MODEL_PATH}"
    data.train_files="${TRAIN_FILE}"
    training.output_dir="${OUTPUT_DIR}"
    training.num_train_epochs="${NUM_EPOCHS}"
    training.per_device_train_batch_size="${PER_DEVICE_BS}"
    training.gradient_accumulation_steps="${GRAD_ACCUM}"
    training.learning_rate="${LR}"
    training.report_to="${REPORT_TO:-none}"
    lora.enabled=false
    distillation.max_length="${MAX_LENGTH}"
    distillation.use_naive_random_mask=true
    distillation.trajectory_dataset_path=null
)

cd "${DISTILL_DIR}"
if [[ "${N_GPUS}" -le 1 ]]; then
    echo "[1/2] Launching SFT on a single GPU (DeepSpeed ZeRO-2) ..."
    "${PYTHON}" "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
elif [[ "${USE_DEEPSPEED}" == "1" ]]; then
    if ! "${PYTHON}" -c "import deepspeed" 2>/dev/null; then
        echo "ERROR: USE_DEEPSPEED=1 but deepspeed is not installed in ${PYTHON}" >&2
        exit 1
    fi
    echo "[1/2] Launching SFT with the deepspeed launcher (${N_GPUS} GPUs, ZeRO-2) ..."
    "${PYTHON}" -m deepspeed.launcher.runner --num_gpus="${N_GPUS}" "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
elif [[ -x "${TORCHRUN}" ]]; then
    echo "[1/2] Launching SFT with torchrun (${N_GPUS} GPUs, DeepSpeed ZeRO-2) ..."
    "${TORCHRUN}" --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
else
    echo "[1/2] Launching SFT with torch.distributed.run (${N_GPUS} GPUs, DeepSpeed ZeRO-2) ..."
    "${PYTHON}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="${N_GPUS}" \
        "${TRAIN_SCRIPT}" "${TRAIN_ARGS[@]}"
fi

echo "[2/2] Exporting the model to ${EXPORT_DIR} ..."
bash "${ROOT}/scripts/Misc/sync_verl_modeling.sh" "${MODEL_PATH}" "${RUN_DIR}"
bash "${ROOT}/scripts/Misc/export_sfted_model.sh" "${RUN_DIR}" "${EXPORT_DIR}"
