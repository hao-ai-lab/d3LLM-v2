#!/usr/bin/env bash
# Full-parameter SFT of the autoregressive Qwen baselines with LlamaFactory. The data, packing length, epochs,
# batch size and learning rate are those of the SDAR SFT run of the same task (dLLM_sft/SDAR/run_sft.sh);
# <reason> tags of the data are replaced with <think> tags (prepare_data.py).
#
# Usage:
#   bash dLLM_sft/AR/run_sft.sh <preset>
#
# Presets (SFT model written to dLLM_sft/sfted_models/):
#   alfworld-qwen3-8b    Qwen3-8B-sfted-alfworld
#   webshop-qwen3-8b     Qwen3-8B-sfted-webshop
#   webshop-qwen2.5-7b   Qwen2.5-7B-Instruct-sfted-webshop
#
# Optional environment variables:
#   BASE_MODEL OUTPUT_DIR           paths (default: Hugging Face cache of the base model / sfted_models/<name>)
#   NPROC CUDA_VISIBLE_DEVICES      GPUs (default 4 on 0,1,2,3)
#   CONDA_BASE                      conda root (default: conda info --base)
# The conda environment llamafactory_sdar is created by "bash dLLM_sft/SDAR/run_sft.sh install-env".
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
AR_DIR="${ROOT}/dLLM_sft/AR"
SDAR_TRAIN="${ROOT}/dLLM_sft/SDAR/training/llama_factory_sdar"
source "${ROOT}/scripts/Misc/find_cuda_home.sh"

PRESET="${1:-}"
case "${PRESET}" in
    alfworld-qwen3-8b)  HF_ID="Qwen/Qwen3-8B";            TRAIN_YAML="alfworld_qwen3_8b.yaml";  OUT_NAME="Qwen3-8B-sfted-alfworld" ;;
    webshop-qwen3-8b)   HF_ID="Qwen/Qwen3-8B";            TRAIN_YAML="webshop_qwen3_8b.yaml";   OUT_NAME="Qwen3-8B-sfted-webshop" ;;
    webshop-qwen2.5-7b) HF_ID="Qwen/Qwen2.5-7B-Instruct"; TRAIN_YAML="webshop_qwen2.5_7b.yaml"; OUT_NAME="Qwen2.5-7B-Instruct-sfted-webshop" ;;
    *)
        sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac

OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dLLM_sft/sfted_models/${OUT_NAME}}"
NPROC="${NPROC:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

CONDA_ENV_NAME="llamafactory_sdar"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
TORCHRUN="${CONDA_ENV_PREFIX}/bin/torchrun"
if [[ ! -x "${TORCHRUN}" ]]; then
    echo "ERROR: conda environment '${CONDA_ENV_NAME}' not found; run 'bash dLLM_sft/SDAR/run_sft.sh install-env' once to create it" >&2
    exit 1
fi
if [[ -e "${OUTPUT_DIR}" ]]; then
    echo "ERROR: output already exists: ${OUTPUT_DIR}; refusing to overwrite." >&2
    exit 1
fi

DATA_DIR="${AR_DIR}/work"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/ar_sft_${PRESET}_$(date +%Y%m%d_%H%M%S).log}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_ar_sft_${PRESET}_$$}"

# ── Step 0: data and base model ──────────────────────────────
"${CONDA_ENV_PREFIX}/bin/python" "${AR_DIR}/prepare_data.py"
BASE_MODEL="${BASE_MODEL:-$("${CONDA_ENV_PREFIX}/bin/python" -c "from huggingface_hub import snapshot_download; print(snapshot_download('${HF_ID}'))")}"

echo "======================================================"
echo " AR SFT: ${PRESET}  -  $(date)"
echo " base model : ${BASE_MODEL}"
echo " config     : dLLM_sft/AR/configs/${TRAIN_YAML}"
echo " data dir   : ${DATA_DIR}"
echo " output     : ${OUTPUT_DIR}"
echo " log        : ${LOG_FILE}"
echo "======================================================"

# ── Step 1: training ─────────────────────────────────────────
# DeepSpeed locates nvcc through CUDA_HOME.
TRAIN_CUDA_HOME="$(find_cuda_home 12 "${CONDA_ENV_PREFIX}")"
if [[ -z "${TRAIN_CUDA_HOME}" ]]; then
    echo "ERROR: no nvcc of CUDA 12 found; run 'bash dLLM_sft/SDAR/run_sft.sh install-env' to install the toolkit into the environment" >&2
    exit 1
fi
echo "CUDA_HOME for training: ${TRAIN_CUDA_HOME}"

mkdir -p "$(dirname "${LOG_FILE}")" "${TRITON_CACHE_DIR}"

echo "Launching SFT with ${NPROC} GPU(s) ..."
(
    export CUDA_VISIBLE_DEVICES TRITON_CACHE_DIR
    export CUDA_HOME="${TRAIN_CUDA_HOME}"
    export PATH="${CONDA_ENV_PREFIX}/bin:${TRAIN_CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${SDAR_TRAIN}/src:${PYTHONPATH:-}"
    export HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=4

    cd "${SDAR_TRAIN}"
    "${TORCHRUN}" \
        --standalone --nnodes 1 \
        --nproc_per_node "${NPROC}" \
        ./src/llamafactory/launcher.py \
        "${AR_DIR}/configs/${TRAIN_YAML}" \
        model_name_or_path="${BASE_MODEL}" \
        output_dir="${OUTPUT_DIR}" \
        dataset_dir="${DATA_DIR}"
) 2>&1 | tee "${LOG_FILE}"
echo "Training complete: $(date)"
