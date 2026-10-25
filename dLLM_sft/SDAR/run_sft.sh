#!/usr/bin/env bash
# SFT for SDAR backbones with LlamaFactory. The base model and output directory, and the data directory
# (dLLM_sft/data/sdar) are passed to LlamaFactory on the command line.
#
# Usage:
#   bash dLLM_sft/SDAR/run_sft.sh <preset>
#
# Presets (released SFT model in dLLM_sft/sfted_models/):
#   webshop-8b-b4   SDAR-8B-Chat-sfted-webshop
#   webshop-4b-b4   SDAR-4B-Chat-sfted-webshop
#   webshop-8b-b8   SDAR-8B-Chat-b8-sfted-webshop
#   alfworld-8b-b4  SDAR-8B-Chat-sfted-alfworld
#   install-env     only create / repair the conda environment (e.g. to finish the flash-attn build
#                   before a GPU is allocated)
#
# Optional environment variables:
#   BASE_MODEL OUTPUT_DIR DATA_DIR  paths (defaults below)
#   NPROC CUDA_VISIBLE_DEVICES      GPUs (default 4 on 0,1,2,3)
#   FLASH_ATTN_WHL                  prebuilt flash-attn wheel; if empty, flash-attn is built from source
#   CUDA_HOME                       CUDA toolkit with nvcc of CUDA 12 (default: PATH, /usr/local/cuda, otherwise the
#                                   toolkit is installed into the conda environment)
#   MAX_JOBS                        parallel compile jobs for the flash-attn build (default: min(CPUs, RAM_GB/6))
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SDAR_DIR="${ROOT}/dLLM_sft/SDAR"
SDAR_TRAIN="${SDAR_DIR}/training/llama_factory_sdar"
DATA_DIR="${DATA_DIR:-${ROOT}/dLLM_sft/data}"

source "${ROOT}/scripts/Misc/find_cuda_home.sh"

INSTALL_ONLY=0
PRESET="${1:-}"
case "${PRESET}" in
    webshop-8b-b4)  MODEL_NAME="SDAR-8B-Chat";    TRAIN_YAML="examples/train_full_sdar/webshop_8b_b4.yaml";  OUT_NAME="SDAR-8B-Chat-sfted-webshop" ;;
    webshop-4b-b4)  MODEL_NAME="SDAR-4B-Chat";    TRAIN_YAML="examples/train_full_sdar/webshop_4b_b4.yaml";  OUT_NAME="SDAR-4B-Chat-sfted-webshop" ;;
    webshop-8b-b8)  MODEL_NAME="SDAR-8B-Chat-b8"; TRAIN_YAML="examples/train_full_sdar/webshop_8b_b8.yaml";  OUT_NAME="SDAR-8B-Chat-b8-sfted-webshop" ;;
    alfworld-8b-b4) MODEL_NAME="SDAR-8B-Chat";    TRAIN_YAML="examples/train_full_sdar/alfworld_8b_b4.yaml"; OUT_NAME="SDAR-8B-Chat-sfted-alfworld" ;;
    install-env)    MODEL_NAME="";                TRAIN_YAML="";                                             OUT_NAME="";  INSTALL_ONLY=1 ;;
    *)
        sed -n '2,/^set -euo/p' "$0" | sed '$d' | sed 's/^# \{0,1\}//'
        exit 1
        ;;
esac

BASE_MODEL="${BASE_MODEL:-${SDAR_DIR}/training/model/${MODEL_NAME}}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/dLLM_sft/sfted_models/ckpt/${OUT_NAME}}"
NPROC="${NPROC:-4}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
FLASH_ATTN_WHL="${FLASH_ATTN_WHL:-}"

CONDA_ENV_NAME="llamafactory_sdar"
CONDA_BASE="${CONDA_BASE:-$(conda info --base)}"
CONDA_ENV_PREFIX="${CONDA_BASE}/envs/${CONDA_ENV_NAME}"
PYTHON="${CONDA_ENV_PREFIX}/bin/python"
TORCHRUN="${CONDA_ENV_PREFIX}/bin/torchrun"
TRAIN_YAML_PATH="${SDAR_TRAIN}/${TRAIN_YAML}"
LOG_FILE="${LOG_FILE:-${ROOT}/logs/sdar_sft_${PRESET}_$(date +%Y%m%d_%H%M%S).log}"
TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/tmp/${USER}/triton_sdar_sft_${PRESET}_$$}"

TORCH_VERSION="2.9.1+cu129"
TORCH_CUDA="12.9"
TORCH_INDEX="https://download.pytorch.org/whl/cu129"
TRANSFORMERS_VERSION="4.52.4"

if [[ "${INSTALL_ONLY}" != 1 ]]; then
    echo "======================================================"
    echo " SDAR SFT: ${PRESET}  -  $(date)"
    echo " base model : ${BASE_MODEL}"
    echo " config     : ${TRAIN_YAML}"
    echo " data dir   : ${DATA_DIR}/sdar"
    echo " output     : ${OUTPUT_DIR}"
    echo " log        : ${LOG_FILE}"
    echo "======================================================"

    for p in "${TRAIN_YAML_PATH}" "${DATA_DIR}/sdar/dataset_info.json" "${BASE_MODEL}"; do
        if [[ ! -e "${p}" ]]; then
            echo "ERROR: not found: ${p}" >&2
            exit 1
        fi
    done
    # dLLM_sft/SDAR/training/model/<name> holds the code and config for SFT; the weights are linked from verl/models/<name>
    # (see README, "Prepare the SDAR base models").
    if ! compgen -G "${BASE_MODEL}/*.safetensors" >/dev/null; then
        echo "ERROR: no weights (*.safetensors) in ${BASE_MODEL}. Download the base model and link the weights:" >&2
        echo "       hf download JetLM/${MODEL_NAME} --include \"*.safetensors\" --local-dir verl/models/${MODEL_NAME}" >&2
        echo "       ln -sf \"\$PWD\"/verl/models/${MODEL_NAME}/*.safetensors dLLM_sft/SDAR/training/model/${MODEL_NAME}/" >&2
        exit 1
    fi
fi

# ── Step 0: conda environment ────────────────────────────────
# The environment counts as complete only when a matching CUDA toolkit exists and torch, flash-attn, deepspeed and
# LlamaFactory can all be imported with the pinned versions, so an interrupted installation is resumed.
env_ready() {
    [[ -x "${PYTHON}" && -x "${TORCHRUN}" ]] || return 1
    [[ -n "$(find_cuda_home "${TORCH_CUDA%%.*}" "${CONDA_ENV_PREFIX}")" ]] || return 1
    (cd /tmp && "${PYTHON}" - "${TORCH_VERSION}" "${TRANSFORMERS_VERSION}" "${SDAR_TRAIN}/src" <<'PY'
import importlib.util
import sys

import torch
import transformers
import accelerate
import peft
import datasets
import flash_attn
import deepspeed

torch_version, transformers_version, llamafactory_src = sys.argv[1:4]
assert torch.__version__ == torch_version, f"torch {torch.__version__} != {torch_version}"
assert transformers.__version__ == transformers_version, f"transformers {transformers.__version__} != {transformers_version}"
sys.path.insert(0, llamafactory_src)
assert importlib.util.find_spec("llamafactory") is not None, "llamafactory not importable"
PY
    ) >/dev/null 2>&1
}

env_pip() { "${PYTHON}" -m pip "$@"; }

# DeepSpeed and the flash-attn build need nvcc with the same CUDA major version as torch. A matching system toolkit is
# used if there is one; otherwise the toolkit is installed into the conda environment.
ensure_cuda_toolkit() {
    CUDA_TOOLKIT="$(find_cuda_home "${TORCH_CUDA%%.*}" "${CONDA_ENV_PREFIX}")"
    if [[ -z "${CUDA_TOOLKIT}" ]]; then
        echo "No nvcc matching torch's CUDA ${TORCH_CUDA} found; installing the toolkit into '${CONDA_ENV_NAME}' ..."
        conda install -n "${CONDA_ENV_NAME}" --override-channels -c nvidia -c conda-forge \
            "cuda-compiler=${TORCH_CUDA}" "cuda-libraries-dev=${TORCH_CUDA}" -y
        CUDA_TOOLKIT="$(find_cuda_home "${TORCH_CUDA%%.*}" "${CONDA_ENV_PREFIX}")"
    fi
    if [[ -z "${CUDA_TOOLKIT}" ]]; then
        echo "ERROR: no usable nvcc for CUDA ${TORCH_CUDA}. Install a CUDA ${TORCH_CUDA%%.*}.x toolkit and set CUDA_HOME." >&2
        exit 1
    fi
    echo "CUDA toolkit: ${CUDA_TOOLKIT}"
}

install_flash_attn() {
    if [[ -n "${FLASH_ATTN_WHL}" ]]; then
        env_pip install "${FLASH_ATTN_WHL}"
        return
    fi

    export CUDA_HOME="${CUDA_TOOLKIT}"
    export PATH="${CUDA_TOOLKIT}/bin:${PATH}"
    # conda's CUDA packages keep the headers in targets/<arch>-linux/include, which torch's cpp_extension does not search.
    if [[ ! -f "${CUDA_TOOLKIT}/include/cuda_runtime_api.h" ]]; then
        local cuda_inc
        for cuda_inc in "${CUDA_TOOLKIT}"/targets/*/include; do
            if [[ -f "${cuda_inc}/cuda_runtime_api.h" ]]; then
                export CUDA_INC_PATH="${cuda_inc}"
                break
            fi
        done
    fi

    local cpus mem_gb jobs
    cpus="$(nproc)"
    mem_gb="$(awk '/MemAvailable/ {print int($2 / 1048576)}' /proc/meminfo)"
    jobs=$(( mem_gb / 6 ))
    (( jobs > cpus )) && jobs="${cpus}"
    (( jobs < 1 )) && jobs=1
    export MAX_JOBS="${MAX_JOBS:-${jobs}}"
    echo "Building flash-attn from source: CUDA_HOME=${CUDA_HOME}, torch CUDA ${TORCH_CUDA}, MAX_JOBS=${MAX_JOBS}"
    env_pip install -v "flash-attn==2.8.3" --no-build-isolation
}

if env_ready; then
    echo "[0/2] conda environment '${CONDA_ENV_NAME}' is ready."
else
    if [[ ! -x "${PYTHON}" ]]; then
        echo "[0/2] creating conda environment '${CONDA_ENV_NAME}' (python 3.12) ..."
        conda create -n "${CONDA_ENV_NAME}" python=3.12 -y
    else
        echo "[0/2] conda environment '${CONDA_ENV_NAME}' is incomplete, resuming installation ..."
    fi

    env_pip install "torch==${TORCH_VERSION}" "torchvision==0.24.1" "torchaudio==2.9.1" --index-url "${TORCH_INDEX}"
    # LlamaFactory requires numpy<2, and recent pyarrow releases need numpy>=2, so pyarrow is capped.
    env_pip install \
        -e "${SDAR_TRAIN}[torch,metrics]" \
        "transformers==${TRANSFORMERS_VERSION}" \
        "deepspeed==0.16.9" \
        "accelerate==1.7.0" \
        "peft==0.15.2" \
        "pyarrow<22" \
        "tensorboard"
    env_pip install ninja packaging psutil wheel setuptools
    ensure_cuda_toolkit
    install_flash_attn

    if ! env_ready; then
        echo "ERROR: conda environment '${CONDA_ENV_NAME}' is still incomplete after installation" >&2
        exit 1
    fi
    echo "[0/2] conda environment '${CONDA_ENV_NAME}' installed."
fi

if [[ "${INSTALL_ONLY}" == 1 ]]; then
    exit 0
fi

# ── Step 1: training ─────────────────────────────────────────
# DeepSpeed locates nvcc through CUDA_HOME.
TRAIN_CUDA_HOME="$(find_cuda_home "${TORCH_CUDA%%.*}" "${CONDA_ENV_PREFIX}")"
echo "CUDA_HOME for training: ${TRAIN_CUDA_HOME}"

mkdir -p "$(dirname "${LOG_FILE}")" "${TRITON_CACHE_DIR}"
rm -rf "${HOME}/.cache/huggingface/modules/transformers_modules/$(basename "${BASE_MODEL}")"

echo "[1/2] Launching SFT with ${NPROC} GPU(s) ..."
(
    export CUDA_VISIBLE_DEVICES TRITON_CACHE_DIR
    export CUDA_HOME="${TRAIN_CUDA_HOME}"
    export PATH="${CONDA_ENV_PREFIX}/bin:${TRAIN_CUDA_HOME}/bin:${PATH}"
    export LD_LIBRARY_PATH="${CONDA_ENV_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
    export PYTHONPATH="${SDAR_TRAIN}/src:${PYTHONPATH:-}"

    cd "${SDAR_TRAIN}"
    "${TORCHRUN}" \
        --nnodes 1 \
        --nproc_per_node "${NPROC}" \
        ./src/llamafactory/launcher.py \
        "./${TRAIN_YAML}" \
        model_name_or_path="${BASE_MODEL}" \
        output_dir="${OUTPUT_DIR}" \
        dataset_dir="${DATA_DIR}/sdar"
) 2>&1 | tee "${LOG_FILE}"

# ── Step 2: copy verl model code and export the HuggingFace model ──
bash "${ROOT}/scripts/Misc/sync_verl_modeling.sh" "${BASE_MODEL}" "${OUTPUT_DIR}"
bash "${ROOT}/scripts/Misc/export_sfted_model.sh" "${OUTPUT_DIR}"

echo "[2/2] Training complete: $(date)"
