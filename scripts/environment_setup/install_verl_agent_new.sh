#!/usr/bin/env bash
# Build the main conda environment (default name: verl-dllm-agent): pinned dependencies,
# transformers 4.57.1 and the local SGLang (sglang_dllm). If an environment with the same name
# exists, the script stops unless RECREATE=1 is set, in which case it is removed and rebuilt.
# Developed with conda 25.7.0 (libmamba solver); conda older than 23.10 is rejected.
#
# Afterwards run the setup script of each environment you need: setup_webshop.sh,
# setup_alfworld.sh, setup_search.sh.
#
# Optional environment variables:
#   RECREATE              set to 1 to remove and rebuild an existing environment (default: 0)
#   PROXY                 HTTP(S) proxy, exported as http(s)_proxy
#   HF_ENDPOINT           HuggingFace mirror
#   CONDA                 conda executable (default: conda on PATH)
#   ENV_NAME              environment name (default: verl-dllm-agent)
#   TORCH_CUDA_ARCH_LIST  GPU architectures for building flash-attn, e.g. "8.0" or "8.0 9.0"
#                         (default: detected from the visible GPUs)
#   MAX_JOBS              parallel flash-attn compile jobs (default: chosen from cores and free memory)
set -euo pipefail

[[ -n "${PROXY:-}" ]] && export http_proxy=$PROXY https_proxy=$PROXY HTTP_PROXY=$PROXY HTTPS_PROXY=$PROXY
[[ -n "${HF_ENDPOINT:-}" ]] && export HF_ENDPOINT

CONDA=${CONDA:-$(command -v conda)}
ENV_NAME=${ENV_NAME:-verl-dllm-agent}
ENV_PREFIX="$($CONDA info --base)/envs/${ENV_NAME}"
PYTHON=${ENV_PREFIX}/bin/python3
PIP="$PYTHON -m pip"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { echo "[$(date '+%H:%M:%S')] === $* ==="; }

if [[ -z "$ENV_NAME" || "$ENV_NAME" == "base" ]]; then
  echo "ERROR: invalid ENV_NAME '${ENV_NAME}'" >&2
  exit 1
fi
if [[ -d "$ENV_PREFIX" && "${RECREATE:-0}" != "1" ]]; then
  echo "ERROR: conda env ${ENV_NAME} already exists at ${ENV_PREFIX}." >&2
  echo "       Re-run with RECREATE=1 to remove and rebuild it, or set ENV_NAME to another name." >&2
  exit 1
fi

# conda older than 23.10 solves with the slow classic solver, which can stall for a very long
# time on `conda install`/`conda create`, so stop here. The setup_*.sh scripts need the same conda.
CONDA_MIN_VERSION=23.10.0
conda_version="$($CONDA --version | awk '{print $2}')"
if [[ "$(printf '%s\n%s\n' "$CONDA_MIN_VERSION" "$conda_version" | sort -V | head -n1)" != "$CONDA_MIN_VERSION" ]]; then
  cat >&2 <<EOF
ERROR: conda ${conda_version} is older than ${CONDA_MIN_VERSION} (classic solver, very slow).
       Install a separate Miniforge and run this script with its conda; the base conda is not modified:

         curl -L -O https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-\$(uname -m).sh
         bash Miniforge3-Linux-\$(uname -m).sh -b -p ~/miniforge3
         CONDA=~/miniforge3/bin/conda bash scripts/environment_setup/$(basename "${BASH_SOURCE[0]}")

       The environment is then created under ~/miniforge3/envs/${ENV_NAME}.
       Use the same CONDA for setup_webshop.sh and setup_search.sh.
EOF
  exit 1
fi

if [[ -d "$ENV_PREFIX" ]]; then
  log "[1/7] Remove existing conda env ${ENV_NAME} (RECREATE=1)"
  $CONDA remove -n "$ENV_NAME" --all -y
fi
log "[1/7] Create conda env ${ENV_NAME}"
$CONDA create -n "$ENV_NAME" python==3.10 -y
$PIP install pip==26.0.1 uv==0.10.4

log "[2/7] SGLang dependencies (commit e2eb5bf), then the local SGLang"
${ENV_PREFIX}/bin/uv pip install --python "$PYTHON" \
  "sglang[all] @ git+https://github.com/sgl-project/sglang.git@e2eb5bf#subdirectory=python"
$PIP uninstall sglang -y
$PIP install -e "${REPO}/sglang_dllm/python" --no-build-isolation
$PIP install flashinfer-python==0.6.3 flashinfer-cubin==0.6.3

log "[3/7] Other dependencies"
$PIP install \
  numpy==2.2.6 socksio==1.0.0 "ray[data,train,tune,serve]==2.54.0" tensordict==0.11.0 \
  hydra-core==1.3.2 liger-kernel==0.7.0 codetiming==1.4.0 accelerate==1.10.1 peft==0.17.1 \
  wandb==0.25.0 uvicorn==0.41.0 fastapi==0.129.0 torchdata==0.11.0 cachetools==7.0.2 \
  bs64==0.0.1 pyserini==1.2.0 Werkzeug==3.1.6 spacy==3.8.11 requests_mock==1.12.1 rich==14.3.3 \
  bs4==0.0.2 selenium==4.41.0 rank_bm25==0.2.2 gdown==5.2.1 beautifulsoup4==4.14.3 \
  cleantext==1.1.4 thefuzz==0.22.1 train==0.0.5 gym==0.24.0
# The decord extra has no wheel on aarch64; fall back to the plain package.
$PIP install "qwen-vl-utils[decord]==0.0.14" || $PIP install qwen-vl-utils==0.0.14

log "[4/7] Python 3.10.19"
$CONDA install python=3.10.19 -n "$ENV_NAME" -y || echo "[WARN] Python update failed, continuing"

log "[5/7] flash-attn"
# flash-attn must be compiled against the installed torch. --no-cache-dir keeps pip from reusing a
# wheel that an earlier build cached for another torch version (import then fails with an
# undefined symbol). The compile needs ninja (otherwise torch falls back to a serial build that
# takes hours) and an nvcc whose CUDA major version equals torch's (e.g. the system nvcc may be
# 13.x while torch is cu128). If no matching nvcc is found, the matching toolkit is installed
# into the environment with conda.
torch_cuda="$($PYTHON -c 'import torch; print(torch.version.cuda or "")')"
if [[ -z "$torch_cuda" ]]; then
  echo "ERROR: the installed torch has no CUDA support, cannot build flash-attn." >&2
  exit 1
fi
nvcc_major() { "$1" --version 2>/dev/null | sed -n 's/.*release \([0-9]\+\)\..*/\1/p'; }
nvcc_bin="${CUDA_HOME:+${CUDA_HOME}/bin/nvcc}"
[[ -x "${nvcc_bin}" ]] || nvcc_bin="$(command -v nvcc || true)"
if [[ -z "$nvcc_bin" || "$(nvcc_major "$nvcc_bin")" != "${torch_cuda%%.*}" ]]; then
  log "No nvcc matching torch's CUDA ${torch_cuda} (found: ${nvcc_bin:-none}); installing the toolkit into ${ENV_NAME}"
  $CONDA install -n "$ENV_NAME" --override-channels -c nvidia -c conda-forge \
    "cuda-compiler=${torch_cuda}" "cuda-libraries-dev=${torch_cuda}" -y
  export CUDA_HOME="${ENV_PREFIX}"
  nvcc_bin="${CUDA_HOME}/bin/nvcc"
  if [[ "$(nvcc_major "$nvcc_bin")" != "${torch_cuda%%.*}" ]]; then
    echo "ERROR: no usable nvcc for CUDA ${torch_cuda}. Install a CUDA ${torch_cuda%%.*}.x toolkit and set CUDA_HOME." >&2
    exit 1
  fi
else
  export CUDA_HOME="$(dirname "$(dirname "$(readlink -f "$nvcc_bin")")")"
fi
export PATH="${CUDA_HOME}/bin:${ENV_PREFIX}/bin:${PATH}"
# conda's CUDA packages keep the headers in ${CUDA_HOME}/targets/<arch>-linux/include, not in
# ${CUDA_HOME}/include, which is the only place torch's cpp_extension looks (besides CUDA_INC_PATH).
# nvcc finds them by itself, but the g++ compile of flash-attn's .cpp files does not. conda sets
# this up when the environment is activated, which this script does not do.
if [[ ! -f "${CUDA_HOME}/include/cuda_runtime_api.h" ]]; then
  for cuda_inc in "${CUDA_HOME}"/targets/*/include; do
    if [[ -f "${cuda_inc}/cuda_runtime_api.h" ]]; then
      export CUDA_INC_PATH="$cuda_inc"
      break
    fi
  done
  if [[ -z "${CUDA_INC_PATH:-}" ]]; then
    echo "ERROR: cuda_runtime_api.h not found under ${CUDA_HOME}; the CUDA development headers are missing." >&2
    exit 1
  fi
fi
log "flash-attn build: torch CUDA ${torch_cuda}, nvcc $("$nvcc_bin" --version | sed -n 's/.*release \([0-9.]\+\).*/\1/p'), CUDA_HOME=${CUDA_HOME}"
$PIP install ninja
# flash-attn ignores TORCH_CUDA_ARCH_LIST and builds sm80/90/100/120 unless FLASH_ATTN_CUDA_ARCHS
# is set. Use TORCH_CUDA_ARCH_LIST when given ("8.0 9.0+PTX" -> "80;90"), otherwise the compute
# capability of the visible GPUs (8.x -> 80, 9.x -> 90, 10.x -> 100, 12.x -> 120), otherwise build
# every supported architecture. MAX_JOBS is left unset on purpose: flash-attn then picks
# min(cores/2, available memory/9GB) parallel jobs (each nvcc job peaks at ~9GB); a larger value
# runs out of memory and the build fails. Export MAX_JOBS to override it.
detect_gpu_archs() {
  $PYTHON -c '
import torch
arch = {8: "80", 9: "90", 10: "100", 12: "120"}
found = {arch.get(torch.cuda.get_device_capability(i)[0]) for i in range(torch.cuda.device_count())}
print(";".join(sorted(a for a in found if a)))' 2>/dev/null || true
}
if [[ -n "${TORCH_CUDA_ARCH_LIST:-}" ]]; then
  FLASH_ATTN_CUDA_ARCHS="$(echo "$TORCH_CUDA_ARCH_LIST" | sed 's/+PTX//g; s/\.//g; s/[ ,;]\+/;/g')"
else
  FLASH_ATTN_CUDA_ARCHS="$(detect_gpu_archs)"
fi
if [[ -n "$FLASH_ATTN_CUDA_ARCHS" ]]; then
  export FLASH_ATTN_CUDA_ARCHS
  log "flash-attn architectures: ${FLASH_ATTN_CUDA_ARCHS}, MAX_JOBS=${MAX_JOBS:-16}"
else
  echo "[WARN] No supported GPU detected; building flash-attn for every architecture it supports (slower)."
  echo "       Set TORCH_CUDA_ARCH_LIST (e.g. \"8.0\") to build for specific GPUs only."
fi
FLASH_ATTENTION_FORCE_BUILD=TRUE FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE \
  $PIP install flash-attn==2.8.3 --no-build-isolation --no-cache-dir

log "[6/7] transformers / huggingface_hub (matches sglang_dllm)"
$PIP install transformers==4.57.1 huggingface_hub==0.36.2

log "[7/7] Verify"
$PYTHON - <<'EOF'
import torch, transformers, peft, ray, flash_attn
print(f"torch {torch.__version__} (cuda {torch.version.cuda}), transformers {transformers.__version__}, "
      f"peft {peft.__version__}, ray {ray.__version__}, flash_attn {flash_attn.__version__}")
EOF
log "Done. Next: setup_webshop.sh, setup_alfworld.sh, setup_search.sh"
