#!/usr/bin/env bash
# relaxed mode: never kill your shell/session on errors
set +e
set +u
set +o pipefail 2>/dev/null || true

# ============================================================
# NVL72 (aarch64) env setup for sglang
# - Activates conda env
# - Pins CUDA toolchain to shared CUDA 12.9
# - Uses conda CUDA headers (targets/sbsa-linux) for crt/host_defines.h
# - Pins host compiler to conda GCC/G++ and makes cc1plus discoverable
# - Forces nvcc to use g++ as host compiler
# - Avoids _conda_clang_precmd crash by defining CONDA_BACKUP_HOST
# - Prints "env setup ok ✅" on success; does NOT exit your node on failure
# ============================================================

# ---- user config ----
ENV_NAME="${ENV_NAME:-verl-agent-new}"
CUDA_ROOT="${CUDA_ROOT:-/home/shared-bin/cuda-12.9}"
NCCL_HOME="${NCCL_HOME:-/home/shared-bin/nccl/build}"

# ---- avoid zsh conda clang hook crash (variable required by hook) ----
export CONDA_BACKUP_HOST="${CONDA_BACKUP_HOST:-}"

# ---- helper: try to source conda.sh if conda function isn't available ----
_load_conda_sh() {
  if [[ -n "${CONDA_EXE:-}" ]]; then
    return 0
  fi
  if [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/anaconda3/etc/profile.d/conda.sh"
    return $?
  fi
  if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
    return $?
  fi
  return 1
}

# ---- activate conda env ----
if ! _load_conda_sh; then
  echo "ERROR: conda.sh not found. Please run: conda init <your shell> (or source conda.sh manually)." >&2
  echo "env setup FAILED ❌" >&2
  return 1 2>/dev/null || true
fi

conda activate "${ENV_NAME}" >/dev/null 2>&1
if [[ $? -ne 0 ]]; then
  _fallback_prefix=""
  if [[ -d "$HOME/anaconda3/envs/${ENV_NAME}" ]]; then
    _fallback_prefix="$HOME/anaconda3/envs/${ENV_NAME}"
  elif [[ -d "$HOME/miniconda3/envs/${ENV_NAME}" ]]; then
    _fallback_prefix="$HOME/miniconda3/envs/${ENV_NAME}"
  fi
  if [[ -n "${_fallback_prefix}" ]]; then
    export CONDA_PREFIX="${_fallback_prefix}"
    export PATH="${CONDA_PREFIX}/bin:${PATH}"
    echo "WARN: conda activate ${ENV_NAME} failed; using fallback prefix ${CONDA_PREFIX}" >&2
  else
    echo "ERROR: conda activate ${ENV_NAME} failed" >&2
    echo "env setup FAILED ❌" >&2
    return 1 2>/dev/null || true
  fi
fi

# ---- shared CUDA toolchain ----
export CUDA_HOME="${CUDA_ROOT}"

# ---- CUDA headers from conda (sbsa-linux layout) ----
if [[ -z "${CONDA_PREFIX:-}" ]]; then
  echo "ERROR: CONDA_PREFIX not set after conda activate." >&2
  echo "env setup FAILED ❌" >&2
  return 1 2>/dev/null || true
fi
export CUDA_HEADERS="${CONDA_PREFIX}/targets/sbsa-linux"

# ---- host compiler (conda) ----
export CC="${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-gcc"
export CXX="${CONDA_PREFIX}/bin/aarch64-conda-linux-gnu-g++"
export CUDAHOSTCXX="${CXX}"

# ---- reset potentially polluted vars (important for JIT builds) ----
unset CPATH C_INCLUDE_PATH CPLUS_INCLUDE_PATH
unset CFLAGS CXXFLAGS CPPFLAGS
unset NVCCFLAGS

# ---- ensure cc1plus discoverable ----
_cc1plus_path="$("${CXX}" -print-prog-name=cc1plus 2>/dev/null)"
_cc1plus_dir=""
if [[ -n "${_cc1plus_path}" ]]; then
  _cc1plus_dir="$(cd "$(dirname "${_cc1plus_path}")" >/dev/null 2>&1 && pwd -P)"
fi

# ---- PATH order: cc1plus dir -> CUDA toolchain -> conda -> existing PATH ----
if [[ -n "${_cc1plus_dir}" && -d "${_cc1plus_dir}" ]]; then
  export PATH="${_cc1plus_dir}:${CUDA_ROOT}/bin:${CUDA_ROOT}/nvvm/bin:${CONDA_PREFIX}/bin:${PATH}"
else
  export PATH="${CUDA_ROOT}/bin:${CUDA_ROOT}/nvvm/bin:${CONDA_PREFIX}/bin:${PATH}"
fi

# ---- libraries ----
export LD_LIBRARY_PATH="${CUDA_ROOT}/lib64:${NCCL_HOME}/lib:${CONDA_PREFIX}/lib:/usr/lib/aarch64-linux-gnu:${LD_LIBRARY_PATH:-}"
export LIBRARY_PATH="${NCCL_HOME}/lib:${CONDA_PREFIX}/lib:/usr/lib/aarch64-linux-gnu:${LIBRARY_PATH:-}"

# ---- includes ----
export CPATH="${CUDA_HEADERS}/include:${NCCL_HOME}/include"
export C_INCLUDE_PATH="${CUDA_HEADERS}/include:${CONDA_PREFIX}/include"
export CPLUS_INCLUDE_PATH="${CUDA_HEADERS}/include:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cuda_runtime/include:${CONDA_PREFIX}/lib/python3.10/site-packages/nvidia/cublas/include"

# ---- nvcc: force host compiler to g++ ----
export NVCC_PREPEND_FLAGS="--compiler-bindir=${CXX} -allow-unsupported-compiler"

# ---- sglang ----
export SGLANG_SKIP_SGL_KERNEL_VERSION_CHECK=1
export SGLANG_DISABLE_CUDNN_CHECK=1

# ============================================================
# Sanity checks (non-fatal; we just report + return 1 on failure)
# ============================================================
ok=1

echo "ENV_NAME=${ENV_NAME}"
echo "CONDA_PREFIX=${CONDA_PREFIX}"
echo "CUDA_ROOT=${CUDA_ROOT}"
echo "CUDA_HOME=${CUDA_HOME}"
echo "CUDA_HEADERS=${CUDA_HEADERS}"
echo "NCCL_HOME=${NCCL_HOME}"
echo "CC=${CC}"
echo "CXX=${CXX}"
echo "CUDAHOSTCXX=${CUDAHOSTCXX}"
echo "NVCC_PREPEND_FLAGS=${NVCC_PREPEND_FLAGS}"

echo "which nvcc:  $(command -v nvcc  2>/dev/null || echo NOT_FOUND)"
echo "which cicc:  $(command -v cicc  2>/dev/null || echo NOT_FOUND)"
echo "which ptxas: $(command -v ptxas 2>/dev/null || echo NOT_FOUND)"
echo "which cc1plus: $(command -v cc1plus 2>/dev/null || echo NOT_FOUND)"

if [[ ! -x "${CUDA_ROOT}/bin/nvcc" ]]; then
  echo "ERROR: nvcc not found at ${CUDA_ROOT}/bin/nvcc" >&2
  ok=0
fi

if [[ ! -f "${CUDA_HEADERS}/include/crt/host_defines.h" ]]; then
  echo "ERROR: missing ${CUDA_HEADERS}/include/crt/host_defines.h" >&2
  ok=0
fi

if ! command -v cc1plus >/dev/null 2>&1; then
  echo "ERROR: cc1plus not on PATH" >&2
  ok=0
else
  cc1plus --help >/dev/null 2>&1
  if [[ $? -ne 0 ]]; then
    echo "ERROR: cc1plus not runnable" >&2
    ok=0
  fi
fi

# quick nvcc smoke test (fast; no GPU required)
cat > /tmp/nvcc_smoke_test.cu <<'EOF'
#include <vector>
__global__ void k() {}
int main(){ std::vector<int> v; k<<<1,1>>>(); return 0; }
EOF
nvcc /tmp/nvcc_smoke_test.cu -o /tmp/nvcc_smoke_test >/dev/null 2>&1
if [[ $? -ne 0 ]]; then
  echo "ERROR: nvcc smoke test failed. Run: nvcc -v /tmp/nvcc_smoke_test.cu -o /tmp/nvcc_smoke_test" >&2
  ok=0
fi

if [[ "${ok}" -eq 1 ]]; then
  echo "env setup ok ✅"
  return 0 2>/dev/null || true
else
  echo "env setup FAILED ❌" >&2
  return 1 2>/dev/null || true
fi
