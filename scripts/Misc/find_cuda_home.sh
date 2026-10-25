# Source this file to get find_cuda_home <cuda_major> [fallback_dir].
#
# Prints a CUDA toolkit root whose bin/nvcc has the given CUDA major version, searched in this order: $CUDA_HOME,
# $CUDA_PATH, the nvcc in PATH, /usr/local/cuda and the optional fallback_dir (e.g. a conda environment prefix).
# Prints nothing if there is no such toolkit.
find_cuda_home() {
    local major="$1" fallback="${2:-}" candidate nvcc found
    local candidates=("${CUDA_HOME:-}" "${CUDA_PATH:-}")
    nvcc="$(command -v nvcc || true)"
    if [[ -n "${nvcc}" ]]; then
        candidates+=("$(cd "$(dirname "$(readlink -f "${nvcc}")")/.." && pwd)")
    fi
    candidates+=(/usr/local/cuda "${fallback}")
    for candidate in "${candidates[@]}"; do
        [[ -n "${candidate}" && -x "${candidate}/bin/nvcc" ]] || continue
        found="$("${candidate}/bin/nvcc" --version 2>/dev/null | sed -n 's/.*release \([0-9]*\)\..*/\1/p')"
        if [[ "${found}" == "${major}" ]]; then
            echo "${candidate}"
            return 0
        fi
    done
    return 0
}
