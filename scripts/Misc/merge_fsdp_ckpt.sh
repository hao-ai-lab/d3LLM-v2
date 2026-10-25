#!/usr/bin/env bash
# Merge the FSDP actor checkpoint of a training run into a HuggingFace model.
#
# Usage:
#   bash scripts/Misc/merge_fsdp_ckpt.sh <ckpt_step_dir> <base_model> [out_dir]
#
#   ckpt_step_dir  checkpoint of one step, e.g. checkpoints/verl_dllm/<experiment>/global_step_150
#   base_model     starting model of the run; its model code (*.py) and tokenizer files are reused
#   out_dir        output directory, default <ckpt_step_dir>/actor_merged_hf
set -euo pipefail

if [[ $# -lt 2 ]]; then
  sed -n '4,9p' "$0" | sed 's/^# \{0,1\}//'
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"

CKPT="$(cd "$1" && pwd)"
BASE="$(cd "$2" && pwd)"
ACTOR="${CKPT}/actor"
OUT="${3:-${CKPT}/actor_merged_hf}"

if [[ ! -d "${ACTOR}" ]]; then
  echo "ERROR: actor checkpoint not found: ${ACTOR}" >&2
  exit 1
fi

echo "actor : ${ACTOR}"
echo "base  : ${BASE}"
echo "output: ${OUT}"

shopt -s nullglob
for f in "${BASE}"/*.py; do
  [[ -f "${ACTOR}/$(basename "${f}")" ]] || cp "${f}" "${ACTOR}/"
done
shopt -u nullglob

mkdir -p "${OUT}"
find "${BASE}" -maxdepth 1 -type f ! -name '*.safetensors' ! -name 'model.safetensors.index.json' \
  -exec cp {} "${OUT}/" \;

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
"${PYTHON}" "${REPO_ROOT}/scripts/Misc/model_merger.py" merge \
  --backend fsdp --local_dir "${ACTOR}" --target_dir "${OUT}" < <(yes)
