#!/usr/bin/env bash
# 把 verl/models 中对应模型的代码文件（*.py）同步到训练输出目录及其 checkpoint-* 子目录。
#
# 用法：
#   bash scripts/Misc/sync_verl_modeling.sh BASE_MODEL OUTPUT_DIR [OUTPUT_DIR ...]
#
# 对应目录按 BASE_MODEL 的目录名查找，找不到时去掉 -b16 / -bs32 这类 block 后缀重试。
set -euo pipefail

VERL_MODELS="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)/verl/models"

if [[ $# -lt 2 ]]; then
  sed -n '4,5p' "$0"
  exit 1
fi

name="$(basename "${1%/}")"
ref=""
for candidate in "${name}" "$(sed -E 's/-bs?[0-9]+$//' <<< "${name}")"; do
  if compgen -G "${VERL_MODELS}/${candidate}/modeling_*.py" > /dev/null; then
    ref="${VERL_MODELS}/${candidate}"
    break
  fi
done
if [[ -z "${ref}" ]]; then
  echo "[sync_verl_modeling] verl/models 下没有与 $1 对应的目录，跳过"
  exit 0
fi
shift

for out in "$@"; do
  for target in "${out}" "${out}"/checkpoint-*; do
    [[ -f "${target}/config.json" ]] || continue
    cp -f "${ref}"/*.py "${target}/"
    echo "[sync_verl_modeling] ${ref} -> ${target}"
  done
done
