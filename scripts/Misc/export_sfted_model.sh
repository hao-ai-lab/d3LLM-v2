#!/usr/bin/env bash
# 从 SFT 训练输出中取出 HF 格式的模型文件，复制到单独的模型目录。
#
# 用法：
#   bash scripts/Misc/export_sfted_model.sh SRC [DST]
#
# SRC 根目录下有 config.json 时从 SRC 复制模型文件，否则从编号最大的 checkpoint-* 复制。
# DST 默认为 SRC 上两级目录下的 <SRC 目录名>，
# 例如 dLLM_sft/sfted_models/ckpt/SDAR-8B-Chat-sfted-webshop -> dLLM_sft/sfted_models/SDAR-8B-Chat-sfted-webshop。
# 训练记录（trainer_state.json、日志等）不复制；DST 已存在时先删除再导出。
set -euo pipefail

if [[ $# -lt 1 ]]; then
  sed -n '4,5p' "$0"
  exit 1
fi

src="${1%/}"
dst="${2:-$(dirname "$(dirname "${src}")")/$(basename "${src}")}"

model_dir="${src}"
if [[ ! -f "${model_dir}/config.json" ]]; then
  model_dir="$(ls -d "${src}"/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)"
fi
if [[ -z "${model_dir}" || ! -f "${model_dir}/config.json" ]]; then
  echo "[export_sfted_model] ${src} 中没有找到模型" >&2
  exit 1
fi
rm -rf "${dst}"
mkdir -p "${dst}"
for f in "${model_dir}"/*; do
  [[ -f "${f}" ]] || continue
  case "$(basename "${f}")" in
    trainer_state.json | training_args.bin | all_results.json | train_results.json | eval_results.json | \
      trainer_log.jsonl | training_loss.png | rng_state*.pth | scheduler.pt | optimizer.pt | latest | zero_to_fp32.py)
      continue ;;
  esac
  cp "${f}" "${dst}/"
done
echo "[export_sfted_model] ${model_dir} -> ${dst}"
