#!/usr/bin/env bash
# Prepare the Search (Search-R1) environment: tool-calling package, training data,
# the separate retriever conda environment, and the e5 index / wiki-18 corpus.
# Run install_verl_agent_new.sh first.
#
# Optional environment variables:
#   ENV_NAME       main conda environment (default: verl-dllm-agent)
#   RETRIEVER_ENV  retriever conda environment (default: retriever)
#   CONDA          conda executable (default: conda on PATH)
#   PYTHON         Python of the main environment (default: the one of ENV_NAME)
#   DATA_DIR       index and corpus directory (default: ~/data/searchR1)
#   TRAIN_DIR      processed training data directory (default: ~/data/searchR1_processed_direct)
#   TORCH_INDEX_URL  PyTorch wheel index of the retriever environment (default: cu129)
#   HF_ENDPOINT    HuggingFace mirror
set -euo pipefail

CONDA=${CONDA:-$(command -v conda)}
CONDA_BASE="$($CONDA info --base)"
ENV_NAME=${ENV_NAME:-verl-dllm-agent}
RETRIEVER_ENV=${RETRIEVER_ENV:-retriever}
PYTHON=${PYTHON:-${CONDA_BASE}/envs/${ENV_NAME}/bin/python3}
RETRIEVER_PYTHON=${CONDA_BASE}/envs/${RETRIEVER_ENV}/bin/python
DATA_DIR=${DATA_DIR:-$HOME/data/searchR1}
TRAIN_DIR=${TRAIN_DIR:-$HOME/data/searchR1_processed_direct}
TORCH_INDEX_URL=${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
[[ -n "${HF_ENDPOINT:-}" ]] && export HF_ENDPOINT

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: ${PYTHON}. Run install_verl_agent_new.sh first (ENV_NAME=${ENV_NAME})." >&2
  exit 1
fi

echo ">>> [1/4] Search tool package (skyrl-gym)"
"$PYTHON" -m pip install -e "${REPO}/agent_system/environments/env_package/search/third_party"

echo ">>> [2/4] Training data -> ${TRAIN_DIR}"
if [[ -s "${TRAIN_DIR}/train.parquet" && -s "${TRAIN_DIR}/test.parquet" ]]; then
  echo "Already processed, skipping."
else
  (cd "$REPO" && PYTHONPATH="$REPO" "$PYTHON" scripts/training/data_preprocess/preprocess_search_r1_dataset.py --local_dir "$TRAIN_DIR")
fi

echo ">>> [3/4] Retriever environment (${RETRIEVER_ENV})"
if [[ ! -x "$RETRIEVER_PYTHON" ]]; then
  # faiss-gpu is not available on PyPI, so the retriever runs in its own conda environment.
  $CONDA create -n "$RETRIEVER_ENV" python=3.10 -y
  $CONDA install -n "$RETRIEVER_ENV" numpy==1.26.4 -y
  "$CONDA" install -n "$RETRIEVER_ENV" faiss-gpu --override-channels -c conda-forge -y
  "$RETRIEVER_PYTHON" -m pip install torch==2.8.0 --index-url "$TORCH_INDEX_URL"
  "$RETRIEVER_PYTHON" -m pip install transformers==5.5.4 datasets==4.8.4 pyserini==1.2.0 \
    huggingface_hub==1.11.0 uvicorn==0.44.0 fastapi==0.136.0
else
  echo "Already exists, skipping."
fi

echo ">>> [4/4] e5 index and wiki-18 corpus -> ${DATA_DIR}"
if [[ -s "${DATA_DIR}/e5_Flat.index" && -s "${DATA_DIR}/wiki-18.jsonl" ]]; then
  echo "Already downloaded, skipping."
else
  "$RETRIEVER_PYTHON" "${REPO}/scripts/training/search/searchr1_download.py" --local_dir "$DATA_DIR"
  cat "${DATA_DIR}"/part_* > "${DATA_DIR}/e5_Flat.index"
  gzip -d "${DATA_DIR}/wiki-18.jsonl.gz"
fi

echo ">>> Search is ready. The training scripts start the retrieval server automatically."
