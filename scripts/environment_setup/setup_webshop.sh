#!/usr/bin/env bash
# Prepare the WebShop environment: product catalog (from HuggingFace), spaCy models, OpenJDK and Lucene search indexes.
# Run install_verl_agent_new.sh first.
#
# Optional environment variables:
#   ENV_NAME     main conda environment (default: verl-dllm-agent)
#   CONDA        conda executable (default: conda on PATH)
#   PYTHON       Python executable (default: the one of ENV_NAME)
#   HF_ENDPOINT  HuggingFace mirror
set -euo pipefail

CONDA=${CONDA:-$(command -v conda)}
ENV_NAME=${ENV_NAME:-verl-dllm-agent}
ENV_PREFIX="$($CONDA info --base)/envs/${ENV_NAME}"
PYTHON=${PYTHON:-${ENV_PREFIX}/bin/python3}
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WEBSHOP_DIR=${REPO}/agent_system/environments/env_package/webshop/webshop
[[ -n "${HF_ENDPOINT:-}" ]] && export HF_ENDPOINT

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: ${PYTHON}. Run install_verl_agent_new.sh first (ENV_NAME=${ENV_NAME})." >&2
  exit 1
fi

echo ">>> [1/4] Product catalog -> ${WEBSHOP_DIR}/data"
"$PYTHON" -c "
from huggingface_hub import snapshot_download
snapshot_download('d3LLM/webshop_catalog', repo_type='dataset', local_dir='${WEBSHOP_DIR}/data')"

echo ">>> [2/4] spaCy models"
if ! "$PYTHON" -c "import en_core_web_lg, en_core_web_sm" 2>/dev/null; then
  # The wheels are git-ignored; download them from the spaCy release page when missing.
  for model in lg sm; do
    whl="${WEBSHOP_DIR}/en_core_web_${model}-3.8.0-py3-none-any.whl"
    if [[ ! -f "$whl" ]]; then
      curl -fL -o "${whl}.part" \
        "https://github.com/explosion/spacy-models/releases/download/en_core_web_${model}-3.8.0/en_core_web_${model}-3.8.0-py3-none-any.whl"
      mv "${whl}.part" "$whl"
    fi
    "$PYTHON" -m pip install "$whl"
  done
fi

echo ">>> [3/4] OpenJDK"
[[ -x "${ENV_PREFIX}/lib/jvm/bin/java" ]] || \
  $CONDA install -c conda-forge openjdk=25.0.1 -n "$ENV_NAME" -y

echo ">>> [4/4] Lucene search indexes -> ${WEBSHOP_DIR}/search_engine/indexes*"
# The documents (search_engine/resources*) are tracked in git; the indexes are generated from them and git-ignored.
SEARCH_DIR=${WEBSHOP_DIR}/search_engine
if [[ -d "${SEARCH_DIR}/indexes" && -d "${SEARCH_DIR}/indexes_100" && -d "${SEARCH_DIR}/indexes_1k" && -d "${SEARCH_DIR}/indexes_100k" ]]; then
  echo "indexes already exist, skipped"
else
  (cd "${SEARCH_DIR}" && \
    JAVA_HOME="${ENV_PREFIX}/lib/jvm" PATH="$(dirname "$PYTHON"):${ENV_PREFIX}/lib/jvm/bin:${PATH}" bash run_indexing.sh)
fi

echo ">>> WebShop is ready."
