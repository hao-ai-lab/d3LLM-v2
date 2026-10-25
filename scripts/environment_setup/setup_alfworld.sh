#!/usr/bin/env bash
# Prepare the ALFWorld environment: Python packages and the PDDL / game files.
# Run install_verl_agent_new.sh first.
#
# Optional environment variables:
#   ENV_NAME       main conda environment (default: verl-dllm-agent)
#   CONDA          conda executable (default: conda on PATH)
#   PYTHON         Python executable (default: the one of ENV_NAME)
#   ALFWORLD_DATA  where the game files are stored (default: ~/.cache/alfworld)
set -euo pipefail

CONDA=${CONDA:-$(command -v conda)}
ENV_NAME=${ENV_NAME:-verl-dllm-agent}
PYTHON=${PYTHON:-$($CONDA info --base)/envs/${ENV_NAME}/bin/python3}
export ALFWORLD_DATA=${ALFWORLD_DATA:-$HOME/.cache/alfworld}

if [[ ! -x "$PYTHON" ]]; then
  echo "ERROR: Python not found: ${PYTHON}. Run install_verl_agent_new.sh first (ENV_NAME=${ENV_NAME})." >&2
  exit 1
fi

echo ">>> [1/2] Python packages"
"$PYTHON" -m pip install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld==0.4.2

echo ">>> [2/2] PDDL and game files -> ${ALFWORLD_DATA}"
if [[ -d "${ALFWORLD_DATA}/json_2.1.1" ]]; then
  echo "Already downloaded, skipping."
else
  "$(dirname "$PYTHON")/alfworld-download" -f
fi

echo ">>> ALFWorld is ready."
