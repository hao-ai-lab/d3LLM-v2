# Shared environment defaults for the launch scripts. Source it after REPO_ROOT is set:
#   source "${REPO_ROOT}/scripts/Misc/common_env.sh"
#
# All variables can be overridden from the environment:
#   CLUSTER_ENV_SCRIPT  optional script sourced first (e.g. scripts/Misc/cluster/setup_sglang_env_nvl72.sh)
#   PYTHON              Python of the main environment (default: python of the active environment)
#   JAVA_HOME           Java for WebShop (default: <env>/lib/jvm)
#   RETRIEVER_PYTHON    Python of the retriever environment for Search (default: sibling env "retriever")

if [[ -n "${CLUSTER_ENV_SCRIPT:-}" ]]; then
  _common_env_opts="$(set +o)"
  _common_env_flags="$-"
  source "${CLUSTER_ENV_SCRIPT}"
  eval "${_common_env_opts}"
  if [[ "${_common_env_flags}" == *e* ]]; then set -e; fi
  unset _common_env_opts _common_env_flags
fi

PYTHON=${PYTHON:-$(command -v python || true)}
if [[ -z "${PYTHON}" ]]; then
  echo "ERROR: python not found. Activate the environment (conda activate verl-dllm-agent) or set PYTHON." >&2
  exit 1
fi
ENV_PREFIX="$(cd "$(dirname "${PYTHON}")/.." && pwd)"

# CUDA runtime installed by pip (nvidia-cuda-runtime-*) is not on the dynamic loader path. Child processes such as the
# SGLang scheduler (which preloads the memory-saver library linked against libcudart.so.12) fail to start when the
# system CUDA is another major version. The directories are appended, so libraries provided by the system keep priority.
_cuda_wheel_libs="$("${PYTHON}" - <<'PYEOF' 2>/dev/null || true
import glob, importlib.util, os

spec = importlib.util.find_spec("nvidia")
roots = list(spec.submodule_search_locations or []) if spec else []
current = set(filter(None, os.environ.get("LD_LIBRARY_PATH", "").split(os.pathsep)))
dirs = {os.path.dirname(p) for r in roots for p in glob.glob(os.path.join(r, "*", "lib", "libcudart.so.*"))}
print(os.pathsep.join(sorted(dirs - current)))
PYEOF
)"
if [[ -n "${_cuda_wheel_libs}" ]]; then
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}${_cuda_wheel_libs}"
fi
unset _cuda_wheel_libs

# An inherited JAVA_HOME is only kept if it contains libjvm.so (needed by pyjnius for WebShop); otherwise the Java
# installed into the environment by setup_webshop.sh is used.
if [[ -z "$(find "${JAVA_HOME:-/nonexistent}" -maxdepth 5 -name libjvm.so -print -quit 2>/dev/null)" ]]; then
  if [[ -n "${JAVA_HOME:-}" ]]; then
    echo "WARNING: JAVA_HOME=${JAVA_HOME} has no libjvm.so, using ${ENV_PREFIX}/lib/jvm instead." >&2
  fi
  JAVA_HOME="${ENV_PREFIX}/lib/jvm"
fi
export JAVA_HOME
RETRIEVER_PYTHON=${RETRIEVER_PYTHON:-$(dirname "${ENV_PREFIX}")/retriever/bin/python}
