#!/usr/bin/env bash
# Run one WebShop model on one GPU. Defaults to one active sequence; set
# BENCH_BATCH_SIZE to measure aggregate batched throughput.

set -euo pipefail

MODEL_KIND=${1:?Usage: benchmark_tps_single_gpu.sh <sdar|llada|qwen> <model_path> <run_name>}
MODEL_PATH=${2:?Usage: benchmark_tps_single_gpu.sh <sdar|llada|qwen> <model_path> <run_name>}
RUN_NAME=${3:?Usage: benchmark_tps_single_gpu.sh <sdar|llada|qwen> <model_path> <run_name>}
case "${MODEL_KIND}" in
  sdar|llada|qwen) ;;
  *)
    echo "ERROR: MODEL_KIND must be sdar, llada, or qwen, got ${MODEL_KIND}" >&2
    exit 2
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

if [[ ! -f "${MODEL_PATH}/config.json" ]]; then
  echo "ERROR: model config not found: ${MODEL_PATH}/config.json" >&2
  exit 2
fi

_CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
export CUDA_VISIBLE_DEVICES="${_CUDA_VISIBLE_DEVICES}"
export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
export TOKENIZERS_PARALLELISM=false
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/ray_tps_${SLURM_JOB_ID:-$$}_${RUN_NAME}"}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
unset RAY_ADDRESS

VAL_DATA_SIZE=${VAL_DATA_SIZE:-500}
MAX_RESPONSE_LENGTH=${MAX_RESPONSE_LENGTH:-768}
GPU_MEMORY_UTILIZATION=${GPU_MEMORY_UTILIZATION:-0.45}
MC_NUM=${MC_NUM:-3}
ENV_SEED=${ENV_SEED:-1}
MAX_STEPS=${MAX_STEPS:-15}
BENCH_BATCH_SIZE=${BENCH_BATCH_SIZE:-1}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-${BENCH_BATCH_SIZE}}
if [[ "${MODEL_KIND}" == "llada" ]]; then
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-3326}
  DATA_TRUNCATION=${DATA_TRUNCATION:-left}
else
  MAX_PROMPT_LENGTH=${MAX_PROMPT_LENGTH:-4096}
  DATA_TRUNCATION=${DATA_TRUNCATION:-error}
fi
DATA_DIR=${DATA_DIR:-"${HOME}/data/webshop_tps_${SLURM_JOB_ID:-local}_${RUN_NAME}"}
RUN_DIR=${RUN_DIR:-"${REPO_ROOT}/outputs/webshop_tps/runs/${RUN_NAME}"}
mkdir -p "${RAY_TMPDIR}" "${DATA_DIR}/text" "${RUN_DIR}" "${REPO_ROOT}/temp/invalid"

"${PYTHON}" -m scripts.training.data_preprocess.prepare \
  --mode text \
  --local_dir "${DATA_DIR}" \
  --train_data_size 1 \
  --val_data_size "${VAL_DATA_SIZE}"

COMMON_ARGS=(
  "data.train_files=${DATA_DIR}/text/train.parquet"
  "data.val_files=${DATA_DIR}/text/test.parquet"
  "data.train_batch_size=1"
  "data.val_batch_size=${BENCH_BATCH_SIZE}"
  "data.max_prompt_length=${MAX_PROMPT_LENGTH}"
  "data.max_response_length=${MAX_RESPONSE_LENGTH}"
  "data.filter_overlong_prompts=True"
  "data.filter_overlong_prompts_workers=1"
  "data.truncation=${DATA_TRUNCATION}"
  "data.return_raw_chat=True"
  "data.trust_remote_code=True"
  "actor_rollout_ref.model.path=${MODEL_PATH}"
  "actor_rollout_ref.model.trust_remote_code=True"
  "actor_rollout_ref.actor.ppo_mini_batch_size=1"
  "actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.actor.fsdp_config.param_offload=True"
  "actor_rollout_ref.actor.fsdp_config.optimizer_offload=True"
  "actor_rollout_ref.rollout.name=sglang"
  "actor_rollout_ref.rollout.tensor_model_parallel_size=1"
  "actor_rollout_ref.rollout.max_num_seqs=${MAX_NUM_SEQS}"
  "actor_rollout_ref.rollout.n=1"
  "actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEMORY_UTILIZATION}"
  "actor_rollout_ref.rollout.enable_chunked_prefill=False"
  "actor_rollout_ref.rollout.free_cache_engine=False"
  "actor_rollout_ref.rollout.val_kwargs.n=1"
  "actor_rollout_ref.rollout.val_kwargs.temperature=0.0"
  "actor_rollout_ref.rollout.val_kwargs.do_sample=True"
  "actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1"
  "actor_rollout_ref.ref.fsdp_config.param_offload=True"
  "env.env_name=Webshop"
  "env.seed=${ENV_SEED}"
  "env.max_steps=${MAX_STEPS}"
  "env.rollout.n=1"
  "env.resources_per_worker.num_cpus=0.2"
  "trainer.logger=['console']"
  "trainer.project_name=webshop_tps"
  "trainer.experiment_name=${RUN_NAME}"
  "trainer.balance_batch=False"
  "trainer.resume_mode=disable"
  "trainer.n_gpus_per_node=1"
  "trainer.nnodes=1"
  "trainer.default_local_dir=${RUN_DIR}"
  "trainer.val_before_train=True"
  "trainer.val_only=True"
  "trainer.log_val_generations=0"
  "+ray_init.address=local"
  "+ray_init.num_cpus=8"
  "+ray_init.object_store_memory=4294967296"
)

if [[ "${MODEL_KIND}" == "sdar" ]]; then
  "${PYTHON}" -m verl.trainer.main_dllm_new \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    +actor_rollout_ref.model.name=sdar \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.max_model_len=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH + 2)) \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[SDARDecoderLayer]' \
    +actor_rollout_ref.rollout.block_length=4 \
    +actor_rollout_ref.dllm_config.mc_num="${MC_NUM}" \
    +actor_rollout_ref.dllm_config.n_l=1 \
    +actor_rollout_ref.dllm_config.actor_type=rl \
    +actor_rollout_ref.dllm_config.noisy_backend=gaussian \
    +actor_rollout_ref.dllm_config.log_prob_level=block-level \
    +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
    +actor_rollout_ref.dllm_config.block_size=4 \
    +actor_rollout_ref.rollout.mask_invalid_token=True \
    +actor_rollout_ref.rollout.fixed_teacher=False \
    +actor_rollout_ref.rollout.dllm_algorithm=LowConfidence \
    +actor_rollout_ref.rollout.dllm_algorithm_config="${REPO_ROOT}/verl/trainer/config/sdar_dllm.yaml" \
    +actor_rollout_ref.rollout.mc_num="${MC_NUM}" \
    +actor_rollout_ref.rollout.n_l=1 \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    "${COMMON_ARGS[@]}"
elif [[ "${MODEL_KIND}" == "llada" ]]; then
  "${PYTHON}" -m verl.trainer.main_dllm_new \
    algorithm.adv_estimator=grpo \
    algorithm.norm_adv_by_std_in_grpo=True \
    algorithm.use_kl_in_reward=False \
    +data.enable_thinking=False \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    +actor_rollout_ref.model.name=llada \
    actor_rollout_ref.model.use_remove_padding=False \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[LLaDALlamaBlock]' \
    +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap='[LLaDALlamaBlock]' \
    +actor_rollout_ref.dllm_config.mc_num="${MC_NUM}" \
    +actor_rollout_ref.dllm_config.n_l=1 \
    +actor_rollout_ref.dllm_config.actor_type=rl \
    +actor_rollout_ref.dllm_config.noisy_backend=random \
    +actor_rollout_ref.dllm_config.log_prob_level=sequence-level \
    +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True \
    +actor_rollout_ref.dllm_config.p_mask_normalization=True \
    +actor_rollout_ref.dllm_config.approx_per_mc_update=False \
    +actor_rollout_ref.dllm_config.block_size=-1 \
    +actor_rollout_ref.actor.cfg_scale=0.0 \
    +actor_rollout_ref.actor.use_block_log_prob=False \
    +actor_rollout_ref.rollout.dllm_algorithm=FullAttnMultiBlock \
    +actor_rollout_ref.rollout.dllm_algorithm_config="${ROLLOUT_YAML:-${REPO_ROOT}/verl/trainer/config/llada.yaml}" \
    actor_rollout_ref.rollout.max_model_len=4096 \
    +actor_rollout_ref.rollout.mask_invalid_token=True \
    +actor_rollout_ref.rollout.num_workers=1 \
    +actor_rollout_ref.rollout.num_gpus_per_worker=1 \
    +actor_rollout_ref.rollout.mc_num="${MC_NUM}" \
    +actor_rollout_ref.rollout.n_l=1 \
    +actor_rollout_ref.rollout.cfg_scale=0.0 \
    +actor_rollout_ref.rollout.dllm_full_response_mask=True \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    env.webshop.use_sdar_prompt=True \
    "${COMMON_ARGS[@]}"
else
  "${PYTHON}" -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    +data.apply_chat_template_kwargs.enable_thinking=True \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=False \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.rollout_only=True \
    actor_rollout_ref.rollout.load_format=auto \
    actor_rollout_ref.rollout.dtype=bfloat16 \
    actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer \
    actor_rollout_ref.actor.use_invalid_action_penalty=False \
    "${COMMON_ARGS[@]}"
fi
