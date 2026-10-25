#!/usr/bin/env bash
# d3LLaDA-8B Search LoRA online trajectory distillation.
# FullAttnMultiBlock rollout + block-trajectory curriculum CE.
set -uo pipefail
set -x

export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
export SEARCH_HISTORY_LOG=${SEARCH_HISTORY_LOG:-1}

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the merged RL checkpoint (global_step_*/actor_merged_hf) to distill}
ROLLOUT_YAML="${REPO_ROOT}/verl/trainer/config/llada.yaml"

TRAIN_DATA=${TRAIN_DATA:-"$HOME/data/searchR1_processed_direct/train.parquet"}
VAL_DATA=${VAL_DATA:-"$HOME/data/searchR1_processed_direct/test.parquet"}
RETRIEVER_INDEX=${RETRIEVER_INDEX:-"$HOME/data/searchR1/e5_Flat.index"}
RETRIEVER_CORPUS=${RETRIEVER_CORPUS:-"$HOME/data/searchR1/wiki-18.jsonl"}
SERVER_URL=${SERVER_URL:-"http://127.0.0.1:8000/retrieve"}
SERVER_PID=""

cleanup() {
  if [[ -n "${SERVER_PID}" ]] && kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo ">>> 清理 Retrieval Server (PID: ${SERVER_PID})..."
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

for required_file in \
  "${MODEL_PATH}/config.json" \
  "${TRAIN_DATA}" \
  "${VAL_DATA}" \
  "${RETRIEVER_INDEX}" \
  "${RETRIEVER_CORPUS}" \
  "${ROLLOUT_YAML}"; do
  if [[ ! -e "${required_file}" ]]; then
    echo "ERROR: required file not found: ${required_file}" >&2
    exit 1
  fi
done

echo ">>> [1/2] 启动 Retrieval Server..."
"${RETRIEVER_PYTHON}" "${REPO_ROOT}/scripts/training/search/retriever/retrieval_server.py" \
  --index_path "${RETRIEVER_INDEX}" \
  --corpus_path "${RETRIEVER_CORPUS}" \
  --topk "${RETRIEVER_TOPK:-3}" \
  --retriever_name e5 \
  --retriever_model "${RETRIEVER_MODEL:-intfloat/e5-base-v2}" \
  --faiss_gpu \
  --port "${RETRIEVER_PORT:-8000}" \
  > "${REPO_ROOT}/retrieval_server.log" 2>&1 &
SERVER_PID=$!

success=false
for ((i=1; i<=${MAX_RETRIES:-60}; i++)); do
  if curl -s -o /dev/null "${SERVER_URL}" --connect-timeout 2; then
    success=true
    break
  fi
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: Retrieval Server 意外退出，请查看 retrieval_server.log" >&2
    exit 1
  fi
  sleep "${RETRY_INTERVAL:-5}"
done
if [[ "${success}" != "true" ]]; then
  echo "ERROR: Retrieval Server 启动超时" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-/tmp/flashinfer_d3llada_search_distill_lora_${SLURM_JOB_ID:-$$}}
export RAY_TMPDIR=${RAY_TMPDIR:-/tmp/ray_${SLURM_JOB_ID:-$$}}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}
unset RAY_ADDRESS
mkdir -p "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"


# 每个 MC 样本从 rollout 的真实 decode trajectory 中抽取一个 denoising state。
mc_num=${MC_NUM:-16}
n_l=${N_L:-1}
lora_rank=${LORA_RANK:-128}
lora_alpha=${LORA_ALPHA:-256}
lora_dropout=${LORA_DROPOUT:-0.0}
lora_target_modules=${LORA_TARGET_MODULES:-all-linear}
trajectory_block_size=${TRAJECTORY_BLOCK_SIZE:-32}
curriculum_min_mask_ratio=${CURRICULUM_MIN_MASK_RATIO:-0.1}
curriculum_initial_max_mask_ratio=${CURRICULUM_INITIAL_MAX_MASK_RATIO:-0.2}
curriculum_max_mask_ratio=${CURRICULUM_MAX_MASK_RATIO:-1.0}
curriculum_total_steps=${CURRICULUM_TOTAL_STEPS:-${TOTAL_TRAINING_STEPS:-300}}
train_data_size=${TRAIN_DATA_SIZE:-64}
val_data_size=${VAL_DATA_SIZE:-512}
group_size=1
num_cpus_per_env_worker=${NUM_CPUS_PER_ENV_WORKER:-0.1}
max_prompt_length=${MAX_PROMPT_LENGTH:-3580}
max_response_length=${MAX_RESPONSE_LENGTH:-512}
max_model_len=4096
if (( max_prompt_length + max_response_length >= max_model_len )); then
  echo "ERROR: MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH=$((max_prompt_length + max_response_length)) 必须小于 max_model_len=${max_model_len}" >&2
  exit 1
fi

BASE_EXP_NAME=${EXP_NAME:-dllm_search_d3llada8b_distill_lora}
VERSION_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)${JOB_TAG:+_${JOB_TAG}}"
EXP_NAME="${BASE_EXP_NAME}_${VERSION_TAG}"
CKPT_DIR=${CKPT_DIR:-"${REPO_ROOT}/checkpoints/verl_dllm/${EXP_NAME}"}
mkdir -p "${CKPT_DIR}"
export SEARCH_HISTORY_RUN_ID=${SEARCH_HISTORY_RUN_ID:-"${VERSION_TAG}"}
exec > >(tee "${CKPT_DIR}/output.out") 2> >(tee "${CKPT_DIR}/error.err" >&2)

cp "$0" "${CKPT_DIR}/$(basename "$0")"
cp "${ROLLOUT_YAML}" "${CKPT_DIR}/llada.yaml"
git -C "${REPO_ROOT}" rev-parse HEAD > "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true
git -C "${REPO_ROOT}" diff HEAD >> "${CKPT_DIR}/git_commit.txt" 2>/dev/null || true

RESUME_FROM_PATH=${RESUME_FROM_PATH:-}
HYDRA_RESUME_ARGS=("trainer.resume_mode=disable" "trainer.resume_from_path=null")
if [[ -n "${RESUME_FROM_PATH}" ]]; then
  if [[ ! -d "${RESUME_FROM_PATH}" ]]; then
    echo "ERROR: resume path not found: ${RESUME_FROM_PATH}" >&2
    exit 1
  fi
  HYDRA_RESUME_ARGS=("trainer.resume_mode=resume_path" "trainer.resume_from_path=${RESUME_FROM_PATH}")
fi

echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== LoRA Online Distillation | full-attn block trajectory curriculum | block=${trajectory_block_size} mc_num=${mc_num} ==="
echo "=== LoRA: rank=${lora_rank} alpha=${lora_alpha} dropout=${lora_dropout} target_modules=${lora_target_modules} ==="
echo "=== curriculum: U(${curriculum_min_mask_ratio}, upper), initial_upper=${curriculum_initial_max_mask_ratio}, final_upper=${curriculum_max_mask_ratio}, steps=${curriculum_total_steps} ==="
echo "=== SDAR <reason> Search prompt | group_size=1 ==="
echo "=== prompt=${max_prompt_length} response=${max_response_length} context=${max_model_len} ==="
echo "=== EXP_NAME=${EXP_NAME} CKPT_DIR=${CKPT_DIR} ==="

TRAIN_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  algorithm.use_kl_in_reward=False
  data.train_files="${TRAIN_DATA}"
  data.val_files="${VAL_DATA}"
  data.train_batch_size=$train_data_size
  data.val_batch_size=$val_data_size
  data.max_prompt_length=$max_prompt_length
  data.max_response_length=$max_response_length
  +data.num_val_samples=${NUM_VAL_SAMPLES:-512}
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers=1
  data.truncation='left'
  data.return_raw_chat=True
  data.trust_remote_code=True
  data.not_use_template=False
  env.search.use_sdar_prompt=True
  +actor_rollout_ref.model.name=llada
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.lora_rank=$lora_rank
  actor_rollout_ref.model.lora_alpha=$lora_alpha
  +actor_rollout_ref.model.lora_dropout=$lora_dropout
  actor_rollout_ref.model.target_modules=$lora_target_modules
  actor_rollout_ref.model.use_remove_padding=False
  +actor_rollout_ref.model.attn_implementation=flash_attention_2
  actor_rollout_ref.actor.optim.lr=${LR:-1e-5}
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH:--1}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH:-2}
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock]
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock]
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.ref.strategy=fsdp
  actor_rollout_ref.actor.fsdp_config.param_offload=True
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=True
  actor_rollout_ref.actor.use_dynamic_bsz=False
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  +actor_rollout_ref.dllm_config.mc_num=$mc_num
  +actor_rollout_ref.dllm_config.n_l=$n_l
  +actor_rollout_ref.dllm_config.actor_type=distill
  +actor_rollout_ref.dllm_config.noisy_backend=block_trajectory_curriculum
  +actor_rollout_ref.dllm_config.block_random_block_size=$trajectory_block_size
  +actor_rollout_ref.dllm_config.curriculum_min_mask_ratio=$curriculum_min_mask_ratio
  +actor_rollout_ref.dllm_config.curriculum_initial_max_mask_ratio=$curriculum_initial_max_mask_ratio
  +actor_rollout_ref.dllm_config.curriculum_max_mask_ratio=$curriculum_max_mask_ratio
  +actor_rollout_ref.dllm_config.curriculum_total_steps=$curriculum_total_steps
  +actor_rollout_ref.dllm_config.log_prob_level=sequence-level
  +actor_rollout_ref.dllm_config.normalize_log_prob_by_length=True
  +actor_rollout_ref.dllm_config.p_mask_normalization=True
  +actor_rollout_ref.dllm_config.approx_per_mc_update=False
  +actor_rollout_ref.dllm_config.block_size=-1
  +actor_rollout_ref.actor.cfg_scale=0.0
  +actor_rollout_ref.actor.use_block_log_prob=False
  +actor_rollout_ref.actor.distill_success_threshold=${DISTILL_THRESH:-1.0}
  +actor_rollout_ref.actor.distill_all=${DISTILL_ALL:-False}
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.grad_clip=${GRAD_CLIP:-1.0}
  actor_rollout_ref.actor.use_invalid_action_penalty=True
  actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_PENALTY:-1.0}
  actor_rollout_ref.rollout.name=$ENGINE
  +actor_rollout_ref.rollout.dllm_algorithm=FullAttnMultiBlock
  +actor_rollout_ref.rollout.dllm_algorithm_config="${ROLLOUT_YAML}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB:-4}
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  +actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.25}
  actor_rollout_ref.rollout.max_model_len=$max_model_len
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}
  +actor_rollout_ref.rollout.mask_invalid_token=${MASK_INVALID_TOKEN:-True}
  +actor_rollout_ref.rollout.num_workers=${ROLLOUT_WORKERS:-1}
  +actor_rollout_ref.rollout.num_gpus_per_worker=${ROLLOUT_GPUS:-4}
  +actor_rollout_ref.rollout.mc_num=$mc_num
  +actor_rollout_ref.rollout.n_l=$n_l
  +actor_rollout_ref.rollout.cfg_scale=0.0
  +actor_rollout_ref.rollout.dllm_full_response_mask=True
  actor_rollout_ref.rollout.temperature=${TRAIN_TEMP:-0.6}
  actor_rollout_ref.rollout.do_sample=True
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.temperature=0.0
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=${REF_LOGPROB_MB:-8}
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  +actor_rollout_ref.ref.use_dynamic_bsz=False
  env.env_name=search
  env.seed=0
  env.max_steps=${MAX_STEPS:-4}
  env.history_length=${HISTORY_LENGTH:-4}
  env.search.search_url="${SERVER_URL}"
  env.rollout.n=$group_size
  env.resources_per_worker.num_cpus=$num_cpus_per_env_worker
  trainer.logger=['console','wandb']
  trainer.project_name='verl_dllm'
  trainer.experiment_name=$EXP_NAME
  trainer.balance_batch=False
  "${HYDRA_RESUME_ARGS[@]}"
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4}
  trainer.nnodes=${NNODES:-1}
  trainer.save_freq=${SAVE_FREQ:-50}
  trainer.test_freq=${TEST_FREQ:-10}
  trainer.total_epochs=${TOTAL_EPOCHS:-300}
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-300}
  trainer.default_local_dir=$CKPT_DIR
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
  trainer.val_only=False
  +ray_init.address=local
)

echo ">>> [2/2] 开始 d3LLaDA Search LoRA block trajectory curriculum distillation..."
"${PYTHON}" -m verl.trainer.main_dllm_new "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?

if [[ "${RUN_MERGE:-0}" == "1" ]]; then
  echo "RUN_MERGE=1 is not supported for LoRA checkpoints."
  echo "Use global_step_*/actor/lora_adapter together with MODEL_PATH."
fi

echo "=== Finished Search d3LLaDA-8B LoRA Online Block Trajectory Curriculum Distillation ==="
exit "${TRAIN_EXIT}"
