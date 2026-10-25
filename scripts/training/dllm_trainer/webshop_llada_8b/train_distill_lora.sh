#!/usr/bin/env bash
# d3LLaDA-8B WebShop LoRA online trajectory distillation.
# FullAttnMultiBlock rollout + block-trajectory curriculum CE.

set -uo pipefail
set -x

export PYTHONUNBUFFERED=1
export VERL_LOGGING_LEVEL=INFO
export HYDRA_FULL_ERROR=${HYDRA_FULL_ERROR:-1}
export TOKENIZERS_PARALLELISM=false
export TORCHDYNAMO_DISABLE=${TORCHDYNAMO_DISABLE:-1}

ENGINE=${1:-sglang}
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"
source "${REPO_ROOT}/scripts/Misc/common_env.sh"
cd "${REPO_ROOT}"

MODEL_PATH=${MODEL_PATH:?set MODEL_PATH to the merged RL checkpoint (global_step_*/actor_merged_hf) to distill}
ROLLOUT_YAML=${ROLLOUT_YAML:-"${REPO_ROOT}/verl/trainer/config/llada.yaml"}

for required_path in \
  "${MODEL_PATH}/config.json" \
  "${MODEL_PATH}/model.safetensors" \
  "${MODEL_PATH}/modeling_llada.py" \
  "${MODEL_PATH}/tokenizer.json" \
  "${MODEL_PATH}/tokenizer_config.json" \
  "${ROLLOUT_YAML}"; do
  if [[ ! -f "${required_path}" ]]; then
    echo "ERROR: required file not found: ${required_path}" >&2
    exit 1
  fi
done
if [[ ! -x "${PYTHON}" ]]; then
  echo "ERROR: Python is not executable: ${PYTHON}" >&2
  exit 1
fi

export PYTHONPATH="${REPO_ROOT}/sglang_dllm/python:${PYTHONPATH:-}"
export PATH="${JAVA_HOME}/bin:${PATH}"
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1,2,3}
export SGLANG_DISABLE_CUDNN_CHECK=${SGLANG_DISABLE_CUDNN_CHECK:-1}
export SGLANG_FLASHINFER_WORKSPACE_SIZE=${SGLANG_FLASHINFER_WORKSPACE_SIZE:-$((4 * 1024 * 1024 * 1024))}
export FLASHINFER_CACHE_DIR=${FLASHINFER_CACHE_DIR:-"/tmp/flashinfer_d3llada_webshop_distill_lora_${SLURM_JOB_ID:-$$}"}
export RAY_TMPDIR=${RAY_TMPDIR:-"/tmp/r_${SLURM_JOB_ID:-$$}"}
export NCCL_TIMEOUT=${NCCL_TIMEOUT:-10800000}
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=${TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC:-10800}
export GLOO_TIMEOUT_MS=${GLOO_TIMEOUT_MS:-10800000}
unset RAY_ADDRESS
mkdir -p "${FLASHINFER_CACHE_DIR}" "${RAY_TMPDIR}" "${REPO_ROOT}/temp/invalid"

# LoRA / distillation defaults. Every value can be overridden by environment.
lora_rank=${LORA_RANK:-32}
lora_alpha=${LORA_ALPHA:-64}
lora_dropout=${LORA_DROPOUT:-0.0}
lora_target_modules=${LORA_TARGET_MODULES:-all-linear}
mc_num=${MC_NUM:-16}
n_l=${N_L:-1}
trajectory_block_size=${TRAJECTORY_BLOCK_SIZE:-32}
curriculum_min_mask_ratio=${CURRICULUM_MIN_MASK_RATIO:-0.1}
curriculum_initial_max_mask_ratio=${CURRICULUM_INITIAL_MAX_MASK_RATIO:-0.2}
curriculum_max_mask_ratio=${CURRICULUM_MAX_MASK_RATIO:-1.0}
curriculum_total_steps=${CURRICULUM_TOTAL_STEPS:-${TOTAL_TRAINING_STEPS:-300}}

train_data_size=${TRAIN_DATA_SIZE:-16}
val_data_size=${VAL_DATA_SIZE:-128}
group_size=1
max_prompt_length=${MAX_PROMPT_LENGTH:-3326}
max_response_length=${MAX_RESPONSE_LENGTH:-768}
max_model_len=4096
if (( max_prompt_length + max_response_length >= max_model_len )); then
  echo "ERROR: MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH=$((max_prompt_length + max_response_length)) 必须小于 max_model_len=${max_model_len}" >&2
  exit 2
fi

BASE_EXP_NAME=${EXP_NAME:-dllm_webshop_d3llada8b_distill_lora}
VERSION_TAG="${SLURM_JOB_ID:-$$}_$(date +%Y%m%d_%H%M%S)${JOB_TAG:+_${JOB_TAG}}"
EXP_NAME="${BASE_EXP_NAME}_${VERSION_TAG}"
CKPT_DIR=${CKPT_DIR:-"${REPO_ROOT}/checkpoints/verl_dllm/${EXP_NAME}"}
DATA_DIR=${DATA_DIR:-"${HOME}/data/verl-agent_${EXP_NAME}"}
mkdir -p "${CKPT_DIR}" "${DATA_DIR}/text"
exec > >(tee -a "${CKPT_DIR}/output.out") 2> >(tee -a "${CKPT_DIR}/error.err" >&2)

rm -f "${DATA_DIR}/text/train.parquet" "${DATA_DIR}/text/test.parquet"
"${PYTHON}" -m scripts.training.data_preprocess.prepare \
  --mode text \
  --local_dir "${DATA_DIR}" \
  --train_data_size "${train_data_size}" \
  --val_data_size "${val_data_size}"

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

stop_ray() {
  if command -v ray >/dev/null 2>&1; then
    ray stop --force >/dev/null 2>&1 || true
  fi
}
trap stop_ray EXIT

echo "=== d3LLaDA-8B WebShop LoRA Online Trajectory Distillation ==="
echo "=== MODEL_PATH=${MODEL_PATH} ==="
echo "=== LoRA: rank=${lora_rank}, alpha=${lora_alpha}, dropout=${lora_dropout}, target=${lora_target_modules} ==="
echo "=== trajectory: block=${trajectory_block_size}, mc_num=${mc_num}, curriculum_steps=${curriculum_total_steps} ==="
echo "=== train=${train_data_size}, val=${val_data_size}, group=${group_size} ==="
echo "=== context=${max_prompt_length}+${max_response_length}=${max_model_len} ==="
echo "=== EXP_NAME=${EXP_NAME}, CKPT_DIR=${CKPT_DIR} ==="

TRAIN_ARGS=(
  algorithm.adv_estimator=grpo
  algorithm.norm_adv_by_std_in_grpo=True
  algorithm.use_kl_in_reward=False
  data.train_files="${DATA_DIR}/text/train.parquet"
  data.val_files="${DATA_DIR}/text/test.parquet"
  data.train_batch_size=$train_data_size
  data.val_batch_size=$val_data_size
  data.max_prompt_length=$max_prompt_length
  data.max_response_length=$max_response_length
  data.filter_overlong_prompts=True
  data.filter_overlong_prompts_workers=1
  data.truncation=left
  data.return_raw_chat=True
  data.trust_remote_code=True
  data.not_use_template=False
  +data.enable_thinking=False
  +actor_rollout_ref.model.name=llada
  actor_rollout_ref.model.path="${MODEL_PATH}"
  actor_rollout_ref.model.trust_remote_code=True
  actor_rollout_ref.model.lora_rank=$lora_rank
  actor_rollout_ref.model.lora_alpha=$lora_alpha
  +actor_rollout_ref.model.lora_dropout=$lora_dropout
  actor_rollout_ref.model.target_modules=$lora_target_modules
  actor_rollout_ref.model.use_remove_padding=False
  +actor_rollout_ref.model.attn_implementation=flash_attention_2
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.actor.optim.lr=${LR:-5e-6}
  actor_rollout_ref.actor.optim.weight_decay=0.01
  actor_rollout_ref.actor.ppo_mini_batch_size=${MINI_BATCH:--1}
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=${MICRO_BATCH:-2}
  +actor_rollout_ref.actor.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock]
  +actor_rollout_ref.ref.fsdp_config.wrap_policy.transformer_layer_cls_to_wrap=[LLaDALlamaBlock]
  actor_rollout_ref.actor.strategy=fsdp
  actor_rollout_ref.ref.strategy=fsdp
  actor_rollout_ref.actor.fsdp_config.param_offload=${PARAM_OFFLOAD:-True}
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=${OPTIMIZER_OFFLOAD:-True}
  actor_rollout_ref.actor.use_dynamic_bsz=False
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
  +actor_rollout_ref.actor.distill_success_threshold=${DISTILL_THRESH:-10.0}
  +actor_rollout_ref.actor.distill_all=${DISTILL_ALL:-False}
  +actor_rollout_ref.actor.distill_filter_invalid_actions=${DISTILL_FILTER_INVALID_ACTIONS:-True}
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.grad_clip=${GRAD_CLIP:-1.0}
  actor_rollout_ref.actor.use_invalid_action_penalty=True
  actor_rollout_ref.actor.invalid_action_penalty_coef=${INVALID_PENALTY:-1.0}
  actor_rollout_ref.rollout.name=$ENGINE
  +actor_rollout_ref.rollout.dllm_algorithm=FullAttnMultiBlock
  +actor_rollout_ref.rollout.dllm_algorithm_config="${ROLLOUT_YAML}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=${LOGPROB_MB:-4}
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.engine_kwargs.sglang.attention_backend=flashinfer
  actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.25}
  actor_rollout_ref.rollout.max_model_len=$max_model_len
  actor_rollout_ref.rollout.enforce_eager=False
  actor_rollout_ref.rollout.free_cache_engine=False
  actor_rollout_ref.rollout.max_num_seqs=${ROLLOUT_MAX_NUM_SEQS:-16}
  +actor_rollout_ref.rollout.mask_invalid_token=${MASK_INVALID_TOKEN:-True}
  +actor_rollout_ref.rollout.num_workers=${ROLLOUT_WORKERS:-1}
  +actor_rollout_ref.rollout.num_gpus_per_worker=${ROLLOUT_GPUS:-4}
  +actor_rollout_ref.rollout.fixed_teacher=False
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
  env.env_name=Webshop
  env.webshop.use_sdar_prompt=True
  env.seed=${ENV_SEED:-1}
  env.max_steps=${MAX_STEPS:-15}
  env.rollout.n=$group_size
  env.resources_per_worker.num_cpus=${NUM_CPUS_PER_ENV_WORKER:-0.1}
  trainer.logger=['console','wandb']
  trainer.project_name=verl_dllm
  trainer.experiment_name=$EXP_NAME
  trainer.balance_batch=False
  trainer.adjust_batch_samples_participate_in_gradient=${ADJUST_BATCH_SAMPLES_PARTICIPATE_IN_GRADIENT:-False}
  "${HYDRA_RESUME_ARGS[@]}"
  trainer.n_gpus_per_node=${N_GPUS_PER_NODE:-4}
  trainer.nnodes=${NNODES:-1}
  trainer.save_freq=${SAVE_FREQ:-30}
  trainer.test_freq=${TEST_FREQ:-10}
  trainer.total_epochs=${TOTAL_EPOCHS:-300}
  trainer.total_training_steps=${TOTAL_TRAINING_STEPS:-300}
  trainer.save_trajectory_data=False
  trainer.default_local_dir="${CKPT_DIR}"
  trainer.val_before_train=${VAL_BEFORE_TRAIN:-True}
  trainer.val_only=False
  +ray_init.address=local
)

"${PYTHON}" -m verl.trainer.main_dllm_new "${TRAIN_ARGS[@]}"
TRAIN_EXIT=$?

if [[ "${RUN_MERGE:-0}" == "1" ]]; then
  echo "RUN_MERGE=1 is not supported for LoRA checkpoints."
  echo "Use global_step_*/actor/lora_adapter together with MODEL_PATH."
fi

echo "=== Finished d3LLaDA-8B WebShop LoRA Online Trajectory Distillation ==="
exit "${TRAIN_EXIT}"
