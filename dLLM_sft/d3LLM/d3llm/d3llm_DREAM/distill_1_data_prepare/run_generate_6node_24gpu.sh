#!/bin/bash
#SBATCH --job-name=dream_multi_try
#SBATCH --partition=all
#SBATCH --nodes=6
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --time=256:00:00
#SBATCH --output=logs/DREAM_stage_1/dream_%j_generate_multi_try.out
#SBATCH --error=logs/DREAM_stage_1/dream_%j_generate_multi_try.err

# Set common variables
USER_HOME="/home/$USER"
IMAGE_NAME="docker://nvcr.io#nvidia/pytorch:25.08-py3"
CONTAINER_IMAGE="$USER_HOME/nvidia+pytorch+25.08-py3.sqsh"

# NCCL environment variables (important for multi-node)
export NCCL_CUMEM_ENABLE=1
export NCCL_MNNVL_ENABLE=2
export UCX_NET_DEVICES=eth0
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_NET_GDR_LEVEL=2
CONTAINER_ENV_LIST="NCCL_CUMEM_ENABLE,NCCL_MNNVL_ENABLE,UCX_NET_DEVICES,NCCL_DEBUG,NCCL_IB_DISABLE,NCCL_NET_GDR_LEVEL"

# Create logs directory if not exists
mkdir -p logs/DREAM_stage_1

# Configuration
NUM_GPUS=24
STEPS=512
GEN_LENGTH=512
BLOCK_LENGTH=32
OUTPUT_DIR="trajectory_data_dream_32"
MAX_DATA_NUM=-1

# Change to workspace directory
cd /home/$USER/Codes/d3LLM

# Option 1: Run with container (recommended for cluster)
# Make sure to import the container first if you haven't:
# cd "$USER_HOME" && enroot import "$IMAGE_NAME"

# srun --container-image="$CONTAINER_IMAGE" \
#      --container-env="${CONTAINER_ENV_LIST}" \
#      --container-mounts="/home/$USER:/home/$USER" \
#      python d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_multinode.py \
#      --num_gpus $NUM_GPUS \
#      --steps $STEPS \
#      --gen_length $GEN_LENGTH \
#      --block_length $BLOCK_LENGTH \
#      --output_dir $OUTPUT_DIR \
#      --max_data_num $MAX_DATA_NUM

# Option 2: Run directly (if Python environment is configured on compute node)
srun python d3llm/d3llm_DREAM/distill_1_data_prepare/d3llm_dream_generate_multinode.py \
    --num_gpus $NUM_GPUS \
    --steps $STEPS \
    --gen_length $GEN_LENGTH \
    --block_length $BLOCK_LENGTH \
    --output_dir $OUTPUT_DIR \
    --max_data_num $MAX_DATA_NUM

echo "Job completed!"

