#!/bin/bash
#SBATCH --job-name=d3llm_gen_24gpu
#SBATCH --partition=all
#SBATCH --nodes=6
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=4
#SBATCH --time=24:00:00
#SBATCH --output=logs/LLaDA_stage_1/llada_generate_24gpu_%j.out
#SBATCH --error=logs/LLaDA_stage_1/llada_generate_24gpu_%j.err

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
mkdir -p logs/LLaDA_stage_1

# Configuration
NUM_GPUS=24
STEPS=256
GEN_LENGTH=256
BLOCK_LENGTH=32
OUTPUT_DIR="trajectory_data_llada_32"

# Change to workspace directory
cd /home/$USER/Codes/d3LLM

# Option 1: Run with container (recommended for cluster)
# Make sure to import the container first if you haven't:
# cd "$USER_HOME" && enroot import "$IMAGE_NAME"

# srun --container-image="$CONTAINER_IMAGE" \
#      --container-env="${CONTAINER_ENV_LIST}" \
#      --container-mounts="/home/$USER:/home/$USER" \
#      python d3llm_LLaDA/distill_1_data_prepare/d3llm_llada_generate_multinode.py \
#      --num_gpus $NUM_GPUS \
#      --steps $STEPS \
#      --gen_length $GEN_LENGTH \
#      --block_length $BLOCK_LENGTH \
#      --output_dir $OUTPUT_DIR

# Option 2: Run directly (if Python environment is configured on compute node)
srun python d3llm/d3llm_LLaDA/distill_1_data_prepare/d3llm_llada_generate_multinode.py \
    --num_gpus $NUM_GPUS \
    --steps $STEPS \
    --gen_length $GEN_LENGTH \
    --block_length $BLOCK_LENGTH \
    --output_dir $OUTPUT_DIR

echo "Job completed!"

