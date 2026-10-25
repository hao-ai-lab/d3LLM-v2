#########################################################
# ----------------- humaneval ---------------------------

# Qwen2.5-Coder-7B-Instruct, humaneval
export CUDA_VISIBLE_DEVICES=1
cd ~/Codes/d3LLM/utils/utils_DreamCoder/code_eval
CKPT_DIR=Qwen/Qwen2.5-Coder-7B-Instruct
PYTHONPATH=evalplus python -m evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend hf --temperature 0.1


# Vanilla Dream-Coder, humaneval, vanilla decoding
cd ~/Codes/d3LLM/utils/utils_DreamCoder/code_eval
CKPT_DIR=Dream-org/Dream-Coder-v0-Instruct-7B
PYTHONPATH=evalplus python -m evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend dllm --temperature 0.1

# d3LLM Dream-Coder, humaneval, multi_block decoding
cd ~/Codes/d3LLM/utils/utils_DreamCoder/code_eval
export PYTHONPATH=evalplus
CKPT_DIR=d3LLM/d3LLM_Dream_Coder
python -m evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset humaneval --backend dllm --temperature 0. --generation_method generation_multi_block --alg entropy_threshold --threshold 0.5 --block_length 32 --block_add_threshold 0.1 --decoded_token_threshold 0.95 --cache_delay_iter 32 --early_stop True --torch_compile True

#########################################################
# ---------------------- mbpp ---------------------------

# Qwen2.5-Coder-7B-Instruct, mbpp
export CUDA_VISIBLE_DEVICES=1
cd ~/Codes/d3LLM/utils/utils_DreamCoder/code_eval
CKPT_DIR=Qwen/Qwen2.5-Coder-7B-Instruct
PYTHONPATH=evalplus python -m evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend hf --temperature 0.1

# Vanilla Dream-Coder, mbpp, vanilla decoding
cd ~/Codes/d3LLM/utils/utils_DreamCoder/code_eval
CKPT_DIR=Dream-org/Dream-Coder-v0-Instruct-7B
PYTHONPATH=evalplus python -m evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend dllm --temperature 0.1

# d3LLM Dream-Coder, mbpp, multi_block decoding
cd ~/Codes/d3LLM/utils/utils_DreamCoder/code_eval
export PYTHONPATH=evalplus
CKPT_DIR=d3LLM/d3LLM_Dream_Coder
python -m evalplus.evaluate --model $CKPT_DIR --trust_remote_code True --max_new_tokens 512 --diffusion_steps 512 --dataset mbpp --backend dllm --temperature 0. --generation_method generation_multi_block --alg entropy_threshold --threshold 0.5 --block_length 32 --block_add_threshold 0.1 --decoded_token_threshold 0.95 --cache_delay_iter 32 --early_stop True --torch_compile True

