# Qwen2.5-7B-Instruct, mbpp_instruct
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
PYTHONPATH=. \
accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=Qwen/Qwen2.5-7B-Instruct,temperature=0.0" \
    --tasks mbpp_instruct \
    --num_fewshot 0 \
    --batch_size 16 \
    --output_path evals_results/mbpp_instruct \
    --log_samples \
    --confirm_run_unsafe_code \
    --gen_kwargs do_sample=False,max_gen_toks=256
    
## Vanilla Dream, TPF=1.0:
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# Fast-dLLM Dream (dual cache):
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Dream-org/Dream-v0-Instruct-7B,trust_remote_code=True,max_new_tokens=256,diffusion_steps=8,dtype=bfloat16,temperature=0.,alg=confidence_threshold,threshold=0.9,generation_method=Fast_dllm_v1,use_cache=True,dual_cache=True,block_length=32 --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/fast_dllm_dual_cache --log_samples --confirm_run_unsafe_code --apply_chat_template


# dParallel-Dream, TPF=1.0:
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Zigeng/dParallel_Dream_7B_Instruct,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# dParallel-Dream, entropy-threshold=0.5:
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 12334 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=Zigeng/dParallel_Dream_7B_Instruct,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype="bfloat16",temperature=0.,alg="entropy_threshold",dParallel=True,threshold=0.5 --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# d3LLM-Dream, TPF=1.0:
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.1,top_p=0.9,alg=entropy,dParallel=False --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# d3LLM-Dream: generate_multi_block (no delay):
export HF_ALLOW_CODE_EVAL=1
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=True,pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.4,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=10000 --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template


# d3LLM-Dream: generate_multi_block, delay=2:
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1
cd ~/Codes/d3LLM/utils/utils_Dream/eval_instruct
accelerate launch --main_process_port 46666 -m lm_eval --model diffllm --model_args torch_compile=False,pretrained=d3LLM/d3LLM_Dream,trust_remote_code=True,max_new_tokens=256,diffusion_steps=256,dtype=bfloat16,temperature=0.,alg=entropy_threshold,dParallel=False,threshold=0.45,generation_method=generation_multi_block,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,cache_delay_iter=2,refresh_interval=10000,early_stop=True --tasks mbpp_instruct --device cuda --batch_size 1 --num_fewshot 4 --output_path ./eval_tmp/multi_block_cot --log_samples --confirm_run_unsafe_code --apply_chat_template
