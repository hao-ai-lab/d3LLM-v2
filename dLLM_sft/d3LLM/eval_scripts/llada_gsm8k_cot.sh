# Qwen2.5-7B-Instruct, gsm8k_cot_zeroshot
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1
cd ~/Codes/d3LLM/utils/lm-evaluation-harness
PYTHONPATH=~/Codes/d3LLM/utils/lm-evaluation-harness:$PYTHONPATH \
accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=Qwen/Qwen2.5-7B-Instruct,temperature=0.0" \
    --tasks gsm8k_cot_zeroshot \
    --num_fewshot 0 \
    --batch_size 32 \
    --output_path evals_results/gsm8k_cot_zeroshot   \
    --log_samples \
    --confirm_run_unsafe_code \
    --gen_kwargs do_sample=False,max_gen_toks=256


# Vanilla LLaDA, TPF=1.0:
cd ~/Codes/d3LLM/utils/utils_LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='GSAI-ML/LLaDA-8B-Instruct',gen_length=256,steps=256,block_length=32,show_speed=True,task="gsm8k_cot_zeroshot" --batch_size 1


# Fast-dLLM LLaDA:
cd ~/Codes/d3LLM/utils/utils_LLaDA
rm -rf null/
accelerate launch --main_process_port 12335 eval_llada.py --model llada_dist --model_args model_path=GSAI-ML/LLaDA-8B-Instruct,steps=8,gen_length=256,block_length=32,remasking=low_confidence,threshold=0.9,save_dir=null,show_speed=True,task="gsm8k_cot_zeroshot",generation_method=Fast_dllm_v1,use_cache=True,dual_cache=True --tasks gsm8k_cot_zeroshot --batch_size 1 --output_path evals_results/llada_fast_dllm_dual_cache


# dParallel-LLaDA, TPF=1.0:
cd ~/Codes/d3LLM/utils/utils_LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=256,steps=256,block_length=32,show_speed=True,task="gsm8k_cot_zeroshot" --batch_size 1

# dParallel-LLaDA, entropy-threshold:
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29601 eval_llada.py --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='Zigeng/dParallel-LLaDA-8B-instruct',gen_length=256,steps=256,block_length=32,show_speed=True,threshold=0.5,task="gsm8k_cot_zeroshot" --batch_size 1

# D2F TPF = 1.0
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 29520 \
    --num_processes 4 \
    eval_d2f.py \
    --model dream_lora \
    --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0,block_size=32,block_add_threshold=1.0,skip_threshold=1.0,decoded_token_threshold=1.0,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp \
    --tasks gsm8k_cot \
    --num_fewshot 1 \
    --batch_size 1 \
    --output_path eval_tmp \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --fewshot_as_multiturn

# D2F
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 29520 \
    --num_processes 4 \
    eval_d2f.py \
    --model dream_lora \
    --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0,block_size=32,block_add_threshold=0.7,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp \
    --tasks gsm8k_cot \
    --num_fewshot 1 \
    --batch_size 1 \
    --output_path eval_tmp \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --fewshot_as_multiturn



# d3LLM-LLaDA, TPF=1.0:
cd ~/Codes/d3LLM/utils/utils_LLaDA
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch --main_process_port 29600 eval_llada.py --tasks gsm8k_cot_zeroshot --num_fewshot 0 \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='d3LLM/d3LLM_LLaDA',gen_length=256,steps=256,block_length=32,show_speed=True,task="gsm8k_cot_zeroshot" --batch_size 1


# d3LLM-LLaDA: generate_multi_block:
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=gsm8k_cot_zeroshot
length=256
block_length=32
num_fewshot=0
steps=256
accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='d3LLM/d3LLM_LLaDA',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k_cot_zeroshot",generation_method="generate_multi_block",early_stop=True


# d3LLM-LLaDA: generate_multi_block_kv_cache, delay=2:
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=gsm8k_cot_zeroshot
length=256
block_length=32
num_fewshot=0
steps=256
accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path='d3LLM/d3LLM_LLaDA',gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.5,task="gsm8k_cot_zeroshot",generation_method="generate_multi_block_kv_cache",cache_delay_iter=2,refresh_interval=3,early_stop=True