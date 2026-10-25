# Qwen2.5-7B-Instruct, humaneval
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export HF_ALLOW_CODE_EVAL=1
cd ~/Codes/d3LLM/utils/lm-evaluation-harness
PYTHONPATH=~/Codes/d3LLM/utils/lm-evaluation-harness:$PYTHONPATH \
accelerate launch -m lm_eval \
    --model hf \
    --model_args "pretrained=Qwen/Qwen2.5-7B-Instruct,temperature=0.0" \
    --tasks humaneval \
    --num_fewshot 0 \
    --batch_size 32 \
    --output_path evals_results/humaneval \
    --log_samples \
    --confirm_run_unsafe_code \
    --gen_kwargs do_sample=False,max_gen_toks=256
latest_jsonl=$(find evals_results/humaneval -name "samples_humaneval_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python ~/Codes/d3LLM/utils/utils_LLaDA/postprocess_code_humaneval.py "$latest_jsonl" || echo "No jsonl file found"


cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256

# 1. GSAI-ML/LLaDA-8B-Instruct, TPF=1.0
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir='GSAI-ML'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="${task}" \
--output_path evals_results/${output_dir}/${task}-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"

# 2. GSAI-ML/LLaDA-8B-Instruct, Fast-dLLM
rm -rf null/
model_path='GSAI-ML/LLaDA-8B-Instruct'
output_dir='GSAI-ML'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 12335 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},steps=8,gen_length=${length},block_length=${block_length},remasking=low_confidence,threshold=0.9,save_dir=null,show_speed=True,task=${task},generation_method=Fast_dllm_v1,use_cache=True,dual_cache=True \
--output_path evals_results/${output_dir}/${task}-fast-dllm-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-fast-dllm-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"

# 3. Zigeng/dParallel-LLaDA-8B-instruct, TPF=1.0
model_path='Zigeng/dParallel-LLaDA-8B-instruct'
output_dir='Zigeng'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="${task}" \
--output_path evals_results/${output_dir}/${task}-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"

# 4. Zigeng/dParallel-LLaDA-8B-instruct, entropy-threshold
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
model_path='Zigeng/dParallel-LLaDA-8B-instruct'
output_dir='Zigeng'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.45,task="${task}" \
--output_path evals_results/${output_dir}/${task}-entropy-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-entropy-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"



# 5. D2F, TPF = 1.0
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=mbpp
length=256
num_fewshot=3
output_dir='d2f'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 29520 \
    --num_processes 4 \
    eval_d2f.py \
    --model dream_lora \
    --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0,block_size=32,block_add_threshold=1.0,skip_threshold=1.0,decoded_token_threshold=1.0,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results/${output_dir}/${task}-tpf1.0-ns${num_fewshot}-${length} \
    --log_samples \
    --confirm_run_unsafe_code
latest_jsonl=$(find evals_results/${output_dir}/${task}-tpf1.0-ns${num_fewshot}-${length} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"

# 6. D2F
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=mbpp
length=256
num_fewshot=3
output_dir='d2f'
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch \
    --main_process_port 29520 \
    --num_processes 4 \
    eval_d2f.py \
    --model dream_lora \
    --model_args pretrained=GSAI-ML/LLaDA-8B-Instruct,lora_path=SJTU-Deng-Lab/D2F_LLaDA_Instruct_8B_Lora,max_new_tokens=256,diffusion_steps=256,add_bos_token=true,temperature=0,block_size=32,block_add_threshold=0.9,skip_threshold=0.9,decoded_token_threshold=0.95,dtype=bfloat16,sampling_strategy=default,save_dir=eval_tmp \
    --tasks ${task} \
    --num_fewshot ${num_fewshot} \
    --batch_size 1 \
    --output_path evals_results/${output_dir}/${task}-ns${num_fewshot}-${length} \
    --log_samples \
    --confirm_run_unsafe_code \
    --apply_chat_template \
    --fewshot_as_multiturn
latest_jsonl=$(find evals_results/${output_dir}/${task}-ns${num_fewshot}-${length} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"


# 7. d3LLM-LLaDA, TPF=1.0
model_path='d3LLM/d3LLM_LLaDA'
output_dir='d3llm'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 29600 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,task="${task}" \
--output_path evals_results/${output_dir}/${task}-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"


# 8. d3LLM-LLaDA, generate_multi_block
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
model_path='d3LLM/d3LLM_LLaDA'
output_dir='d3llm'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.4,task="${task}",generation_method="generate_multi_block",block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,early_stop=True \
--output_path evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"


# 9. d3LLM-LLaDA, generate_multi_block_kv_cache, delay=1:
cd ~/Codes/d3LLM/utils/utils_LLaDA
export HF_ALLOW_CODE_EVAL=1
export HF_DATASETS_TRUST_REMOTE_CODE=true
task=mbpp
length=256
block_length=32
num_fewshot=3
steps=256
model_path='d3LLM/d3LLM_LLaDA'
output_dir='d3llm'
METHOD_NAME_ENCODED=$(echo "${model_path}" | sed 's|/|__|g')
accelerate launch --main_process_port 29601 eval_llada.py --tasks ${task} --num_fewshot ${num_fewshot} \
--confirm_run_unsafe_code --model llada_dist \
--model_args model_path=${model_path},gen_length=${length},steps=${steps},block_length=${block_length},show_speed=True,threshold=0.4,task="${task}",generation_method="generate_multi_block_kv_cache",cache_delay_iter=1,refresh_interval=10000,block_add_threshold=0.1,decoded_token_threshold=0.95,block_length=32,early_stop=True \
--output_path evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length} --log_samples
latest_jsonl=$(find evals_results/${output_dir}/${task}-multi-block-ns${num_fewshot}-${length}/${METHOD_NAME_ENCODED} -name "samples_${task}_*.jsonl" -type f 2>/dev/null | head -n 1)
[ -n "$latest_jsonl" ] && python postprocess_code_mbpp.py "$latest_jsonl" || echo "No jsonl file found"

