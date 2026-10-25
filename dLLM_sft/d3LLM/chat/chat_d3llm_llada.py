from pathlib import Path
import time
import torch
import tqdm
from transformers import AutoTokenizer, AutoConfig
import sys
# Add project root to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from d3llm.d3llm_LLaDA.d3llm_llada_generate_util import generate_multi_block_kv_cache
from utils.utils_LLaDA.model.modeling_llada import LLaDAModelLM

# Model path
m = "d3LLM/d3LLM_LLaDA"

tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
device = torch.device("cuda:0")

# Load LLaDA model
print("Loading LLaDA model...")
config = AutoConfig.from_pretrained(m)
config.flash_attention = True

model = LLaDAModelLM.from_pretrained(
    m,
    config=config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)
model = model.to(device).eval()

# Generation parameters
gen_params = {
    "steps": 256,
    "max_new_tokens": 256,
    "block_size": 32,
    "temperature": 0.,
    "mask_id": 126336,
    "threshold": 0.5,
    "block_add_threshold": 0.1,
    "decoded_token_threshold": 0.95,
    "cache_delay_iter": 2,
    "early_stop": True,
}

print("\n" + "="*80)
print("d3LLM-LLaDA Chat Mode (type 'quit' or 'exit' to end)")
print("="*80)

# Warmup model
test_questions_path = Path(__file__).parent.parent / 'utils' / 'serve' / 'test_question.txt'
test_questions = []
try:
    with open(test_questions_path, 'r') as f:
        content = f.read()
        # Split by empty lines to get individual questions
        questions = [q.strip() for q in content.split('\n\n') if q.strip()]
        test_questions = questions
except Exception as e:
    print(f"Warning: Could not load test questions: {e}. Using fallback warmup.")
    test_questions = ["Write a hello world program in Python."] * 10

with torch.no_grad():
    num_warmups = min(10, len(test_questions))
    for i in tqdm.tqdm(range(num_warmups), desc="Warming up model"):
        prompt_text = test_questions[i % len(test_questions)]
        inputs = tokenizer(prompt_text, return_tensors="pt")
        input_ids = inputs['input_ids'].to(device)
        output, nfe = generate_multi_block_kv_cache(model, input_ids, **gen_params)
print("Warmup complete.\n")

print("\033[31mNote that because our distillation data primarily consists of **coding** and **math reasoning** tasks, acceleration may only appear on prompts of these tasks.\033[0m")

while True:
    # Get user input
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    if not user_input:
        continue
    
    # Prepare input
    messages = [{"role": "user", "content": user_input}]
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text)['input_ids']
    input_ids = torch.tensor(input_ids).to(device).unsqueeze(0)
    
    # Generate response
    start_time = time.time()
    with torch.no_grad():
        output, nfe = generate_multi_block_kv_cache(model, input_ids, **gen_params)
    end_time = time.time()
    
    # Decode response
    full_response = tokenizer.decode(output[0, input_ids.shape[1]:], skip_special_tokens=True)
    assistant_response = full_response.split(tokenizer.eos_token)[0].strip()
    
    print("\n\033[34mAssistant:\n \033[0m")
    print("\033[34m" + assistant_response + "\033[0m")
    
    # Calculate statistics
    num_generated_tokens = len(tokenizer(assistant_response, add_special_tokens=False)['input_ids'])
    elapsed_time = end_time - start_time
    tps = num_generated_tokens / elapsed_time if elapsed_time > 0 else 0
    tpf = num_generated_tokens / nfe if nfe > 0 else 0
    
    print(f"\n[Stats] Tokens: {num_generated_tokens} | Time: {elapsed_time:.2f}s | "
          f"NFE: {nfe} | TPS (token/s): {tps:.2f} | TPF (token/forward): {tpf:.2f}")
