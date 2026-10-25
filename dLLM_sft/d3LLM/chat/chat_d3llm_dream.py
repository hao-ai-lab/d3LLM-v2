import sys
from pathlib import Path
import types
import time
import torch
import tqdm
from transformers import AutoTokenizer

# Add project root to sys.path to enable imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from utils.utils_Dream.model.modeling_dream import DreamModel
from utils.utils_Dream.model.configuration_dream import DreamConfig

# Add d3llm_DREAM for multi-block generation (already added above)
from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin

# Model path
m = "d3LLM/d3LLM_Dream"
# m = "d3LLM/d3LLM_Dream_Coder"

tokenizer = AutoTokenizer.from_pretrained(m, trust_remote_code=True)
device = torch.device("cuda:0")

# Load Dream model using original DreamModel (not dInfer wrapper)
print("Loading Dream model...")
model_config = DreamConfig.from_pretrained(m, trust_remote_code=True)

# Enable Flash Attention 2 in config
try:
    model_config._attn_implementation = "flash_attention_2"
    print("Flash Attention 2 configuration set")
except Exception as e:
    print(f"Warning: Could not set Flash Attention 2 in config: {e}")

model = DreamModel.from_pretrained(
    m, 
    config=model_config,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16
)
model = model.to(device).eval()

print("Compiling model with torch.compile...")
model = torch.compile(model, mode="reduce-overhead")
print("Model compilation complete.")

model.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, model)
model._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, model)
model._sample_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin._sample_multi_block_kv_cache, model)
model._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, model)

# Multi-block generation parameters
multi_block_params = {
    "attention_mask": None,
    "max_new_tokens": 256,
    "output_history": False,
    "return_dict_in_generate": True,
    "steps": 256,
    "temperature": 0.,
    "alg": "entropy_threshold",
    "threshold": 0.5,
    "block_size": 32,
    "block_add_threshold": 0.1,
    "decoded_token_threshold": 0.95,
    "cache_delay_iter": 10000,
    "early_stop": True,
}

print("\n" + "="*80)
print("d3LLM-Dream Chat Mode (type 'quit' or 'exit' to end)")
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
        attention_mask = inputs.get('attention_mask', torch.ones_like(input_ids)).to(device)
        output, nfe = model.generate_multi_block(
            input_ids,
            **multi_block_params
        )
print("Warmup complete.\n")

print("\033[31mNote that because our distillation data primarily consists of **coding** and **math reasoning** tasks, acceleration may only appear on prompts of these tasks.\033[0m")

messages = []
while True:
    # Get user input
    user_input = input("\nYou: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("Goodbye!")
        break
    if not user_input:
        continue
    
    # Add user message to conversation history
    messages = [
        {"role": "user", "content": user_input}
    ]
    
    # Prepare input
    prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    input_ids = tokenizer(prompt_text, return_tensors="pt")['input_ids'].to(device)
    
    # Generate response
    # print("\nAssistant: ", end="", flush=True)
    start_time = time.time()
    with torch.no_grad():
        output, nfe = model.generate_multi_block(input_ids, **multi_block_params)
    end_time = time.time()
    
    # Decode response
    full_response = tokenizer.decode(output.sequences[0], skip_special_tokens=True)
    # Extract only the assistant's last response
    assistant_response = full_response.split(tokenizer.eos_token)[0].split("assistant\n")[-1].strip()
    
    print("\n\033[34mAssistant:\n \033[0m")
    print("\033[34m" + assistant_response + "\033[0m")
    
    # Calculate statistics - use tokenizer to count actual tokens
    num_generated_tokens = len(tokenizer.encode(assistant_response, add_special_tokens=False))
    elapsed_time = end_time - start_time
    tps = num_generated_tokens / elapsed_time if elapsed_time > 0 else 0  # Token per second
    tpf = num_generated_tokens / nfe if nfe > 0 else 0  # Token per forward
    
    print(f"\n[Stats] Tokens: {num_generated_tokens} | Time: {elapsed_time:.2f}s | "
          f"NFE: {nfe} | TPS (token/s): {tps:.2f} | TPF (token/forward): {tpf:.2f}")
    