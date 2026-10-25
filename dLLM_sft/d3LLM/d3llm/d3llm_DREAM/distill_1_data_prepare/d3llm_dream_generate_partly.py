# Copyright 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0
# Modified from Dream repos: https://github.com/HKUNLP/Dream


import sys
import os
import re
import types
from typing import Optional

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/')
sys.path.append('./')
sys.path.append('../')

from transformers import AutoTokenizer, AutoModel
import torch
from torch.nn import functional as F


def extract_boxed_answer(text):
    """Extract answer from \\boxed{} format"""
    if "\\boxed" not in text:
        return None

    idx = text.rfind("\\boxed")
    if idx < 0:
        return None

    i = idx
    num_left_braces = 0
    while i < len(text):
        if text[i] == "{":
            num_left_braces += 1
        elif text[i] == "}":
            num_left_braces -= 1
            if num_left_braces == 0:
                answer = text[idx + 7 : i]  # Skip "\\boxed{"
                return answer.strip()
        i += 1
    return None


def normalize_answer(ans):
    """Normalize answer for comparison"""
    if ans is None:
        return None
    ans = str(ans).strip().lower()
    ans = re.sub(r"[,\s]", "", ans)  # Remove commas and spaces
    return ans


def check_answer_correctness(generated_text, ground_truth):
    """Check if generated answer matches ground truth"""
    # If ground_truth is None or empty, consider it as correct (e.g., for code generation tasks)
    if ground_truth is None or ground_truth == "":
        return True
    
    pred_ans = extract_boxed_answer(generated_text)
    gt_ans = (
        extract_boxed_answer(ground_truth)
        if "\\boxed" in ground_truth
        else ground_truth.strip()
    )

    pred_norm = normalize_answer(pred_ans)
    gt_norm = normalize_answer(gt_ans)

    if pred_norm is None or gt_norm is None:
        return False

    return pred_norm == gt_norm


def sample_tokens_with_entropy(logits, temperature=1.0):
    """
    Sample tokens and return corresponding entropy values
    
    Args:
        logits: Model output logits [batch_size, vocab_size]
        temperature: Temperature parameter
    
    Returns:
        entropy: Entropy value at each position [batch_size]
        samples: Sampled token ids [batch_size]
    """
    # Calculate entropy from original logits (for threshold judgment)
    original_probs = torch.softmax(logits, dim=-1)
    log_probs = torch.log(original_probs + 1e-8)
    entropy = -torch.sum(original_probs * log_probs, dim=-1)
    
    # Then perform sampling
    if temperature == 0:
        # Greedy decoding: directly select the token with largest logits
        samples = torch.argmax(logits, dim=-1)
    else:
        # Apply temperature
        scaled_logits = logits / temperature
        # Convert to probabilities and sample
        probs = torch.softmax(scaled_logits, dim=-1)
        samples = torch.multinomial(probs, num_samples=1).squeeze(-1)
    
    return entropy, samples


@torch.no_grad()
def generate_teacher_model_trajectory(
    model,
    tokenizer,
    input_ids,
    attention_mask=None,
    steps=256,
    gen_length=256,
    block_length=32,
    temperature=0.0,
    threshold=0.5,
    mask_token_id=None,
):
    """Generate trajectory for DREAM teacher model with block-wise diffusion decoding"""
    
    # Bind generation methods to model
    from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin
    if not hasattr(model, '_sample_original'):
        model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, model)
        model._sample_original = types.MethodType(DreamGenerationMixin._sample, model)
        model._prepare_inputs = types.MethodType(DreamGenerationMixin._prepare_inputs, model)
        model._prepare_generation_config = types.MethodType(DreamGenerationMixin._prepare_generation_config, model)
        model._prepare_special_tokens = types.MethodType(DreamGenerationMixin._prepare_special_tokens, model)
        model._prepare_generated_length = types.MethodType(DreamGenerationMixin._prepare_generated_length, model)
        model._validate_generated_length = types.MethodType(DreamGenerationMixin._validate_generated_length, model)
        # _expand_inputs_for_generation is a staticmethod, so we assign it directly
        model._expand_inputs_for_generation = DreamGenerationMixin._expand_inputs_for_generation
    
    # Create custom _sample method that records trajectory
    trajectory = []
    
    def _sample_with_trajectory(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor],
        generation_config,
        threshold: Optional[float] = 0.5,
        block_length: Optional[int] = 32,
    ):
        # init values
        output_history = generation_config.output_history
        return_dict_in_generate = generation_config.return_dict_in_generate
        max_length = generation_config.max_length
        mask_token_id_val = generation_config.mask_token_id
        steps_val = generation_config.steps
        temperature_val = generation_config.temperature
        alg = generation_config.alg

        histories = [] if (return_dict_in_generate and output_history) else None

        # pad input_ids to max_length
        x = F.pad(input_ids, (0, max_length - input_ids.shape[1]), value=mask_token_id_val)
        gen_length_val = max_length - input_ids.shape[1]
        
        # Handle block configuration
        if block_length is None:
            block_length = gen_length_val  # Default: single block (original behavior)
        
        assert gen_length_val % block_length == 0, f"gen_length ({gen_length_val}) must be divisible by block_length ({block_length})"
        num_blocks = gen_length_val // block_length
        
        assert steps_val % num_blocks == 0, f"steps ({steps_val}) must be divisible by num_blocks ({num_blocks})"
        steps_per_block = steps_val // num_blocks
        timesteps = torch.linspace(1, generation_config.eps, steps_per_block + 1, device=x.device)

        if attention_mask is not None and torch.any(attention_mask == 0.0):
            # we do not mask the [MASK] tokens so value = 1.0
            attention_mask = F.pad(attention_mask, (0, max_length - attention_mask.shape[1]), value=1.0)
            tok_idx = attention_mask.long().cumsum(-1) - 1
            tok_idx.masked_fill_(attention_mask == 0, 1)
            # attention_mask is of shape [B, N]
            # broadcast to [B, 1, N, N]
            attention_mask = torch.logical_and(
                attention_mask.unsqueeze(1).unsqueeze(-2),
                attention_mask.unsqueeze(1).unsqueeze(-1),
            )
        else:
            tok_idx = None
            attention_mask = "full"

        # Process each block
        i = 0
        for num_block in range(num_blocks):
            
            current_block_start = input_ids.shape[1] + num_block * block_length
            current_block_end = current_block_start + block_length
                
            while True:  
                i += 1  
                mask_index = (x == mask_token_id_val)      

                model_output = self(x, attention_mask, tok_idx)

                mask_index[:, current_block_end:] = 0
                
                logits = model_output.logits
                logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

                if alg == 'entropy_threshold':
                    mask_logits = logits[mask_index]
                    
                    # Calculate entropy instead of confidence
                    entropy, x0 = sample_tokens_with_entropy(mask_logits, temperature=temperature_val)
                    
                    x_ = torch.zeros_like(x, device=self.device, dtype=torch.long) + mask_token_id_val
                    full_entropy = torch.full_like(x, torch.inf, device=self.device, dtype=logits.dtype)
                    
                    x_[mask_index] = x0.clone()
                    full_entropy[mask_index] = entropy
                    
                    current_transfer_tokens = (x[:, current_block_start:current_block_end] == mask_token_id_val).sum()
                    
                    # Select tokens with lowest entropy (high certainty)
                    selected_entropy, select_index = torch.topk(full_entropy, current_transfer_tokens, largest=False)
                    transfer_index = torch.zeros_like(x, device=x.device, dtype=torch.bool)
                    
                    select_index = select_index.to(x.device)
                    transfer_index[0, select_index[0]] = True
                    for k in range(1, current_transfer_tokens):
                        # Only decode tokens with entropy below threshold
                        if selected_entropy[0, k] < threshold:
                            transfer_index[0, select_index[0, k]] = True
                        else:
                            transfer_index[0, select_index[0, k]] = False
                    x[transfer_index] = x_[transfer_index].clone()

                # Store trajectory after each step
                trajectory.append(x.clone())

                if (x[:, current_block_start:current_block_end] == mask_token_id_val).sum() == 0:
                    break
        
        from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamModelOutput
        if return_dict_in_generate:
            return DreamModelOutput(sequences=x, history=histories), i
        else:
            return x, i
    
    # Temporarily replace _sample method
    original_sample = model._sample_original
    model._sample = types.MethodType(_sample_with_trajectory, model)
    
    try:
        # Generate with trajectory recording
        output, nfe = model.diffusion_generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=gen_length,
            output_history=False,
            return_dict_in_generate=True,
            steps=steps,
            temperature=temperature,
            top_p=None,
            alg="entropy_threshold",
            alg_temp=0.1,
            top_k=None,
            block_length=block_length,
            threshold=threshold,
        )
        
        final_output = output.sequences
        
    finally:
        # Restore original _sample method
        model._sample = original_sample
    
    return final_output, trajectory, nfe


def main(
    start_idx,
    end_idx,
    steps=256,
    gen_length=256,
    block_length=32,
    output_file="trajectory_data.json",
    max_data_num=-1,
):
    from datasets import load_dataset
    from tqdm import tqdm
    import json

    device = "cuda"

    # Load DREAM teacher model
    model_path = "Dream-org/Dream-v0-Instruct-7B"
    # model_path = "Dream-org/Dream-Coder-v0-Instruct-7B"
    teacher_model = AutoModel.from_pretrained(
        model_path, 
        torch_dtype=torch.bfloat16, 
        trust_remote_code=True
    ).to(device).eval()
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=True
    )

    # Load dataset
    dataset = load_dataset("Zigeng/dParallel_Dream_Distill_Data", split="train")
    # dataset = load_dataset("d3LLM/Ling-Coder-dParallel-merged-512-120k", split="train")

    # Apply max_data_num limit
    if max_data_num > 0:
        end_idx = min(end_idx, start_idx + max_data_num)

    results = []
    total_count = 0
    incorrect_count = 0
    
    for idx in tqdm(
        range(start_idx, min(end_idx, len(dataset))), desc="Generating trajectories"
    ):
        sample = dataset[idx]
        prompt_text = sample["question"]
        ground_truth = sample.get("gt_answer", None)

        # Prepare messages for chat template
        messages = [
            {"role": "user", "content": prompt_text}
        ]
        
        # Apply chat template
        inputs = tokenizer.apply_chat_template(
            messages, return_tensors="pt", return_dict=True, add_generation_prompt=True
        )
        input_ids = inputs.input_ids.to(device=device)
        attention_mask = inputs.attention_mask.to(device=device)
        
        # Store prompt_ids as list
        prompt_ids = input_ids[0].cpu().tolist()

        # Retry mechanism: try up to 5 times if answer is incorrect
        # Temperature increases: 0.0, 0.1, 0.2, 0.3, 0.4
        max_attempts = 5
        for attempt in range(max_attempts):
            current_temperature = attempt * 0.1
            
            # Generate trajectory
            final_output, trajectory, nfe = generate_teacher_model_trajectory(
                teacher_model,
                tokenizer,
                input_ids,
                attention_mask=attention_mask,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=current_temperature,
                threshold=-float('inf'),
            )

            # Decode generated text and check correctness
            generated_text = tokenizer.decode(final_output[0], skip_special_tokens=True)
            llm_answer = extract_boxed_answer(generated_text)
            # is_correct = check_answer_correctness(generated_text, ground_truth)
            is_correct = True   # TODO: default to be True for now
            
            # If correct or this is the last attempt, break
            if is_correct or attempt == max_attempts - 1:
                break
            
            print(f"Attempt {attempt + 1}/{max_attempts} failed for idx {idx} (temperature={current_temperature:.1f}), retrying...", flush=True)

        # Store result: convert tensors to lists for JSON serialization
        results.append(
            {
                "idx": idx,
                "question": prompt_text,
                "prompt_ids": prompt_ids,
                "trajectory": [traj[0].cpu().tolist() for traj in trajectory],
                "final_output": final_output[0].cpu().tolist(),
                "generated_text": generated_text,
                "llm_answer": llm_answer,
                "gt_answer": ground_truth,
                "is_correct": is_correct,
                "nfe": nfe,
            }
        )
        
        # Update statistics and print real-time status
        total_count += 1
        if not is_correct:
            incorrect_count += 1
        
        correct_count = total_count - incorrect_count
        accuracy = (correct_count / total_count * 100) if total_count > 0 else 0
        error_rate = (incorrect_count / total_count * 100) if total_count > 0 else 0
        
        if total_count % 10 == 0:
            print(f"[idx {idx}] Status: {'✓ Correct' if is_correct else '✗ Incorrect'} | "
                f"Total: {total_count} | Correct: {correct_count} ({accuracy:.2f}%) | "
                f"Incorrect: {incorrect_count} ({error_rate:.2f}%)", flush=True)

    # Save results
    with open(output_file, "w") as f:
        json.dump(results, f)

    print(f"Saved {len(results)} trajectories to {output_file}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--start_idx", type=int, required=True)
    parser.add_argument("--end_idx", type=int, required=True)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--gen_length", type=int, default=256)
    parser.add_argument("--block_length", type=int, default=32)
    parser.add_argument("--output_file", type=str, default="trajectory_data.json")
    parser.add_argument(
        "--max_data_num",
        type=int,
        default=-1,
        help="Max number of samples to generate (-1 for no limit)",
    )
    args = parser.parse_args()

    main(
        args.start_idx,
        args.end_idx,
        args.steps,
        args.gen_length,
        args.block_length,
        args.output_file,
        args.max_data_num,
    )
