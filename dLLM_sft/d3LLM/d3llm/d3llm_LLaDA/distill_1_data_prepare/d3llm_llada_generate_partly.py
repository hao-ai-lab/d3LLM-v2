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
# Modified from LLaDA repos: https://github.com/ML-GSAI/LLaDA


import sys
import os
import re

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append('/')

from transformers import AutoTokenizer
import torch
import numpy as np
import torch.nn.functional as F
from utils.utils_LLaDA.model.modeling_llada import LLaDAModelLM


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


def add_gumbel_noise(logits, temperature):
    """
    The Gumbel max is a method for sampling categorical distributions.
    According to arXiv:2409.02908, for MDM, low-precision Gumbel Max improves perplexity score but reduces generation quality.
    Thus, we use float64.
    """
    if temperature == 0:
        return logits
    logits = logits.to(torch.float64)
    noise = torch.rand_like(logits, dtype=torch.float64)
    gumbel_noise = (-torch.log(noise)) ** temperature
    return logits.exp() / gumbel_noise


def get_num_transfer_tokens(mask_index, steps):
    """
    In the reverse process, the interval [0, 1] is uniformly discretized into steps intervals.
    Furthermore, because LLaDA employs a linear noise schedule (as defined in Eq. (8)),
    the expected number of tokens transitioned at each step should be consistent.

    This function is designed to precompute the number of tokens that need to be transitioned at each step.
    """
    mask_num = mask_index.sum(dim=1, keepdim=True)

    base = mask_num // steps
    remainder = mask_num % steps

    num_transfer_tokens = (
        torch.zeros(
            mask_num.size(0), steps, device=mask_index.device, dtype=torch.int64
        )
        + base
    )

    for i in range(mask_num.size(0)):
        num_transfer_tokens[i, : remainder[i]] += 1

    return num_transfer_tokens


def get_transfer_index(
    logits, temperature, remasking, mask_index, x, num_transfer_tokens, threshold=None
):
    logits_with_noise = add_gumbel_noise(logits, temperature=temperature)
    x0 = torch.argmax(logits_with_noise, dim=-1)  # b, l

    if remasking == "low_confidence":
        p = F.softmax(logits.to(torch.float64), dim=-1)
        x0_p = torch.squeeze(
            torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)), -1
        )  # b, l
    elif remasking == "random":
        x0_p = torch.rand((x0.shape[0], x0.shape[1]), device=x0.device)
    else:
        raise NotImplementedError(remasking)

    x0 = torch.where(mask_index, x0, x)
    confidence = torch.where(mask_index, x0_p, -np.inf)

    transfer_index = torch.zeros_like(x0, dtype=torch.bool, device=x0.device)
    if threshold is not None:
        num_transfer_tokens = mask_index.sum(dim=1, keepdim=True)
    for j in range(confidence.shape[0]):
        _, select_index = torch.topk(confidence[j], k=num_transfer_tokens[j])
        transfer_index[j, select_index] = True
        if threshold is not None:
            for k in range(1, num_transfer_tokens[j]):
                if confidence[j, select_index[k]] < threshold:
                    transfer_index[j, select_index[k]] = False
    return x0, transfer_index


@torch.no_grad()
def generate(
    model,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
):
    """
    Args:
        model: Mask predictor.
        prompt: A tensor of shape (1, L).
        steps: Sampling steps, less than or equal to gen_length.
        gen_length: Generated answer length.
        block_length: Block length, less than or equal to gen_length. If less than gen_length, it means using semi_autoregressive remasking.
        temperature: Categorical distribution sampling temperature.
        remasking: Remasking strategy. 'low_confidence' or 'random'.
        mask_id: The toke id of [MASK] is 126336.
    """
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length

    assert steps % num_blocks == 0
    steps = steps // num_blocks

    nfe = 0
    for num_block in range(num_blocks):
        block_mask_index = (
            x[
                :,
                prompt.shape[1]
                + num_block * block_length : prompt.shape[1]
                + (num_block + 1) * block_length,
            ]
            == mask_id
        )
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps)
        i = 0
        while True:
            nfe += 1
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, prompt.shape[1] + (num_block + 1) * block_length :] = 0

            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, i],
                threshold,
            )

            x[transfer_index] = x0[transfer_index]
            i += 1
            if (
                x[
                    :,
                    prompt.shape[1]
                    + num_block * block_length : prompt.shape[1]
                    + (num_block + 1) * block_length,
                ]
                == mask_id
            ).sum() == 0:
                break
    return x, nfe


@torch.no_grad()
def generate_teacher_model_trajectory(
    model,
    tokenizer,
    prompt,
    steps=128,
    gen_length=128,
    block_length=128,
    temperature=0.0,
    remasking="low_confidence",
    mask_id=126336,
    threshold=None,
):
    """Generate trajectory for teacher model with block-wise diffusion decoding"""
    x = torch.full((1, prompt.shape[1] + gen_length), mask_id, dtype=torch.long).to(
        model.device
    )
    x[:, : prompt.shape[1]] = prompt.clone()

    assert gen_length % block_length == 0
    num_blocks = gen_length // block_length
    assert steps % num_blocks == 0
    steps_per_block = steps // num_blocks

    trajectory = []  # Store full sequence at each step

    for num_block in range(num_blocks):
        block_start = prompt.shape[1] + num_block * block_length
        block_end = prompt.shape[1] + (num_block + 1) * block_length
        block_mask_index = x[:, block_start:block_end] == mask_id
        num_transfer_tokens = get_num_transfer_tokens(block_mask_index, steps_per_block)

        for i in range(steps_per_block):
            mask_index = x == mask_id
            logits = model(x).logits
            mask_index[:, block_end:] = 0  # Only unmask current block

            x0, transfer_index = get_transfer_index(
                logits,
                temperature,
                remasking,
                mask_index,
                x,
                num_transfer_tokens[:, i],
                threshold,
            )
            x[transfer_index] = x0[transfer_index]
            trajectory.append(x.clone())

            if (x[:, block_start:block_end] == mask_id).sum() == 0:
                break

    return x, trajectory


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

    teacher_model = (
        LLaDAModelLM.from_pretrained(
            "GSAI-ML/LLaDA-8B-Instruct",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16,
        )
        .to(device)
        .eval()
    )
    tokenizer = AutoTokenizer.from_pretrained(
        "GSAI-ML/LLaDA-8B-Instruct", trust_remote_code=True
    )

    dataset = load_dataset("Zigeng/dParallel_LLaDA_Distill_Data", split="train")

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

        # Tokenize prompt
        prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
        prompt_tensor = torch.tensor(prompt_ids).to(device).unsqueeze(0)

        # Retry mechanism: try up to 5 times if answer is incorrect
        # Temperature increases: 0.0, 0.1, 0.2, 0.3, 0.4
        max_attempts = 5
        for attempt in range(max_attempts):
            current_temperature = attempt * 0.1
            
            # Generate trajectory
            final_output, trajectory = generate_teacher_model_trajectory(
                teacher_model,
                tokenizer,
                prompt_tensor,
                steps=steps,
                gen_length=gen_length,
                block_length=block_length,
                temperature=current_temperature,
                remasking="low_confidence",
            )

            # Decode generated text and check correctness
            generated_text = tokenizer.decode(final_output[0], skip_special_tokens=True)
            llm_answer = extract_boxed_answer(generated_text)
            is_correct = check_answer_correctness(generated_text, ground_truth)
            
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
