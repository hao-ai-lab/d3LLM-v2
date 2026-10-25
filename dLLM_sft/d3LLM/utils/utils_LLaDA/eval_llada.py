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

'''
This file is inspired by the code from https://github.com/ML-GSAI/SMDM
'''
import accelerate
import torch
import re
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoModel, AutoConfig
import sys
from pathlib import Path

# Add d3llm_LLaDA to path for multi-block generation methods
d3llm_path = Path(__file__).resolve().parent.parent.parent / 'd3llm' /'d3llm_LLaDA'
if str(d3llm_path) not in sys.path:
    sys.path.insert(0, str(d3llm_path))
# Import dparallel functions (will be used conditionally)
from d3llm_llada_generate_util import (
    generate_multi_block,
    generate_multi_block_kv_cache
)
# Note: LLaDAModelLM will be imported conditionally in __init__ based on generation_method
import json
import time
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
    find_executable_batch_size,
)
from datetime import timedelta
import subprocess

import sys
# Add specific lm_eval path to sys.path
lm_eval_path = Path(__file__).parent.parent.parent / 'utils' / 'utils_Dream' / 'eval_instruct'
sys.path.insert(0, str(lm_eval_path))
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model

def set_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@register_model("llada_dist")
class LLaDAEvalHarness(LM):
    def __init__(
        self,
        model_path='',
        mask_id=126336,
        max_length=4096,
        batch_size=16,
        mc_num=128,
        is_check_greedy=True,
        steps=1024,
        gen_length=1024,
        block_length=1024,
        remasking='low_confidence',
        device="cuda",
        use_cache=False,
        threshold=None,
        save_dir=None,
        show_speed=True,
        dual_cache=False,
        task="null",
        generation_method="generate",
        block_add_threshold=0.1,
        decoded_token_threshold=0.95,
        cache_delay_iter=10000,
        refresh_interval=10000,
        temperature=0.0,
        early_stop=False,
        **kwargs,
    ):
        '''
        Args:
            model_path: LLaDA-8B-Base model path.
            mask_id: The token id of [MASK] is 126336.
            max_length: the max sequence length.
            batch_size: mini batch size.
            mc_num: Monte Carlo estimation iterations
            is_check_greedy: For certain metrics like LAMBADA, the evaluation requires the model to verify whether the answer 
                             is generated through greedy sampling conditioned on the prompt (note that this differs from conditional
                             generation). We implement this verification through the suffix_greedy_prediction() function, which 
                             returns a True/False judgment used for accuracy calculation. 
                             When is_check_greedy is set to True, the lm-evaluation-harness library automatically invokes this function. 
                             However, since none of the metrics in the LLaDA paper (https://arxiv.org/abs/2502.09992) require this functionality, 
                             we recommend setting is_check_greedy to False. This configuration causes suffix_greedy_prediction() to return False 
                             by default, significantly accelerating the evaluation process.
            cfg_scale: Unsupervised classifier-free guidance scale.
        '''
        super().__init__()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])

        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        # Validate generation method first
        valid_methods = [
            "generate",
            "generate_with_prefix_cache",
            "generate_with_dual_cache",
            "generate_multi_block",
            "generate_multi_block_kv_cache",
            "Fast_dllm_v1"
        ]
        if generation_method not in valid_methods:
            raise ValueError(f"Invalid generation_method: {generation_method}. Must be one of {valid_methods}")
        self.generation_method = generation_method
        
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        
        # Import model based on generation method
        if self.generation_method == "Fast_dllm_v1":
            # Use Fast-dLLM-v1 model for compatibility
            
            # Debug: Print current state before modifications
            print(f"\n{'='*80}")
            print(f"DEBUG: Initializing Fast-dLLM-v1 model (generation_method={self.generation_method})")
            print(f"DEBUG: __file__ = {__file__}")
            
            # Add Fast-dLLM-v1/llada to path at the BEGINNING to ensure priority
            # Go up to project root: utils/utils_LLaDA -> utils -> d3LLM
            fast_llada_path = str(Path(__file__).resolve().parent.parent.parent / 'baseline' / 'Fast_dLLM_v1' / 'llada')
            print(f"DEBUG: fast_llada_path = {fast_llada_path}")
            print(f"DEBUG: Path exists? {Path(fast_llada_path).exists()}")
            
            # Remove if already exists to re-insert at front
            if fast_llada_path in sys.path:
                print(f"DEBUG: Removing existing Fast-dLLM-v1 path from sys.path")
                while fast_llada_path in sys.path:
                    sys.path.remove(fast_llada_path)
            
            sys.path.insert(0, fast_llada_path)
            print(f"DEBUG: Inserted Fast-dLLM-v1 path at sys.path[0]")
            print(f"DEBUG: sys.path[:3] = {sys.path[:3]}")
            
            # Verify the model directory and file exist
            model_dir = Path(fast_llada_path) / 'model'
            modeling_file = model_dir / 'modeling_llada.py'
            print(f"DEBUG: model directory exists? {model_dir.exists()}")
            print(f"DEBUG: modeling_llada.py exists? {modeling_file.exists()}")
            if modeling_file.exists():
                print(f"DEBUG: modeling_llada.py path: {modeling_file}")
            
            # Clear any cached imports to force re-import from Fast-dLLM-v1
            modules_to_clear = [k for k in list(sys.modules.keys()) 
                               if 'modeling_llada' in k or 'configuration_llada' in k]
            if modules_to_clear:
                print(f"DEBUG: Clearing cached modules: {modules_to_clear}")
                for mod in modules_to_clear:
                    del sys.modules[mod]
            
            print(f"DEBUG: Attempting to import from model.modeling_llada (Fast-dLLM-v1)...")
            from model.modeling_llada import LLaDAModelLM as FastLLaDAModelLM
            print(f"DEBUG: Successfully imported LLaDAModelLM from {FastLLaDAModelLM.__module__}")
            print(f"{'='*80}\n")
            
            self.model = FastLLaDAModelLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                config=config, 
                **model_kwargs
            )
        else:
            # Use dparallel model for other methods
            
            # Debug: Print current state before modifications
            print(f"\n{'='*80}")
            print(f"DEBUG: Initializing dparallel model (generation_method={self.generation_method})")
            print(f"DEBUG: __file__ = {__file__}")
            print(f"DEBUG: Path(__file__).resolve().parent = {Path(__file__).resolve().parent}")
            
            # Step 1: Remove any Fast-dLLM-v1 paths to avoid conflicts
            fast_paths = [p for p in sys.path if 'Fast_dLLM_v1' in p]
            if fast_paths:
                print(f"DEBUG: Removing Fast-dLLM-v1 paths from sys.path: {fast_paths}")
                for p in fast_paths:
                    while p in sys.path:
                        sys.path.remove(p)
            
            # Step 2: Ensure utils_LLaDA is at the front of sys.path
            dparallel_llada_path = str(Path(__file__).resolve().parent)
            print(f"DEBUG: dparallel_llada_path = {dparallel_llada_path}")
            
            # Remove if already exists to re-insert at front
            if dparallel_llada_path in sys.path:
                print(f"DEBUG: Removing existing dparallel path from sys.path")
                while dparallel_llada_path in sys.path:
                    sys.path.remove(dparallel_llada_path)
            
            # Insert at the very front
            sys.path.insert(0, dparallel_llada_path)
            print(f"DEBUG: Inserted dparallel path at sys.path[0]")
            print(f"DEBUG: sys.path[:3] = {sys.path[:3]}")
            
            # Step 3: Clear ALL modeling_llada and configuration_llada related modules
            modules_to_clear = [k for k in list(sys.modules.keys()) 
                               if 'modeling_llada' in k or 'configuration_llada' in k]
            if modules_to_clear:
                print(f"DEBUG: Clearing cached modules: {modules_to_clear}")
                for mod in modules_to_clear:
                    del sys.modules[mod]
            
            # Step 4: Verify the model directory exists
            model_dir = Path(dparallel_llada_path) / 'model'
            modeling_file = model_dir / 'modeling_llada.py'
            print(f"DEBUG: model directory exists? {model_dir.exists()}")
            print(f"DEBUG: modeling_llada.py exists? {modeling_file.exists()}")
            
            # Step 5: Import the model
            print(f"DEBUG: Attempting to import from model.modeling_llada...")
            from model.modeling_llada import LLaDAModelLM
            print(f"DEBUG: Successfully imported LLaDAModelLM from {LLaDAModelLM.__module__}")
            print(f"{'='*80}\n")
            
            self.model = LLaDAModelLM.from_pretrained(
                model_path, 
                trust_remote_code=True, 
                torch_dtype=torch.bfloat16, 
                config=config, 
                **model_kwargs
            )
        self.model.eval()
        
        # Flag to track if model has been compiled
        self._model_compiled = False

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.model.to(self.accelerator.device)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        
        # Set batch_size to 4 per GPU
        if isinstance(batch_size, str) and batch_size.lower() == 'auto':
            num_gpus = torch.cuda.device_count()
            if num_gpus == 0:
                self.batch_size = 4  # default for CPU
            else:
                # Fixed batch size: 4 per GPU
                self.batch_size = 4 * num_gpus
            print(f"Auto-detected {num_gpus} GPU(s), setting batch_size to {self.batch_size} (4 per GPU)")
        else:
            self.batch_size = int(batch_size)
            print(f"Using fixed batch_size: {self.batch_size}")
        
        assert mc_num % self.batch_size == 0, f"mc_num ({mc_num}) must be divisible by batch_size ({self.batch_size})"
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        # self.is_instruct = True if ('instruct' in model_path.lower() or '1.5' in model_path.lower()) else False
        # self.is_instruct = True if ('instruct' in model_path.lower() or '1.5' in model_path.lower() or 'd3llm' in model_path.lower()) else False
        self.is_instruct = True if 'humaneval' in task.lower() or 'mbpp' in task.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        self.task = task
        self.cfg = 0
        self.temperature = temperature
        
        # Multi-block generation parameters
        self.block_add_threshold = block_add_threshold
        self.decoded_token_threshold = decoded_token_threshold
        self.cache_delay_iter = cache_delay_iter
        self.refresh_interval = refresh_interval
        self.early_stop = early_stop
        # generation_method already validated and set before model creation
        print(f"Using generation method: {self.generation_method}")
        print(f"  temperature: {self.temperature}")
        if self.generation_method in ["generate_multi_block", "generate_multi_block_kv_cache"]:
            print(f"  block_add_threshold: {self.block_add_threshold}")
            print(f"  decoded_token_threshold: {self.decoded_token_threshold}")
            print(f"  early_stop: {self.early_stop}")
        if self.generation_method == "generate_multi_block_kv_cache":
            print(f"  cache_delay_iter: {self.cache_delay_iter}")
    @property
    def rank(self):
        return self._rank
    
    @property
    def world_size(self):
        return self._world_size

    def _forward_process(self, batch, prompt_index):
        b, l = batch.shape

        target_len = (l - prompt_index.sum()).item()
        k = torch.randint(1, target_len + 1, (), device=batch.device)

        x = torch.round(torch.linspace(float(k), k + (b - 1) * (target_len / b), steps=b, device=batch.device)).long()
        x = ((x - 1) % target_len) + 1
        assert x.min() >= 1 and x.max() <= target_len

        indices = torch.arange(target_len, device=batch.device).repeat(b, 1)
        is_mask = indices < x.unsqueeze(1)

        for i in range(b):
            is_mask[i] = is_mask[i][torch.randperm(target_len)]

        is_mask = torch.cat((torch.zeros(b, prompt_index.sum(), dtype=torch.bool, device=batch.device), is_mask), dim=1)

        noisy_batch = torch.where(is_mask, self.mask_id, batch)

        return noisy_batch, (x / target_len).unsqueeze(1).repeat(1, l)

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        if self.cfg > 0.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.mask_id
            batch = torch.cat([batch, un_batch])

        logits = self.model(batch).logits

        if self.cfg > 0.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + (self.cfg + 1) * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def get_loglikelihood(self, prefix, target):
        seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)

        loss_acc = []
        for _ in range(self.mc_num // self.batch_size):
            perturbed_seq, p_mask = self._forward_process(seq, prompt_index)

            mask_indices = perturbed_seq == self.mask_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())

        return - sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def suffix_greedy_prediction(self, prefix, target):
        if not self.is_check_greedy:
            return False

        seq = torch.full((1, len(prefix) + len(target)), self.mask_id, device=self.device)
        prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        prefix, target = prefix.to(self.device), target.to(self.device)
        seq[0, :len(prefix)] = prefix

        for i in range(len(target)):
            mask_index = (seq == self.mask_id)
            logits = self.get_logits(seq, prompt_index)[mask_index]
            x0 = torch.argmax(logits, dim=-1)

            p = torch.softmax(logits.to(torch.float32), dim=-1)
            confidence = torch.gather(p, dim=-1, index=torch.unsqueeze(x0, -1)).squeeze(dim=-1)
            _, index = torch.sort(confidence, descending=True)
            x0[index[1:]] = self.mask_id
            seq[mask_index] = x0.clone()
        correct = target == seq[0, len(prefix):]
        correct = torch.all(correct)
        return correct

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer(context + continuation)["input_ids"]
        context_enc = self.tokenizer(context)["input_ids"]

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests):
        def _tokenize(e):
            prefix, target = self._encode_pair(e["prefix"], e["target"])
            return {
                "prefix_text": e["prefix"],
                "target_text": e["target"],
                "prefix": prefix,
                "target": target,
            }

        ds = []
        ds = [{"prefix": req.args[0], "target": req.args[1]} for req in requests]
        ds = Dataset.from_list(ds)
        ds = ds.map(_tokenize)
        ds = ds.with_format("torch")
        prompt_len = [len(x["prefix"]) + len(x["target"]) for x in ds]

        assert max(prompt_len) <= 4096

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                ll = self.get_loglikelihood(prefix, target)

                is_target_greedy_dec = self.suffix_greedy_prediction(prefix, target)

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))

        torch.cuda.empty_cache()
        return out

    def loglikelihood_rolling(self, requests):
        raise NotImplementedError
    
    def _get_gpu_stats(self):
        """Get GPU memory and utilization statistics"""
        try:
            # Try gpustat first (more readable)
            import json
            result = subprocess.run(
                ['gpustat', '--json'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                gpu_data = json.loads(result.stdout)
                total_memory_used = 0
                total_memory_total = 0
                total_utilization = 0
                num_gpus = len(gpu_data['gpus'])
                
                for gpu in gpu_data['gpus']:
                    total_memory_used += gpu['memory.used']
                    total_memory_total += gpu['memory.total']
                    total_utilization += gpu['utilization.gpu']
                
                if num_gpus > 0:
                    avg_memory_used = total_memory_used / num_gpus
                    avg_memory_total = total_memory_total / num_gpus
                    avg_utilization = total_utilization / num_gpus
                    memory_percent = (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else 0
                    
                    return {
                        'gpu_memory_used_mb': avg_memory_used,
                        'gpu_memory_total_mb': avg_memory_total,
                        'gpu_memory_percent': memory_percent,
                        'gpu_utilization_percent': avg_utilization,
                        'num_gpus': num_gpus
                    }
        except Exception:
            # Catch all exceptions from gpustat
            pass
        
        # Fallback to nvidia-smi
        try:
            result = subprocess.run(
                ['nvidia-smi', '--query-gpu=memory.used,memory.total,utilization.gpu', '--format=csv,noheader,nounits'],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                total_memory_used = 0
                total_memory_total = 0
                total_utilization = 0
                num_gpus = len(lines)
                
                for line in lines:
                    parts = line.split(',')
                    if len(parts) == 3:
                        total_memory_used += float(parts[0].strip())
                        total_memory_total += float(parts[1].strip())
                        total_utilization += float(parts[2].strip())
                
                if num_gpus > 0:
                    avg_memory_used = total_memory_used / num_gpus
                    avg_memory_total = total_memory_total / num_gpus
                    avg_utilization = total_utilization / num_gpus
                    memory_percent = (total_memory_used / total_memory_total * 100) if total_memory_total > 0 else 0
                    
                    return {
                        'gpu_memory_used_mb': avg_memory_used,
                        'gpu_memory_total_mb': avg_memory_total,
                        'gpu_memory_percent': memory_percent,
                        'gpu_utilization_percent': avg_utilization,
                        'num_gpus': num_gpus
                    }
        except Exception:
            # Catch all exceptions from nvidia-smi
            pass
        
        return None
    
    def generate_until(self, requests):
        # Apply torch.compile FIRST, before any generation calls (matching diffllm.py order)
        # This should only happen once, and after the model is on the correct device
        # if not self._model_compiled:
        #     print(f"Compiling model with torch.compile (model is on {self.model.device})...")
        #     print("First compilation may take some time. Subsequent forward passes will be faster.")
        #     self.model = torch.compile(self.model, mode="reduce-overhead")
        #     self._model_compiled = True
        #     print("Model compilation complete.")
        
        output = []
        num_tokens = 0
        num_tokens_before_until = 0
        num_nfe = 0
        processed_count = 0
        warmup_steps = 10  # Skip first 10 iterations for statistics
        if self.save_dir is not None:
            os.makedirs(self.save_dir, exist_ok=True)
            rank = self.rank
            save_path = os.path.join(self.save_dir, f'rank_{rank}.jsonl')
            print(f"save_path: {save_path}")
            if os.path.exists(save_path):
                print(f"load from {save_path}")
                with open(save_path, 'r', encoding='utf-8') as f:
                    output = [json.loads(line) for line in f]
                    processed_count = len(output)
                print(f"processed_count: {processed_count}")
        run_time = 0
        # Only rank 0 shows progress bar
        iterator = tqdm(requests, desc="Generating...") if self.rank == 0 else requests
        for i, req in enumerate(iterator):
            start_time = time.time()

            if i < processed_count:
                continue
            
            question = req.args[0]
            if self.is_instruct:
                tail = r" Please reason step by step, and put your final answer within \boxed{}."
                if self.task == "gsm8k":
                    # add chain of thought prompt
                    m = [{"role": "user", "content": question + tail}]
                elif self.task == "humaneval" or self.task == "humaneval_plus":
                    # add instructions according to humaneval_instruct task
                    start = "Write a solution to the following problem and make sure that it passes the tests:\n```python\n{{ prompt }}\n```\n "
                    question = start.replace("{{ prompt }}", question)
                    m = [{"role": "user", "content": question}]
                else:
                    m = [{"role": "user", "content":  question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input)['input_ids']
            else:
                user_input = question
                input_ids = self.tokenizer(user_input)['input_ids']

        
            stop_tokens = req.args[1]['until']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            
            # Call the selected generation method
            if self.generation_method == "Fast_dllm_v1":
                # Use Fast-dLLM-v1 generation methods
                # Ensure Fast-dLLM-v1/llada is at the front of sys.path
                fast_llada_path = str(Path(__file__).parent.parent / 'Fast_dLLM_v1' / 'llada')
                if fast_llada_path in sys.path:
                    sys.path.remove(fast_llada_path)
                sys.path.insert(0, fast_llada_path)
                
                # Clear cached generate module to force re-import from Fast-dLLM-v1
                if 'generate' in sys.modules:
                    del sys.modules['generate']
                
                if self.use_cache:
                    if self.dual_cache:
                        from generate import generate_with_dual_cache
                        generated_answer, nfe = generate_with_dual_cache(
                            self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                            block_length=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                            mask_id=self.mask_id, threshold=self.threshold
                        )
                    else:
                        from generate import generate_with_prefix_cache
                        generated_answer, nfe = generate_with_prefix_cache(
                            self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                            block_length=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                            mask_id=self.mask_id, threshold=self.threshold
                        )
                else:
                    from generate import generate
                    generated_answer, nfe = generate(
                        self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                        block_length=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                        mask_id=self.mask_id, threshold=self.threshold
                    )
            elif self.generation_method == "generate":
                # Import dparallel generate function
                from generate import generate
                generated_answer, nfe = generate(
                    self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                    block_length=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold
                )
            elif self.generation_method == "generate_with_prefix_cache":
                # Import dparallel generate function
                from generate import generate_with_prefix_cache
                generated_answer, nfe = generate_with_prefix_cache(
                    self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                    block_length=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold
                )
            elif self.generation_method == "generate_with_dual_cache":
                # Import dparallel generate function
                from generate import generate_with_dual_cache
                generated_answer, nfe = generate_with_dual_cache(
                    self.model, input_ids, steps=self.steps, gen_length=self.gen_length, 
                    block_length=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold
                )
            elif self.generation_method == "generate_multi_block":
                generated_answer, nfe = generate_multi_block(
                    self.model, input_ids, steps=self.steps, max_new_tokens=self.gen_length, 
                    block_size=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold,
                    block_add_threshold=self.block_add_threshold, 
                    decoded_token_threshold=self.decoded_token_threshold,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stop=self.early_stop,
                )
            elif self.generation_method == "generate_multi_block_kv_cache":
                generated_answer, nfe = generate_multi_block_kv_cache(
                    self.model, input_ids, steps=self.steps, max_new_tokens=self.gen_length, 
                    block_size=self.block_length, temperature=self.temperature, remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold,
                    block_add_threshold=self.block_add_threshold, 
                    decoded_token_threshold=self.decoded_token_threshold,
                    cache_delay_iter=self.cache_delay_iter,
                    refresh_interval=self.refresh_interval,
                    eos_token_id=self.tokenizer.eos_token_id,
                    early_stop=self.early_stop,
                )
            else:
                raise ValueError(f"Unknown generation method: {self.generation_method}")

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                original_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=True)
                generated_answer_ids = torch.tensor(self.tokenizer(original_answer)["input_ids"])
                if self.show_speed and i >= warmup_steps:
                    tokens_before_until = self.tokenizer(original_answer, add_special_tokens=False)['input_ids']
                    num_tokens_before_until += len([t for t in tokens_before_until if t != 126081])
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                generated_answer = original_answer
            else:
                original_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=False)
                if self.show_speed and i >= warmup_steps:
                    tokens_before_until = self.tokenizer(original_answer, add_special_tokens=False)['input_ids']
                    num_tokens_before_until += len([t for t in tokens_before_until if t != 126081])
                
                for stop_seq in stop_tokens:
                    if stop_seq in original_answer:
                        original_answer = original_answer.split(stop_seq)[0]

                # remove special tokens
                generated_answer_ids = torch.tensor(self.tokenizer(original_answer)["input_ids"])
                if self.show_speed and i >= warmup_steps:
                    num_tokens += (generated_answer_ids != 126081).sum()
                    num_nfe += nfe
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            output.append(generated_answer)

            if self.save_dir is not None:
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            end_time = time.time()
            if i >= warmup_steps:
                run_time += end_time - start_time
            
            # Only rank 0 prints per-sample details
            if self.rank == 0:
                print('=' * 20)
                print('question: ', question[:min(len(question), 100)])
                print('answer: ', generated_answer[:min(len(generated_answer), 100)])
                print('=' * 20, end='\n\n')

                # Print NFE:
                print(f"NFE (Number of Function Evaluations): {nfe}")
                
                # Print GPU stats
                gpu_stats = self._get_gpu_stats()
                if gpu_stats:
                    print(f"\n{'='*60}")
                    print(f"GPU Stats at Step {i + 1}:")
                    print(f"  Number of GPUs: {gpu_stats['num_gpus']}")
                    print(f"  GPU Memory Used: {gpu_stats['gpu_memory_used_mb']:.2f} MB")
                    print(f"  GPU Memory Total: {gpu_stats['gpu_memory_total_mb']:.2f} MB")
                    print(f"  GPU Memory Usage: {gpu_stats['gpu_memory_percent']:.2f}%")
                    print(f"  GPU Utilization: {gpu_stats['gpu_utilization_percent']:.2f}%")
                    print(f"{'='*60}\n")
                else:
                    print("No GPU stats available for step", i + 1)
                
                if i >= warmup_steps:
                    print(f"Total time taken (after warmup): {run_time} seconds")
                    print(f"Total NFE is {num_nfe}")
                    
                    print(f"\nAFTER 'until' truncation:")
                    print(f"  Total tokens: {num_tokens}")
                    if run_time > 0:
                        print(f"  Throughput: {num_tokens / run_time:.2f} tokens/s")
                        if num_nfe > 0:
                            print(f"  Tokens per forward: {num_tokens / num_nfe:.2f}")
            
        return output


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    