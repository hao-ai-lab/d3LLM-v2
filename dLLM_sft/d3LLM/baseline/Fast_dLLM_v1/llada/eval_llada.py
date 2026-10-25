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
from pathlib import Path
import random
import numpy as np
import torch.nn.functional as F
from datasets import Dataset
from lm_eval.__main__ import cli_evaluate
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from tqdm import tqdm
import os
from transformers import AutoTokenizer, AutoConfig
from generate import generate, generate_with_prefix_cache, generate_with_dual_cache
from model.modeling_llada import LLaDAModelLM
import json
import time
import sys
# Add d3llm_LLaDA to path for multi-block generation methods
d3llm_path = Path(__file__).parent.parent.parent / 'd3llm_LLaDA'
sys.path.insert(0, str(d3llm_path))
from d3llm_llada_generate_util import generate_multi_block, generate_multi_block_kv_cache
sys.path.append('../')
sys.path.append('../../../')
from utils.utils_LLaDA.model.modeling_llada import LLaDAModelLM as LLaDAModelLM_dparallel
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
        batch_size=32,
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
        show_speed=False,
        dual_cache=False,
        generation_method="generate",
        block_add_threshold=0.1,
        decoded_token_threshold=0.95,
        cache_delay_iter=10000,
        lazy_cache_refresh_num=0,
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

        accelerator = accelerate.Accelerator()
        if accelerator.num_processes > 1:
            self.accelerator = accelerator
        else:
            self.accelerator = None
        
        # Use dparallel model for multi-block methods, otherwise use Fast-dLLM v1 model
        model_kwargs = {}
        if self.accelerator is not None:
            model_kwargs.update({'device_map': {'': f'{self.accelerator.device}'}})
        config = AutoConfig.from_pretrained(model_path)
        config.flash_attention = True
        
        if generation_method in ["generate_multi_block", "generate_multi_block_kv_cache"]:
            self.model = LLaDAModelLM_dparallel.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        else:
            self.model = LLaDAModelLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config, **model_kwargs)
        self.model.eval()

        self.device = torch.device(device)
        if self.accelerator is not None:
            self.model = self.accelerator.prepare(self.model)
            self.device = torch.device(f'{self.accelerator.device}')
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else: 
            self.model = self.model.to(device)

        self.mask_id = mask_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

        self.mc_num = mc_num
        self.batch_size = int(batch_size)
        assert mc_num % self.batch_size == 0
        self.sampling_eps = 0.
        self.max_length = max_length
        self.is_check_greedy = is_check_greedy

        self.steps = steps
        self.gen_length = gen_length
        self.block_length = block_length
        self.remasking = remasking
        self.use_cache = use_cache
        self.threshold = threshold
        self.is_instruct = True if 'instruct' in model_path.lower() or 'd3llm' in model_path.lower() else False
        self.save_dir = save_dir
        self.show_speed = show_speed
        self.dual_cache = dual_cache
        
        # Multi-block generation parameters
        self.generation_method = generation_method
        self.block_add_threshold = block_add_threshold
        self.decoded_token_threshold = decoded_token_threshold
        self.cache_delay_iter = cache_delay_iter
        self.lazy_cache_refresh_num = lazy_cache_refresh_num
        
        # Print generation method info
        print(f"Using generation method: {self.generation_method}")
        if self.generation_method in ["generate_multi_block", "generate_multi_block_kv_cache"]:
            print(f"  block_add_threshold: {self.block_add_threshold}")
            print(f"  decoded_token_threshold: {self.decoded_token_threshold}")
        if self.generation_method == "generate_multi_block_kv_cache":
            print(f"  cache_delay_iter: {self.cache_delay_iter}")
    @property
    def rank(self):
        return self._rank if self.accelerator is not None else 0
    
    @property
    def world_size(self):
        return self._world_size if self.accelerator is not None else 1

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
    
    
    def generate_until(self, requests):
        output = []
        num_tokens = 0
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

            if i < processed_count:
                continue
                
            question = req.args[0]
            if self.is_instruct:
                m = [{"role": "user", "content": question}]
                user_input = self.tokenizer.apply_chat_template(m, add_generation_prompt=True, tokenize=False)
                input_ids = self.tokenizer(user_input)['input_ids']
            else:
                user_input = question
                input_ids = self.tokenizer(user_input)['input_ids']

            stop_tokens = req.args[1]['until']
            input_ids = torch.tensor(input_ids).to(self.device).unsqueeze(0)
            
            start_time = time.time()
            
            # Call the selected generation method
            if self.generation_method == "generate_multi_block":
                generated_answer, nfe = generate_multi_block(
                    self.model, input_ids, steps=self.steps, max_new_tokens=self.gen_length, 
                    block_size=self.block_length, temperature=0., remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold,
                    block_add_threshold=self.block_add_threshold, 
                    decoded_token_threshold=self.decoded_token_threshold
                )
            elif self.generation_method == "generate_multi_block_kv_cache":
                generated_answer, nfe = generate_multi_block_kv_cache(
                    self.model, input_ids, steps=self.steps, max_new_tokens=self.gen_length, 
                    block_size=self.block_length, temperature=0., remasking=self.remasking, 
                    mask_id=self.mask_id, threshold=self.threshold,
                    block_add_threshold=self.block_add_threshold, 
                    decoded_token_threshold=self.decoded_token_threshold,
                    cache_delay_iter=self.cache_delay_iter,
                    lazy_cache_refresh_num=self.lazy_cache_refresh_num
                )
            elif self.use_cache:
                if self.dual_cache:
                    generated_answer, nfe = generate_with_dual_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold)
                else:
                    generated_answer, nfe = generate_with_prefix_cache(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold)
            else:
                generated_answer, nfe = generate(self.model, input_ids, steps=self.steps, gen_length=self.gen_length, block_length=self.block_length, 
                                        temperature=0, remasking=self.remasking, mask_id=self.mask_id, threshold=self.threshold)

            end_time = time.time()

            if self.is_instruct and 'task_id' in req.doc and str(req.doc['task_id']).lower().startswith('humaneval'):
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=True)
            else:
                generated_answer = self.tokenizer.decode(generated_answer[0][input_ids.shape[1]:], skip_special_tokens=False)
                for stop_seq in stop_tokens:
                    if stop_seq in generated_answer:
                        generated_answer = generated_answer.split(stop_seq)[0]

                # remove special tokens
                generated_answer_ids = torch.tensor(self.tokenizer(generated_answer)["input_ids"])
                generated_answer = self.tokenizer.decode(generated_answer_ids, skip_special_tokens=True)
            
            # Count tokens by re-encoding the generated answer
            if self.show_speed and i >= warmup_steps:
                num_tokens += len(self.tokenizer(generated_answer)["input_ids"])
                num_nfe += nfe
            
            output.append(generated_answer)

            if self.save_dir is not None:
                with open(save_path, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(generated_answer, ensure_ascii=False) + '\n')

            if i >= warmup_steps:
                run_time += end_time - start_time
            
            # Only rank 0 prints per-sample details
            if self.rank == 0:
                print('=' * 20)
                print('question: ', question[:min(len(question), 100)])
                print('answer: ', generated_answer[:min(len(generated_answer), 100)])
                print('=' * 20, end='\n\n')

                print(f"NFE (Number of Function Evaluations): {nfe}")
                
                if i >= warmup_steps:
                    print(f"Total time taken (after warmup): {run_time:.2f} seconds")
                    print(f"Total NFE is {num_nfe}")
                    if run_time > 0 and num_nfe > 0:
                        tokens_per_second = num_tokens / run_time
                        print(f"Throughput: {tokens_per_second:.2f} tokens/s")
                        print(f"Total tokens generated: {num_tokens}")
                        print(f"Token per step: {num_tokens / num_nfe:.2f}")
        
        # Aggregate statistics across all ranks
        if self.show_speed and self.accelerator is not None:
            # Gather statistics from all ranks
            num_tokens_tensor = torch.tensor([num_tokens], device=self.device, dtype=torch.float32)
            num_nfe_tensor = torch.tensor([num_nfe], device=self.device, dtype=torch.float32)
            run_time_tensor = torch.tensor([run_time], device=self.device, dtype=torch.float32)
            
            torch.distributed.all_reduce(num_tokens_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(num_nfe_tensor, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(run_time_tensor, op=torch.distributed.ReduceOp.SUM)
            
            if self.rank == 0:
                total_tokens = int(num_tokens_tensor.item())
                total_nfe = int(num_nfe_tensor.item())
                total_time = run_time_tensor.item()  # Calculate average manually
                
                print(f"\n{'='*60}")
                print(f"Aggregated Statistics Across {self.world_size} Ranks:")
                print(f"  Total tokens generated: {total_tokens}")
                print(f"  Total NFE: {total_nfe}")
                print(f"  Average time taken: {total_time:.2f} seconds")
                if total_time > 0 and total_nfe > 0:
                    print(f"  Overall throughput: {total_tokens / total_time:.2f} tokens/s")
                    print(f"  Overall token per step: {total_tokens / total_nfe:.2f}")
                print(f"{'='*60}\n")
        
        return output


if __name__ == "__main__":
    set_seed(1234)
    cli_evaluate()
    