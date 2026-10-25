import sys
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')
sys.path.append('../../../../../')

import logging
import gc
from datetime import timedelta
from typing import List, Optional, Tuple, Type, TypeVar, Union
from collections import deque

import torch
import torch.nn.functional as F
import transformers
from accelerate import (
    Accelerator,
    InitProcessGroupKwargs,
)
from datasets import Dataset
from packaging import version
from tqdm import tqdm
import types
from lm_eval import utils
from lm_eval.api.instance import Instance
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval.models.utils import get_dtype
import time
import subprocess
eval_logger = logging.getLogger(__name__)
T = TypeVar("T", bound="LM")

def empty_cache_by_memory(threshold_gb=70):
    """
    Empty CUDA cache if allocated memory exceeds threshold
    Args:
        threshold_gb: Memory threshold in GB
    """
    if torch.cuda.is_available():
        # Get current memory allocated
        allocated = torch.cuda.memory_allocated() / 1024**3  # Convert to GB

        if allocated > threshold_gb:
            # Clear cache
            gc.collect()
            torch.cuda.empty_cache()
            print(f"Cache cleared. Memory freed: {allocated:.2f} GB")

@register_model("diffllm")
class DiffLLM(LM):
    def __init__(
        self,
        pretrained: Union[str, transformers.PreTrainedModel],
        batch_size: Optional[Union[int, str]] = 1,
        device: Optional[str] = "cuda",
        dtype: Optional[Union[str, torch.dtype]] = "auto",
        max_prompt_len: Optional[int] = 1024,
        max_new_tokens: Optional[int] = 128,
        nll_type: Optional[str] = "mc",
        log_type: Optional[str] = "ftb",
        classifier_free_guidance: Optional[float] = 1.0,
        pad_to_max_len: Optional[bool] = False,
        sampling_eps: Optional[float] = 1e-3,
        diffusion_steps: Optional[int] = 32,
        trust_remote_code: Optional[bool] = True,
        parallelize: Optional[bool] = False,
        autogptq: Optional[Union[bool, str]] = False,
        torch_compile: Optional[bool] = True,
        **kwargs,
    ) -> None:
        super().__init__()

        # prepare for parallelism
        assert isinstance(device, str)
        assert isinstance(pretrained, str)
        assert isinstance(batch_size, (int, str))

        gpus = torch.cuda.device_count()
        accelerator_kwargs = InitProcessGroupKwargs(timeout=timedelta(weeks=52))
        accelerator = Accelerator(kwargs_handlers=[accelerator_kwargs])
        if accelerator.num_processes > 1:
            self.accelerator = accelerator

        if "npu" in accelerator.device.type:
            gpus = torch.npu.device_count()

        # using one process with no model parallelism
        if not (parallelize or accelerator.num_processes > 1):
            # use user-passed device
            device_list = set(
                ["cuda", "cpu"]
                + [f"cuda:{i}" for i in range(gpus)]
                + ["mps", "mps:0"]
                + [f"npu:{i}" for i in range(gpus)]
            )
            if device and device in device_list:
                self._device = torch.device(device)
                eval_logger.info(f"Using device '{device}'")
                if device in ("mps", "mps:0") and version.parse(
                    torch.__version__
                ) < version.parse("2.1"):
                    raise RuntimeError(
                        f"mps requires torch >= 2.1. You have {torch.__version__}"
                    )
            else:
                eval_logger.info("Device not specified")
                eval_logger.info(f"Cuda Available? {torch.cuda.is_available()}")
                self._device = (
                    torch.device("cuda")
                    if torch.cuda.is_available()
                    else torch.device("cpu")
                )
        else:  # Parallelism managed by accelerate
            if device != "cuda":
                eval_logger.info(
                    f"Using `accelerate launch` or `parallelize=True`, device '{device}' will be overridden when placing model."
                )
            # TODO: include in warning that `load_in_8bit` etc. affect this too
            self._device = (
                self.accelerator.device
                if hasattr(self, "accelerator")
                else torch.device(device)
            )

        self.batch_size_per_gpu = batch_size
        if isinstance(batch_size, str):
            self.batch_size_per_gpu = int(batch_size)
        
        # Store torch_compile setting
        self.torch_compile = torch_compile
        self._model_compiled = False
        
        # Initialize generation_method before creating model
        self.generation_method = kwargs.get("generation_method", "generation")
        
        self._create_model_and_tokenizer(pretrained, dtype, trust_remote_code)

        if isinstance(pretrained, str):
            if gpus >= 1 or str(self.device) == "mps":
                # TODO: can remove this whole snippet except in the mps case, perhaps?
                if not (parallelize or autogptq or hasattr(self, "accelerator")):
                    # place model onto device requested manually,
                    # if not using HF Accelerate or device_map
                    # or any other option that preloads model onto device
                    try:
                        self.model.to(self.device)
                    except ValueError:
                        eval_logger.debug(
                            "Failed to place model onto specified device. This may be because the model is quantized via `bitsandbytes` or `device_map` is provided. If the desired GPU is being used, this message is safe to ignore."
                        )
            # multigpu data-parallel support when launched with accelerate
            if gpus > 1:
                if accelerator.num_processes > 1:
                    if parallelize:
                        eval_logger.warning(
                            "You are both using a HF Accelerate `device_map` (`--model_args parallelize=True`) and launching via `accelerate launch`. This will attempt to do model and data parallelism depending on the resources available."
                        )
                    elif gpus > accelerator.num_processes:
                        eval_logger.warning(
                            "WARNING: The number of total system GPUs does not match the number of spawned processes. "
                            "If you would like to use data parallelism, please launch the script "
                            "with 'accelerate launch *script*'. "
                            f"Current run will proceed with {accelerator.num_processes} devices."
                        )
                        if self.accelerator.is_local_main_process:
                            eval_logger.info(
                                f"Using {gpus} devices with data parallelism"
                            )

                    self._device = torch.device(f"{accelerator.device}")
                    self.accelerator = accelerator

                    self._rank = self.accelerator.local_process_index
                    self._world_size = self.accelerator.num_processes
                else:
                    # if we aren't launching via accelerate, ditch
                    self._rank = 0
                    self._world_size = 1
        else:
            # if a PreTrainedModel was passed into HFLM, we forgo distributed setup.
            eval_logger.warning(
                "Passed an already-initialized model through `pretrained`, assuming single-process call to evaluate() or custom distributed integration"
            )
            self._rank = 0
            self._world_size = 1

        # generation params
        self.max_prompt_len = max_prompt_len
        self.max_new_tokens = max_new_tokens
        self.diffusion_steps = diffusion_steps
        self.temperature = kwargs.get("temperature", 0.1)
        self.top_p = kwargs.get("top_p", None)
        self.alg = kwargs.get("alg", "entropy")
        self.alg_temp = kwargs.get("alg_temp", 0.0)
        self.top_k = kwargs.get("top_k", None)
        self.dParallel = kwargs.get("dParallel", False)
        self.threshold = kwargs.get("threshold", 0.45)
        self.block_length = kwargs.get("block_length", 32)
        # generation_method already set before model creation
        self.block_add_threshold = kwargs.get("block_add_threshold", 1.0)
        self.decoded_token_threshold = kwargs.get("decoded_token_threshold", 1.0)
        self.cache_delay_iter = kwargs.get("cache_delay_iter", 10000)
        self.refresh_interval = kwargs.get("refresh_interval", 10000)
        self.early_stop = kwargs.get("early_stop", False)
        self.use_cache = kwargs.get("use_cache", False)
        self.dual_cache = kwargs.get("dual_cache", False)

        # loglikelihood params
        self.nll_type = nll_type
        self.log_type = log_type
        self.classifier_free_guidance = classifier_free_guidance
        self.pad_to_max_len = pad_to_max_len
        self.sampling_eps = sampling_eps


    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    @property
    def device(self):
        return self._device

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _create_model_and_tokenizer(self, pretrained, dtype, trust_remote_code):
        # Use flash_attention_2 only if torch_compile is enabled
        attn_impl = "flash_attention_2" if self.torch_compile else None
        
        # Import model based on generation method
        if self.generation_method == "Fast_dllm_v1":
            # Use Fast-dLLM-v1 model for compatibility with its generation methods
            from baseline.Fast_dLLM_v1.dream.model.modeling_dream import DreamModel
            from baseline.Fast_dLLM_v1.dream.model.configuration_dream import DreamConfig
        else:
            # Use utils_Dream model for other methods
            from utils.utils_Dream.model.modeling_dream import DreamModel, DreamConfig
        
        model_config = DreamConfig.from_pretrained(pretrained)
        self.model = (
            DreamModel.from_pretrained(
                pretrained,
                config=model_config,
                torch_dtype=get_dtype(dtype),
                trust_remote_code=False,
                attn_implementation=attn_impl,
            )
            .eval()
        ).to(self.device)
        
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            pretrained, trust_remote_code=trust_remote_code
        )

    def tok_decode(self, tokens, skip_special_tokens=True):
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special_tokens)

    def tok_encode(self, text, add_special_tokens=True):
        return self.tokenizer(
            text, return_tensors="pt", add_special_tokens=add_special_tokens
        ).input_ids
    @classmethod
    def create_from_arg_string(
        cls: Type[T], arg_string: str, additional_config: Optional[dict] = None
    ) -> T:
        """
        Creates an instance of the LM class using the given argument string and additional config.

        Parameters:
        - arg_string: A string containing arguments in the format key1=value1,key2=value2.
        - additional_config: Optional dictionary containing additional configuration parameters.

        Returns:
        - Instance of the LM class.
        """
        additional_config = {} if additional_config is None else additional_config
        args = utils.simple_parse_args_string(arg_string)
        args2 = {k: v for k, v in additional_config.items() if v is not None}
        return cls(**args, **args2)

    def apply_chat_template(
        self, chat_history, add_generation_prompt: bool = True
    ) -> str:
        """
        Method to apply a chat template to a list of chat history between user and model.
        """
        chat_templated = self.tokenizer.apply_chat_template(
            chat_history,
            tokenize=False,
            add_generation_prompt=add_generation_prompt,
            continue_final_message=not add_generation_prompt,
        )

        return chat_templated

    @property
    def tokenizer_name(self) -> str:
        return self.tokenizer.name_or_path.replace("/", "__")

    def _generate_batch(self, prompts: List[str]) -> List[str]:
        # tokenize
        prompt_ids = self.tokenizer(prompts, return_tensors="pt", padding=True, padding_side="left").input_ids
        prompt_ids = prompt_ids[:, -self.max_prompt_len:]
        attn_mask = prompt_ids.ne(self.tokenizer.pad_token_id)
        prompt_ids = prompt_ids.to(device=self.device)
        attn_mask = attn_mask.to(device=self.device)

        # generate
        # NOTE: now fixed
        if self.generation_method == "Fast_dllm_v1":
            # Fast-dLLM-v1 generation
            generation_ids, nfe = self.model.diffusion_generate(
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
                threshold=self.threshold,
                block_length=self.block_length,
                dual_cache=self.dual_cache,
            )
        else:
            generation_ids, nfe = self.model.diffusion_generate(
                prompt_ids,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                alg=self.alg,
                alg_temp=self.alg_temp,
                threshold=self.threshold,
                block_length=self.block_length,
            )

        # decode
        responses = []
        for idx, (p, g) in enumerate(zip(prompt_ids, generation_ids.sequences)):
            generated_ids = g[len(p):].tolist()
            full_response = self.tokenizer.decode(generated_ids)
            response = full_response.split(self.tokenizer.eos_token)[0]
            responses.append(response)
            
            # # Debug info for first sample
            # if idx == 0:
            #     # print(f"\n[Generation Debug]")
            #     # print(f"  Prompt length: {len(p)} tokens")
            #     # print(f"  Full sequence length: {len(g)} tokens")
            #     # print(f"  Generated region length: {len(generated_ids)} tokens")
            #     # print(f"  EOS token id: {self.tokenizer.eos_token_id}")
            #     # print(f"  EOS token string: '{self.tokenizer.eos_token}'")
                
            #     # Count EOS in generated region
            #     # eos_count = generated_ids.count(self.tokenizer.eos_token_id)
            #     # print(f"  Number of EOS in generated region: {eos_count}")
                
            #     # Find first EOS position
            #     # if self.tokenizer.eos_token_id in generated_ids:
            #         # first_eos_pos = generated_ids.index(self.tokenizer.eos_token_id)
            #         # print(f"  First EOS at position: {first_eos_pos} (relative to generation start)")
            #         # print(f"  Tokens before first EOS: {first_eos_pos}")
                    
            #         # Analyze tokens before first EOS
            #         # tokens_before_eos = generated_ids[:first_eos_pos]
            #         # print(f"  First 20 generated token IDs: {tokens_before_eos[:20]}")
            #         # print(f"  Last 20 generated token IDs before EOS: {tokens_before_eos[-20:]}")
                    
            #         # Check for special tokens
            #         # mask_token_id = getattr(self.tokenizer, 'mask_token_id', None)
            #         # if mask_token_id is not None:
            #         #     mask_count = tokens_before_eos.count(mask_token_id)
            #             # print(f"  MASK tokens in region before EOS: {mask_count}")
                    
            #         # Decode the region before EOS
            #         # text_before_eos = self.tokenizer.decode(tokens_before_eos)
            #         # print(f"  Text before EOS length: {len(text_before_eos)} chars")
            #         # print(f"  Text before EOS preview (first 200 chars): {text_before_eos[:200]}")
            #         # print(f"  Text before EOS preview (last 200 chars): ...{text_before_eos[-200:]}")
                    
            #         # CRITICAL: Re-tokenize the decoded text to see the mismatch
            #         # retokenized = self.tokenizer(text_before_eos, add_special_tokens=False)['input_ids']
            #         # print(f"\n  [CRITICAL COMPARISON]")
            #         # print(f"  Original token IDs before EOS: {first_eos_pos} tokens")
            #         # print(f"  Re-tokenized from decoded text: {len(retokenized)} tokens")
            #         # print(f"  Difference: {first_eos_pos - len(retokenized)} tokens lost!")
                    
            #         # if first_eos_pos != len(retokenized):
            #         #     print(f"  WARNING: Tokenization mismatch detected!")
            #         #     print(f"  This explains why external statistics are different.")
            #     # else:
            #     #     print(f"  No EOS found in generated region")
                
            #     print(f"  Full response length (before split): {len(full_response)} chars")
            #     print(f"  Final response length (after split): {len(response)} chars")

        return responses, nfe

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

    def generate_until(self, requests: List[Instance], disable_tqdm: bool = False):

        # Apply torch.compile only if enabled
        if self.torch_compile and not self._model_compiled:
            eval_logger.info(f"Compiling model with torch.compile (model is on {self.model.device})...")
            eval_logger.info("First compilation may take some time. Subsequent forward passes will be faster.")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self._model_compiled = True
            eval_logger.info("Model compilation complete.")

        # Bind custom generation methods
        if self.dParallel:
            from model.generation_utils_semiar import DreamGenerationMixin
            self.model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, self.model)
            self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)
        elif self.generation_method == "Fast_dllm_v1":
            # Use Fast-dLLM-v1 generation methods
            if self.use_cache:
                from baseline.Fast_dLLM_v1.dream.model.generation_utils_block import DreamGenerationMixin as FastDLLMGenerationMixin
            else:
                from baseline.Fast_dLLM_v1.dream.model.generation_utils import DreamGenerationMixin as FastDLLMGenerationMixin
            
            self.model.diffusion_generate = types.MethodType(FastDLLMGenerationMixin.diffusion_generate, self.model)
            self.model._sample = types.MethodType(FastDLLMGenerationMixin._sample, self.model)
        elif self.generation_method == "generation_multi_block":
            # Use d3llm multi-block generation
            from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin
            
            self.model.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, self.model)
            self.model._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, self.model)

            # self.model.generate_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin.generate_multi_block_kv_cache, self.model)
            self.model._sample_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin._sample_multi_block_kv_cache, self.model)

            self.model._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, self.model)
            
            # Override diffusion_generate to use generate_multi_block
            def diffusion_generate_wrapper(
                model_self,
                input_ids,
                attention_mask=None,
                max_new_tokens=None,
                output_history=False,
                return_dict_in_generate=True,
                steps=None,
                temperature=None,
                top_p=None,
                top_k=None,
                alg=None,
                alg_temp=None,
                threshold=None,
                block_length=None,
            ):
                # Call generate_multi_block instead
                from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationConfig
                
                generation_config = DreamGenerationConfig(
                    max_length=input_ids.shape[1] + max_new_tokens,
                    mask_token_id=model_self.config.mask_token_id,
                    temperature=temperature if temperature is not None else self.temperature,
                    alg=alg if alg is not None else self.alg,
                    return_dict_in_generate=return_dict_in_generate,
                )
                
                result, nfe = model_self.generate_multi_block(
                    inputs=input_ids,
                    generation_config=generation_config,
                    threshold=threshold if threshold is not None else self.threshold,
                    block_size=block_length if block_length is not None else self.block_length,
                    block_add_threshold=self.block_add_threshold,
                    decoded_token_threshold=self.decoded_token_threshold,
                    cache_delay_iter=self.cache_delay_iter,
                    refresh_interval=self.refresh_interval,
                    early_stop=self.early_stop,
                )
                
                return result, nfe
            
            self.model.diffusion_generate = types.MethodType(diffusion_generate_wrapper, self.model)
        else:
            from model.generation_utils import DreamGenerationMixin
            self.model.diffusion_generate = types.MethodType(DreamGenerationMixin.diffusion_generate, self.model)
            self.model._sample = types.MethodType(DreamGenerationMixin._sample, self.model)

        res = []

        pbar = tqdm(
            total=len(requests),
            disable=(disable_tqdm or (self.rank != 0)),
            desc="Running generate_until requests",
        )

        
        
        total_nfe = 0
        total_tokens = 0
        total_tokens_before_until = 0
        run_time = 0
        step_count = 0
        warmup_steps = 10  # Skip first 10 iterations for statistics
        recent_stats = deque(maxlen=10)  # Track last 10 rounds (time, tokens, nfe)
        
        for batch_idx in range(0, len(requests), self.batch_size):
            batch_requests = requests[batch_idx : batch_idx + self.batch_size]
            contexts, gen_args = zip(*[req.arguments for req in batch_requests])
            start_time = time.time()
            responses, nfe = self._generate_batch(contexts)
            end_time = time.time()
            
            # Count tokens in responses
            for i, r in enumerate(responses):
                original_response = r
                
                # Tokenize before 'until' truncation
                tokens_before_until = self.tokenizer(original_response, add_special_tokens=False)['input_ids']
                tokens_before_len = len(tokens_before_until)
                
                # Count tokens before 'until' truncation
                if step_count >= warmup_steps:
                    total_tokens_before_until += tokens_before_len
                
                for s in gen_args[0]['until']:
                    r = r.split(s)[0]
                responses[i] = r
                
                # Tokenize after 'until' truncation
                tokens_after_until = self.tokenizer(r, add_special_tokens=False)['input_ids']
                tokens_after_len = len(tokens_after_until)
                
                # Debug: show the effect of 'until' processing
                if self.rank == 0 and i == 0 and step_count >= warmup_steps:
                    print(f"\n[Until Processing Debug]")
                    print(f"  Until strings: {gen_args[0]['until']}")
                    print(f"  Token length before 'until': {tokens_before_len} tokens")
                    print(f"  Token length after 'until': {tokens_after_len} tokens")
                    print(f"  Tokens removed: {tokens_before_len - tokens_after_len}")
                    
                    # Print original response in red if it was truncated
                    # if original_response != r:
                    # print(f"\n\033[91m[Original Response Before 'until' truncation]:\033[0m")
                    # print(f"\033[91m{original_response}\033[0m")
                    # print(f"\033[91mOriginal: {tokens_before_len} tokens, Truncated to: {tokens_after_len} tokens\033[0m")

            res.extend(responses)
            pbar.update(len(contexts))
            
            # Only count statistics after warmup
            if step_count >= warmup_steps:
                total_nfe += nfe
                round_time = end_time - start_time
                round_tokens = 0
                # Count tokens (excluding special tokens)
                for idx, response in enumerate(responses):
                    response_tokens = self.tokenizer(response, add_special_tokens=False)['input_ids']
                    total_tokens += len(response_tokens)
                    round_tokens += len(response_tokens)
                    
                    # Debug: show detailed token info for first response
                    # if self.rank == 0 and idx == 0:
                    #     print(f"\n[Token Analysis]")
                    #     print(f"  Raw response length: {len(response)} chars")
                    #     print(f"  Response tokens: {len(response_tokens)} tokens")
                    #     print(f"  First 10 tokens: {response_tokens[:10]}")
                    #     print(f"  Response preview: {response[:200]}...")
                run_time += round_time
                recent_stats.append((round_time, round_tokens, nfe))
            
            if self.rank == 0:
                print(f"Context:\n{contexts[0][:200]}{'...' if len(contexts[0]) > 200 else ''}")
                response_preview = responses[0][:200] + ('...' if len(responses[0]) > 200 else '')
                print("\033[91mTruncated Response:")
                print(response_preview)
                print("\033[0m")
                print(f"Truncated Token length: {tokens_after_len}")
                print(f"Original Token length: {tokens_before_len}")
                print(f"NFE (Number of Function Evaluations): {nfe}")
                
                step_count += 1
                
                if step_count <= warmup_steps:
                    print(f"[Warmup {step_count}/{warmup_steps}] - Statistics not counted")
                else:
                    # Print GPU stats
                    gpu_stats = self._get_gpu_stats()
                    if gpu_stats:
                        print(f"\n{'='*60}")
                        print(f"GPU Stats at Step {step_count}:")
                        print(f"  Number of GPUs: {gpu_stats['num_gpus']}")
                        print(f"  GPU Memory Used: {gpu_stats['gpu_memory_used_mb']:.2f} MB")
                        print(f"  GPU Memory Total: {gpu_stats['gpu_memory_total_mb']:.2f} MB")
                        print(f"  GPU Memory Usage: {gpu_stats['gpu_memory_percent']:.2f}%")
                        print(f"  GPU Utilization: {gpu_stats['gpu_utilization_percent']:.2f}%")
                        print(f"{'='*60}\n")
                    else:
                        print("No GPU stats available for step", step_count)
                    
                    print(f"Total time taken (after warmup): {run_time:.2f} seconds")
                    print(f"Total NFE (after warmup): {total_nfe}")
                    
                    print(f"\nBEFORE 'until' truncation:")
                    print(f"  Total tokens: {total_tokens_before_until}")
                    if run_time > 0:
                        print(f"  Throughput: {total_tokens_before_until / run_time:.2f} tokens/s")
                        if total_nfe > 0:
                            print(f"  Tokens per forward: {total_tokens_before_until / total_nfe:.2f}")
                    
                    print(f"\nAFTER 'until' truncation:")
                    print(f"  Total tokens: {total_tokens}")
                    if run_time > 0:
                        print(f"  Throughput: {total_tokens / run_time:.2f} tokens/s")
                        if total_nfe > 0:
                            print(f"  Tokens per forward: {total_tokens / total_nfe:.2f}")
                    
                    # Print last 10 rounds average
                    if len(recent_stats) > 0:
                        avg_time = sum(t for t, _, _ in recent_stats)
                        avg_tokens = sum(tk for _, tk, _ in recent_stats)
                        avg_nfe = sum(n for _, _, n in recent_stats)
                        print(f"\nLast {len(recent_stats)} rounds average:")
                        if avg_time > 0:
                            print(f"  Avg TPS: {avg_tokens / avg_time:.2f} tokens/s")
                            if avg_nfe > 0:
                                print(f"  Avg TPF: {avg_tokens / avg_nfe:.2f} tokens/forward")
                    
                    if total_tokens_before_until > 0:
                        print(f"\nTokens removed by 'until': {total_tokens_before_until - total_tokens} ({100*(total_tokens_before_until - total_tokens)/total_tokens_before_until:.1f}%)")
                    print("="*60 + "\n")

        
        # Print final statistics (excluding warmup)
        print("\n" + "="*60)
        print(f"Final Statistics (excluding first {warmup_steps} warmup iterations):")
        print(f"Total time taken: {run_time:.2f} seconds")
        print(f"Total NFE: {total_nfe}")
        
        print(f"\nBEFORE 'until' truncation:")
        print(f"  Total tokens: {total_tokens_before_until}")
        if run_time > 0:
            print(f"  Throughput: {total_tokens_before_until / run_time:.2f} tokens/s")
            if total_nfe > 0:
                print(f"  Tokens per forward: {total_tokens_before_until / total_nfe:.2f}")
        
        print(f"\nAFTER 'until' truncation:")
        print(f"  Total tokens: {total_tokens}")
        if run_time > 0:
            print(f"  Throughput: {total_tokens / run_time:.2f} tokens/s")
            if total_nfe > 0:
                print(f"  Tokens per forward: {total_tokens / total_nfe:.2f}")
        
        if total_tokens_before_until > 0:
            print(f"\nTokens removed by 'until': {total_tokens_before_until - total_tokens} ({100*(total_tokens_before_until - total_tokens)/total_tokens_before_until:.1f}%)")
        
        effective_samples = len(requests) - warmup_steps
        if effective_samples > 0:
            print(f"\nAverage time per request: {run_time / effective_samples:.2f} seconds")
            print(f"Average tokens per request (after 'until'): {total_tokens / effective_samples:.1f}")
        print("="*60 + "\n")

        return res

    def _forward_process(self, batch):
        b, l = batch.shape
        # sample from U[0, 1] following https://arxiv.org/pdf/2107.00630 I.1
        u0 = torch.rand(1, device=batch.device, dtype=torch.float32)
        indices = torch.arange(b, device=batch.device).float()
        t = (u0 + indices / b) % 1

        p_mask = (1 - self.sampling_eps) * t + self.sampling_eps

        p_mask = p_mask[:, None].repeat(1, l)

        mask_indices = torch.rand((b, l), device=batch.device) < p_mask
        # always unmask bos and eos
        mask_indices[:, 0] = False
        mask_indices[:, -1] = False

        noisy_batch = torch.where(mask_indices, self.tokenizer.mask_token_id, batch)
        return noisy_batch, p_mask

    @torch.no_grad()
    def get_logits(self, batch, prompt_index):
        '''
        prompt_index : 1D bool tensor, length=batch.shape[1]
        '''
        if self.classifier_free_guidance > 1.:
            assert len(prompt_index) == batch.shape[1]
            prompt_index = prompt_index.unsqueeze(0).repeat(batch.shape[0], 1)
            un_batch = batch.clone()
            un_batch[prompt_index] = self.tokenizer.mask_token_id
            batch = torch.cat([batch, un_batch])

        if self.pad_to_max_len:
            raise NotImplementedError
        else:
            input = batch

        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            logits = self.model(input, 'full').logits
            # since bos always unmask, the first logits will not be used
            logits = torch.cat([logits[:,:1], logits[:, :-1]], dim=1)

        if self.classifier_free_guidance > 1.:
            logits, un_logits = torch.chunk(logits, 2, dim=0)
            logits = un_logits + self.cfg * (logits - un_logits)
        return logits[:, :batch.shape[1]]

    @torch.no_grad()
    def _eval_target_nll_mc(self, prefix, target):
        if prefix is None:
            seq = target[None, :]
        else:
            seq = torch.concatenate([prefix, target])[None, :]
        seq = seq.repeat((self.batch_size, 1)).to(self.device)

        if self.log_type == 'ftb':
            prompt_index = torch.arange(seq.shape[1], device=self.device) < len(prefix)
        else:
            prompt_index = torch.arange(seq.shape[1], device=self.device) >= len(prefix)

        loss_acc = []
        mc_num = self.diffusion_steps
        for _ in range(max(mc_num // self.batch_size, 1)):
            perturbed_seq = seq.clone()
            perturbed_seq_, p_mask = self._forward_process(seq)
            if self.log_type == 'ftb':
                perturbed_seq[:, -len(target):] = perturbed_seq_[:, -len(target):]
            elif self.log_type == 'btf':
                perturbed_seq[:, :len(prefix)] = perturbed_seq_[:, :len(prefix)]
            elif self.log_type == 'union':
                perturbed_seq = perturbed_seq_
            else:
                raise NotImplementedError(self.log_type)

            mask_indices = perturbed_seq == self.tokenizer.mask_token_id

            logits = self.get_logits(perturbed_seq, prompt_index)

            loss = F.cross_entropy(logits[mask_indices], seq[mask_indices], reduction='none') / p_mask[mask_indices]
            loss = loss.sum() / self.batch_size
            loss_acc.append(loss.item())
            del logits, loss, perturbed_seq, perturbed_seq_, p_mask, mask_indices
            empty_cache_by_memory(threshold_gb=70)

        return sum(loss_acc) / len(loss_acc)

    @torch.no_grad()
    def _eval_target_nll_ar(self, prefix, target):
        prefix, target = prefix.unsqueeze(0), target.unsqueeze(0) # 1*l1, 1*l2
        assert self.log_type in ['ftb', 'btf']
        assert self.nll_type in ['ar_ftb', 'ar_btf']

        if self.log_type == 'ftb':
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) < prefix.shape[1]
        else:
            prompt_index = torch.arange(prefix.shape[1] + target.shape[1], device=self.device) >= prefix.shape[1]

        if self.log_type == 'ftb':
            perturbed_ = target.repeat(target.shape[1], 1).clone().contiguous() # l2*l2
        else:
            perturbed_ = prefix.repeat(prefix.shape[1], 1).clone().contiguous() # l1*l1

        mask_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            mask_index = torch.triu(mask_index)
        else:
            mask_index = torch.tril(mask_index)
        perturbed_[mask_index] = self.tokenizer.mask_token_id
        if self.log_type == 'ftb':
            perturbed_seq = torch.cat([prefix.repeat(perturbed_.shape[0], 1), perturbed_], dim=-1)
        else:
            perturbed_seq = torch.cat([perturbed_, target.repeat(perturbed_.shape[0], 1)], dim=-1)

        logits_ = []
        num = len(perturbed_seq) // self.batch_size if len(perturbed_seq) % self.batch_size == 0 else len(perturbed_seq) // self.batch_size + 1
        for i in range(num):
            end = (i + 1) * self.batch_size if (i + 1) * self.batch_size < len(perturbed_seq) else len(perturbed_seq)
            perturbed_seq_ = perturbed_seq[i * self.batch_size: end]
            perturbed_seq_ = perturbed_seq_.to(self.device)
            if len(perturbed_seq_.shape) == 1:
                perturbed_seq_ = perturbed_seq_.unsqueeze(0)
            logits = self.get_logits(perturbed_seq_, prompt_index)
            logits_.append(logits.cpu())
        logits = torch.cat(logits_, dim=0)

        temp_index = torch.ones((perturbed_.shape[1], perturbed_.shape[1]), dtype=torch.bool)
        if self.nll_type == 'ar_ftb':
            temp_index = torch.triu(temp_index, diagonal=1)
        else:
            temp_index = torch.tril(temp_index, diagonal=-1)
        mask_index[temp_index] = False
        if self.log_type == 'ftb':
            logits_index = torch.cat([torch.zeros((perturbed_.shape[1], prefix.shape[1]), dtype=torch.bool), mask_index], dim=-1)
        else:
            logits_index = torch.cat([mask_index, torch.zeros((perturbed_.shape[1], target.shape[1]), dtype=torch.bool)], dim=-1)

        if self.log_type == 'ftb':
            loss = F.cross_entropy(logits[logits_index], target[0], reduction='sum').cpu().item()
        else:
            loss = F.cross_entropy(logits[logits_index], prefix[0], reduction='sum').cpu().item()
        return loss

    def _encode_pair(self, context, continuation):
        n_spaces = len(context) - len(context.rstrip())
        if n_spaces > 0:
            continuation = context[-n_spaces:] + continuation
            context = context[:-n_spaces]

        whole_enc = self.tokenizer.encode(context + continuation) + [
            self.tokenizer.eos_token_id
        ]
        context_enc = self.tokenizer.encode(context)

        context_enc_len = len(context_enc)
        continuation_enc = whole_enc[context_enc_len:]

        return context_enc, continuation_enc

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
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

        out = []
        with torch.no_grad():
            for elem in tqdm(ds, desc="Computing likelihood..."):
                prefix = elem["prefix"]
                target = elem["target"]

                if self.nll_type == 'mc':
                    ll = -self._eval_target_nll_mc(prefix, target)
                    if self.log_type == 'union':
                        ll = ll / (len(target) + len(prefix))
                elif self.nll_type == 'ar_ftb' or self.nll_type == 'ar_btf':
                    ll = -self._eval_target_nll_ar(prefix, target)
                else:
                    raise NotImplementedError(self.nll_type)

                # TODO: greedy decoding
                is_target_greedy_dec = False

                out.append((ll, 1.0 if is_target_greedy_dec else 0.0))
        return out

    def loglikelihood_rolling(self, requests: List[Instance]) -> List[float]:
        raise NotImplementedError