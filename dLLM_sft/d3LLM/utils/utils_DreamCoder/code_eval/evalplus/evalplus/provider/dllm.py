from typing import List
import time
import types

import torch
from transformers import AutoModel, AutoTokenizer

from evalplus.provider.base import DecoderBase
from evalplus.provider.utility import (
    extra_eos_for_direct_completion,
    make_raw_chat_prompt,
)

import sys
sys.path.append('/')
sys.path.append('../')
sys.path.append('../../')
sys.path.append('../../../')

class DLLMDecoder(DecoderBase):
    def __init__(
        self,
        name: str,
        dataset: str,
        force_base_prompt: bool = False,
        attn_implementation: str = "eager",
        device_map: str = None,
        gguf_file: str = None,
        **kwargs,
    ):
        # extract custom kwargs for dllm
        self.diffusion_steps = kwargs.pop("diffusion_steps", 256)
        self.top_p = kwargs.pop("top_p", 0.9)
        self.alg = kwargs.pop("alg", "entropy")
        self.alg_temp = kwargs.pop("alg_temp", 0.0)

        self.fast_dllm = kwargs.pop("fast_dllm", False)
        self.generation_method = kwargs.pop("generation_method", "generation")
        self.threshold = kwargs.pop("threshold", 0.45)
        self.block_length = kwargs.pop("block_length", 32)
        self.block_add_threshold = kwargs.pop("block_add_threshold", 1.0)
        self.decoded_token_threshold = kwargs.pop("decoded_token_threshold", 1.0)
        self.cache_delay_iter = kwargs.pop("cache_delay_iter", 10000)
        self.early_stop = kwargs.pop("early_stop", False)
        self.torch_compile = kwargs.pop("torch_compile", False)

        super().__init__(name=name, **kwargs)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Performance tracking
        self.stats = {"num_tokens": 0, "num_nfe": 0, "run_time": 0, "call_count": 0}
        self._model_compiled = False

        # Use flash_attention_2 only if torch_compile is enabled
        if self.torch_compile and attn_implementation == "eager":
            attn_implementation = "flash_attention_2"

        model_kwargs = {
            "device_map": device_map,
            "trust_remote_code": self.trust_remote_code,
            "torch_dtype": getattr(torch, self.dtype),
            "attn_implementation": attn_implementation,  # "eager", "flash_attention_2", "sdpa"
            "gguf_file": gguf_file,
        }

        self.skip_special_tokens = True

        print(f"{model_kwargs = }")

        self.force_base_prompt = force_base_prompt

        # gguf format embeds tokenizer and is not compatible with hf tokenizer `use_fast` param
        tokenizer_kwargs = {"trust_remote_code": self.trust_remote_code}
        if gguf_file is not None:
            tokenizer_kwargs["gguf_file"] = gguf_file
        self.tokenizer = AutoTokenizer.from_pretrained(name, **tokenizer_kwargs)
        if self.is_direct_completion():  # no chat template
            self.eos += extra_eos_for_direct_completion(dataset)
        else:  # with chat template
            self.eos += ["\n```\n"]

        print(f"{self.eos = }")
        if self.fast_dllm:
            from src.inference.fast_dllm.modeling_dream import DreamModel
            from src.inference.fast_dllm.generation_utils_block import (
                DreamGenerationMixin,
            )

            self.model = DreamModel.from_pretrained(name, **model_kwargs)
            self.model.diffusion_generate = types.MethodType(
                DreamGenerationMixin.diffusion_generate, self.model
            )
            self.model._sample = types.MethodType(
                DreamGenerationMixin._sample, self.model
            )
        elif self.generation_method == "generation_multi_block":
            from utils.utils_Dream.model.modeling_dream import DreamModel, DreamConfig
            
            model_config = DreamConfig.from_pretrained(name)
            self.model = DreamModel.from_pretrained(name, config=model_config, **model_kwargs)
        else:
            self.model = AutoModel.from_pretrained(name, **model_kwargs)
        
        self.model = self.model.to(self.device)
        
        # Bind custom generation methods after model is loaded
        if self.generation_method == "generation_multi_block":
            self._bind_multi_block_methods()

    def _bind_multi_block_methods(self):
        """Bind multi-block generation methods to the model."""
        from d3llm.d3llm_DREAM.d3llm_dream_generate_util import DreamGenerationMixin as D3LLMGenerationMixin
        
        self.model.generate_multi_block = types.MethodType(D3LLMGenerationMixin.generate_multi_block, self.model)
        self.model._sample_multi_block = types.MethodType(D3LLMGenerationMixin._sample_multi_block, self.model)
        self.model._sample_multi_block_kv_cache = types.MethodType(D3LLMGenerationMixin._sample_multi_block_kv_cache, self.model)
        self.model._prepare_inputs = types.MethodType(D3LLMGenerationMixin._prepare_inputs, self.model)
        
        # Create a reference to self for the wrapper closure
        decoder_self = self
        
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
            
            final_temperature = temperature if temperature is not None else decoder_self.temperature
            final_alg = alg if alg is not None else decoder_self.alg
            final_threshold = threshold if threshold is not None else decoder_self.threshold
            final_block_length = block_length if block_length is not None else decoder_self.block_length
            
            # Debug: print parameters for first call
            # if not hasattr(decoder_self, '_debug_printed'):
            #     print(f"\n[DEBUG wrapper] Parameters:")
            #     print(f"  input_ids.shape: {input_ids.shape}")
            #     print(f"  max_new_tokens: {max_new_tokens}")
            #     print(f"  temperature: {final_temperature}")
            #     print(f"  alg: {final_alg}")
            #     print(f"  threshold: {final_threshold}")
            #     print(f"  block_size: {final_block_length}")
            #     print(f"  block_add_threshold: {decoder_self.block_add_threshold}")
            #     print(f"  decoded_token_threshold: {decoder_self.decoded_token_threshold}")
            #     print(f"  cache_delay_iter: {decoder_self.cache_delay_iter}")
            #     print(f"  early_stop: {decoder_self.early_stop}")
            #     print(f"  mask_token_id: {model_self.config.mask_token_id}")
            #     decoder_self._debug_printed = True
            
            generation_config = DreamGenerationConfig(
                max_length=input_ids.shape[1] + max_new_tokens,
                mask_token_id=model_self.config.mask_token_id,
                temperature=final_temperature,
                alg=final_alg,
                return_dict_in_generate=return_dict_in_generate,
            )
            
            result, nfe = model_self.generate_multi_block(
                inputs=input_ids,
                generation_config=generation_config,
                threshold=final_threshold,
                block_size=final_block_length,
                block_add_threshold=decoder_self.block_add_threshold,
                decoded_token_threshold=decoder_self.decoded_token_threshold,
                cache_delay_iter=decoder_self.cache_delay_iter,
                early_stop=decoder_self.early_stop,
            )
            
            return result, nfe
        
        self.model.diffusion_generate = types.MethodType(diffusion_generate_wrapper, self.model)

    def is_direct_completion(self) -> bool:
        return self.force_base_prompt or self.tokenizer.chat_template is None

    @torch.inference_mode()
    def codegen(
        self, prompt: str, do_sample: bool = True, num_samples: int = 200
    ) -> List[str]:
        warmup_steps = 10  # Skip first 10 iterations for statistics
        
        # Apply torch.compile only if enabled
        if self.torch_compile and not self._model_compiled:
            print("Compiling model with torch.compile...")
            print("First compilation may take some time. Subsequent forward passes will be faster.")
            self.model = torch.compile(self.model, mode="reduce-overhead")
            self._model_compiled = True
            print("Model compilation complete.")
        
        # Auto-adjust sampling parameters for temperature=0
        if self.temperature == 0:
            do_sample = False
            num_samples = 1

        prompt = (
            prompt
            if self.is_direct_completion()
            else make_raw_chat_prompt(
                prompt, self.instruction_prefix, self.response_prefix, self.tokenizer
            )
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        input_tokens = inputs.input_ids
        attn_mask = inputs.attention_mask

        gen_kwargs = {}
        if do_sample:
            gen_kwargs["top_p"] = 0.95
            gen_kwargs["temperature"] = self.temperature

        start_time = time.time()
        if self.generation_method == "generation_multi_block":
            result, nfe = self.model.diffusion_generate(
                input_tokens,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                alg=self.alg,
                alg_temp=self.alg_temp,
                threshold=self.threshold,
                block_length=self.block_length,
            )
            outputs = result.sequences
        else:
            outputs = self.model.diffusion_generate(
                input_tokens,
                attention_mask=attn_mask,
                max_new_tokens=self.max_new_tokens,
                output_history=False,
                return_dict_in_generate=True,
                steps=self.diffusion_steps,
                temperature=self.temperature,
                top_p=self.top_p,
                alg=self.alg,
                alg_temp=self.alg_temp,
            ).sequences
            nfe = self.diffusion_steps
        end_time = time.time()

        gen_strs = self.tokenizer.batch_decode(
            outputs[:, input_tokens.size(-1) :],
            skip_special_tokens=self.skip_special_tokens,
        )
        
        # # Debug: print first generation
        # if self.stats["call_count"] == 0:
        #     print(f"\n[DEBUG] First generation:")
        #     print(f"  Input tokens size: {input_tokens.size()}")
        #     print(f"  Output tokens size: {outputs.size()}")
        #     print(f"  Generated token IDs (first 20): {outputs[0, input_tokens.size(-1):input_tokens.size(-1)+20].tolist()}")
        #     print(f"  Generated token IDs (last 20): {outputs[0, -20:].tolist()}")
        #     print(f"  Unique token IDs in generation: {torch.unique(outputs[0, input_tokens.size(-1):]).tolist()[:10]}")
        #     print(f"  Decoded text (first 200 chars): '{gen_strs[0][:200]}'")
        #     print(f"  EOS list: {self.eos}")
        
        outputs = []
        # removes eos tokens.
        for output in gen_strs:
            min_index = 10000
            for eos in self.eos:
                if eos in output:
                    min_index = min(min_index, output.index(eos))
            outputs.append(output[:min_index].replace("\t", "    "))

        # Accumulate statistics after warmup
        if self.stats["call_count"] >= warmup_steps:
            num_tokens = len(self.tokenizer.encode(outputs[0], add_special_tokens=False))
            self.stats["num_tokens"] += num_tokens
            self.stats["num_nfe"] += nfe
            self.stats["run_time"] += end_time - start_time
        
        self.stats["call_count"] += 1

        print(f"Context:\n{prompt}\n\nGenerated:\n{outputs[0]}")
        print(self.max_new_tokens, self.diffusion_steps, outputs)
        
        # Print real-time statistics after warmup
        if self.stats["call_count"] > warmup_steps:
            print(f"\nNFE (Number of Function Evaluations): {nfe}")
            print(f"Total time taken (after warmup): {self.stats['run_time']:.2f} seconds")
            print(f"Total NFE is {self.stats['num_nfe']}")
            if self.stats["run_time"] > 0 and self.stats["num_nfe"] > 0:
                tokens_per_second = self.stats["num_tokens"] / self.stats["run_time"]
                print(f"Throughput: {tokens_per_second:.2f} tokens/s")
                print(f"Total tokens generated: {self.stats['num_tokens']}")
                print(f"Token per step: {self.stats['num_tokens'] / self.stats['num_nfe']:.2f}")
        
        return outputs
