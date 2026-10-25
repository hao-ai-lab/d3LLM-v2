# Evaluation Scripts

### Supported Methods

We include comprehensive evaluation code for:

- ✅ **d3LLM** (our method)
- ✅ [**AR Model (e.g., Qwen-2.5-7B-it)**](https://arxiv.org/abs/2412.15115) - Autoregressive baselines
- ✅ [**Vanilla LLaDA**](https://arxiv.org/abs/2502.09992) - Original LLaDA model
- ✅ [**Vanilla Dream**](https://arxiv.org/abs/2508.15487) - Original Dream model
- ✅ [**Fast-dLLM**](https://arxiv.org/abs/2505.22618) - Training-free acceleration with KV cache
- ✅ [**D2F**](https://arxiv.org/abs/2508.09192) - Discrete diffusion forcing
- ✅ [**dParallel**](https://arxiv.org/abs/2509.26488) - Distilled dLLMs
- ✅ [**Fast-dLLM v2**](https://arxiv.org/abs/2509.26328) - Block-wise diffusion

### Supported Benchmarks

```bash
# GSM8K
bash dream_gsm8k_cot.sh
bash llada_gsm8k_cot.sh

# MATH
bash dream_math.sh
bash llada_math.sh

# Code Generation (HumanEval & MBPP)
bash dream_humaneval.sh
bash dream_mbpp.sh
bash llada_humaneval.sh
bash llada_mbpp.sh
bash dream-coder.sh

# Long-Context GSM8K
bash dream_long_gsm8k.sh
bash llada_long_gsm8k.sh
```
