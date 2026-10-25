# SFT Data

The data used to train the released SFT models (`dLLM_sft/sfted_models/*`). All files are tracked by git.

| File | Samples | Used by |
| - | - | - |
| `sdar/webshop_sft.jsonl` | 1990 | `SDAR-8B-Chat-sfted-webshop`, `SDAR-8B-Chat-b8-sfted-webshop`, `SDAR-4B-Chat-sfted-webshop` (datasets: `webshop_sft`) |
| `sdar/alfworld_sft.jsonl` | 242 | `SDAR-8B-Chat-sfted-alfworld` (dataset: `alfworld_sft`) |
| `d3llm/webshop_sft_tags.parquet` | 1990 | `d3LLM_LLaDA-sfted-webshop` |

`sdar/*.jsonl` are ShareGPT-style trajectories (one `user` and one `assistant` message per sample, with `<reason></reason><action></action>` in the response). [sdar/dataset_info.json](./sdar/dataset_info.json) registers them for LlamaFactory; the SDAR launcher passes this directory as `dataset_dir`.

The d3LLM parquet (columns `question`, `llm_response`) is converted from the WebShop jsonl and can be regenerated with:
```bash
python dLLM_sft/d3LLM/d3llm/d3llm_LLaDA/webshop_training/convert_sdar_sharegpt_to_llada_parquet.py \
  --input dLLM_sft/data/sdar/webshop_sft.jsonl \
  --output dLLM_sft/data/d3llm/webshop_sft_tags.parquet \
  --model-name verl/models/d3LLM_LLaDA --max-length 4096 --reason-format tags
```
