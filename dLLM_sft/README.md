# dLLM SFT

SFT code and data for the starting models of the RL stage.

| Path | Content |
| - | - |
| [data/](./data/) | Training data of the released SFT models ([README](./data/README.md)) |
| [SDAR/](./SDAR/) | SDAR backbones: LlamaFactory (`training/llama_factory_sdar`), training yamls (`examples/train_full_sdar`), model code (`training/model`) and the launcher [run_sft.sh](./SDAR/run_sft.sh) |
| [AR/](./AR/) | Autoregressive Qwen baselines trained with the same LlamaFactory environment, data and training budget as SDAR ([README](./AR/README.md)) |
| [d3LLM/](./d3LLM/) | d3LLM-LLaDA: training code (`d3llm/d3llm_LLaDA/distill_2_training`), WebShop scripts (`d3llm/d3llm_LLaDA/webshop_training`), the launcher [run_sft.sh](./d3LLM/run_sft.sh) and the evaluation tools of d3LLM |

## Workflow

1. `data/` provides the trajectories. SDAR reads the jsonl files through `data/sdar/dataset_info.json`; AR converts the same jsonl files (`<reason>` to `<think>`); d3LLM reads the parquet in `data/d3llm/`, which is converted from the SDAR WebShop jsonl.
2. The SDAR launcher trains the model into `dLLM_sft/sfted_models/ckpt/<name>`; the AR launcher writes the model directly to `dLLM_sft/sfted_models/<name>`.
3. For SDAR, [scripts/Misc/sync_verl_modeling.sh](../scripts/Misc/sync_verl_modeling.sh) copies the model code from `verl/models` into the training output, which the RL log-likelihood path requires.
4. For SDAR, [scripts/Misc/export_sfted_model.sh](../scripts/Misc/export_sfted_model.sh) exports the HuggingFace model to `dLLM_sft/sfted_models/<name>`.

## Reproduce the released SFT models

```bash
bash dLLM_sft/SDAR/run_sft.sh webshop-8b-b4   # SDAR-8B-Chat-sfted-webshop
bash dLLM_sft/SDAR/run_sft.sh webshop-4b-b4   # SDAR-4B-Chat-sfted-webshop
bash dLLM_sft/SDAR/run_sft.sh alfworld-8b-b4  # SDAR-8B-Chat-sfted-alfworld
JOB_ID=<job_id> bash dLLM_sft/SDAR/run_sft.sh webshop-8b-b8   # SDAR-8B-Chat-b8-sfted-webshop
bash dLLM_sft/AR/run_sft.sh alfworld-qwen3-8b                 # Qwen3-8B-sfted-alfworld
bash dLLM_sft/AR/run_sft.sh webshop-qwen3-8b                  # Qwen3-8B-sfted-webshop
bash dLLM_sft/AR/run_sft.sh webshop-qwen2.5-7b                # Qwen2.5-7B-Instruct-sfted-webshop
bash dLLM_sft/d3LLM/run_sft.sh webshop-llada-8b               # d3LLM_LLaDA-sfted-webshop
```

See the [repository README](../README.md#stage-1-sft) for the options.
