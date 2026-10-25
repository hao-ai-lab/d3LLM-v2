import sys
import os

sys.path.append("./distill_2_training")
import torch
import torch.nn.functional as F
import yaml
from datasets import load_from_disk, load_dataset
from transformers import AutoModel, AutoTokenizer, Trainer, TrainingArguments
from peft import LoraConfig, get_peft_model
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
import random
import pickle
import os
import re
import hashlib
import subprocess
from ast import literal_eval


def load_config(config_path: str) -> Dict[str, Any]:
    """Loading a YAML Configuration File"""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def override_config(config: Dict[str, Any], overrides: List[str]) -> Dict[str, Any]:
    """Override config values from command line args like 'training.learning_rate=0.000001'"""
    for override in overrides:
        # Skip DeepSpeed/distributed training args (--local_rank, etc.)
        if override.startswith('--') or '=' not in override:
            continue
        key_path, value = override.split('=', 1)
        keys = key_path.split('.')
        
        # Navigate to nested dict
        target = config
        for key in keys[:-1]:
            target = target.setdefault(key, {})
        
        # Convert value to appropriate type
        final_key = keys[-1]
        old_value = target.get(final_key)
        
        # Try literal_eval for lists/dicts (e.g., "[16,32,32]")
        if isinstance(old_value, (list, dict)) or value.startswith(('[', '{')):
            try:
                target[final_key] = literal_eval(value)
            except (ValueError, SyntaxError):
                target[final_key] = value
        elif isinstance(old_value, bool):
            target[final_key] = value.lower() in ('true', '1', 'yes')
        elif isinstance(old_value, int):
            target[final_key] = int(value)
        elif isinstance(old_value, float):
            target[final_key] = float(value)
        elif value.lower() == "null":
            target[final_key] = None
        else:
            target[final_key] = value
    
    return config


LLADA_REASONING_PREFIX = "Reasoning:\n"
_OPEN_TAG_RE = re.compile(r"<think>\s*", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"\s*</think>\s*", re.IGNORECASE)

# 按 prompt 过滤占位步骤（与 scripts/convert_distill_to_d3llm_parquet.py 对齐）
# episode done 后 env_manager 将此固定字符串作为后续步骤的 built_observation
try:
    from agent_system.environments.prompts.webshop import WEBSHOP_DONE_TEMPLATE as _WEBSHOP_DONE_TEMPLATE
except ImportError:
    _WEBSHOP_DONE_TEMPLATE = "Reply: <reason> done </reason> <action> done </action>"
_DONE_PROMPT_NORM = " ".join(_WEBSHOP_DONE_TEMPLATE.split()).lower()


def _find_repo_root() -> Optional[str]:
    for key in ("VERL_AGENT_ROOT", "REPO_ROOT"):
        root = os.environ.get(key)
        if root and os.path.isdir(os.path.join(root, "agent_system")):
            return root
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(10):
        if os.path.isdir(os.path.join(d, "agent_system")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return None


def _webshop_prompt_constants():
    repo_root = _find_repo_root()
    if repo_root and repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from agent_system.environments.prompts.webshop import (  # noqa: WPS433
            WEBSHOP_DEFAULT_THINKING_INSTRUCTION,
            WEBSHOP_DEFAULT_THINKING_INSTRUCTION_NO_TAGS,
        )
        return WEBSHOP_DEFAULT_THINKING_INSTRUCTION, WEBSHOP_DEFAULT_THINKING_INSTRUCTION_NO_TAGS
    except ImportError:
        tagged = (
            "You should first reason step-by-step about the current situation, then think carefully "
            "which admissible action best advances the shopping goal. This reasoning process MUST be "
            "enclosed within <think> </think> tags. "
        )
        no_tags = (
            "You should first reason step-by-step about the current situation, then think carefully "
            "which admissible action best advances the shopping goal. "
        )
        return tagged, no_tags


def qwen_user_prompt_to_llada(text: str) -> str:
    tagged, no_tags = _webshop_prompt_constants()
    if not text:
        return text
    out = text.replace(tagged, no_tags)
    out = re.sub(
        r"This reasoning process MUST be enclosed within\s*<think>\s*</think>\s*tags\.\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return out


def qwen_response_to_llada(text: str) -> str:
    if not text:
        return text
    out = _OPEN_TAG_RE.sub(LLADA_REASONING_PREFIX, text)
    out = _CLOSE_TAG_RE.sub("\n", out)
    if LLADA_REASONING_PREFIX not in out and not out.lstrip().startswith("Reasoning:"):
        if "<action>" in out and not out.lstrip().startswith("<action>"):
            out = LLADA_REASONING_PREFIX + out.lstrip()
    return out.strip()


def _normalize_steps(steps) -> List[Dict[str, Any]]:
    if hasattr(steps, "tolist"):
        steps = steps.tolist()
    return list(steps)


def _is_done_prompt(text: str) -> bool:
    """Episode ended: env replaces observation with WEBSHOP_DONE_TEMPLATE."""
    normalized = " ".join((text or "").split()).lower()
    return bool(normalized) and normalized == _DONE_PROMPT_NORM


def webshop_steps_to_pairs(steps) -> List[Dict[str, str]]:
    """Pair each raw_response with the *previous* step's built_observation.

    History JSONL logs at end of env.step(): built_text is the prompt for the
    *next* model call, while raw_response is the output for the *current* call.
    Filters done-placeholder steps by prompt (not response) for robustness.
    """
    steps = _normalize_steps(steps)
    pairs: List[Dict[str, str]] = []
    for i, step in enumerate(steps):
        raw = step.get("raw_response")
        if not raw or i == 0:
            continue
        built = steps[i - 1].get("built_observation") or ""
        if not built or _is_done_prompt(built):
            continue
        response = qwen_response_to_llada(str(raw))
        pairs.append(
            {
                "question": qwen_user_prompt_to_llada(built),
                "llm_response": response,
            }
        )
    return pairs


def _sample_token_length(tokenizer, question: str, response: str) -> int:
    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + response + tokenizer.eos_token
    return len(tokenizer(full_text, add_special_tokens=False)["input_ids"])


def load_training_dataset(tokenizer, data_cfg: Dict[str, Any], distill_config: Dict[str, Any]):
    from datasets import Dataset as HFDataset, load_dataset
    import pandas as pd

    train_files = data_cfg.get("train_files")
    if not train_files:
        return load_dataset("Zigeng/dParallel_LLaDA_Distill_Data", split="train")

    train_path = os.path.expanduser(train_files)
    print(f"Loading local dataset: {train_path}")
    pdf = pd.read_parquet(train_path)

    data_format = data_cfg.get("format", "auto")
    if data_format == "auto":
        data_format = "webshop_distill" if "steps" in pdf.columns else "qa"

    if data_format == "qa":
        print(f"Dataset format: question/llm_response ({len(pdf)} rows)")
        return HFDataset.from_pandas(pdf[["question", "llm_response"]], preserve_index=False)

    max_length = distill_config.get("max_length", 512)
    success_only = data_cfg.get("success_only", True)
    if success_only and "won" in pdf.columns:
        pdf = pdf[pdf["won"] == True]  # noqa: E712
        print(f"Success trajectories: {len(pdf)}")

    rows: List[Dict[str, str]] = []
    dropped = 0
    for _, row in pdf.iterrows():
        for pair in webshop_steps_to_pairs(row["steps"]):
            if _sample_token_length(tokenizer, pair["question"], pair["llm_response"]) > max_length:
                dropped += 1
                continue
            rows.append(pair)

    print(
        f"WebShop distill: kept {len(rows)} step samples, "
        f"dropped {dropped} over max_length={max_length}"
    )
    return HFDataset.from_list(rows)


def get_deepspeed_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Creating a DeepSpeed ​​Configuration"""
    return {
        "train_batch_size": "auto",
        "train_micro_batch_size_per_gpu": "auto",
        "gradient_accumulation_steps": "auto",
        "gradient_clipping": "auto",
        "zero_allow_untested_optimizer": True,
        "bf16": {"enabled": "auto"},
        "zero_optimization": {
            "stage": 2,
            "offload_optimizer": {"device": "cpu", "pin_memory": True},
            "allgather_partitions": True,
            "allgather_bucket_size": 2e8,
            "reduce_scatter": True,
            "reduce_bucket_size": 2e8,
            "overlap_comm": True,
            "contiguous_gradients": True,
        },
    }


def prepare_model(config: Dict[str, Any]):
    """Prepare the model and tokenizer with optional LoRA"""

    # Setting torch dtype
    torch_dtype = getattr(torch, config["model"]["torch_dtype"])

    # Loading the model and tokenizer
    model = AutoModel.from_pretrained(
        config["model"]["name"],
        torch_dtype=torch_dtype,
        trust_remote_code=config["model"]["trust_remote_code"],
    )

    tokenizer = AutoTokenizer.from_pretrained(
        config["model"]["name"], trust_remote_code=config["model"]["trust_remote_code"]
    )

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Apply LoRA if enabled in config
    lora_config_dict = config.get("lora")
    if lora_config_dict and lora_config_dict.get("enabled", False):
        print("=" * 80)
        print("Applying LoRA configuration...")
        lora_config = LoraConfig(
            r=lora_config_dict.get("r", 128),
            lora_alpha=lora_config_dict.get("lora_alpha", 128),
            target_modules=lora_config_dict.get("target_modules", [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]),
            lora_dropout=lora_config_dict.get("lora_dropout", 0.0),
            bias=lora_config_dict.get("bias", "none"),
            task_type=lora_config_dict.get("task_type", "CAUSAL_LM")
        )
        
        model = get_peft_model(model, lora_config)
        
        # Print the number of trainable parameters
        model.print_trainable_parameters()
        print("=" * 80)
    else:
        print("=" * 80)
        print("LoRA is disabled. Training full model.")
        print("=" * 80)

    return model, tokenizer


def select_trajectory_by_ratio(trajectories, mask_ratio, mask_token_id, block_start, block_end):
    """Select the trajectory step with mask ratio closest to target mask ratio in the current block
    
    Args:
        trajectories: List of trajectory steps (each step is a full sequence)
        mask_ratio: Target mask ratio [0, 1]
        mask_token_id: ID of the mask token (not used, kept for API compatibility)
        block_start: Start index of current block
        block_end: End index of current block
    
    Returns:
        The trajectory step with the closest mask ratio in the specified block region
    """
    if not trajectories or len(trajectories) == 0:
        return None
    
    # Trajectories are ordered by denoising steps (from fully masked to clean)
    # Calculate the target number of unmasked tokens in the current block
    block_length = block_end - block_start
    num_unmasked = int((1 - mask_ratio) * block_length)
    
    # Map to trajectory index: earlier steps have more masks
    # index = block_start + number of unmasked tokens so far
    target_idx = block_start + num_unmasked
    target_idx = min(target_idx, len(trajectories) - 1)
    
    return trajectories[target_idx]


def naive_random_mask(trajectories, mask_ratio, mask_token_id, block_start, block_end):
    """Baseline: randomly mask final trajectory by mask_ratio in specified block region"""
    
    return None


def forward_process_with_trajectory(
    input_ids,
    prompt_lengths,
    trajectory_batch,
    full_lengths=None,
    mask_token_id=126336,
    block_size=32,
    mask_ratio=0.5,  # Changed from min/max_ratio to single mask_ratio (progressively increases during training)
    use_naive_random_mask=False,
    use_complementary_loss=False,
    eps=1e-3,
    debug=False,
):
    """Block-wise semi-autoregressive forward masking using teacher trajectories

    Key points:
    1. Randomly select a block to predict
    2. Sample a trajectory step based on mask ratio (for the current block)
    3. Use trajectory tokens for the current block region
    4. Compute loss ONLY on masked positions in the current block
    
    Args:
        mask_ratio: The mask ratio for current training step (linearly increases from min to max during training)
        use_naive_random_mask: If True, use naive random masking baseline instead of trajectory selection
        use_complementary_loss: If True, also return complementary masked batch for dParallel loss
    """
    b, l = input_ids.shape
    device = input_ids.device

    noisy_batch = input_ids.clone()  # Start with clean target
    noisy_batch_rev = input_ids.clone() if use_complementary_loss else None
    masked_indices = torch.zeros_like(input_ids, dtype=torch.bool)
    masked_indices_rev = torch.zeros_like(input_ids, dtype=torch.bool) if use_complementary_loss else None

    for i in range(b):
        prompt_len = prompt_lengths[i].item()
        # response_len = l - prompt_len
        full_len = full_lengths[i].item() if full_lengths is not None else l
        response_len = full_len - prompt_len

        if response_len > 0:
            max_blocks = response_len // block_size

            # Randomly select which block to predict
            num_blocks_to_mask = random.randint(0, max_blocks)
            num_tokens_to_mask = num_blocks_to_mask * block_size

            mask_start = prompt_len + num_tokens_to_mask
            if num_blocks_to_mask == max_blocks:
                mask_end = l
            else:
                mask_end = mask_start + block_size

            # Use the progressive mask ratio (increases linearly during training)
            t = mask_ratio

            # Get trajectory using selected method
            traj_fn = naive_random_mask if use_naive_random_mask else select_trajectory_by_ratio
            traj_step = traj_fn(
                trajectory_batch[i], t, mask_token_id, mask_start - prompt_len, mask_end - prompt_len
            )

            # DEBUG
            if debug and i == 0:
                print(f"Sample {i}: traj_step is None: {traj_step is None}")
                if traj_step is not None:
                    print(f"  traj_step length: {len(traj_step)}, expected: {l}")
                print(f"  trajectory_batch[{i}] type: {type(trajectory_batch[i])}")
                if trajectory_batch[i]:
                    print(f"  trajectory_batch[{i}] length: {len(trajectory_batch[i])}")

            # Check if trajectory length matches input length
            seg_len = mask_end - mask_start
            if traj_step is not None and len(traj_step) == l:
                # Use trajectory to extract mask information
                traj_tensor = torch.tensor(traj_step, device=device, dtype=torch.long)
                seg_mask = (traj_tensor[mask_start:mask_end] == mask_token_id)
            else:
                # Fallback to random masking
                p_mask = (1 - eps) * t + eps
                seg_mask = torch.rand(seg_len, device=device) < p_mask
            
            # Apply mask
            masked_indices[i, mask_start:mask_end] = seg_mask
            if use_complementary_loss:
                masked_indices_rev[i, mask_start:mask_end] = ~seg_mask
            
            noisy_batch[i, mask_start:mask_end] = torch.where(
                masked_indices[i, mask_start:mask_end], mask_token_id, input_ids[i, mask_start:mask_end]
            )
            if use_complementary_loss:
                noisy_batch_rev[i, mask_start:mask_end] = torch.where(
                    masked_indices_rev[i, mask_start:mask_end], mask_token_id, input_ids[i, mask_start:mask_end]
                )
            
            # Mask future tokens
            noisy_batch[i, mask_end:l] = mask_token_id
            if use_complementary_loss:
                noisy_batch_rev[i, mask_end:l] = mask_token_id

    if use_complementary_loss:
        return noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev
    return noisy_batch, masked_indices


class DLMTrainer(Trainer):
    """Trajectory-based diffusion language model trainer"""

    def __init__(
        self,
        mask_token_id=126336,
        temperature=0.5,
        entropy_weight=2.0,
        progressive_block_sizes=None,
        min_mask_ratio=0.2,
        max_mask_ratio=0.8,
        use_naive_random_mask=False,
        use_complementary_loss=False,
        trajectory_dataset=None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.mask_token_id = mask_token_id
        self.temperature = temperature
        self.entropy_weight = entropy_weight
        self.progressive_block_sizes = progressive_block_sizes or [32]
        self.current_block_size = self.progressive_block_sizes[0]
        self.min_mask_ratio = min_mask_ratio
        self.max_mask_ratio = max_mask_ratio
        self.use_naive_random_mask = use_naive_random_mask
        self.use_complementary_loss = use_complementary_loss
        self.trajectory_dataset = trajectory_dataset
    
    def get_current_block_size(self):
        """Calculate current block size based on epoch progress (linear interpolation)
        
        Within each epoch, block size increases linearly from progressive_block_sizes[i] 
        to progressive_block_sizes[i+1].
        
        Returns:
            int: Current block size
        """
        if self.state.epoch is None:
            return self.progressive_block_sizes[0]
        
        current_epoch = self.state.epoch
        num_epochs = self.args.num_train_epochs
        
        # Calculate which epoch interval we're in
        epoch_idx = int(current_epoch)
        epoch_idx = min(epoch_idx, len(self.progressive_block_sizes) - 1)
        
        # Get start and end block sizes for this epoch
        start_block_size = self.progressive_block_sizes[epoch_idx]
        
        # If this is the last epoch or only one block size, use current
        if epoch_idx >= len(self.progressive_block_sizes) - 1:
            return int(start_block_size)
        
        end_block_size = self.progressive_block_sizes[epoch_idx + 1]
        
        # Linear interpolation within the epoch
        epoch_progress = current_epoch - epoch_idx  # 0.0 to 1.0 within current epoch
        interpolated_size = start_block_size + epoch_progress * (end_block_size - start_block_size)
        
        return int(interpolated_size)
    
    def get_current_mask_ratio(self):
        """Calculate current mask ratio based on training progress (linear schedule)
        
        The mask ratio increases linearly from min_mask_ratio to max_mask_ratio
        over the course of training (based on global steps for finer granularity).
        
        Returns:
            float: Current mask ratio
        """
        # Use global_step for finer-grained progression (iteration-level)
        if self.state.max_steps > 0:
            current_step = self.state.global_step
            total_steps = self.state.max_steps
            
            # Linear interpolation from min to max
            progress = min(current_step / total_steps, 1.0)  # 0.0 to 1.0
            current_ratio = self.min_mask_ratio + progress * (self.max_mask_ratio - self.min_mask_ratio)
            
            return current_ratio
        else:
            # Fallback to min_mask_ratio if step info not available
            return self.min_mask_ratio
    
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
    
    def log(self, logs: Dict[str, float], *args, **kwargs) -> None:
        """Override log to add GPU statistics and current mask ratio"""
        # Get GPU stats and add to logs
        gpu_stats = self._get_gpu_stats()
        if gpu_stats:
            logs.update(gpu_stats)
            # # Print GPU stats for visibility
            # if self.state.global_step % self.args.logging_steps == 0:
            #     print(f"\n{'='*60}")
            #     print(f"GPU Stats (Step {self.state.global_step}):")
            #     print(f"  Num GPUs: {gpu_stats['num_gpus']}")
            #     print(f"  Avg Memory Used: {gpu_stats['gpu_memory_used_mb']:.0f} MB / {gpu_stats['gpu_memory_total_mb']:.0f} MB ({gpu_stats['gpu_memory_percent']:.1f}%)")
            #     print(f"  Avg GPU Utilization: {gpu_stats['gpu_utilization_percent']:.1f}%")
            #     print(f"{'='*60}\n")
        
        # Add current mask ratio and block size to logs
        logs['mask_ratio'] = self.get_current_mask_ratio()
        logs['block_size'] = self.get_current_block_size()
        
        # Call parent log method
        super().log(logs, *args, **kwargs)

    def _maybe_log_save_evaluate(self, tr_loss, grad_norm, model, trial, epoch, *args, **kwargs):
        """Override for logging and evaluation"""
        # Block size is now dynamically calculated per iteration via get_current_block_size()
        # No need to manually update self.current_block_size here
        
        return super()._maybe_log_save_evaluate(
            tr_loss, grad_norm, model, trial, epoch, *args, **kwargs
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        input_ids = inputs["input_ids"]
        prompt_lengths = inputs["prompt_lengths"]
        full_lengths = inputs["full_lengths"]
        sample_indices = inputs["sample_idx"]
        
        # Dynamically load trajectories from trajectory_dataset based on sample_idx
        trajectories = []
        for idx in sample_indices.cpu().tolist():
            if self.trajectory_dataset is not None and idx < len(self.trajectory_dataset):
                traj = self.trajectory_dataset[idx]["trajectory"]
            else:
                traj = []
            trajectories.append(traj)
        
        # Get current mask ratio (progressively increases from min to max during training)
        current_mask_ratio = self.get_current_mask_ratio()
        # uniform from [current_mask_ratio, self.max_mask_ratio]
        current_mask_ratio = random.uniform(current_mask_ratio, self.max_mask_ratio)

        # Get current block size (linearly interpolates within each epoch)
        current_block_size = self.get_current_block_size()

        # Block-wise semi-autoregressive forward masking with trajectory
        if self.use_complementary_loss:
            noisy_batch, noisy_batch_rev, masked_indices, masked_indices_rev = forward_process_with_trajectory(
                input_ids, prompt_lengths, trajectories, full_lengths,
                mask_token_id=self.mask_token_id, block_size=current_block_size,
                mask_ratio=current_mask_ratio, use_naive_random_mask=self.use_naive_random_mask,
                use_complementary_loss=True, debug=(self.state.global_step < 3),
            )
        else:
            noisy_batch, masked_indices = forward_process_with_trajectory(
                input_ids, prompt_lengths, trajectories, full_lengths,
                mask_token_id=self.mask_token_id, block_size=current_block_size,
                mask_ratio=current_mask_ratio, use_naive_random_mask=self.use_naive_random_mask,
                debug=(self.state.global_step < 3),
            )

        # DEBUG: Check if we have any masked tokens
        num_masked = masked_indices.sum().item()
        if num_masked == 0:
            print(f"\n{'='*60}")
            print(f"WARNING: No masked tokens found!")
            print(f"Input shape: {input_ids.shape}")
            print(f"Prompt lengths: {prompt_lengths.tolist()}")
            print(f"Num trajectories: {len(trajectories)}")
            if len(trajectories) > 0:
                print(f"First trajectory length: {len(trajectories[0]) if trajectories[0] else 'None'}")
                if trajectories[0] and len(trajectories[0]) > 0:
                    print(f"First trajectory[0] length: {len(trajectories[0][0]) if len(trajectories[0]) > 0 else 'Empty'}")
            print(f"{'='*60}\n")

        # Forward pass
        outputs = model(input_ids=noisy_batch)
        logits = outputs.logits

        # Cross-entropy loss on masked tokens
        ce_loss = 0.0 * logits.sum()
        if masked_indices.sum() > 0:
            token_loss = F.cross_entropy(logits[masked_indices], input_ids[masked_indices], reduction='none')
            ce_loss = torch.sum(token_loss) / masked_indices.sum()

        # Complementary CE loss
        ce_loss_rev = 0.0 * logits.sum()
        if self.use_complementary_loss and masked_indices_rev.sum() > 0:
            logits_rev = model(input_ids=noisy_batch_rev).logits
            ce_loss_rev = F.cross_entropy(logits_rev[masked_indices_rev], input_ids[masked_indices_rev])

        # Entropy regularization on correctly predicted masked tokens
        entropy_loss = 0.0 * logits.sum()
        if masked_indices.sum() > 0:
            probs = F.softmax(logits / self.temperature, dim=-1)
            H_tok = -(probs * torch.log(probs + 1e-12)).sum(dim=-1)
            correct_mask = (logits.argmax(dim=-1) == input_ids) & masked_indices
            num_correct = correct_mask.sum()
            if num_correct > 0:
                entropy_loss = (H_tok * correct_mask).sum() / num_correct

        total_loss = (ce_loss + ce_loss_rev + self.entropy_weight * entropy_loss) / 4.0

        return (total_loss, outputs) if return_outputs else total_loss


def main():
    # 1. Load configuration, model and tokenizer
    import os
    from datetime import datetime
    import shutil
    from zoneinfo import ZoneInfo

    config_name = os.environ.get("D3LLM_TRAIN_CONFIG", "d3llm_train.yaml")
    config_path = os.path.join(os.path.dirname(__file__), config_name)
    config = load_config(config_path)
    
    # Override config from command line args
    config = override_config(config, sys.argv[1:])
    
    # Save modified config as d3llm_train_used.yaml for backup
    used_config_path = os.path.join(os.path.dirname(__file__), "d3llm_train_used.yaml")
    with open(used_config_path, 'w', encoding='utf-8') as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)

    # Print all configuration parameters
    print(f"=" * 80)
    print("Configuration Parameters:")
    print(f"=" * 80)
    import json
    print(json.dumps(config, indent=2, ensure_ascii=False))
    print(f"=" * 80)
    
    # Get SLURM job ID if available
    slurm_job_id = os.environ.get("SLURM_JOB_ID", "local")
    
    # Use timestamp from environment (set by shell script) or generate new one
    timestamp = os.environ.get("TRAINING_TIMESTAMP")
    if not timestamp:
        # Fallback: generate timestamp if not provided by shell script
        san_diego_tz = ZoneInfo("America/Los_Angeles")
        timestamp = datetime.now(san_diego_tz).strftime("%m%d_%H%M%S")
    
    base_output_dir = config["training"]["output_dir"]
    output_dir = f"{base_output_dir}_{slurm_job_id}_{timestamp}"
    
    # Create W&B run name with the same format
    wandb_run_name = f"{os.path.basename(base_output_dir)}_{slurm_job_id}_{timestamp}"
    
    # Update config with timestamped output_dir and run_name
    config["training"]["output_dir"] = output_dir
    config["training"]["run_name"] = wandb_run_name
    
    print(f"=" * 80)
    print(f"SLURM Job ID: {slurm_job_id}")
    print(f"Output directory: {output_dir}")
    print(f"W&B Run name: {wandb_run_name}")
    print(f"=" * 80)
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Backup training code and config to output directory
    source_dir = os.path.dirname(__file__)  # d3llm_LLaDA/distill_2_training
    backup_dir = os.path.join(output_dir, "training_code_backup")
    
    print(f"Backing up training code from {source_dir} to {backup_dir}...")
    
    # Use copytree with dirs_exist_ok=True (Python 3.8+) to avoid removal issues
    try:
        shutil.copytree(source_dir, backup_dir, dirs_exist_ok=True)
        print(f"Training code backed up successfully!")
    except Exception as e:
        print(f"Warning: Failed to backup training code: {e}")
        print(f"Continuing with training anyway...")
    
    print(f"=" * 80)

    training_args = TrainingArguments(
        **config["training"],
        deepspeed=get_deepspeed_config(config),
        ddp_find_unused_parameters=False,
        label_names=["input_ids", "prompt_lengths", "full_lengths", "sample_idx"],
    )

    model, tokenizer = prepare_model(config)

    # 2. Load trajectory dataset and create mapping
    distill_config = config.get("distillation", {})
    trajectory_dataset_path = distill_config.get("trajectory_dataset_path")
    
    if trajectory_dataset_path:
        # Handle relative path from script directory (only if it's a local path)
        if not os.path.isabs(trajectory_dataset_path):
            potential_path = os.path.join(os.path.dirname(__file__), "..", trajectory_dataset_path)
            if os.path.exists(potential_path):
                trajectory_dataset_path = potential_path
        
        # Create cache directory and file path
        cache_dir = os.path.join(os.path.dirname(trajectory_dataset_path), "cache")
        os.makedirs(cache_dir, exist_ok=True)
        
        # Generate cache key based on dataset path and max_length (important for preprocessing)
        max_length = distill_config.get("max_length", 512)
        cache_params = f"{trajectory_dataset_path}_maxlen{max_length}".encode()
        cache_key = hashlib.md5(cache_params).hexdigest()
        cache_file = os.path.join(cache_dir, f"trajectory_preprocessed_{cache_key}.pkl")
        
        # Try to load preprocessed trajectory dataset from cache first
        if os.path.exists(cache_file):
            try:
                print(f"Loading preprocessed trajectory dataset from cache: {cache_file}")
                with open(cache_file, 'rb') as f:
                    trajectory_dataset = pickle.load(f)
                print(f"Successfully loaded {len(trajectory_dataset)} preprocessed samples from cache!")
            except Exception as e:
                print(f"Failed to load cache: {e}")
                print(f"Will process dataset from scratch...")
                trajectory_dataset = None
        else:
            print(f"Preprocessed cache not found at {cache_file}. Processing from scratch...")
            trajectory_dataset = None
        
        # If cache doesn't exist or failed to load, process dataset
        if trajectory_dataset is None:
            print(f"Loading trajectory dataset from {trajectory_dataset_path}...")
            # Support both local and remote HuggingFace datasets
            if os.path.isdir(trajectory_dataset_path):
                trajectory_dataset = load_from_disk(trajectory_dataset_path)
            else:
                trajectory_dataset = load_dataset(trajectory_dataset_path, split="train")
            
            # Filter correct samples only
            num_proc = distill_config.get("num_proc", 8)
            print(f"Filtering correct trajectory samples with {num_proc} processes...")
            # trajectory_dataset = trajectory_dataset.filter(lambda x: x["is_correct"], num_proc=num_proc)
            print(f"Loaded {len(trajectory_dataset)} correct trajectory samples")
            
            # Preprocess trajectory dataset: truncate and pad each step to max_length
            pad_token_id = tokenizer.eos_token_id
            mask_token_id = 126336
            
            def preprocess_trajectory_sample(examples):
                """Preprocess trajectory samples: truncate and pad each step to max_length"""
                processed_trajectories = []
                
                for traj in examples["trajectory"]:
                    if traj:
                        padded_traj = []
                        for step in traj:
                            if len(step) < max_length:
                                # Pad with mask_token to max_length
                                padding_length = max_length - len(step)
                                padded_step = step + [mask_token_id] * padding_length
                            else:
                                # Truncate to max_length
                                padded_step = step[:max_length]
                            padded_traj.append(padded_step)
                        processed_trajectories.append(padded_traj)
                    else:
                        processed_trajectories.append([])
                
                return {
                    "trajectory": processed_trajectories,
                }
            
            print(f"Preprocessing trajectories (truncate/pad to max_length={max_length})...")
            trajectory_dataset = trajectory_dataset.map(
                preprocess_trajectory_sample,
                batched=True,
                num_proc=num_proc,
                desc="Preprocessing trajectories"
            )
            print(f"Trajectory preprocessing completed!")
            
            # Save preprocessed dataset to cache
            try:
                print(f"Saving preprocessed trajectory dataset to cache: {cache_file}")
                with open(cache_file, 'wb') as f:
                    pickle.dump(trajectory_dataset, f)
                print(f"Preprocessed cache saved successfully!")
            except Exception as e:
                print(f"Warning: Failed to save cache: {e}")
        
        print(f"Preprocessed trajectory dataset ready with {len(trajectory_dataset)} samples")
    else:
        print(f"No trajectory dataset specified, using random masking")
        trajectory_dataset = None
    
    # 3. Load training dataset (WebShop distill steps or question/llm_response parquet)
    data_cfg = config.get("data", {})
    dataset = load_training_dataset(tokenizer, data_cfg, distill_config)
    
    # Limit dataset size for testing if max_samples is specified
    max_samples = distill_config.get("max_samples")
    if max_samples is not None and max_samples > 0:
        original_size = len(dataset)
        dataset = dataset.select(range(min(max_samples, original_size)))
        print(f"=" * 80)
        print(f"[Testing Mode] Limited dataset from {original_size} to {len(dataset)} samples")
        
        # Also limit trajectory dataset to match
        if trajectory_dataset is not None:
            traj_original_size = len(trajectory_dataset)
            trajectory_dataset = trajectory_dataset.select(range(min(max_samples, traj_original_size)))
            print(f"[Testing Mode] Limited trajectory dataset from {traj_original_size} to {len(trajectory_dataset)} samples")
        print(f"=" * 80)
    
    # 4. Check tokenized dataset cache
    cache_dir = os.path.join(os.path.dirname(__file__), "..", "cache")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Generate cache key based on dataset configuration
    cache_params = {
        "model_name": config["model"]["name"],
        "max_samples": max_samples,
        "max_length": distill_config.get("max_length", 512),
        "dataset_size": len(dataset),
        "train_files": data_cfg.get("train_files", ""),
        "data_format": data_cfg.get("format", "auto"),
        "success_only": data_cfg.get("success_only", True),
        "pairing": "prev_built",
    }
    cache_key_str = str(cache_params).encode()
    cache_key = hashlib.md5(cache_key_str).hexdigest()
    cache_file_tokenized = os.path.join(cache_dir, f"tokenized_dataset_{cache_key}.pkl")
    
    # Try to load tokenized dataset from cache
    if os.path.exists(cache_file_tokenized):
        try:
            print(f"=" * 80)
            print(f"Loading tokenized dataset from cache: {cache_file_tokenized}")
            with open(cache_file_tokenized, 'rb') as f:
                tokenized_dataset = pickle.load(f)
            print(f"Successfully loaded tokenized dataset with {len(tokenized_dataset)} samples from cache!")
            print(f"=" * 80)
        except Exception as e:
            print(f"Failed to load tokenized dataset cache: {e}")
            print(f"Will tokenize from scratch...")
            tokenized_dataset = None
    else:
        print(f"Tokenized dataset cache not found. Will tokenize from scratch...")
        tokenized_dataset = None
    
    # If cache doesn't exist or failed to load, perform tokenization
    if tokenized_dataset is None:
        # Format each sample, generate the complete text and record the number of tokens in the prompt section
        def format_example(example):
            texts = []
            prompt_lengths = []
            full_lengths = []
            
            for question, response in zip(example["question"], example["llm_response"]):
                # prompt text
                messages = [{"role": "user", "content": question}]
                prompt_text = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                
                # response text
                answer_text = response + tokenizer.eos_token
                
                # complete text
                full_text = prompt_text + answer_text
                texts.append(full_text)
                
                # Calculate the number of tokens in the prompt part
                prompt_token_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
                prompt_lengths.append(len(prompt_token_ids))
                
                # Calculate full text length
                full_token_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"]
                full_lengths.append(len(full_token_ids))
            
            return {"text": texts, "prompt_length": prompt_lengths, "full_length": full_lengths}
        
        print(f"Formatting dataset...")
        formatted_dataset = dataset.map(
            format_example,
            batched=True,
        )
        
        def tokenize_function(examples, indices):
            tokenized = tokenizer(
                examples["text"],
                truncation=True,
                padding=False,
                max_length=distill_config.get("max_length", 512),
                add_special_tokens=False,
            )
            
            prompt_lengths, full_lengths = [], []
            for ids, raw_pl in zip(tokenized["input_ids"], examples["prompt_length"]):
                actual_len = len(ids)
                pl = min(int(raw_pl), actual_len - 1)
                pl = max(0, pl)
                prompt_lengths.append(pl)
                full_lengths.append(actual_len)
            tokenized["prompt_lengths"] = prompt_lengths
            tokenized["full_lengths"] = full_lengths

            # Store original dataset index for dynamic trajectory loading during training
            tokenized["sample_idx"] = list(indices)
            
            return tokenized
        
        print(f"Tokenizing dataset...")
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            with_indices=True,
        )
        
        # Save tokenized dataset to cache
        try:
            print(f"Saving tokenized dataset to cache: {cache_file_tokenized}")
            with open(cache_file_tokenized, 'wb') as f:
                pickle.dump(tokenized_dataset, f)
            print(f"Tokenized dataset cache saved successfully!")
        except Exception as e:
            print(f"Warning: Failed to save tokenized dataset cache: {e}")
    
    from dataclasses import dataclass
    from typing import Dict, List, Any
    
    @dataclass
    class MaskDiffusionDataCollator:
        tokenizer: Any
        max_length: int = 512
        
        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
            input_ids = [torch.tensor(f["input_ids"]) for f in features]
            prompt_lengths = [f["prompt_lengths"] for f in features]
            full_lengths = [f["full_lengths"] for f in features]
            sample_indices = [f["sample_idx"] for f in features]
            
            target_length = self.max_length
            
            pad_token_id = self.tokenizer.eos_token_id
            
            # right padding
            padded_input_ids = []
            for ids in input_ids:
                current_length = len(ids)
                if current_length < target_length:
                    # Right padding with EOS token
                    padding_length = target_length - current_length
                    padded_ids = torch.cat([
                        ids,
                        torch.full((padding_length,), pad_token_id, dtype=ids.dtype)
                    ])
                else:
                    # Truncate to target_length
                    padded_ids = ids[:target_length]
                
                padded_input_ids.append(padded_ids)
            
            batch = {
                "input_ids": torch.stack(padded_input_ids),
                "prompt_lengths": torch.tensor(prompt_lengths, dtype=torch.long),
                "full_lengths": torch.tensor(full_lengths, dtype=torch.long),
                "sample_idx": torch.tensor(sample_indices, dtype=torch.long),
            }
            
            return batch
    
    data_collator_fixed = MaskDiffusionDataCollator(
        tokenizer=tokenizer,
        max_length=distill_config.get("max_length", 512)
    )

    # 5. Create trainer and train
    progressive_block_sizes = distill_config.get("progressive_block_sizes", [32])
    num_epochs = config["training"]["num_train_epochs"]

    # Validate progressive_block_sizes length
    if len(progressive_block_sizes) != num_epochs:
        print(f"Warning: progressive_block_sizes length ({len(progressive_block_sizes)}) != num_epochs ({num_epochs})")
        print(f"Using last block size ({progressive_block_sizes[-1]}) for remaining epochs")
        progressive_block_sizes = progressive_block_sizes + [
            progressive_block_sizes[-1]
        ] * (num_epochs - len(progressive_block_sizes))

    # 6. DLM Trainer
    trainer = DLMTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator_fixed,
        mask_token_id=126336,
        temperature=distill_config.get("temperature", 0.5),
        entropy_weight=distill_config.get("entropy_weight", 2.0),
        progressive_block_sizes=progressive_block_sizes,
        min_mask_ratio=distill_config.get("min_mask_ratio", 0.2),
        max_mask_ratio=distill_config.get("max_mask_ratio", 0.8),
        use_naive_random_mask=distill_config.get("use_naive_random_mask", False),
        use_complementary_loss=distill_config.get("use_complementary_loss", False),
        trajectory_dataset=trajectory_dataset,  # Pass trajectory_dataset for dynamic loading
    )

    print(f"Training with progressive block sizes: {trainer.progressive_block_sizes}")
    print(f"Starting with block size: {trainer.current_block_size}")
    print(f"Progressive mask ratio: [{trainer.min_mask_ratio}, {trainer.max_mask_ratio}]")
    print(f"Temperature: {trainer.temperature}, Entropy weight: {trainer.entropy_weight}")

    # 6. start training
    trainer.train()


if __name__ == "__main__":
    main()
