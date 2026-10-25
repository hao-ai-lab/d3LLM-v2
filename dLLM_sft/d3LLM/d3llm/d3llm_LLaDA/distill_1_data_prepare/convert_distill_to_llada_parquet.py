"""Convert WebShop distill_trajectories.parquet (steps format) to LLaDA training parquet.

Reads a parquet with 'steps' column, pairs each raw_response with the previous
built_observation, applies Qwen→LLaDA text conversion (<think> tags → Reasoning:
prefix), filters by max_length using the LLaDA tokenizer, and writes a parquet
with 'question' and 'llm_response' columns ready for d3llm_llada_train.py.

Usage:
    python distill_1_data_prepare/convert_distill_to_llada_parquet.py \\
        --input  /path/to/distill_trajectories.parquet \\
        --output /path/to/d3llm_llada_train.parquet \\
        --model_name GSAI-ML/LLaDA-8B-Instruct \\
        --max_length 4096 \\
        [--no_success_only]
"""
import argparse
import os
import re
import sys


def _find_repo_root() -> str:
    d = os.path.dirname(os.path.abspath(__file__))
    for _ in range(8):
        if os.path.isdir(os.path.join(d, "agent_system")):
            return d
        parent = os.path.dirname(d)
        if parent == d:
            break
        d = parent
    return os.path.dirname(os.path.abspath(__file__))


_REPO_ROOT = _find_repo_root()
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ── LLaDA text-format constants ─────────────────────────────────────────────
LLADA_REASONING_PREFIX = "Reasoning:\n"
_OPEN_TAG_RE = re.compile(r"<think>\s*", re.IGNORECASE)
_CLOSE_TAG_RE = re.compile(r"\s*</think>\s*", re.IGNORECASE)

try:
    from agent_system.environments.prompts.webshop import WEBSHOP_DONE_TEMPLATE as _WEBSHOP_DONE_TEMPLATE
except ImportError:
    _WEBSHOP_DONE_TEMPLATE = "Reply: <reason> done </reason> <action> done </action>"
_DONE_PROMPT_NORM = " ".join(_WEBSHOP_DONE_TEMPLATE.split()).lower()


def _find_webshop_prompt_constants():
    try:
        from agent_system.environments.prompts.webshop import (
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


_TAGGED_INSTRUCTION, _NO_TAGS_INSTRUCTION = _find_webshop_prompt_constants()


def qwen_user_prompt_to_llada(text: str) -> str:
    """Remove <think> tag instructions from user prompt."""
    if not text:
        return text
    out = text.replace(_TAGGED_INSTRUCTION, _NO_TAGS_INSTRUCTION)
    out = re.sub(
        r"This reasoning process MUST be enclosed within\s*<think>\s*</think>\s*tags\.\s*",
        "",
        out,
        flags=re.IGNORECASE,
    )
    return out


def qwen_response_to_llada(text: str) -> str:
    """Convert <think>...</think> to Reasoning:\\n...\\n in model response."""
    if not text:
        return text
    out = _OPEN_TAG_RE.sub(LLADA_REASONING_PREFIX, text)
    out = _CLOSE_TAG_RE.sub("\n", out)
    if LLADA_REASONING_PREFIX not in out and not out.lstrip().startswith("Reasoning:"):
        if "<action>" in out and not out.lstrip().startswith("<action>"):
            out = LLADA_REASONING_PREFIX + out.lstrip()
    return out.strip()


def _is_done_prompt(text: str) -> bool:
    normalized = " ".join((text or "").split()).lower()
    return bool(normalized) and normalized == _DONE_PROMPT_NORM


def webshop_steps_to_pairs(steps):
    """Pair raw_response with the previous step's built_observation.

    Applies Qwen→LLaDA text conversion on both question and llm_response.
    """
    if hasattr(steps, "tolist"):
        steps = steps.tolist()
    steps = list(steps)
    pairs = []
    for i, step in enumerate(steps):
        raw = step.get("raw_response")
        if not raw or i == 0:
            continue
        built = steps[i - 1].get("built_observation") or ""
        if not built or _is_done_prompt(built):
            continue
        pairs.append({
            "question": qwen_user_prompt_to_llada(built),
            "llm_response": qwen_response_to_llada(str(raw)),
        })
    return pairs


def sample_token_length(tokenizer, question: str, response: str) -> int:
    messages = [{"role": "user", "content": question}]
    prompt_text = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True, tokenize=False
    )
    full_text = prompt_text + response + tokenizer.eos_token
    return len(tokenizer(full_text, add_special_tokens=False)["input_ids"])


def main():
    parser = argparse.ArgumentParser(
        description="Convert WebShop distill trajectories to LLaDA training parquet."
    )
    parser.add_argument("--input", required=True, help="Input parquet with 'steps' column")
    parser.add_argument("--output", required=True, help="Output parquet with question/llm_response columns")
    parser.add_argument("--model_name", default="GSAI-ML/LLaDA-8B-Instruct",
                        help="Tokenizer model name for length filtering")
    parser.add_argument("--max_length", type=int, default=4096,
                        help="Maximum token length; longer samples are dropped")
    parser.add_argument("--success_only", dest="success_only", action="store_true", default=True,
                        help="Keep only won==True episodes (default: True)")
    parser.add_argument("--no_success_only", dest="success_only", action="store_false",
                        help="Keep all episodes regardless of success")
    args = parser.parse_args()

    import pandas as pd
    from transformers import AutoTokenizer
    try:
        from tqdm import tqdm
    except ImportError:
        tqdm = None

    print(f"Loading tokenizer: {args.model_name}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)

    print(f"Loading input parquet: {args.input}")
    df = pd.read_parquet(args.input)
    print(f"  Total episodes: {len(df)}")

    if args.success_only and "won" in df.columns:
        df = df[df["won"] == True]  # noqa: E712
        print(f"  Success-only filter: {len(df)} episodes kept")

    rows = []
    dropped = 0
    iterable = df.iterrows()
    if tqdm is not None:
        iterable = tqdm(iterable, total=len(df), desc="Processing episodes", unit="ep")
    for _, row in iterable:
        for pair in webshop_steps_to_pairs(row["steps"]):
            if sample_token_length(tokenizer, pair["question"], pair["llm_response"]) > args.max_length:
                dropped += 1
                continue
            rows.append(pair)

    print(f"Step pairs kept: {len(rows)}, dropped (over max_length={args.max_length}): {dropped}")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    out_df = pd.DataFrame(rows, columns=["question", "llm_response"])
    out_df.to_parquet(args.output, index=False)
    size_kb = os.path.getsize(args.output) // 1024
    print(f"Saved → {args.output}  ({size_kb} KB, {len(rows)} rows)")


if __name__ == "__main__":
    main()
