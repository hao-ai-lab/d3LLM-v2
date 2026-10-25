#!/usr/bin/env python3
"""Convert SDAR WebShop ShareGPT JSONL to d3LLM LLaDA QA parquet."""

import argparse
import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer


WEBSHOP_DEFAULT_THINKING_INSTRUCTION_NO_TAGS = (
    "You should first reason step-by-step about the current situation, then think "
    "carefully which admissible action best advances the shopping goal. "
)
WEBSHOP_SDAR_THINKING_INSTRUCTION = (
    "You should first reason step-by-step about the current situation, then think "
    "carefully which admissible action best advances the shopping goal. This reasoning "
    "process MUST be enclosed within <reason> </reason> tags. "
)


REASON_OPEN_RE = re.compile(r"<(?:reason|think)>\s*", re.IGNORECASE)
REASON_CLOSE_RE = re.compile(r"\s*</(?:reason|think)>\s*", re.IGNORECASE)
TAGGED_INSTRUCTION_RE = re.compile(
    r"This reasoning process MUST be enclosed within\s*"
    r"<(?:reason|think)>\s*</(?:reason|think)>\s*tags\.\s*",
    re.IGNORECASE,
)


def convert_question(text: str, reason_format: str = "prefix") -> str:
    if reason_format == "tags":
        return text
    text = text.replace(
        WEBSHOP_SDAR_THINKING_INSTRUCTION,
        WEBSHOP_DEFAULT_THINKING_INSTRUCTION_NO_TAGS,
    )
    return TAGGED_INSTRUCTION_RE.sub("", text)


def convert_response(text: str, reason_format: str = "prefix") -> str:
    if reason_format == "tags":
        return text.strip()
    converted = REASON_OPEN_RE.sub("Reasoning:\n", text)
    converted = REASON_CLOSE_RE.sub("\n", converted).strip()
    if not converted.startswith("Reasoning:"):
        converted = f"Reasoning:\n{converted}"
    return converted


def token_length(tokenizer, question: str, response: str) -> int:
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )
    full_text = prompt + response + tokenizer.eos_token
    return len(tokenizer(full_text, add_special_tokens=False)["input_ids"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="SDAR ShareGPT JSONL")
    parser.add_argument("--output", required=True, help="Output QA parquet")
    parser.add_argument("--model-name", required=True, help="LLaDA tokenizer path")
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument(
        "--reason-format",
        choices=("prefix", "tags"),
        default="prefix",
        help="Use LLaDA 'Reasoning:' prefix or preserve SDAR <reason> tags",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_path = Path(args.output).expanduser().resolve()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        trust_remote_code=True,
    )

    rows = []
    dropped = 0
    max_seen = 0
    with input_path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            sample = json.loads(line)
            messages = sample.get("messages")
            if (
                not isinstance(messages, list)
                or len(messages) != 2
                or messages[0].get("role") != "user"
                or messages[1].get("role") != "assistant"
            ):
                raise ValueError(
                    f"Line {line_number}: expected exactly one user/assistant pair"
                )

            question = convert_question(
                str(messages[0].get("content", "")), args.reason_format
            )
            response = convert_response(
                str(messages[1].get("content", "")), args.reason_format
            )
            if not question.strip() or not response.strip():
                raise ValueError(f"Line {line_number}: empty question or response")
            if args.reason_format == "tags" and (
                "<reason>" not in question.lower()
                or "<reason>" not in response.lower()
                or "</reason>" not in response.lower()
            ):
                raise ValueError(
                    f"Line {line_number}: missing required <reason> tags"
                )

            length = token_length(tokenizer, question, response)
            max_seen = max(max_seen, length)
            if length > args.max_length:
                dropped += 1
                continue
            rows.append({"question": question, "llm_response": response})

    if not rows:
        raise RuntimeError("No samples remained after conversion")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows, columns=["question", "llm_response"]).to_parquet(
        output_path,
        index=False,
    )
    print(
        f"Converted {len(rows)} samples; dropped={dropped}; "
        f"max_tokens_seen={max_seen}; output={output_path}"
    )


if __name__ == "__main__":
    main()
