#!/usr/bin/env python3
"""Validate that a merged Hugging Face model directory is complete."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def require_nonempty(path: Path) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing or empty file: {path}")


def validate(model_dir: Path) -> None:
    require_nonempty(model_dir / "config.json")
    require_nonempty(model_dir / "tokenizer_config.json")
    tokenizer_json = model_dir / "tokenizer.json"
    if tokenizer_json.exists():
        require_nonempty(tokenizer_json)
    else:
        require_nonempty(model_dir / "vocab.json")
        require_nonempty(model_dir / "merges.txt")

    index_path = model_dir / "model.safetensors.index.json"
    single_shard = model_dir / "model.safetensors"
    if index_path.exists():
        require_nonempty(index_path)
        index = json.loads(index_path.read_text(encoding="utf-8"))
        shards = set(index.get("weight_map", {}).values())
        if not shards:
            raise ValueError(f"empty weight_map: {index_path}")
        for shard in shards:
            require_nonempty(model_dir / shard)
    else:
        require_nonempty(single_shard)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("model_dir", type=Path)
    args = parser.parse_args()
    validate(args.model_dir)
    print(f"OK: {args.model_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
