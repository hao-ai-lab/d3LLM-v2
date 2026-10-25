"""Convert the SDAR SFT data for the Qwen baselines: <reason> / </reason> -> <think> / </think>.

Writes alfworld_sft_think.jsonl, webshop_sft_think.jsonl and dataset_info.json to dLLM_sft/AR/work/.
"""
import json
from pathlib import Path

AR_DIR = Path(__file__).resolve().parent
SOURCE = AR_DIR.parent / "data/sdar"
WORK = AR_DIR / "work"

WORK.mkdir(exist_ok=True)
info = {}
for task in ("alfworld", "webshop"):
    lines = []
    for line in (SOURCE / f"{task}_sft.jsonl").read_text().splitlines():
        row = json.loads(line)
        for message in row["messages"]:
            message["content"] = message["content"].replace("<reason>", "<think>").replace("</reason>", "</think>")
        lines.append(json.dumps(row, ensure_ascii=False))
    (WORK / f"{task}_sft_think.jsonl").write_text("\n".join(lines) + "\n")
    info[f"{task}_sft_think"] = {
        "file_name": f"{task}_sft_think.jsonl", "formatting": "sharegpt",
        "columns": {"messages": "messages"},
        "tags": {"role_tag": "role", "content_tag": "content",
                 "user_tag": "user", "assistant_tag": "assistant"},
    }
    print(f"{task}: {len(lines)} examples")
(WORK / "dataset_info.json").write_text(json.dumps(info, indent=2) + "\n")
