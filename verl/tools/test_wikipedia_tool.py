"""Interactive test for WikipediaTool.

Usage:
    python verl/tools/test_wikipedia_tool.py [query]

If no query is given on the command line, the script will prompt for one.
"""

import asyncio
import json
import sys

from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.tools.wikipedia_tool import WikipediaTool

_TOOL_SCHEMA = OpenAIFunctionToolSchema(**{
    "type": "function",
    "function": {
        "name": "wiki_search",
        "description": "Searches Wikipedia.",
        "parameters": {
            "type": "object",
            "properties": {
                "query_list": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "list of queries to search",
                }
            },
            "required": ["query_list"],
        },
    },
})

_CONFIG = {
    "top_k_results": 3,
    "doc_content_chars_max": 4000,
    "lang": "en",
    "timeout": 30,
}


async def main(query: str):
    tool = WikipediaTool(_CONFIG, _TOOL_SCHEMA)
    instance_id = await tool.create()

    print(f"\n[tool call] wiki_search(query_list=[{query!r}])\n")
    result_text, reward, metrics = await tool.execute(instance_id, {"query_list": [query]})

    data = json.loads(result_text)
    print(data["result"])
    print(f"\n[metrics] {metrics}")

    await tool.release(instance_id)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        q = " ".join(sys.argv[1:])
    else:
        q = input("Enter search query: ").strip()

    asyncio.run(main(q))
