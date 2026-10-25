# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import json
import logging
import os
from typing import Any, Optional, Tuple
from uuid import uuid4

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def _search_wikipedia_sync(query: str, top_k: int, doc_content_chars_max: int, lang: str) -> str:
    """Synchronous Wikipedia search using LangChain's WikipediaAPIWrapper."""
    from langchain_community.utilities import WikipediaAPIWrapper

    wrapper = WikipediaAPIWrapper(
        lang=lang,
        top_k_results=top_k,
        load_all_available_meta=False,
        doc_content_chars_max=doc_content_chars_max,
    )
    return wrapper.run(query)


class WikipediaTool(BaseTool):
    """Online Wikipedia search tool using LangChain's WikipediaAPIWrapper.

    Requires: pip install langchain langchain-community wikipedia

    Each call to execute() accepts a list of queries (query_list) and returns
    the concatenated Wikipedia search results for all queries.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the Wikipedia search
        calc_reward: Return accumulated results (no scalar reward)
        release: Release the tool instance
    """

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """Initialize WikipediaTool.

        Config keys:
            top_k_results (int): Number of Wikipedia articles to return per query. Default: 3
            doc_content_chars_max (int): Max chars per article. Default: 4000
            lang (str): Wikipedia language code. Default: "en"
            timeout (float): Per-query timeout in seconds. Default: 30
        """
        super().__init__(config, tool_schema)
        self._instance_dict: dict = {}

        self.top_k = int(config.get("top_k_results", 3))
        self.doc_content_chars_max = int(config.get("doc_content_chars_max", 4000))
        self.lang = config.get("lang", "en")
        self.timeout = float(config.get("timeout", 30))

        # Eagerly verify the dependency is available
        try:
            from langchain_community.utilities import WikipediaAPIWrapper  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "WikipediaTool requires langchain-community and wikipedia. "
                "Install them with: pip install langchain langchain-community wikipedia"
            ) from e

        logger.info(f"Initialized WikipediaTool: top_k={self.top_k}, lang={self.lang}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> str:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {"results": []}
        return instance_id

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> Tuple[str, float, dict]:
        """Execute Wikipedia search for all queries in query_list.

        Args:
            instance_id: Tool instance ID
            parameters: Must contain "query_list" (list of str)

        Returns:
            (result_text, reward, metrics)
        """
        query_list = parameters.get("query_list")
        if not query_list or not isinstance(query_list, list):
            msg = "Error: 'query_list' is missing, empty, or not a list."
            logger.error(f"[WikipediaTool] {msg} parameters={parameters}")
            return json.dumps({"result": msg}), 0.0, {}

        loop = asyncio.get_event_loop()
        results = []
        metrics = {"query_count": 0, "errors": 0}

        for query in query_list:
            try:
                raw = await asyncio.wait_for(
                    loop.run_in_executor(
                        None,
                        _search_wikipedia_sync,
                        query,
                        self.top_k,
                        self.doc_content_chars_max,
                        self.lang,
                    ),
                    timeout=self.timeout,
                )
                results.append(f"Query: {query}\n{raw}")
                metrics["query_count"] += 1
            except asyncio.TimeoutError:
                msg = f"Timeout searching Wikipedia for: {query}"
                logger.warning(f"[WikipediaTool] {msg}")
                results.append(f"Query: {query}\n{msg}")
                metrics["errors"] += 1
            except Exception as e:
                msg = f"Error searching Wikipedia for '{query}': {e}"
                logger.warning(f"[WikipediaTool] {msg}")
                results.append(f"Query: {query}\n{msg}")
                metrics["errors"] += 1

        result_text = json.dumps({"result": "\n\n".join(results)})

        if instance_id in self._instance_dict:
            self._instance_dict[instance_id]["results"].append(result_text)

        return result_text, 0.0, metrics

    async def calc_reward(self, instance_id: str, **kwargs):
        return self._instance_dict.get(instance_id, {}).get("results", [])

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
