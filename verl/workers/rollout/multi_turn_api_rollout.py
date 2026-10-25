"""
MultiTurnAPIRollout: Multi-turn rollout with tool calling via an OpenAI-compatible API.

Unlike APIRollout (which does single-turn only), this rollout:
- Drives the multi-turn loop itself (like SGLangRollout._async_rollout_a_request)
- Calls an external API server for each generation step via /v1/chat/completions
- Supports OpenAI-native function calling (tool_calls in assistant messages)
- finish_answer is a built-in virtual tool that terminates the loop with a final answer
- Constructs the same DataProto output format as SGLangRollout._req_level_generate_sequences

Usage (in eval script):
    actor_rollout_ref.rollout.name=multi_turn_api
    +actor_rollout_ref.rollout.api_base=http://localhost:6768/v1
    +actor_rollout_ref.rollout.api_model=Qwen/Qwen3-1.7B
    actor_rollout_ref.rollout.multi_turn.enable=true
    actor_rollout_ref.rollout.multi_turn.max_turns=8
    actor_rollout_ref.rollout.multi_turn.tool_config_path=...
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import re
import sys
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

import logging

import numpy as np
import torch
from omegaconf import OmegaConf
from openai import AsyncOpenAI
from torch.nn.utils.rnn import pad_sequence
from transformers import PreTrainedTokenizer

from verl import DataProto
from verl.utils.torch_functional import get_response_mask, pad_sequence_to_length
from verl.workers.rollout.base import BaseRollout
from verl.workers.rollout.schemas import (
    AsyncRolloutRequest,
    AsyncRolloutRequestStateEnum,
    FinishReasonTypeEnum,
    Message,
)
from verl.tools.schemas import OpenAIFunctionToolSchema, OpenAIFunctionToolCall, OpenAIFunctionCallSchema

__all__ = ["MultiTurnAPIRollout"]

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Hermes XML tool-call fallback parser
# SGLang qwen25 parser expects JSON inside <tool_call>, but the model's own
# chat template instructs it to output Hermes XML format:
#   <tool_call>
#   <function=name>
#   <parameter=key>value</parameter>
#   </function>
#   </tool_call>
# We parse this format ourselves when tool_calls comes back empty.
# ---------------------------------------------------------------------------

_HERMES_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*<function=(\w+)>(.*?)</function>\s*</tool_call>",
    re.DOTALL,
)
_HERMES_PARAM_RE = re.compile(
    r"<parameter=(\w+)>\s*(.*?)\s*</parameter>",
    re.DOTALL,
)


def _parse_hermes_tool_calls(content: str) -> list:
    """Parse Hermes XML tool calls from model content into OpenAIFunctionToolCall list."""
    calls = []
    for i, m in enumerate(_HERMES_TOOL_CALL_RE.finditer(content)):
        func_name = m.group(1)
        body = m.group(2)
        arguments: dict = {}
        for pm in _HERMES_PARAM_RE.finditer(body):
            key = pm.group(1)
            val_str = pm.group(2).strip()
            try:
                parsed = json.loads(val_str)
                # If a list contains strings that are themselves JSON objects, parse them
                if isinstance(parsed, list):
                    new_parsed = []
                    for item in parsed:
                        if isinstance(item, str):
                            try:
                                new_parsed.append(json.loads(item))
                            except json.JSONDecodeError:
                                new_parsed.append(item)
                        else:
                            new_parsed.append(item)
                    parsed = new_parsed
                arguments[key] = parsed
            except json.JSONDecodeError:
                # Try stripping potential surrounding whitespace/newlines and retry
                val_str_clean = " ".join(val_str.split())
                try:
                    arguments[key] = json.loads(val_str_clean)
                except json.JSONDecodeError:
                    arguments[key] = val_str
        calls.append(
            OpenAIFunctionToolCall(
                id=f"call_{i}",
                function=OpenAIFunctionCallSchema(name=func_name, arguments=arguments),
            )
        )
    return calls


# ---------------------------------------------------------------------------
# Built-in virtual tools (handled directly in the loop, no BaseTool needed)
# ---------------------------------------------------------------------------

_MAKE_PLAN_SCHEMA = {
    "type": "function",
    "function": {
        "name": "make_plan",
        "description": "Break the question into a structured multi-phase plan before starting to search. Call this FIRST before any wiki_search.",
        "parameters": {
            "type": "object",
            "properties": {
                "phases": {
                    "type": "array",
                    "description": "List of phases in the plan.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "todo": {
                                "type": "string",
                                "description": "What to do in this phase.",
                            },
                            "goal": {
                                "type": "string",
                                "description": "How to decide this phase is complete.",
                            },
                        },
                        "required": ["todo", "goal"],
                    },
                }
            },
            "required": ["phases"],
        },
    },
}

_SUMMARIZE_PHASE_SCHEMA = {
    "type": "function",
    "function": {
        "name": "summarize_phase",
        "description": (
            "Call this tool when you believe the current phase is complete. "
            "Summarize what was done, what information was found, and why the phase goal was reached."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "phase": {
                    "type": "integer",
                    "description": "The phase number being summarized.",
                },
                "actions": {
                    "type": "string",
                    "description": "What searches or actions were performed in this phase.",
                },
                "findings": {
                    "type": "string",
                    "description": "Key information found during this phase.",
                },
                "goal_reached": {
                    "type": "string",
                    "description": "Explanation of why the phase goal has been reached.",
                },
            },
            "required": ["phase", "actions", "findings", "goal_reached"],
        },
    },
}

_FINISH_ANSWER_SCHEMA = {
    "type": "function",
    "function": {
        "name": "finish_answer",
        "description": "Submit the final answer to the question. Call this when you have gathered enough information.",
        "parameters": {
            "type": "object",
            "properties": {
                "answer": {
                    "type": "string",
                    "description": "The concise final answer.",
                }
            },
            "required": ["answer"],
        },
    },
}


class MultiTurnAPIRollout(BaseRollout):
    """
    Multi-turn rollout backed by an external OpenAI-compatible API.

    Uses /v1/chat/completions with OpenAI-native function calling.
    finish_answer is a virtual built-in tool that terminates the loop.
    """

    def __init__(self, tokenizer: PreTrainedTokenizer, config):
        super().__init__()
        self.tokenizer = tokenizer
        self.config = config
        self.pad_token_id = tokenizer.pad_token_id

        api_base = config.get("api_base", "http://localhost:8000/v1")
        self.api_model = config.get("api_model", "")
        self.async_client = AsyncOpenAI(base_url=api_base, api_key="EMPTY")

        # Initialize tools from tool_config_path (excludes finish_answer)
        self._tool_schemas, self._tool_map = self._initialize_tools(config)
        # Full tool list: make_plan + real tools + summarize_phase + finish_answer
        self._all_tool_schemas = [_MAKE_PLAN_SCHEMA] + self._tool_schemas + [_SUMMARIZE_PHASE_SCHEMA, _FINISH_ANSWER_SCHEMA]

    # ------------------------------------------------------------------
    # Tool initialization
    # ------------------------------------------------------------------

    def _initialize_tools(self, config):
        tool_config_path = config.multi_turn.get("tool_config_path", None)
        if tool_config_path is None:
            return [], {}

        def initialize_tools_from_config(tools_config) -> list:
            tool_list = []
            for tool_config in tools_config.tools:
                cls_name = tool_config.class_name
                module_name, class_name = cls_name.rsplit(".", 1)
                if module_name not in sys.modules:
                    spec = importlib.util.find_spec(module_name)
                    module = importlib.util.module_from_spec(spec)
                    sys.modules[module_name] = module
                    spec.loader.exec_module(module)
                else:
                    module = sys.modules[module_name]
                tool_cls = getattr(module, class_name)
                tool_schema_dict = OmegaConf.to_container(tool_config.tool_schema, resolve=True)
                tool_schema = OpenAIFunctionToolSchema.model_validate(tool_schema_dict)
                tool = tool_cls(
                    config=OmegaConf.to_container(tool_config.config, resolve=True),
                    tool_schema=tool_schema,
                )
                tool_list.append(tool)
            return tool_list

        tools_config = OmegaConf.load(tool_config_path)
        tool_list = initialize_tools_from_config(tools_config)
        tool_schemas = [tool.get_openai_tool_schema().model_dump(exclude_none=True) for tool in tool_list]
        tool_map = {tool.name: tool for tool in tool_list}

        return tool_schemas, tool_map

    # ------------------------------------------------------------------
    # Tool execution helper
    # ------------------------------------------------------------------

    async def _execute_tool_call(
        self,
        request_id: str,
        tool_name: str,
        arguments: dict,
        tools_kwargs: dict,
    ) -> Tuple[str, float, dict]:
        tool = self._tool_map.get(tool_name)
        if tool is None:
            logger.warning(f"Unknown tool: {tool_name}")
            return json.dumps({"error": f"Unknown tool: {tool_name}"}), 0.0, {}

        # Ensure any string-valued arguments that look like JSON are parsed
        parsed_arguments = {}
        for k, v in arguments.items():
            if isinstance(v, str):
                try:
                    parsed_arguments[k] = json.loads(v)
                except json.JSONDecodeError:
                    parsed_arguments[k] = v
            else:
                parsed_arguments[k] = v

        extra_kwargs = tools_kwargs.get(tool_name, {}).get("execute_kwargs", {})
        try:
            resp_text, reward, metrics = await tool.execute(
                request_id, parsed_arguments, **extra_kwargs
            )
        except Exception as e:
            logger.warning(f"Tool '{tool_name}' execution error: {e}")
            resp_text = json.dumps({"error": str(e)})
            reward, metrics = 0.0, {}
        return resp_text, reward, metrics

    # ------------------------------------------------------------------
    # prompt_ids snapshot & input_ids rebuild helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _serialize_messages(messages) -> list[dict]:
        """Serialize messages to OpenAI API format.

        Handles two requirements:
        - assistant tool_calls: arguments must be JSON string (not dict)
        - tool messages: must include tool_call_id matching the assistant's tool call
        """
        result = []
        # Track last tool_call id per tool name for pairing with tool responses
        last_tool_call_id: dict[str, str] = {}

        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                tc_list = []
                for tc in msg.tool_calls:
                    tc_list.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": json.dumps(tc.function.arguments, ensure_ascii=False),
                        },
                    })
                    last_tool_call_id[tc.function.name] = tc.id
                d = {"role": "assistant", "content": msg.content or "", "tool_calls": tc_list}
                result.append(d)
            elif msg.role == "tool":
                tool_call_id = msg.tool_call_id or next(iter(last_tool_call_id.values()), "tool_call_0")
                result.append({
                    "role": "tool",
                    "tool_call_id": tool_call_id,
                    "content": msg.content or "",
                })
            else:
                d = msg.model_dump(exclude_none=True)
                result.append(d)

        return result

    @staticmethod
    def _serialize_messages_for_template(messages) -> list[dict]:
        """Serialize messages for apply_chat_template (arguments stay as dict)."""
        result = []
        last_tool_call_id: dict[str, str] = {}
        for msg in messages:
            if msg.role == "assistant" and msg.tool_calls:
                tc_list = []
                for tc in msg.tool_calls:
                    tc_list.append({
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,  # keep as dict
                        },
                    })
                    last_tool_call_id[tc.function.name] = tc.id
                result.append({"role": "assistant", "content": msg.content or "", "tool_calls": tc_list})
            elif msg.role == "tool":
                tool_call_id = msg.tool_call_id or next(iter(last_tool_call_id.values()), "tool_call_0")
                result.append({"role": "tool", "tool_call_id": tool_call_id, "content": msg.content or ""})
            else:
                result.append(msg.model_dump(exclude_none=True))
        return result

    def _rebuild_input_ids(self, req) -> None:
        """Rebuild input_ids from req.messages (add_generation_prompt=True)."""
        prompt_text = self.tokenizer.apply_chat_template(
            conversation=self._serialize_messages_for_template(req.messages),
            add_generation_prompt=True,
            tokenize=False,
            tools=self._all_tool_schemas if self._all_tool_schemas else None,
        )
        ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        # Strip auto-appended <think>\n (Qwen3 chat template behaviour)
        think_ids = self.tokenizer.encode("<think>\n", add_special_tokens=False)
        if ids[-len(think_ids):] == think_ids:
            ids = ids[:-len(think_ids)]
        req.input_ids = ids
        req.attention_mask = [1] * len(ids)
        req.position_ids = list(range(len(ids)))
        req.loss_mask = [0] * len(ids)

    # ------------------------------------------------------------------
    # Single-step chat completion (async)
    # ------------------------------------------------------------------

    async def _api_generate_one_step(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
    ) -> dict:
        """Call /v1/chat/completions and return a normalized result dict."""
        try:
            resp = await self.async_client.chat.completions.create(
                model=self.api_model,
                messages=messages,
                tools=self._all_tool_schemas if self._all_tool_schemas else None,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            finish_reason = choice.finish_reason or "stop"
            msg = choice.message
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    OpenAIFunctionToolCall(
                        id=tc.id,
                        function=OpenAIFunctionCallSchema(
                            name=tc.function.name,
                            arguments=json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {}),
                        ),
                    )
                    for tc in msg.tool_calls
                ]
            # Fallback: SGLang qwen25 parser returns empty tool_calls when the model
            # outputs Hermes XML format. Parse the XML ourselves whenever content
            # contains a <tool_call> block, regardless of finish_reason.
            if not tool_calls and "<tool_call>" in (msg.content or ""):
                tool_calls = _parse_hermes_tool_calls(msg.content or "") or None
                if tool_calls:
                    finish_reason = "tool_calls"
            return {
                "content": msg.content or "",
                "tool_calls": tool_calls,  # List[OpenAIFunctionToolCall] or None
                "finish_reason": finish_reason,
            }
        except Exception as e:
            logger.warning(f"API call failed: {e}")
            return {"content": "", "tool_calls": None, "finish_reason": "stop"}

    # ------------------------------------------------------------------
    # Multi-turn loop for a single request
    # ------------------------------------------------------------------

    async def _multi_turn_rollout_one_request(
        self,
        req: AsyncRolloutRequest,
        temperature: float,
        max_tokens: int,
    ) -> AsyncRolloutRequest:
        _req = deepcopy(req)
        finish_reason_type = FinishReasonTypeEnum.STOP
        max_turns = self.config.multi_turn.max_turns or 8

        # Initialize tools
        if self._tool_map and _req.tools_kwargs:
            for name, tool in self._tool_map.items():
                if name in _req.tools_kwargs:
                    create_kwargs = _req.tools_kwargs[name].get("create_kwargs", {})
                    await tool.create(_req.request_id, **create_kwargs)

        # Build initial input_ids
        self._rebuild_input_ids(_req)

        # ------------------------------------------------------------------
        # Main generation loop
        # ------------------------------------------------------------------
        final_answer: Optional[str] = None
        current_turns = 0

        while current_turns < max_turns:
            output = await self._api_generate_one_step(
                self._serialize_messages(_req.messages), temperature, max_tokens
            )
            content = output["content"]
            tool_calls = output["tool_calls"]
            finish_reason = output["finish_reason"]
            current_turns += 1

            if finish_reason == "length":
                _req.messages.append(Message(role="assistant", content=content))
                _req._direct_response_ids = self.tokenizer.encode(
                    content, add_special_tokens=False
                )[: _req.max_response_len]
                finish_reason_type = FinishReasonTypeEnum.LENGTH
                break

            if tool_calls:
                # Append assistant message with tool_calls
                _req.messages.append(Message(
                    role="assistant",
                    content=content,
                    tool_calls=tool_calls,
                ))

                # Handle make_plan: acknowledge and continue (no external call needed)
                make_plan_tc = next((tc for tc in tool_calls if tc.function.name == "make_plan"), None)
                if make_plan_tc is not None:
                    phases = make_plan_tc.function.arguments.get("phases", [])
                    # phases may be a JSON string if the model didn't output valid JSON array
                    if isinstance(phases, str):
                        try:
                            phases = json.loads(phases)
                        except json.JSONDecodeError:
                            phases = []
                    plan_lines = "\n".join(
                        f"Phase {i+1}: TODO: {p.get('todo','') if isinstance(p, dict) else p} | Goal: {p.get('goal','') if isinstance(p, dict) else ''}"
                        for i, p in enumerate(phases)
                    )
                    _req.messages.append(Message(
                        role="tool",
                        tool_call_id=make_plan_tc.id,
                        content=f"Plan recorded:\n{plan_lines}\n\nNow execute the plan step by step.",
                    ))
                    self._rebuild_input_ids(_req)
                    continue

                # Handle summarize_phase: acknowledge and continue
                summarize_tc = next((tc for tc in tool_calls if tc.function.name == "summarize_phase"), None)
                if summarize_tc is not None:
                    args = summarize_tc.function.arguments
                    _req.messages.append(Message(
                        role="tool",
                        tool_call_id=summarize_tc.id,
                        content=(
                            f"Phase {args.get('phase','')} summary recorded. "
                            f"Findings: {args.get('findings','')} "
                            f"Goal reached: {args.get('goal_reached','')}. "
                            "Proceed to the next phase."
                        ),
                    ))
                    self._rebuild_input_ids(_req)
                    continue

                # Check for finish_answer among tool calls
                for tc in tool_calls:
                    if tc.function.name == "finish_answer":
                        final_answer = tc.function.arguments.get("answer", content)
                        break

                if final_answer is not None:
                    _req._direct_response_ids = self.tokenizer.encode(
                        f"<answer>{final_answer}</answer>", add_special_tokens=False
                    )[: _req.max_response_len]
                    finish_reason_type = FinishReasonTypeEnum.STOP
                    break

                # Execute real tool calls in parallel
                real_tool_calls = [
                    tc for tc in tool_calls if tc.function.name != "finish_answer"
                ]
                tool_results = await asyncio.gather(*[
                    self._execute_tool_call(
                        _req.request_id,
                        tc.function.name,
                        tc.function.arguments,
                        _req.tools_kwargs,
                    )
                    for tc in real_tool_calls
                ])

                for tc, (resp_text, reward, metrics) in zip(real_tool_calls, tool_results):
                    _req.messages.append(Message(
                        role="tool",
                        tool_call_id=tc.id,
                        content=resp_text,
                    ))
                    _req.update_metrics(metrics, tc.function.name)
                self._rebuild_input_ids(_req)

            else:
                # No tool call — intermediate reasoning or thinking turn, just continue.
                _req.messages.append(Message(role="assistant", content=content))
                self._rebuild_input_ids(_req)

        # Save full conversation to local file for debugging
        debug_dir = "./debug_conversations"
        os.makedirs(debug_dir, exist_ok=True)
        from random import randint
        debug_path = os.path.join(debug_dir, f"{randint(0, 10000000)}.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(f"request_id: {_req.request_id}\n")
            f.write(f"turns: {current_turns}\n")
            f.write("=" * 60 + "\n")
            for msg in _req.messages:
                f.write(f"[{msg.role}]\n")
                if msg.content:
                    f.write(msg.content + "\n")
                if msg.tool_calls:
                    for tc in msg.tool_calls:
                        f.write(f"  <tool_call> {tc.function.name}({tc.function.arguments})\n")
                f.write("-" * 40 + "\n")
            if hasattr(_req, "_direct_response_ids"):
                decoded = self.tokenizer.decode(_req._direct_response_ids, skip_special_tokens=False)
                f.write("\n[final_answer_decoded]\n")
                f.write(decoded + "\n")

        # Release tools and finalize
        async def calc_reward_and_release(name: str, tool):
            reward = await tool.calc_reward(
                _req.request_id,
                **_req.tools_kwargs[name].get("calc_reward_kwargs", {}),
            )
            await tool.release(
                _req.request_id,
                **_req.tools_kwargs[name].get("release_kwargs", {}),
            )
            return name, reward

        tool_reward_tasks = [
            calc_reward_and_release(name, self._tool_map[name])
            for name in _req.tools_kwargs.keys()
        ]
        tool_reward_scores = dict(await asyncio.gather(*tool_reward_tasks))
        _req.finalize(self.tokenizer, tool_reward_scores, finish_reason_type)
        # Restore directly-set response_ids after finalize() may have overwritten them.
        if hasattr(_req, "_direct_response_ids"):
            _req.response_ids = _req._direct_response_ids
            _req.response_attention_mask = [1] * len(_req.response_ids)
            _req.response_position_ids = list(range(len(_req.response_ids)))
            _req.response_loss_mask = [0] * len(_req.response_ids)
        return _req

    # ------------------------------------------------------------------
    # Preprocess prompts → AsyncRolloutRequest list (mirrors SGLangRollout)
    # ------------------------------------------------------------------

    def _preprocess_prompts(self, prompts: DataProto) -> List[AsyncRolloutRequest]:
        assert "raw_prompt" in prompts.non_tensor_batch, (
            "MultiTurnAPIRollout requires data.return_raw_chat=True"
        )
        req_list = []
        prompt_ids_tensor = prompts.batch["input_ids"]
        prompt_attn_mask = prompts.batch["attention_mask"]

        for data_idx, raw_prompt in enumerate(prompts.non_tensor_batch["raw_prompt"]):
            # raw_prompt may be a numpy array of dicts; convert to plain list
            if not isinstance(raw_prompt, list):
                raw_prompt = raw_prompt.tolist()

            _tools_kwargs = {}
            if self._tool_schemas and "tools_kwargs" in prompts.non_tensor_batch:
                _tools_kwargs = prompts.non_tensor_batch["tools_kwargs"][data_idx]

            # Get prompt ids (strip left padding)
            ids = prompt_ids_tensor[data_idx]
            mask = prompt_attn_mask[data_idx]
            raw_ids = ids[mask.bool()].tolist()

            req = AsyncRolloutRequest(
                batch_data_id=data_idx,
                rollout_offset=0,
                request_id=f"req-{data_idx}",
                state=AsyncRolloutRequestStateEnum.PENDING,
                messages=[Message(**m) if isinstance(m, dict) else m for m in raw_prompt],
                tools_kwargs=_tools_kwargs,
                input_ids=raw_ids,
                prompt_ids=raw_ids,
                attention_mask=[1] * len(raw_ids),
                prompt_attention_mask=[1] * len(raw_ids),
                response_ids=[],
                response_attention_mask=[],
                position_ids=list(range(len(raw_ids))),
                prompt_position_ids=list(range(len(raw_ids))),
                response_position_ids=[],
                loss_mask=[0] * len(raw_ids),
                prompt_loss_mask=[0] * len(raw_ids),
                response_loss_mask=[],
                reward_scores={},
                max_response_len=self.config.response_length,
                max_model_len=int(self.config.get("max_model_len") or (self.config.prompt_length + self.config.response_length)),
            )
            req_list.append(req)
        return req_list

    # ------------------------------------------------------------------
    # Main entry: generate_sequences
    # ------------------------------------------------------------------

    @torch.no_grad()
    def generate_sequences(self, prompts: DataProto) -> DataProto:
        is_validate = prompts.meta_info.get("validate", False)
        temperature = prompts.meta_info.get(
            "temperature",
            self.config.val_kwargs.temperature if is_validate else self.config.get("temperature", 1.0),
        )
        do_sample = prompts.meta_info.get("do_sample", self.config.get("do_sample", True))
        if not do_sample:
            temperature = 0.0
        max_tokens = self.config.response_length

        req_list = self._preprocess_prompts(prompts)

        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            output_req_list = loop.run_until_complete(
                asyncio.gather(
                    *[self._multi_turn_rollout_one_request(req, temperature, max_tokens) for req in req_list]
                )
            )
        finally:
            loop.close()
            asyncio.set_event_loop(None)
        output_req_list = sorted(output_req_list, key=lambda x: x.batch_data_id)

        # Pack results into DataProto (mirrors SGLangRollout._req_level_generate_sequences)
        tgt_device = prompts.batch["input_ids"].device
        prompt_ids_list, response_ids_list = [], []
        prompt_attn_list, response_attn_list = [], []
        prompt_pos_list = []
        prompt_loss_list, response_loss_list = [], []
        messages_list, reward_scores_list = [], []

        for req in output_req_list:
            prompt_ids_list.append(torch.tensor(req.prompt_ids, dtype=torch.int, device=tgt_device))
            response_ids_list.append(torch.tensor(req.response_ids, dtype=torch.int, device=tgt_device))
            prompt_attn_list.append(torch.tensor(req.prompt_attention_mask, dtype=torch.int, device=tgt_device))
            response_attn_list.append(torch.tensor(req.response_attention_mask, dtype=torch.int, device=tgt_device))
            prompt_pos_list.append(torch.tensor(req.prompt_position_ids, dtype=torch.int, device=tgt_device))
            prompt_loss_list.append(torch.tensor(req.prompt_loss_mask, dtype=torch.int, device=tgt_device))
            response_loss_list.append(torch.tensor(req.response_loss_mask, dtype=torch.int, device=tgt_device))
            messages_list.append({"messages": req.messages})
            reward_scores_list.append(req.reward_scores)

        prompt_ids = pad_sequence(prompt_ids_list, batch_first=True, padding_value=self.pad_token_id, padding_side="left")
        if prompt_ids.shape[1] < self.config.prompt_length:
            prompt_ids = pad_sequence_to_length(prompt_ids, self.config.prompt_length, self.pad_token_id, left_pad=True)

        response_ids = pad_sequence(response_ids_list, batch_first=True, padding_value=self.pad_token_id)
        if response_ids.shape[1] < self.config.response_length:
            response_ids = pad_sequence_to_length(response_ids, self.config.response_length, self.pad_token_id)

        prompt_attn = pad_sequence(prompt_attn_list, batch_first=True, padding_value=0, padding_side="left")
        if prompt_attn.shape[1] < self.config.prompt_length:
            prompt_attn = pad_sequence_to_length(prompt_attn, self.config.prompt_length, 0, left_pad=True)

        response_attn = pad_sequence(response_attn_list, batch_first=True, padding_value=0)
        if response_attn.shape[1] < self.config.response_length:
            response_attn = pad_sequence_to_length(response_attn, self.config.response_length, 0)

        prompt_pos = pad_sequence(prompt_pos_list, batch_first=True, padding_value=0, padding_side="left")
        if prompt_pos.shape[1] < self.config.prompt_length:
            prompt_pos = pad_sequence_to_length(prompt_pos, self.config.prompt_length, 0, left_pad=True)

        response_length = response_ids.size(1)
        batch_size = len(output_req_list)
        delta_pos = torch.arange(1, response_length + 1, device=tgt_device).unsqueeze(0).repeat(batch_size, 1)
        response_pos = prompt_pos[:, -1:] + delta_pos

        prompt_loss = pad_sequence(prompt_loss_list, batch_first=True, padding_value=0, padding_side="left")
        if prompt_loss.shape[1] < self.config.prompt_length:
            prompt_loss = pad_sequence_to_length(prompt_loss, self.config.prompt_length, 0, left_pad=True)

        response_loss = pad_sequence(response_loss_list, batch_first=True, padding_value=0)
        if response_loss.shape[1] < self.config.response_length:
            response_loss = pad_sequence_to_length(response_loss, self.config.response_length, 0)

        input_ids = torch.cat([prompt_ids, response_ids], dim=-1)
        attention_mask = torch.cat([prompt_attn, response_attn], dim=-1)
        position_ids = torch.cat([prompt_pos, response_pos], dim=-1)
        loss_mask = torch.cat([prompt_loss, response_loss], dim=-1)

        from tensordict import TensorDict

        batch = TensorDict(
            {
                "prompts": prompt_ids,
                "responses": response_ids,
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "position_ids": position_ids,
                "loss_mask": loss_mask,
            },
            batch_size=batch_size,
        )

        return DataProto(
            batch=batch,
            non_tensor_batch={
                "messages": np.array(messages_list),
                "reward_scores": np.array(reward_scores_list),
            },
        )
