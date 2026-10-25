"""Phase-level speculative inference for MHQA.

Draft model runs one phase ahead; target model verifies in parallel.
A LLM judge compares phase summaries semantically to decide accept/reject.
Two-thread design mirrors rollout_loop.speculative_multi_turn_loop.
"""
from __future__ import annotations

import asyncio
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Optional

from openai import AsyncOpenAI, OpenAI

from verl.workers.rollout.multi_turn_api_rollout import MultiTurnAPIRollout
from verl.workers.rollout.schemas import AsyncRolloutRequest, FinishReasonTypeEnum, Message

import logging
logger = logging.getLogger(__name__)

__all__ = ["SpeculativeMHQARollout"]


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

_JUDGE_PROMPT = """\
You are a strict semantic equivalence judge.
Given two phase summaries from two AI assistants, decide if they reach the SAME essential finding.
Reply with exactly one word: YES or NO.

=== Summary A ===
{a}

=== Summary B ===
{b}

Same finding? YES or NO:"""


def _llm_judge(summary_a: dict, summary_b: dict, client: OpenAI, model: str) -> bool:
    """Return True if two summarize_phase argument dicts convey the same finding."""
    def fmt(s: dict) -> str:
        return f"Actions: {s.get('actions','')}\nFindings: {s.get('findings','')}\nGoal reached: {s.get('goal_reached','')}"
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": _JUDGE_PROMPT.format(a=fmt(summary_a), b=fmt(summary_b))}],
            temperature=0.0,
            max_tokens=4,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        return verdict.startswith("YES")
    except Exception as e:
        logger.warning(f"[Judge] LLM call failed: {e}, defaulting to mismatch")
        return False


# ---------------------------------------------------------------------------
# Phase result
# ---------------------------------------------------------------------------

@dataclass
class _PhaseResult:
    summary_args: Optional[dict]    # arguments of summarize_phase tool call, or None
    messages: List[Message]         # full messages list after this phase
    final_answer: Optional[str]     # set if finish_answer was called
    turns_used: int


# ---------------------------------------------------------------------------
# SpeculativeMHQARollout
# ---------------------------------------------------------------------------

class SpeculativeMHQARollout(MultiTurnAPIRollout):
    """Phase-level speculative MHQA rollout. Inherits all infra from MultiTurnAPIRollout."""

    def __init__(self, config, tokenizer, **kwargs):
        super().__init__(config, tokenizer, **kwargs)
        cfg = config
        self._draft_async_client = AsyncOpenAI(
            base_url=cfg.get("speculative_api_base", "http://localhost:6768/v1"), api_key="EMPTY"
        )
        self._draft_model = cfg.get("speculative_api_model", "")
        # Sync client for judge (called from listener thread)
        self._judge_client = OpenAI(
            base_url=cfg.get("speculative_judge_api_base", cfg.get("api_base", "http://localhost:6767/v1")),
            api_key="EMPTY",
        )
        self._judge_model = cfg.get("speculative_judge_api_model", cfg.get("api_model", ""))

    # ------------------------------------------------------------------
    # Run one complete phase: tool turns until summarize_phase / finish_answer
    # ------------------------------------------------------------------

    async def _run_one_phase(
        self,
        req: AsyncRolloutRequest,
        temperature: float,
        max_tokens: int,
        max_turns: int,
        client: AsyncOpenAI,
        model: str,
    ) -> _PhaseResult:
        """Drive the model (via `client`/`model`) until it calls summarize_phase or finish_answer.

        Returns _PhaseResult with updated req.messages and any summary / final answer.
        Reuses all parent helpers: _serialize_messages, _execute_tool_call, _rebuild_input_ids.
        """
        import json
        turns = 0
        while turns < max_turns:
            # Call the specified model (draft or target)
            output = await self._api_generate_one_step_with(
                self._serialize_messages(req.messages), temperature, max_tokens, client, model
            )
            content, tool_calls, finish_reason = output["content"], output["tool_calls"], output["finish_reason"]
            turns += 1

            if finish_reason == "length":
                req.messages.append(Message(role="assistant", content=content))
                req._direct_response_ids = self.tokenizer.encode(content, add_special_tokens=False)[:req.max_response_len]
                return _PhaseResult(None, req.messages, None, turns)

            if tool_calls:
                req.messages.append(Message(role="assistant", content=content, tool_calls=tool_calls))

                # make_plan → acknowledge
                mp = next((tc for tc in tool_calls if tc.function.name == "make_plan"), None)
                if mp is not None:
                    phases = mp.function.arguments.get("phases", [])
                    if isinstance(phases, str):
                        try: phases = json.loads(phases)
                        except: phases = []
                    plan_lines = "\n".join(
                        f"Phase {i+1}: TODO: {p.get('todo','') if isinstance(p,dict) else p} | Goal: {p.get('goal','') if isinstance(p,dict) else ''}"
                        for i, p in enumerate(phases)
                    )
                    req.messages.append(Message(role="tool", tool_call_id=mp.id,
                        content=f"Plan recorded:\n{plan_lines}\n\nNow execute the plan step by step."))
                    self._rebuild_input_ids(req)
                    continue

                # summarize_phase → phase boundary: return immediately
                sp = next((tc for tc in tool_calls if tc.function.name == "summarize_phase"), None)
                if sp is not None:
                    args = sp.function.arguments
                    req.messages.append(Message(role="tool", tool_call_id=sp.id,
                        content=(f"Phase {args.get('phase','')} summary recorded. "
                                 f"Findings: {args.get('findings','')} "
                                 f"Goal reached: {args.get('goal_reached','')}. "
                                 "Proceed to the next phase.")))
                    self._rebuild_input_ids(req)
                    return _PhaseResult(args, req.messages, None, turns)

                # finish_answer → done
                fa = next((tc for tc in tool_calls if tc.function.name == "finish_answer"), None)
                if fa is not None:
                    answer = fa.function.arguments.get("answer", content)
                    req._direct_response_ids = self.tokenizer.encode(
                        f"<answer>{answer}</answer>", add_special_tokens=False)[:req.max_response_len]
                    return _PhaseResult(None, req.messages, answer, turns)

                # real tool calls (wiki_search etc.)
                real = [tc for tc in tool_calls if tc.function.name not in ("make_plan","summarize_phase","finish_answer")]
                results = await asyncio.gather(*[
                    self._execute_tool_call(req.request_id, tc.function.name, tc.function.arguments, req.tools_kwargs)
                    for tc in real
                ])
                for tc, (resp_text, _, metrics) in zip(real, results):
                    req.messages.append(Message(role="tool", tool_call_id=tc.id, content=resp_text))
                    req.update_metrics(metrics, tc.function.name)
                self._rebuild_input_ids(req)
            else:
                req.messages.append(Message(role="assistant", content=content))
                self._rebuild_input_ids(req)

        return _PhaseResult(None, req.messages, None, turns)

    async def _api_generate_one_step_with(self, messages, temperature, max_tokens, client, model):
        """Like _api_generate_one_step but uses the given client/model instead of self.async_client."""
        import json as _json
        from verl.tools.schemas import OpenAIFunctionToolCall, OpenAIFunctionCallSchema
        from verl.workers.rollout.multi_turn_api_rollout import _parse_hermes_tool_calls
        try:
            resp = await client.chat.completions.create(
                model=model, messages=messages,
                tools=self._all_tool_schemas if self._all_tool_schemas else None,
                temperature=temperature, max_tokens=max_tokens,
            )
            choice = resp.choices[0]
            msg = choice.message
            finish_reason = choice.finish_reason or "stop"
            tool_calls = None
            if msg.tool_calls:
                tool_calls = [
                    OpenAIFunctionToolCall(
                        id=tc.id,
                        function=OpenAIFunctionCallSchema(
                            name=tc.function.name,
                            arguments=_json.loads(tc.function.arguments) if isinstance(tc.function.arguments, str) else (tc.function.arguments or {}),
                        ),
                    ) for tc in msg.tool_calls
                ]
            if not tool_calls and "<tool_call>" in (msg.content or ""):
                tool_calls = _parse_hermes_tool_calls(msg.content or "") or None
                if tool_calls:
                    finish_reason = "tool_calls"
            return {"content": msg.content or "", "tool_calls": tool_calls, "finish_reason": finish_reason}
        except Exception as e:
            logger.warning(f"API call failed ({model}): {e}")
            return {"content": "", "tool_calls": None, "finish_reason": "stop"}

    # ------------------------------------------------------------------
    # Speculative loop (overrides parent)
    # ------------------------------------------------------------------

    async def _multi_turn_rollout_one_request(
        self,
        req: AsyncRolloutRequest,
        temperature: float,
        max_tokens: int,
    ) -> AsyncRolloutRequest:
        _req = deepcopy(req)
        max_turns = self.config.multi_turn.max_turns or 16
        finish_reason_type = FinishReasonTypeEnum.STOP

        # Initialize tools
        if self._tool_map and _req.tools_kwargs:
            for name, tool in self._tool_map.items():
                if name in _req.tools_kwargs:
                    await tool.create(_req.request_id, **_req.tools_kwargs[name].get("create_kwargs", {}))
        self._rebuild_input_ids(_req)

        turns_used = 0

        while turns_used < max_turns:
            # ── One speculative round ─────────────────────────────────────────
            # Each phase: draft and target requests are launched simultaneously
            # from the same prefix. When draft finishes phase N, it immediately
            # launches draft+target for phase N+1 (pipeline parallelism).
            # When target finishes phase N, it runs judge immediately.
            # If judge rejects, the round ends and we restart from target's state.
            #
            # Timeline (matches the diagram):
            #   phase 1: launch d(1)+t(1) simultaneously
            #   d(1) done → launch d(2)+t(2) simultaneously, advance draft prefix
            #   t(1) done → judge d(1) vs t(1)
            #   d(2) done → launch d(3)+t(3) simultaneously
            #   t(2) done → judge d(2) vs t(2)
            #   ...
            #   t(N) reject → abort, use t(N) state as new prefix

            remaining = max_turns - turns_used
            abort = asyncio.Event()
            result_box: list = []   # filled by judge coroutine: (messages, final_answer, turns)

            # Per-phase: stores (draft_task, target_task) launched together
            # judge coroutine consumes target_tasks in order; draft coroutine
            # produces them. We use an asyncio.Queue of target futures so the
            # judge can await them one by one as they arrive.
            target_task_queue: asyncio.Queue = asyncio.Queue()   # items: asyncio.Task[_PhaseResult]
            draft_task_queue: asyncio.Queue = asyncio.Queue()    # items: asyncio.Task[_PhaseResult]

            async def run_draft_pipeline(base_req: AsyncRolloutRequest):
                """Phase-by-phase draft pipeline.
                After each completed phase, immediately launch next draft+target pair."""
                req_d = deepcopy(base_req)
                turns_d = 0
                while turns_d < remaining and not abort.is_set():
                    # Launch both draft and target for this phase simultaneously
                    draft_task = asyncio.create_task(self._run_one_phase(
                        deepcopy(req_d), temperature, max_tokens, remaining - turns_d,
                        self._draft_async_client, self._draft_model,
                    ))
                    target_task = asyncio.create_task(self._run_one_phase(
                        deepcopy(req_d), temperature, max_tokens, remaining - turns_d,
                        self.async_client, self.api_model,
                    ))
                    # Post both to their queues so judge coroutine can pick them up
                    await draft_task_queue.put(draft_task)
                    await target_task_queue.put(target_task)

                    # Wait for draft to finish — its result becomes the next prefix
                    phase = await draft_task
                    turns_d += phase.turns_used
                    req_d.messages = phase.messages

                    if phase.final_answer is not None or phase.summary_args is None:
                        break  # draft finished (answer or hit max_turns)

                # Sentinels so judge knows draft pipeline is done
                await draft_task_queue.put(None)
                await target_task_queue.put(None)

            async def run_judge_pipeline():
                """Consume target tasks in order; judge each phase as soon as target finishes."""
                phase_idx = 0
                while True:
                    target_task = await target_task_queue.get()
                    draft_task  = await draft_task_queue.get()
                    if target_task is None or draft_task is None:
                        # Draft pipeline exhausted without rejection — all phases accepted
                        # Use the last known target state (from latest completed target phase)
                        break

                    phase_idx += 1
                    target_phase: _PhaseResult = await target_task
                    logger.info(f"[Spec] target phase={phase_idx} done, final={target_phase.final_answer is not None}")

                    if target_phase.final_answer is not None:
                        result_box.append((target_phase.messages, target_phase.final_answer, phase_idx))
                        abort.set()
                        return

                    if target_phase.summary_args is None:
                        # Target hit max_turns mid-phase
                        result_box.append((target_phase.messages, None, phase_idx))
                        abort.set()
                        return

                    # Judge: compare target summary against draft summary
                    draft_phase: _PhaseResult = await draft_task
                    match = False
                    if draft_phase.summary_args is not None:
                        match = await asyncio.get_event_loop().run_in_executor(
                            None, _llm_judge,
                            draft_phase.summary_args, target_phase.summary_args,
                            self._judge_client, self._judge_model,
                        )
                    logger.info(f"[Spec] phase={phase_idx}, judge={'ACCEPT' if match else 'REJECT'}")

                    if not match:
                        result_box.append((target_phase.messages, None, phase_idx))
                        abort.set()
                        return
                    # Match: continue to next phase (draft already launched it)

                # All phases accepted — use the last target phase's messages
                # (target_phase from the final iteration, already awaited above)
                if not result_box:
                    result_box.append((target_phase.messages, None, phase_idx))  # type: ignore[possibly-unbound]

            await asyncio.gather(
                run_draft_pipeline(deepcopy(_req)),
                run_judge_pipeline(),
            )

            if not result_box:
                break
            accepted_messages, final_answer, phases_done = result_box[0]
            # Use phases_done as a proxy for turns (exact turn count tracked inside _run_one_phase)
            turns_used += phases_done
            _req.messages = accepted_messages

            if final_answer is not None:
                _req._direct_response_ids = self.tokenizer.encode(
                    f"<answer>{final_answer}</answer>", add_special_tokens=False
                )[:_req.max_response_len]
                break

            self._rebuild_input_ids(_req)

        # ── Finalize (same as parent) ─────────────────────────────────────────
        async def _release(name, tool):
            reward = await tool.calc_reward(_req.request_id, **_req.tools_kwargs[name].get("calc_reward_kwargs", {}))
            await tool.release(_req.request_id, **_req.tools_kwargs[name].get("release_kwargs", {}))
            return name, reward

        scores = dict(await asyncio.gather(*[_release(n, t) for n, t in self._tool_map.items() if n in _req.tools_kwargs]))
        _req.finalize(self.tokenizer, scores, finish_reason_type)
        if hasattr(_req, "_direct_response_ids"):
            _req.response_ids = _req._direct_response_ids
            _req.response_attention_mask = [1] * len(_req.response_ids)
            _req.response_position_ids = list(range(len(_req.response_ids)))
            _req.response_loss_mask = [0] * len(_req.response_ids)
        return _req
