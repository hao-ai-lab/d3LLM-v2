# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

from typing import List
import re

# Strict format (case-insensitive, DOTALL):
#   [whitespace] <think>…</think> [whitespace] <action>…</action> [whitespace]
# Rules enforced:
#   - Nothing before the opening <think>/<reason> except whitespace
#   - Exactly one think/reason block and exactly one action block
#   - Only whitespace between </think> and <action>
#   - Only whitespace after </action>
_STRICT_ACTION_RE = re.compile(
    r'^\s*'
    r'(?:<think>(.*?)</think>|<reason>(.*?)</reason>)'  # exactly one think/reason block
    r'\s*'
    r'<action>(.*?)</action>'                            # exactly one action block
    r'\s*$',
    re.DOTALL | re.IGNORECASE,
)

# Action-only: exactly one <action>…</action> anywhere; reasoning may appear before or after.
# (Do not require end-of-string after </action> — models often append "Reasoning:" after the block.)
_STRICT_ACTION_ONLY_RE = re.compile(
    r'<action>(.*?)</action>',
    re.DOTALL | re.IGNORECASE,
)


def webshop_projection(actions: List[str], avail_actions=None, strict: bool = False):
    """
    Process and validate model actions for WebShop.

    Args:
        actions: list of raw model output strings.
        strict:  when False (default), only requires a <think>/<reason> block and one
                 <action>...</action> block somewhere in the string (original behaviour).
                 When True, enforces the exact format:
                   [whitespace] <think>…</think> [whitespace] <action>…</action> [whitespace]
                 — nothing extra before, between, or after the two blocks.

    Validity rules (both modes):
      - Must contain at least one <action>...</action> block (extracted action fed to env).
      - Must contain a <think>...</think> or <reason>...</reason> block.
      - No Chinese characters.

    Additional rules in strict mode:
      - Only whitespace before the opening <think>/<reason> tag.
      - Exactly one think/reason block and exactly one action block.
      - Only whitespace between </think> and <action>.
      - Only whitespace after </action>.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lower_str = original_str.lower()

        if strict:
            # Check for Chinese characters first — always invalid in strict mode
            if re.search(r'[\u4e00-\u9fff]', original_str):
                start_idx = lower_str.find("<action>")
                end_idx = lower_str.find("</action>")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    actions[i] = lower_str[start_idx + len("<action>"):end_idx].strip()
                else:
                    actions[i] = lower_str[-20:]
                valids[i] = 0
                continue

            # Strict format check: must match exactly the expected structure.
            # Also verify tag counts: the regex uses non-greedy (.*?) with DOTALL,
            # which can backtrack across a first </think> when a second <think> block
            # exists, causing a false positive match. We guard against this by
            # requiring exactly one think/reason open+close tag pair.
            m = _STRICT_ACTION_RE.match(lower_str)
            think_count = lower_str.count("<think>")
            reason_count = lower_str.count("<reason>")
            action_count = lower_str.count("<action>")
            tag_counts_ok = (
                action_count == 1
                and (think_count + reason_count) == 1
                and lower_str.count("</think>") + lower_str.count("</reason>") == 1
                and lower_str.count("</action>") == 1
            )
            if m is None or not tag_counts_ok:
                # Still extract a usable action for env stepping, but mark as invalid
                start_idx = lower_str.find("<action>")
                end_idx = lower_str.find("</action>")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    actions[i] = lower_str[start_idx + len("<action>"):end_idx].strip()
                else:
                    actions[i] = lower_str[-20:]
                valids[i] = 0
                continue
            # group(3) is the action content (groups 1/2 are think/reason captures)
            actions[i] = m.group(3).strip()
            valids[i] = 1
        else:
            # Original lenient behaviour — exactly mirrors the pre-refactor logic
            actions[i] = lower_str  # original did actions[i] = actions[i].lower() in-place

            start_tag = "<action>"
            end_tag = "</action>"
            start_idx = actions[i].find(start_tag)
            end_idx = actions[i].find(end_tag)
            try:
                if start_idx == -1 or end_idx == -1:
                    actions[i] = actions[i][-20:]
                    continue
                extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
                actions[i] = extracted_action
                valids[i] = 1
            except Exception:
                actions[i] = actions[i][-20:]

            # think/reason check (uses original_str.lower() == lower_str, same as original)
            has_think = "<think>" in lower_str and "</think>" in lower_str
            has_reason = "<reason>" in lower_str and "</reason>" in lower_str
            if not has_think and not has_reason:
                valids[i] = 0

            # Chinese check at the end, same order as original
            if re.search(r'[\u4e00-\u9fff]', original_str):
                valids[i] = 0

    # Log summary and first few invalid examples for offline debugging
    n_valid = sum(valids)
    n_total = len(valids)
    if n_total > 0:
        invalid_examples = [(i, actions[i][:80]) for i in range(n_total) if valids[i] == 0][:3]
        mode_tag = "strict" if strict else "lenient"
        print(f"[PROJECTION/{mode_tag}] valid={n_valid}/{n_total} invalid_examples={invalid_examples}")

    return actions, valids


def webshop_projection_action_only(actions: List[str], avail_actions=None):
    """
    Strict projection for the action-only instruction format.

    Expected model output: free-form reasoning (before or after the action block) and
    exactly one <action>…</action> — no <think> or <reason> tags required.

    Validity rules:
      - Exactly one <action>…</action> pair (multiline action body allowed).
      - No Chinese characters.
      - Trailing text after </action> (e.g. "Reasoning: ...") is allowed.

    On failure the extracted action (if findable) is still returned so the
    environment can continue stepping, but valid=0.
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lower_str = original_str.lower()

        def _extract_action_fallback():
            start_idx = lower_str.find("<action>")
            end_idx = lower_str.find("</action>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return lower_str[start_idx + len("<action>"):end_idx].strip()
            return lower_str[-20:]

        # Chinese characters → always invalid
        if re.search(r'[\u4e00-\u9fff]', original_str):
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        # Exactly one <action>…</action> pair
        action_open_count = lower_str.count("<action>")
        action_close_count = lower_str.count("</action>")
        if action_open_count != 1 or action_close_count != 1:
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        m = _STRICT_ACTION_ONLY_RE.search(lower_str)
        if m is None:
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        actions[i] = m.group(1).strip()
        valids[i] = 1

    n_valid = sum(valids)
    n_total = len(valids)
    if n_total > 0:
        invalid_examples = [(i, actions[i][:80]) for i in range(n_total) if valids[i] == 0][:3]
        print(f"[PROJECTION/action_only_strict] valid={n_valid}/{n_total} invalid_examples={invalid_examples}")

    return actions, valids


def webshop_projection_thought_prefix(actions: List[str], avail_actions=None):
    """
    Strict projection for the "Thought: ..." instruction format.

    Expected model output:
        (think) <step-by-step reasoning>
        <action> ... </action>

    Validity rules:
      - Must start with "(think)" (case-insensitive, optional leading whitespace).
      - Exactly one <action>…</action> pair, appearing after the (think) section.
      - No Chinese characters.

    On failure the extracted action (if findable) is still returned so the
    environment can continue stepping, but valid=0.
    """
    _THOUGHT_PREFIX_RE = re.compile(
        r'^\s*\(think\).*<action>(.*?)</action>\s*$',
        re.DOTALL | re.IGNORECASE,
    )

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lower_str = original_str.lower()

        def _extract_action_fallback():
            start_idx = lower_str.find("<action>")
            end_idx = lower_str.find("</action>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return lower_str[start_idx + len("<action>"):end_idx].strip()
            return lower_str[-20:]

        # Chinese characters → always invalid
        if re.search(r'[\u4e00-\u9fff]', original_str):
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        # Exactly one <action>…</action> pair
        if lower_str.count("<action>") != 1 or lower_str.count("</action>") != 1:
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        m = _THOUGHT_PREFIX_RE.match(lower_str)
        if m is None:
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        actions[i] = m.group(1).strip()
        valids[i] = 1

    n_valid = sum(valids)
    n_total = len(valids)
    if n_total > 0:
        invalid_examples = [(i, actions[i][:80]) for i in range(n_total) if valids[i] == 0][:3]
        print(f"[PROJECTION/think_prefix_strict] valid={n_valid}/{n_total} invalid_examples={invalid_examples}")

    return actions, valids


def _is_admissible(action: str, avail: dict) -> bool:
    """
    Check whether *action* (already extracted, lowercased) is actually executable
    by the WebShop environment given *avail* = {has_search_bar, clickables}.

    Rules (mirrors web_agent_text_env.py step() logic):
      - search[X]  → has_search_bar must be True and X must be non-empty.
      - click[X]   → X must be in clickables (also lowercased).
    """
    if avail is None:
        return True  # no info available, assume admissible

    action = action.strip().lower()
    has_search = avail.get("has_search_bar", False)
    clickables = {c.lower() for c in avail.get("clickables", [])}

    if action.startswith("search[") and action.endswith("]"):
        query = action[len("search["):-1].strip()
        return has_search and len(query) > 0

    if action.startswith("click[") and action.endswith("]"):
        target = action[len("click["):-1].strip()
        return target in clickables

    return False


def webshop_projection_admissible(actions: List[str], avail_actions=None):
    """
    Strictest projection: strict format check (same as webshop_projection strict=True)
    followed by an admissible-action check against the environment's legal action set.

    An action is valid only if ALL of the following hold:
      1. No Chinese characters.
      2. Exactly one <think>/<reason> block AND exactly one <action>…</action> pair
         (same tag-count guards as the strict projection).
      3. The overall string matches _STRICT_ACTION_RE.
      4. The extracted action is actually executable by the environment:
           - search[X]  with non-empty X, when has_search_bar=True
           - click[X]   with X in the current clickables list

    *avail_actions* is a list of per-env dicts {has_search_bar, clickables}
    aligned with *actions*.  If None or an element is None, step 4 is skipped
    (falls back to strict-format-only).
    """
    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]
        lower_str = original_str.lower()
        avail = (avail_actions[i] if avail_actions is not None else None)

        def _extract_action_fallback():
            start_idx = lower_str.find("<action>")
            end_idx = lower_str.find("</action>")
            if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                return lower_str[start_idx + len("<action>"):end_idx].strip()
            return lower_str[-20:]

        # 1. Chinese characters → always invalid
        if re.search(r'[\u4e00-\u9fff]', original_str):
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        # 2. Tag-count guard (same as strict projection)
        think_count = lower_str.count("<think>")
        reason_count = lower_str.count("<reason>")
        action_count = lower_str.count("<action>")
        tag_counts_ok = (
            action_count == 1
            and (think_count + reason_count) == 1
            and lower_str.count("</think>") + lower_str.count("</reason>") == 1
            and lower_str.count("</action>") == 1
        )
        if not tag_counts_ok:
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        # 3. Strict regex match
        m = _STRICT_ACTION_RE.match(lower_str)
        if m is None:
            actions[i] = _extract_action_fallback()
            valids[i] = 0
            continue

        extracted = m.group(3).strip()
        actions[i] = extracted

        # 4. Admissibility check against the environment's legal action set
        if not _is_admissible(extracted, avail):
            valids[i] = 0
            continue

        valids[i] = 1

    n_valid = sum(valids)
    n_total = len(valids)
    if n_total > 0:
        invalid_examples = [(i, actions[i][:80]) for i in range(n_total) if valids[i] == 0][:3]
        print(f"[PROJECTION/admissible_strict] valid={n_valid}/{n_total} invalid_examples={invalid_examples}")

    return actions, valids