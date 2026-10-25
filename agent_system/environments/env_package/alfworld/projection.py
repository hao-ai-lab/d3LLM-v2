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

def alfworld_projection(actions: List[str], action_pools: List[List[str]]):
    """
    An function to process the actions
    actions: the list of actions to be processeed, it is a list of strings.
    action_pools: the list of action pools, each pool is a list of strings.
    """

    valids = [0] * len(actions)

    for i in range(len(actions)):
        original_str = actions[i]  # keep the original string
        actions[i] = actions[i].lower()

        # Attempt to extract the substring within <action>...</action>
        start_tag = "<action>"
        end_tag = "</action>"
        start_idx = actions[i].find(start_tag)
        end_idx = actions[i].find(end_tag)
        try:
            if start_idx == -1 or end_idx == -1:
                # If we can't find a valid <action>...</action> block, mark as invalid
                actions[i] = actions[i][-30:]  # 0 is invalid action for Sokoban
                continue

            # Extract just the content between the tags
            extracted_action = actions[i][start_idx + len(start_tag):end_idx].strip().lower()
            
            actions[i] = extracted_action
            valids[i] = 1

        except:
            actions[i] = actions[i][-30:]

        # Require exactly one <action>...</action> and exactly one
        # <think>...</think> or <reason>...</reason> pair (not both, not multiple).
        # SDAR models use <reason> tags instead of <think> tags, so accept either.
        lower_original = original_str.lower()
        think_count = lower_original.count("<think>")
        reason_count = lower_original.count("<reason>")
        action_count = lower_original.count("<action>")
        tag_counts_ok = (
            action_count == 1
            and lower_original.count("</action>") == 1
            and (think_count + reason_count) == 1
            and lower_original.count("</think>") + lower_original.count("</reason>") == 1
        )
        if not tag_counts_ok:
            valids[i] = 0

        # check if contains any Chinese characters
        if re.search(r'[\u4e00-\u9fff]', original_str):
            valids[i] = 0

    n_valid = sum(valids)
    n_total = len(valids)
    if n_valid < n_total:
        invalid_examples = [(i, actions[i][:80]) for i in range(n_total) if valids[i] == 0][:3]
        print(f"[PROJECTION/alfworld] valid={n_valid}/{n_total} invalid_examples={invalid_examples}")

    return actions, valids
