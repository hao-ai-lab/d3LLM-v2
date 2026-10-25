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

from .base import BaseRollout
from .api_rollout import APIRollout
from .naive import NaiveRollout

__all__ = ["BaseRollout", "NaiveRollout", "APIRollout", "MultiTurnAPIRollout", "SpeculativeMHQARollout"]


def __getattr__(name):
    if name == "MultiTurnAPIRollout":
        from .multi_turn_api_rollout import MultiTurnAPIRollout
        return MultiTurnAPIRollout
    if name == "SpeculativeMHQARollout":
        from .speculative_mhqa_rollout import SpeculativeMHQARollout
        return SpeculativeMHQARollout
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
