# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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

import pickle
from typing import Any, List, Optional

import numpy as np
import torch
import torch.distributed as dist


def compute_batch_wall_times(outputs: List[dict[str, Any]]) -> tuple[float, float]:
    """Return batch-level prefill and decode wall times from request timings.

    SGLang gives every request in one batched ``async_generate`` call the same
    creation timestamp.  Request-level prefill/decode durations therefore define
    intervals on a shared timeline.  Summing those durations would count
    concurrent work once per request and understate throughput for batch size > 1.
    """
    intervals = []
    for output in outputs:
        meta_info = output.get("meta_info", {})
        if "prefill_time" not in meta_info or "decode_time" not in meta_info:
            continue
        prefill_time = max(float(meta_info["prefill_time"] or 0.0), 0.0)
        decode_time = max(float(meta_info["decode_time"] or 0.0), 0.0)
        intervals.append((prefill_time, prefill_time + decode_time))

    if not intervals:
        return 0.0, 0.0

    prefill_wall_time = max(first_token_time for first_token_time, _ in intervals)
    decode_wall_time = max(finish_time for _, finish_time in intervals) - min(
        first_token_time for first_token_time, _ in intervals
    )
    return prefill_wall_time, max(decode_wall_time, 0.0)


def broadcast_pyobj(
    data: List[Any],
    rank: int,
    dist_group: Optional[torch.distributed.ProcessGroup] = None,
    src: int = 0,
    force_cpu_device: bool = False,
):
    """from https://github.com/sgl-project/sglang/blob/844e2f227ab0cce6ef818a719170ce37b9eb1e1b/python/sglang/srt/utils.py#L905

    Broadcast inputs from src rank to all other ranks with torch.dist backend.
    The `rank` here refer to the source rank on global process group (regardless
    of dist_group argument).
    """
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not force_cpu_device else "cpu"
    )

    if rank == src:
        if len(data) == 0:
            tensor_size = torch.tensor([0], dtype=torch.long, device=device)
            dist.broadcast(tensor_size, src=src, group=dist_group)
        else:
            serialized_data = pickle.dumps(data)
            size = len(serialized_data)

            tensor_data = torch.ByteTensor(
                np.frombuffer(serialized_data, dtype=np.uint8).copy()
            ).to(device)
            tensor_size = torch.tensor([size], dtype=torch.long, device=device)

            dist.broadcast(tensor_size, src=src, group=dist_group)
            dist.broadcast(tensor_data, src=src, group=dist_group)
        return data
    else:
        tensor_size = torch.tensor([0], dtype=torch.long, device=device)
        dist.broadcast(tensor_size, src=src, group=dist_group)
        size = tensor_size.item()

        if size == 0:
            return []

        tensor_data = torch.empty(size, dtype=torch.uint8, device=device)
        dist.broadcast(tensor_data, src=src, group=dist_group)

        serialized_data = bytes(tensor_data.cpu().numpy())
        data = pickle.loads(serialized_data)
        return data
