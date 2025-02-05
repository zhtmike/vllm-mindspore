#!/usr/bin/env python3
# encoding: utf-8
# Copyright 2025 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================

from typing import List, Optional, Union

import torch
import torch.distributed
from torch.distributed import Backend


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
) -> "GroupCoordinator":
    from vllm.distributed.parallel_state import (
        GroupCoordinator,
        _ENABLE_CUSTOM_ALL_REDUCE,
    )

    if use_custom_allreduce is None:
        use_custom_allreduce = _ENABLE_CUSTOM_ALL_REDUCE

    # TODO(tronzhang): mindspore doesnot support enough communicate cpu ops, set use_message_queue_broadcaster to False now.
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=False,
        use_custom_allreduce=use_custom_allreduce,
        use_tpu_communicator=False,
        use_hpu_communicator=False,
        use_xpu_communicator=False,
        use_message_queue_broadcaster=False,
        group_name=group_name,
    )


def all_reduce_for_GroupCoordinator(self, input_: torch.Tensor) -> torch.Tensor:
    """
    User-facing all-reduce function before we actually call the
    all-reduce operation.

    We need this because Dynamo does not support passing an arbitrary
    object (`self` in this case) to a custom op. We need to pass the
        group name as a string, and then look up the group coordinator from
        the group name, dispatch the all-reduce operation to the group
        coordinator.

    In addition, PyTorch custom ops do not support mutation or returning
    a new tensor in the same op. So we always make the all-reduce operation
    out-of-place.
    """
    # Bypass the function if we are using only 1 GPU.
    if self.world_size == 1:
        return input_

    torch.distributed.all_reduce(input_, group=self.device_group)
    return input_


def init_for_GroupCoordinator(
    self,
    group_ranks: List[List[int]],
    local_rank: int,
    torch_distributed_backend: Union[str, Backend],
    use_pynccl: bool,
    use_custom_allreduce: bool,
    use_tpu_communicator: bool,
    use_hpu_communicator: bool,
    use_xpu_communicator: bool,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
):
    from vllm.distributed.parallel_state import _get_unique_name, _register_group
    from vllm.platforms import current_platform

    group_name = group_name or "anonymous"
    self.unique_name = _get_unique_name(group_name)
    _register_group(self)

    self.rank = torch.distributed.get_rank()
    self.local_rank = local_rank
    self.device_group = None
    self.cpu_group = None

    for ranks in group_ranks:
        device_group = torch.distributed.new_group(
            ranks, backend=torch_distributed_backend)
        # a group with `gloo` backend, to allow direct coordination between
        # processes through the CPU.
        cpu_group = torch.distributed.new_group(ranks, backend="gloo")
        if self.rank in ranks:
            self.ranks = ranks
            self.world_size = len(ranks)
            self.rank_in_group = ranks.index(self.rank)
            self.device_group = device_group
            self.cpu_group = cpu_group

    assert self.cpu_group is not None
    assert self.device_group is not None

    if current_platform.is_cuda_alike():
        self.device = torch.device(f"cuda:{local_rank}")
    else:
        self.device = torch.device("cpu")

    self.use_pynccl = use_pynccl
    self.use_custom_allreduce = use_custom_allreduce
    self.use_tpu_communicator = use_tpu_communicator
    self.use_hpu_communicator = use_hpu_communicator
    self.use_xpu_communicator = use_xpu_communicator

    # lazy import to avoid documentation build error
    from vllm.distributed.device_communicators.custom_all_reduce import (
        CustomAllreduce)
    from vllm.distributed.device_communicators.pynccl import (
        PyNcclCommunicator)

    self.pynccl_comm: Optional[PyNcclCommunicator] = None
    if use_pynccl and self.world_size > 1:
        self.pynccl_comm = PyNcclCommunicator(
            group=self.cpu_group,
            device=self.device,
        )

    self.ca_comm: Optional[CustomAllreduce] = None
    if use_custom_allreduce and self.world_size > 1:
        # Initialize a custom fast all-reduce implementation.
        self.ca_comm = CustomAllreduce(
            group=self.cpu_group,
            device=self.device,
        )

    from vllm.distributed.device_communicators.tpu_communicator import (
        TpuCommunicator)
    self.tpu_communicator: Optional[TpuCommunicator] = None
    if use_tpu_communicator and self.world_size > 1:
        self.tpu_communicator = TpuCommunicator(group=self.cpu_group)

    from vllm.distributed.device_communicators.hpu_communicator import (
        HpuCommunicator)
    self.hpu_communicator: Optional[HpuCommunicator]
    if use_hpu_communicator and self.world_size > 1:
        self.hpu_communicator = HpuCommunicator(group=self.device_group)

    from vllm.distributed.device_communicators.xpu_communicator import (
        XpuCommunicator)
    self.xpu_communicator: Optional[XpuCommunicator] = None
    if use_xpu_communicator and self.world_size > 1:
        self.xpu_communicator = XpuCommunicator(group=self.device_group)

    from vllm.distributed.device_communicators.shm_broadcast import (
        MessageQueue)
    self.mq_broadcaster: Optional[MessageQueue] = None
    if use_message_queue_broadcaster and self.world_size > 1:
        self.mq_broadcaster = MessageQueue.create_from_process_group(
            self.cpu_group, 1 << 22, 6)
