#!/usr/bin/env python3
# Copyright 2025 Huawei Technologies Co., Ltd
# Copyright 2024 The vLLM team.
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
# ============================================================================

# 该文件实现底层通信接口， 要求动静统一， 最后才可以在网络中入图。
# 不要去照搬mindspeed的， 因为训练当中包含太多的特性， 推理只需要非常简单的通信，可以提升性能。

from typing import Any, Dict, Optional, Union

from mindspore import Tensor, nn, ops
from mindspore.communication.comm_func import all_reduce, broadcast
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size,
    get_tp_group, get_world_group)


def tensor_model_parallel_all_reduce(input_: Tensor) -> Tensor:
    if get_tensor_model_parallel_world_size() == 1:
        return input_
    """All-reduce the input tensor across model parallel group."""
    output, _ = all_reduce(input_, group=get_tp_group())
    return output


def broadcast_tensor(tensor, src: int = 0):
    # broadcast tensor to the world group
    return broadcast(tensor, src, group=get_world_group())


def broadcast_tensor_dict(tensor_dict: Optional[Dict[Any, Union[Tensor,
                                                                Any]]] = None,
                          src: int = 0):
    ...
    # if not torch.distributed.is_initialized():
    #     return tensor_dict
    # return get_tp_group().broadcast_tensor_dict(tensor_dict, src)


class ReduceFromModelParallelRegion(nn.Cell):
    "All reduce the input from the model parallel region."

    def __init__(self):
        super().__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tp_group().device_group._name
            self.all_reduce = ops.AllReduce(group=self.tp_group)

    def construct(self, input_):
        if self.world_size == 1:
            return input_
        output = self.all_reduce(input_)
        return output


class GatherFromModelParallelRegion(nn.Cell):
    "Gather the input from model parallel region and concatenate."

    def __init__(self):
        super().__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        self.tp_rank = get_tensor_model_parallel_rank()
        if self.world_size > 1:
            self.tp_group = get_tp_group().device_group._name

    def construct(self,
                  input_: Tensor,
                  dst: int = 0,
                  dim: int = -1) -> Optional[Tensor]:
        # Size and dimension.
        if self.world_size == 1:
            return input_
        output = ops.CollectiveGather(dest_rank=dst,
                                      group=self.tp_group)(input_.transpose(
                                          2, 1, 0))
        if self.tp_rank != dst:
            return ops.depend(ops.zeros_like(input_), output)
        return output.transpose(2, 1, 0)


class AllGatherFromModelParallelRegion(nn.Cell):
    """
    Gather the input from world parallel region and concatenate, simultaneously perform
    transpose operation on input.
    """

    def __init__(self):
        super().__init__()
        self.world_size = get_tensor_model_parallel_world_size()
        if self.world_size > 1:
            self.tp_group = get_tp_group().device_group._name
            self.all_gather_into_tensor = ops.AllGather(group=self.tp_group)

    def construct(self, input_):
        # Size and dimension.
        if self.world_size == 1:
            return input_
        input_ = ops.swapaxes(input_, 0, -1)
        output = self.all_gather_into_tensor(input_)
        output = ops.swapaxes(output, 0, -1)
        return output
