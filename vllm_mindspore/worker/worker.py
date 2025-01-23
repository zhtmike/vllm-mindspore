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
"""Worker functions"""
from typing import Tuple

import torch

from vllm.sequence import ExecuteModelRequest
from vllm.worker.worker_base import WorkerInput

from typing import Tuple

from vllm.model_executor import set_random_seed
from vllm.logger import init_logger

from vllm_mindspore.utils import _create_empty_tensor

logger = init_logger(__name__)


def _warm_up_model(self) -> None:
    # Reset the seed to ensure that the random state is not affected by
    # the model initialization and profiling.
    set_random_seed(self.model_config.seed)


def determine_num_available_blocks(self) -> Tuple[int, int]:
    logger.warning(
        "Cannot get right device memory info, just return here for mindspore!!!!!!!!!!!!"
    )
    # TODO(tronzhang): use env latter...
    return 256, 512


def prepare_worker_input(self, execute_model_req: ExecuteModelRequest) -> WorkerInput:
    virtual_engine = execute_model_req.virtual_engine
    num_steps = execute_model_req.num_steps
    num_seq_groups = len(execute_model_req.seq_group_metadata_list)
    # `blocks_to_swap_in` and `blocks_to_swap_out` are cpu tensors.
    # they contain parameters to launch cudamemcpyasync.
    if execute_model_req.blocks_to_swap_in:
        blocks_to_swap_in = torch.tensor(
            execute_model_req.blocks_to_swap_in, dtype=torch.int64
        ).view(-1, 2)
    else:
        blocks_to_swap_in = _create_empty_tensor(torch.int64)

    if execute_model_req.blocks_to_swap_out:
        blocks_to_swap_out = torch.tensor(
            execute_model_req.blocks_to_swap_out, dtype=torch.int64
        ).view(-1, 2)
    else:
        blocks_to_swap_out = _create_empty_tensor(torch.int64)
    # `blocks_to_copy` is a gpu tensor. The src and tgt of
    # blocks to copy are in the same device, and `blocks_to_copy`
    # can be used directly within cuda kernels.
    if execute_model_req.blocks_to_copy:
        blocks_to_copy = torch.tensor(
            execute_model_req.blocks_to_copy, dtype=torch.int64
        ).view(-1, 2)
    else:
        blocks_to_copy = _create_empty_tensor(torch.int64)

    return WorkerInput(
        num_seq_groups=num_seq_groups,
        blocks_to_swap_in=blocks_to_swap_in,
        blocks_to_swap_out=blocks_to_swap_out,
        blocks_to_copy=blocks_to_copy,
        virtual_engine=virtual_engine,
        num_steps=num_steps,
    )
