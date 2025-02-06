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

