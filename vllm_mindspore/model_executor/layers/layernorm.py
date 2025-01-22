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

from typing import Optional, Tuple, Union

from mindspore import Tensor
from mindspore.ops import rms_norm
from mindspore import mint
import mindspore as ms

from vllm_mindspore.model_executor.custom_op import CustomOp


class RMSNorm(CustomOp):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        var_hidden_size: Optional[int] = None,
    ):
        super().__init__()

        self.hidden_size = hidden_size
        self.variance_epsilon = eps
        self.variance_size_override = (
            None if var_hidden_size == hidden_size else var_hidden_size
        )
        self.weight = ms.Parameter(mint.ones(hidden_size))

    def forward_native(
        self, x: Tensor, residual: Optional[Tensor] = None
    ) -> Union[Tensor, Tuple[Tensor, Tensor]]:
        orig_dtype = x.dtype
        x = x.to(ms.float32)
        if residual is not None:
            x = x + residual.to(ms.float32)
            residual = x.to(orig_dtype)
        output, _ = rms_norm(x, self.weight, self.variance_epsilon)
        output = output.to(orig_dtype)
        if residual is None:
            return output
        else:
            return output, residual
