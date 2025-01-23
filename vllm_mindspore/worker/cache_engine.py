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
"""CacheEngine class for managing the KV cache."""

from typing import List

from vllm.logger import init_logger

logger = init_logger(__name__)

from vllm_mindspore.utils import MsKVCache

import mindspore as ms
from mindspore.common.initializer import Zero

STR_DTYPE_TO_MS_DTYPE = {
    "half": ms.float16,
    "float16": ms.float16,
    "bfloat16": ms.bfloat16,
    "float": ms.float32,
    "fp8": ms.uint8,
    "fp8_e4m3": ms.uint8,
    "fp8_e5m2": ms.uint8,
}


def create_block(shape, dtype, name=None, device=None):
    # from mindspore.ops.function.array_func import empty as empty_tensor
    # blocks = empty_tensor(shape, dtype=dtype, device=device)
    blocks = ms.Parameter(
        ms.Tensor(shape=shape, dtype=dtype, init=Zero()), name=name, requires_grad=False
    )
    return blocks


def ms_allocate_kv_cache(
    self,
    num_blocks: int,
    device: str,
) -> List[MsKVCache]:
    """Allocates KV cache on the specified device."""
    kv_cache_shape = self.attn_backend.get_kv_cache_shape(
        num_blocks, self.block_size, self.num_kv_heads, self.head_size
    )
    kv_cache: List[MsKVCache] = []

    if isinstance(self.dtype, str):
        self.dtype = STR_DTYPE_TO_MS_DTYPE[self.dtype]

    # TODO(tronzhang): A shape with (2, ...) for a kv tensor cannot support in mindspore's tensor and block operation, so split it to two tensor.
    for _ in range(self.num_attention_layers):
        if device == "cpu":
            key_blocks = create_block(kv_cache_shape[1:], self.dtype, device="CPU")
            value_blocks = create_block(kv_cache_shape[1:], self.dtype, device="CPU")
        else:
            key_blocks = create_block(kv_cache_shape[1:], self.dtype, device="Ascend")
            value_blocks = create_block(kv_cache_shape[1:], self.dtype, device="Ascend")
        kv_cache.append((key_blocks, value_blocks))
    return kv_cache


def ms_swap_in(self, src_to_dst: ms.Tensor) -> None:
    for i in range(self.num_attention_layers):
        self.attn_backend.swap_blocks(
            self.cpu_cache[i], self.gpu_cache[i], src_to_dst, False
        )


def ms_swap_out(self, src_to_dst: ms.Tensor) -> None:
    for i in range(self.num_attention_layers):
        self.attn_backend.swap_blocks(
            self.gpu_cache[i], self.cpu_cache[i], src_to_dst, True
        )
