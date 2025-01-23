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

import math
from typing import Any, Dict, List, Optional, Tuple, Union

from mindspore import Tensor
from mindspore import mint
from mindspore import ops
import mindspore as ms
from vllm_mindspore.model_executor.custom_op import CustomOp


def _apply_rotary_emb(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    is_neox_style: bool,
) -> Tensor:
    """
    Args:
        x: [num_tokens, num_heads, head_size]
        cos: [num_tokens, head_size // 2]
        sin: [num_tokens, head_size // 2]
        is_neox_style: Whether to use the Neox-style or GPT-J-style rotary
            positional embeddings.
    """
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = mint.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return mint.cat((o1, o2), dim=-1)
    else:
        return mint.stack((o1, o2), dim=-1).flatten(-2)


class RotaryEmbedding(CustomOp):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache = cache
        # self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (
            base
            ** (mint.arange(0, self.rotary_dim, 2, dtype=ms.float32) / self.rotary_dim)
        )
        return inv_freq

    def _compute_cos_sin_cache(self) -> Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = mint.arange(self.max_position_embeddings, dtype=ms.float32)

        # freqs = ops.einsum("i,j -> ij", t, inv_freq)
        freqs = ops.outer(t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = mint.cat((cos, sin), dim=-1)
        return cache

    def forward_native(
        self,
        positions: Tensor,
        query: Tensor,
        key: Tensor,
        offsets: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]
        cos_sin = self.cos_sin_cache.index_select(0, positions)
        cos, sin = cos_sin.chunk(2, axis=-1)

        query_shape = query.shape
        query = query.view(num_tokens, -1, self.head_size)
        query_rot = query[..., : self.rotary_dim]
        query_pass = query[..., self.rotary_dim :]
        query_rot = _apply_rotary_emb(query_rot, cos, sin, self.is_neox_style)
        query = mint.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

        key_shape = key.shape
        key = key.view(num_tokens, -1, self.head_size)
        key_rot = key[..., : self.rotary_dim]
        key_pass = key[..., self.rotary_dim :]
        key_rot = _apply_rotary_emb(key_rot, cos, sin, self.is_neox_style)
        key = mint.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
        return query, key


_ROPE_DICT: Dict[Tuple, RotaryEmbedding] = {}


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: int,
    is_neox_style: bool = True,
    rope_scaling: Optional[Dict[str, Any]] = None,
    dtype=None,
    partial_rotary_factor: float = 1.0,
) -> RotaryEmbedding:
    if dtype is None:
        # dtype = torch.get_default_dtype()
        dtype = ms.float16

    if rope_scaling is not None:
        # Transforms every value that is a list into a tuple for caching calls
        rope_scaling_tuple = {
            k: tuple(v) if isinstance(v, list) else v for k, v in rope_scaling.items()
        }
        rope_scaling_args = tuple(rope_scaling_tuple.items())
    else:
        rope_scaling_args = None
    if partial_rotary_factor < 1.0:
        rotary_dim = int(rotary_dim * partial_rotary_factor)
    key = (
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        rope_scaling_args,
        dtype,
    )

    if key in _ROPE_DICT:
        return _ROPE_DICT[key]
    rotary_emb = RotaryEmbedding(
        head_size,
        rotary_dim,
        max_position,
        base,
        is_neox_style,
        dtype,
    )
    _ROPE_DICT[key] = rotary_emb
    return rotary_emb
