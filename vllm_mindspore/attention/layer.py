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
"""Common layer for LLM."""
from typing import Optional, List

from mindspore import Tensor
from mindspore import nn
from mindspore.ops.auto_generate import ReshapeAndCache, PagedAttention
from mindspore.ops.operations.nn_ops import FlashAttentionScore
from mindspore import mint
from mindspore import ops
import mindspore as ms

from vllm.attention.backends.abstract import AttentionType, AttentionMetadata


def _pad_to_max_tensor(tensor, max_len, dim=0, pad_value=-1):
    if tensor.shape[dim] == max_len:
        return tensor

    dst_shape = tensor.shape
    pad_tensor = ops.fill(
        type=tensor.dtype,
        shape=(max_len - dst_shape[dim], *dst_shape[dim + 1 :]),
        value=pad_value,
    )
    output = mint.cat([tensor, pad_tensor], dim=dim)
    return output


def _generate_attn_mask(bsz, q_len, kv_len, flatten=False):
    if flatten:
        raise NotImplementedError("flatten not support yet")
    mask = mint.ones((bsz, 1, q_len, kv_len), dtype=ms.uint8)
    mask = mint.triu(mask, diagonal=1)
    return mask


class Attention(nn.Cell):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: Optional[int] = None,
        alibi_slopes: Optional[List[float]] = None,
        cache_config=None,
        quant_config=None,
        blocksparse_params=None,
        logits_soft_cap: Optional[float] = None,
        per_layer_sliding_window: Optional[int] = None,
        prefix: str = "",
    ):
        super().__init__()
        if not num_kv_heads:
            num_kv_heads = num_heads
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)

        if cache_config is not None:
            kv_cache_dtype = cache_config.cache_dtype
            block_size = cache_config.block_size
            is_attention_free = cache_config.is_attention_free
        else:
            kv_cache_dtype = "auto"
            block_size = 16
            is_attention_free = False

        self.paged_attention = PagedAttention(num_heads, self.scale, num_kv_heads)
        self.reshape_and_cache = ReshapeAndCache()

        keep_prob = 1.0
        pre_tokens = 2147483647
        next_tokens = 2147483647
        self.input_layout = "BSH"
        self.sparse_mode = 0
        self.flash_attention = FlashAttentionScore(
            head_num=num_heads,
            keep_prob=keep_prob,
            scale_value=self.scale,
            pre_tokens=pre_tokens,
            next_tokens=next_tokens,
            inner_precise=0,
            input_layout=self.input_layout,
            sparse_mode=self.sparse_mode,
        )

    def construct(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        kv_cache: Tensor,
        attn_metadata: AttentionMetadata,
        attn_type: str = AttentionType.DECODER,
    ):
        block_table = attn_metadata.block_tables
        slot_mapping = attn_metadata.slot_mapping

        if key is not None:
            assert value is not None
        else:
            assert value is None

        if attn_type != AttentionType.ENCODER and (
            kv_cache is not None and kv_cache[0].numel() > 0
        ):
            # KV-cache during decoder-self- or
            # encoder-decoder-cross-attention, but not
            # during encoder attention.
            #
            # Even if there are no new key/value pairs to cache,
            # we still need to break out key_cache and value_cache
            # i.e. for later use by paged attention
            key_cache, value_cache = self.split_kv_cache(
                kv_cache, self.num_kv_heads, self.head_size
            )

            if (key is not None) and (value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    # During cross-attention decode, key & value will be None,
                    # preventing this IF-statement branch from running
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                self.reshape_and_cache(
                    key.view(1, *key.shape),
                    value.view(1, *value.shape),
                    key_cache,
                    value_cache,
                    updated_slot_mapping,
                )

        if prefill_meta := attn_metadata.prefill_metadata:
            assert attn_metadata.seq_lens is not None
            if not prefill_meta.prefill_metadata.chunked_prefill:  # type: ignore
                # 只有图模式atb下才能支持拉平。
                output = self._run_prefill_forward(
                    query, key, value, prefill_meta, attn_type=attn_type
                )
            else:
                # TODO: to support CPP
                raise NotImplementedError("not support CPP yet")

        if decode_meta := attn_metadata.decode_metadata:
            assert (
                attn_type != AttentionType.ENCODER_ONLY
            ), "Encoder-only models should not have decode metadata."
            # Decoding run.
            (
                seq_lens_arg,
                max_seq_len_arg,
                block_tables_arg,
            ) = decode_meta.get_seq_len_block_table_args(attn_type)

            alibi_mask = None
            output = self._run_decode_attention(
                query,
                key_cache,
                value_cache,
                seq_lens_arg,
                block_tables_arg,
                alibi_mask,
            )

        return output

    def _run_decode_attention(
        self, query, key_cache, value_cache, seq_lens_arg, block_tables, alibi_mask=None
    ):
        # TODO: to support alibi mask
        # if self.use_alibi_mask:
        #     return self.paged_attention(query, key_cache, value_cache, batch_valid_length, block_tables, alibi_mask)

        bsz = block_tables.shape[0]
        query = query.reshape(bsz, -1, query.shape[-1])
        output = self.paged_attention(
            query, key_cache, value_cache, block_tables, context_lens=seq_lens_arg
        )
        output = output.reshape(-1, query.shape[-1])
        return output

    def _run_prefill_forward(
        self,
        query,
        key,
        value,
        attn_metadata: AttentionMetadata,
        attn_type: str = AttentionType.DECODER,
    ):
        causal_attn = attn_type == AttentionType.DECODER
        alibi_mask = None
        padding_mask = None
        prefix = None

        seq_lens_q, seq_lens_kv = attn_metadata.get_seq_lens(attn_type)
        seq_lens, _ = attn_metadata.get_seq_lens(attn_type)

        max_q_len = max(seq_lens_q)
        max_kv_len = max(seq_lens_kv)

        start_pos = 0
        q_list = []
        for seq_len_q in seq_lens_q:
            q = query[start_pos : start_pos + seq_len_q, :]
            q = _pad_to_max_tensor(q, max_q_len)
            q_list.append(q)
            start_pos += seq_len_q
        query = mint.stack(q_list)

        k_list = []
        v_list = []
        start_pos = 0
        for seq_len_kv in seq_lens_kv:
            k = key[start_pos : start_pos + seq_len_kv, :]
            k = _pad_to_max_tensor(k, max_kv_len)
            k_list.append(k)

            v = value[start_pos : start_pos + seq_len_kv, :]
            v = _pad_to_max_tensor(v, max_kv_len)
            v_list.append(v)

            start_pos += seq_len_kv

        key = mint.stack(k_list)
        value = mint.stack(v_list)

        batch_size = len(seq_lens_q)
        attn_masks = _generate_attn_mask(
            batch_size, max_q_len, max_kv_len, flatten=False
        )
        _, _, _, output = self.flash_attention(
            query,
            key,
            value,
            alibi_mask,
            None,
            padding_mask,
            attn_masks,
            prefix,
            None,
            None,
        )

        output_list = []
        bsz = output.shape[0]
        for i in range(bsz):
            output_i = output[i]
            seq_len = seq_lens_q[i]
            output_i = output_i[:seq_len, ...]
            output_list.append(output_i)
        output = mint.cat(output_list, dim=0)
        return output

    def split_kv_cache(
        self,
        kv_cache,
        num_kv_heads: int,
        head_size: int,
    ):
        # TODO: to support view operation on mindspore.
        # num_blocks = kv_cache.shape[1]

        # key_cache = kv_cache[0]
        # key_cache = key_cache.view(num_blocks, -1, num_kv_heads, head_size)
        # value_cache = kv_cache[1]
        # value_cache = value_cache.view(num_blocks, -1, num_kv_heads, head_size)
        key_cache = kv_cache[0]
        value_cache = kv_cache[1]
        return key_cache, value_cache
