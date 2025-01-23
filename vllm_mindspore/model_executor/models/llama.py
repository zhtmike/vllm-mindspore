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

from typing import TYPE_CHECKING, Any, Dict, Iterable, List, Optional, Set, Tuple, Type

if TYPE_CHECKING:
    from transformers import LlamaConfig
else:
    LlamaConfig = None

# TODO(tronzhang): use vllm's latter...
from vllm_mindspore.distributed.parallel_state import (
    is_first_pp_rank,
    is_last_pp_rank,
    get_tensor_model_parallel_world_size,
)

from vllm_mindspore.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm_mindspore.model_executor.layers.logits_processor import LogitsProcessor
from vllm_mindspore.attention import Attention
from vllm_mindspore.model_executor.layers.activation import SiluAndMul
from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import (
    DEFAULT_VOCAB_PADDING_SIZE,
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm_mindspore.model_executor.models.utils import (
    PPMissingLayer,
    extract_layer_index,
    make_layers,
    maybe_prefix,
    make_empty_intermediate_tensors_factory,
)
from vllm_mindspore.model_executor.layers.sampler import get_sampler, SamplerOutput
from vllm_mindspore.model_executor.layers.layernorm import RMSNorm
from vllm_mindspore.model_executor.layers.rotary_embedding import get_rope
from vllm_mindspore.model_executor.sampling_metadata import SamplingMetadata

from vllm_mindspore.model_executor.models.model_base import MsModelBase

from vllm.sequence import IntermediateTensors
from vllm.attention import AttentionMetadata

import mindspore as ms
from mindspore import mint
from mindspore import Tensor


def default_weight_loader(param, loaded_weight) -> None:
    param.set_data(loaded_weight)


class LlamaMLP(ms.nn.Cell):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config=None,
        bias: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            input_size=hidden_size,
            output_sizes=[intermediate_size] * 2,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            input_size=intermediate_size,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. "
                "Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def construct(self, x):
        x, _ = self.gate_up_proj(x)
        x = self.act_fn(x)
        x, _ = self.down_proj(x)
        return x


class LlamaAttention(ms.nn.Cell):
    def __init__(
        self,
        config: LlamaConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_theta: float = 10000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        max_position_embeddings: int = 8192,
        quant_config=None,
        bias: bool = False,
        cache_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        layer_idx = extract_layer_index(prefix)
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        # MistralConfig has an optional head_dim introduced by Mistral-Nemo
        self.head_dim = getattr(
            config, "head_dim", self.hidden_size // self.total_num_heads
        )
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.rope_theta = rope_theta
        self.max_position_embeddings = max_position_embeddings

        self.qkv_proj = QKVParallelLinear(
            hidden_size=hidden_size,
            head_size=self.head_dim,
            total_num_heads=self.total_num_heads,
            total_num_kv_heads=self.total_num_kv_heads,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )

        self.o_proj = RowParallelLinear(
            input_size=self.total_num_heads * self.head_dim,
            output_size=hidden_size,
            bias=bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        is_neox_style = True
        if quant_config is not None and quant_config.get_name() == "gguf":
            is_neox_style = False
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
            is_neox_style=is_neox_style,
        )

        if hasattr(config, "interleaved_sliding_window"):
            if isinstance(config.interleaved_sliding_window, int):
                sliding_window = config.interleaved_sliding_window
            elif isinstance(config.interleaved_sliding_window, list):
                sw_idx = layer_idx % len(config.interleaved_sliding_window)
                sliding_window = config.interleaved_sliding_window[sw_idx]
            else:
                raise ValueError(f"{type(sliding_window)} is not supported.")
        else:
            sliding_window = None

        self.attn = Attention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def construct(
        self,
        positions,
        hidden_states,
        kv_cache,
        attn_metadata,
    ):
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], axis=-1)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v, kv_cache, attn_metadata)
        output, _ = self.o_proj(attn_output)
        return output


class LlamaDecoderLayer(ms.nn.Cell):
    def __init__(
        self,
        config: LlamaConfig,
        cache_config=None,
        quant_config=None,
        prefix: str = "",
    ):
        super().__init__()
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 10000)
        rope_scaling = getattr(config, "rope_scaling", None)
        if rope_scaling is not None and getattr(
            config, "original_max_position_embeddings", None
        ):
            rope_scaling["original_max_position_embeddings"] = (
                config.original_max_position_embeddings
            )
        max_position_embeddings = getattr(config, "max_position_embeddings", 8192)
        # Support abacusai/Smaug-72B-v0.1 with attention_bias
        # Support internlm/internlm-7b with bias
        attention_bias = getattr(config, "attention_bias", False) or getattr(
            config, "bias", False
        )
        self.self_attn = LlamaAttention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=getattr(
                config, "num_key_value_heads", config.num_attention_heads
            ),
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            bias=attention_bias,
            cache_config=cache_config,
            prefix=f"{prefix}.self_attn",
        )
        self.mlp = LlamaMLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            bias=getattr(config, "mlp_bias", False),
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def construct(
        self,
        positions,
        hidden_states,
        kv_cache,
        attn_metadata: AttentionMetadata,
        residual,
    ):
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)

        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            kv_cache=kv_cache,
            attn_metadata=attn_metadata,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


class LlamaModel(ms.nn.Cell):
    SUPPORT_LORA = False
    SUPPORT_PP = False

    def __init__(
        self,
        *,
        vllm_config,
        prefix: str = "",
        layer_type: Type[LlamaDecoderLayer] = LlamaDecoderLayer,
    ):
        super().__init__()
        config = vllm_config
        self.config = config
        self.padding_idx = config.pad_token_id
        # TODO: Support lora_config
        # lora_config = config
        # lora_vocab = (lora_config.lora_extra_vocab_size *
        #              (lora_config.max_loras or 1)) if lora_config else 0
        # self.vocab_size = config.vocab_size + lora_vocab
        self.vocab_size = config.vocab_size
        self.org_vocab_size = config.vocab_size
        # TODO: Support quant_config cache_config
        quant_config = None
        cache_config = None

        if is_first_pp_rank() or (config.tie_word_embeddings and is_last_pp_rank()):
            self.embed_tokens = VocabParallelEmbedding(
                self.vocab_size,
                config.hidden_size,
                org_num_embeddings=config.vocab_size,
                quant_config=quant_config,
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        if is_last_pp_rank():
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def get_input_embeddings(self, input_ids: Tensor) -> Tensor:
        return self.embed_tokens(input_ids)

    # NOTE: vllm传下来的input_ids是多batch的token拉平的， 模型需要这种形式的输入。
    def construct(
        self,
        input_ids: Optional[Tensor],
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors=None,
        inputs_embeds: Optional[Tensor] = None,
    ):
        if is_first_pp_rank():
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.get_input_embeddings(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        for i in range(self.start_layer, self.end_layer):  # PP 并行对层进行切分
            layer = self.layers[i]
            hidden_states, residual = layer(
                positions,
                hidden_states,
                kv_caches[i - self.start_layer],
                attn_metadata,
                residual,
            )

        if not is_last_pp_rank():
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states

    def load_weights(self, weights: Iterable[Tuple[str, Tensor]], params_dict):
        loaded_params: Set[str] = set()
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            (".qkv_proj", ".q_proj", "q"),
            (".qkv_proj", ".k_proj", "k"),
            (".qkv_proj", ".v_proj", "v"),
            (".gate_up_proj", ".gate_proj", 0),
            (".gate_up_proj", ".up_proj", 1),
        ]

        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue

            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = param.weight_loader
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_params.add(name)
                    break
            else:
                if name in params_dict:
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(name)

        return loaded_params


class LlamaForCausalLM(MsModelBase):
    def __init__(self, vllm_config, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

        quant_config = vllm_config.quant_config
        self.model = LlamaModel(vllm_config=self.config)

        if is_last_pp_rank():
            self.unpadded_vocab_size = self.config.vocab_size
            # TODO: To support lora
            # if self.lora_config:
            #   self.unpadded_vocab_size += self.lora_config.lora_extra_vocab_size
            # self.unpadded_vocab_size += config.lora_extra_vocab_size
            self.lm_head = ParallelLMHead(
                self.unpadded_vocab_size,
                self.config.hidden_size,
                org_num_embeddings=self.config.vocab_size,
                padding_size=(
                    DEFAULT_VOCAB_PADDING_SIZE
                    # We need bigger padding if using lora for kernel
                    # compatibility
                    if not self.lora_config
                    else self.lora_config.lora_vocab_padding_size
                ),
                quant_config=quant_config,
                prefix=maybe_prefix(prefix, "lm_head"),
            )
            # if self.config.tie_word_embeddings:
            #     self.lm_head = self.lm_head.tie_weights(
            #         self.model.embed_tokens)

            logit_scale = getattr(self.config, "logit_scale", 1.0)
            self.logits_processor = LogitsProcessor(
                self.unpadded_vocab_size, self.config.vocab_size, logit_scale
            )
            self.sampler = get_sampler()
        else:
            self.lm_head = PPMissingLayer()

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

    def tie_lmhead_weights(self):
        self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

    def construct(
        self,
        input_ids,
        positions,
        kv_caches,
        attn_metadata,
        intermediate_tensors=None,
        inputs_embeds=None,
    ):

        # intermediate_tensors 是PP并行中， 传递过来的上一层网络的输出（hidden_state，residual）。
        # inputs_embeds 是预先将token_id转换成embedding传进来的， 可以让网络不需要再做一次embedding操作。
        model_output = self.model(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )
        return model_output

    # 这个接口的定义是vllm定义的， 入参不可改变， 不然vllm会调用不到。
    # TODO：weights入参当前定义为dict格式， 原生为tuple格式， 待插件上层最终决定。
    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        # weights入参传入为完整的、未切分的权重，需要在该函数中实现切分和加载逻辑。
        # 这里vllm默认会每个进程加载一个完整的权重， 可能会有内存重复， 可优化。

        # 在这里调用parameters_dict， 是为了能将self.lm_head考虑进去。
        params_dict = self.parameters_dict()
        self.model.load_weights(weights, params_dict)

    def sample(
        self, logits: Tensor, sampling_metadata: SamplingMetadata
    ) -> Optional[SamplerOutput]:
        next_tokens = self.sampler(logits, sampling_metadata)
        return next_tokens

    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        logits = self.logits_processor(self.lm_head, hidden_states, sampling_metadata)
        return logits
