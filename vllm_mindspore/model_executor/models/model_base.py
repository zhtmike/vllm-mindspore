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

from abc import abstractmethod
from typing import Iterable, List, Optional, Set, Tuple, Union

from vllm.attention import AttentionMetadata
from vllm.config import VllmConfig
from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.model_executor.sampling_metadata import SamplingMetadata
from vllm.sequence import IntermediateTensors

import mindspore
from mindspore import Tensor


class MsModelBase(mindspore.nn.Cell):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        super(MsModelBase, self).__init__()
        config = vllm_config.model_config.hf_config
        lora_config = vllm_config.lora_config

        self.config = config
        self.lora_config = lora_config

    def named_parameters(self):
        return self.parameters_and_names()

    def named_modules(self):
        return self.cells_and_names()

    def get_submodule(self):
        raise RuntimeError("Cannot get submodule for mindspore model now!")

    def eval(self):
        self.set_train(False)
        return self

    def forward(
        self,
        input_ids: Tensor,
        positions: Tensor,
        kv_caches: List[Tensor],
        attn_metadata: AttentionMetadata,
        intermediate_tensors: Optional[IntermediateTensors] = None,
        inputs_embeds: Optional[Tensor] = None,
    ) -> Union[Tensor, IntermediateTensors]:
        raise self.__call__(
            input_ids,
            positions,
            kv_caches,
            attn_metadata,
            intermediate_tensors,
            inputs_embeds,
        )

    @abstractmethod
    def compute_logits(
        self,
        hidden_states: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[Tensor]:
        raise NotImplementedError("Function compute_logits should be Implemented!")

    @abstractmethod
    def sample(
        self,
        logits: Tensor,
        sampling_metadata: SamplingMetadata,
    ) -> Optional[SamplerOutput]:
        raise NotImplementedError("Function sample should be Implemented!")

    @abstractmethod
    def load_weights(self, weights: Iterable[Tuple[str, Tensor]]) -> Set[str]:
        raise NotImplementedError("Function load_weights should be Implemented!")
