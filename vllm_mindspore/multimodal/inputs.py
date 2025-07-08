#!/usr/bin/env python3
# type: ignore
# isort:skip_file
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
from collections import defaultdict
from dataclasses import dataclass
from typing import Union, cast
import numpy as np
import mindspore
from vllm.multimodal.inputs import BaseMultiModalField, BatchedTensorInputs, JSONTree, json_map_leaves,\
    nested_tensors_equal
from vllm.multimodal import MultiModalKwargs
from vllm.utils import is_list_of

NestedTensors = Union[list["NestedTensors"], list[mindspore.Tensor],
                      mindspore.Tensor, tuple[mindspore.Tensor, ...]]


@dataclass
class MultiModalFieldElem:
    """
    Represents a keyword argument corresponding to a multi-modal item
    in :class:`MultiModalKwargs`.
    """

    modality: str
    """
    The modality of the corresponding multi-modal item.
    Each multi-modal item can consist of multiple keyword arguments.
    """

    key: str
    """
    The key of this field in :class:`MultiModalKwargs`,
    i.e. the name of the keyword argument to be passed to the model.
    """

    data: NestedTensors
    """
    The tensor data of this field in :class:`MultiModalKwargs`,
    i.e. the value of the keyword argument to be passed to the model.
    """

    field: "BaseMultiModalField"
    """
    Defines how to combine the tensor data of this field with others
    in order to batch multi-modal items together for model inference.
    """

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, self.__class__):
            return False

        return ((self.modality, self.key) == (other.modality, other.key)
                and nested_tensors_equal(self.data, other.data)
                and type(self.field) == type(other.field))  # noqa: E721


def as_kwargs(
    batched_inputs: BatchedTensorInputs,
    *,
    device=None,
) -> BatchedTensorInputs:
    # replace as_kwargs of vLLM for multi-model
    json_inputs = cast(JSONTree[np.ndarray], batched_inputs)

    json_mapped = json_map_leaves(
        lambda x: mindspore.Tensor(x),
        json_inputs,
    )

    return cast(BatchedTensorInputs, json_mapped)


@staticmethod
def _try_stack(nested_tensors: NestedTensors) -> NestedTensors:
    """
    Stack the inner dimensions that have the same shape in
    a nested list of tensors.

    Thus, a dimension represented by a list means that the inner
    dimensions are different for each element along that dimension.
    """
    if isinstance(nested_tensors, np.ndarray):
        return nested_tensors

    if isinstance(nested_tensors, (int, float)):
        return np.array(nested_tensors)

    stacked = [MultiModalKwargs._try_stack(t) for t in nested_tensors]
    if not is_list_of(stacked, np.ndarray, check="all"):
        # Only tensors (not lists) can be stacked.
        return stacked

    tensors_ = cast(list[np.ndarray], stacked)
    if len(tensors_) == 1:
        # An optimization when `tensors_` contains only one tensor:
        # - produce exactly same result as `torch.stack(tensors_)`
        # - will achieve zero-copy if the tensor is contiguous
        return tensors_[0][None]

    if any(t.shape != tensors_[0].shape for t in tensors_):
        # The tensors have incompatible shapes and can't be stacked.
        return tensors_

    return np.stack(tensors_)
