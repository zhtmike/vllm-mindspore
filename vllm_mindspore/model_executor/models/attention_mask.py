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

"""
infer attention mask.
"""
import numpy as np

from mindspore import Tensor, mint


class LowerTriangularMask:
    r"""
    Provide Infer model attention mask.
    Args:
        dtype (mstype): The dtype of the mask.
        max_model_len (int): The maximum length of the model.

    """

    def __init__(self, dtype, max_model_len):
        self.prefill_mask = Tensor.from_numpy(np.triu(np.ones(shape=(128, 128), dtype=np.float16), k=1)).to(dtype)
        self.decode_mask = Tensor.from_numpy(np.triu(np.ones(shape=(max_model_len, max_model_len), dtype=np.int8), k=1)).to(dtype)
        self.hard_mask = mint.zeros((1, 1), dtype=dtype)

    def gen_attention_mask(self, is_prefill, position_ids, query_lens):
        if is_prefill:
            attention_mask = self.prefill_mask
        else:
            if max(query_lens) > 1:
                attention_mask = mint.index_select(self.decode_mask, 0, position_ids)
            else:
                attention_mask = self.hard_mask
        return attention_mask
