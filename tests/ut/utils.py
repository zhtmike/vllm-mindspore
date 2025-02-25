from collections import OrderedDict

from mindspore import Tensor, mint, nn, mutable
from mindspore.common import dtype as mstype


class FakeConfig:
    """Config for Qwen2"""
    num_hidden_layers = 28
    hidden_size = 3584
    num_attention_heads = 28
    num_key_value_heads = 4
    vocab_size = 152064
    pad_token_id = 151643
    lora_extra_vocab_size = 16
    max_loras = 1
    max_position_embeddings = 32768
    hidden_act = 'silu'
    intermediate_size = 18944
    rms_norm_eps = 1e-6
    tie_word_embeddings = False
    attention_bias = True


_FAKE_CONFIG = FakeConfig()

_DICT_FAKE_INPUTS = {
    "input_ids": mint.arange(0, 17, dtype=mstype.int32).reshape(1, 17),
    "positions": Tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6], dtype=mstype.int64),
    "context_lens": Tensor([5, 5, 7], dtype=mstype.int32),
    batch_valid_length: mutable([5, 5, 7], dynamic_len=True),
    num_prefill_tokens = mutable(1)
    num_decode_tokens = mutable(0)
}


def get_fake_inputs(config: FakeConfig = fake_config):
    pass
