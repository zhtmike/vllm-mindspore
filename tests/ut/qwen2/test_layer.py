import vllm_mindspore  # Add this line on the top of script.
from glob import glob

from vllm.distributed import initialize_model_parallel, init_distributed_environment
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

from mindspore import Tensor,  mint, mutable, set_context
from mindspore.common import dtype as mstype
from mindspore.nn.utils import no_init_parameters

from vllm_mindspore.model_executor.models.qwen2 import Qwen2DecoderLayer
# from .utils import _FAKE_CONFIG


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
# init mindspore context
set_context(mode=1, jit_config={"jit_level": "O0", "infer_boost": "on"},
            save_graphs=True, save_graphs_path='graphs/layer')


# init vllm distributed
distributed_init_method = get_distributed_init_method(get_ip(), get_open_port())
init_distributed_environment(
    world_size=1,
    rank=0,
    distributed_init_method=distributed_init_method,
    local_rank=0,
    backend="nccl"
)
initialize_model_parallel(tensor_model_parallel_size=1, pipeline_model_parallel_size=1)


# create test object
with no_init_parameters():
    qwen_config = _FAKE_CONFIG
    layer = Qwen2DecoderLayer(config=qwen_config)
# test set_inputs
layer.set_inputs(
    Tensor(shape=[None], dtype=mstype.int64),
    Tensor(shape=[None, None, None], dtype=mstype.bfloat16),
    mutable((mutable(Tensor(shape=(None, 16, 4, 128), dtype=mstype.bfloat16)),
             mutable(Tensor(shape=(None, 16, 4, 128), dtype=mstype.bfloat16)),)),
    mutable(1),
    mutable(0),
    Tensor(shape=[None,], dtype=mstype.int32),
    mutable([0, 0, 0], dynamic_len=True),
    Tensor(shape=[None,], dtype=mstype.int32),
    Tensor(shape=[None, None], dtype=mstype.int32),
    None,
)


# test profile_run
layer_profile_out = layer(
    mint.arange((32768), dtype=mstype.int64),
    mint.ones((1, 32768, 3584), dtype=mstype.bfloat16),
    mutable((mutable(mint.empty([0, 16, 4, 128], dtype=mstype.bfloat16)),
             mutable(mint.empty([0, 16, 4, 128], dtype=mstype.bfloat16)))),
    mutable(32768),
    mutable(0),
    mint.ones((32768), dtype=mstype.int32) * -1,
    mutable([2048 for i in range(16)], dynamic_len=True),
    mint.ones((16), dtype=mstype.int32) * 2048,
    mint.empty([0, 0], dtype=mstype.int32),
    None,
)
print(f"layer_profile_out: {[layer_profile_out]}", flush=True)

# test prefill
layer_prefill_out = layer(
    Tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 5, 6], dtype=mstype.int64),
    mint.ones((1, 17, 3584), dtype=mstype.bfloat16),
    mutable((mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)),
             mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)))),
    mutable(1),
    mutable(0),
    mint.ones((17,), dtype=mstype.int32),
    mutable([5, 5, 7], dynamic_len=True),
    Tensor([5, 5, 7], dtype=mstype.int32),
    mint.ones((16, 512), dtype=mstype.int32),
    None,
)
print(f"layer_prefill_out: {[layer_prefill_out]}", flush=True)

# test decode
layer_decode_out = layer(
    Tensor([5, 5, 6], dtype=mstype.int64),
    mint.ones((3, 1, 3584), dtype=mstype.bfloat16),
    mutable((mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)),
             mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)))),
    mutable(0),
    mutable(1),
    mint.ones((3,), dtype=mstype.int32),
    mutable([6, 6, 8], dynamic_len=True),
    Tensor([6, 6, 8], dtype=mstype.int32),
    mint.ones((16, 512), dtype=mstype.int32),
    None,
)
print(f"layer_decode_out: {[layer_decode_out]}", flush=True)

# check graphs
graphs = glob("graphs/layer/20_*.ir")
if len(graphs) != 1:
    raise ValueError()
