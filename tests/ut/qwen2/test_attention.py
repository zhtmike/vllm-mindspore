import vllm_mindspore  # Add this line on the top of script.
from glob import glob

from vllm.distributed import initialize_model_parallel, init_distributed_environment
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

from mindspore import Tensor,  mint, mutable, set_context
from mindspore.common import dtype as mstype
from mindspore.nn.utils import no_init_parameters

from vllm_mindspore.attention.layer import Attention


# init mindspore context
set_context(mode=1, jit_config={"jit_level": "O0", "infer_boost": "on"},
            save_graphs=True, save_graphs_path='graphs/attn')


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
    attn = Attention(
        num_heads=28,
        head_size=128,
        scale=1,
        num_kv_heads=4
    )
# test set_inputs
attn.set_inputs(
    Tensor(shape=[None, None, None], dtype=mstype.bfloat16),  # query
    Tensor(shape=[None, None, None], dtype=mstype.bfloat16),  # key
    Tensor(shape=[None, None, None], dtype=mstype.bfloat16),  # value
    mutable((mutable(Tensor(shape=(None, 16, 4, 128), dtype=mstype.bfloat16)),
             mutable(Tensor(shape=(None, 16, 4, 128), dtype=mstype.bfloat16)),)),
    mutable(1),
    mutable(0),
    Tensor(shape=[None,], dtype=mstype.int32),
    mutable([0, 0, 0], dynamic_len=True),
    Tensor(shape=[None,], dtype=mstype.int32),
    Tensor(shape=[None, None], dtype=mstype.int32),
)


# test profile_run
attn_profile_out = attn(
    mint.ones((1, 32768, 3584), dtype=mstype.bfloat16),
    mint.ones((1, 32768, 512), dtype=mstype.bfloat16),
    mint.ones((1, 32768, 512), dtype=mstype.bfloat16),
    mutable((mutable(mint.empty([0, 16, 4, 128], dtype=mstype.bfloat16)),
             mutable(mint.empty([0, 16, 4, 128], dtype=mstype.bfloat16)))),
    mutable(32768),
    mutable(0),
    mint.ones((32768), dtype=mstype.int32) * -1,
    mutable([2048 for i in range(16)], dynamic_len=True),
    mint.ones((16), dtype=mstype.int32) * 2048,
    mint.empty([0, 0], dtype=mstype.int32)
)
print(f"attn_profile_out: {[attn_profile_out]}", flush=True)

# test prefill
attn_prefill_out = attn(
    mint.ones((1, 17, 3584), dtype=mstype.bfloat16),
    mint.ones((1, 17, 512), dtype=mstype.bfloat16),
    mint.ones((1, 17, 512), dtype=mstype.bfloat16),
    mutable((mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)),
             mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)))),
    mutable(1),
    mutable(0),
    mint.ones((17,), dtype=mstype.int32),
    mutable([5, 5, 7], dynamic_len=True),
    Tensor([5, 5, 7], dtype=mstype.int32),
    mint.ones((16, 512), dtype=mstype.int32)
)
print(f"attn_prefill_out: {[attn_prefill_out]}", flush=True)

# test decode
attn_decode_out = attn(
    mint.ones((3, 1, 3584), dtype=mstype.bfloat16),
    mint.ones((3, 1, 512), dtype=mstype.bfloat16),
    mint.ones((3, 1, 512), dtype=mstype.bfloat16),
    mutable((mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)),
             mutable(mint.zeros([512, 16, 4, 128], dtype=mstype.bfloat16)))),
    mutable(0),
    mutable(1),
    mint.ones((3,), dtype=mstype.int32),
    mutable([6, 6, 8], dynamic_len=True),
    Tensor([6, 6, 8], dtype=mstype.int32),
    mint.ones((16, 512), dtype=mstype.int32)
)
print(f"attn_decode_out: {[attn_decode_out]}", flush=True)

# check graphs
graphs = glob("graphs/attn/20_*.ir")
if len(graphs) != 1:
    raise ValueError()
