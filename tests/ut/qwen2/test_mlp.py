import vllm_mindspore  # Add this line on the top of script.
from glob import glob

from vllm.distributed import initialize_model_parallel, init_distributed_environment
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

from mindspore import Tensor,  mint, mutable, set_context
from mindspore.common import dtype as mstype
from mindspore.nn.utils import no_init_parameters

from vllm_mindspore.model_executor.models.qwen2 import Qwen2MLP
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
            save_graphs=True, save_graphs_path='graphs/mlp')


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
    mlp = Qwen2MLP(
        hidden_size=qwen_config.hidden_size,
        intermediate_size=qwen_config.intermediate_size,
        hidden_act=qwen_config.hidden_act
    )
# test set_inputs
mlp.set_inputs(
    Tensor(shape=[None, None, None], dtype=mstype.bfloat16),
)


# test profile_run
mlp_profile_out = mlp(
    mint.ones((1, 32768, 3584), dtype=mstype.bfloat16),
)
print(f"mlp_profile_out: {[mlp_profile_out]}", flush=True)

# test prefill
mlp_prefill_out = mlp(
    mint.ones((1, 17, 3584), dtype=mstype.bfloat16),
)
print(f"mlp_prefill_out: {[mlp_prefill_out]}", flush=True)

# test decode
mlp_decode_out = mlp(
    mint.ones((3, 1, 3584), dtype=mstype.bfloat16),
)
print(f"mlp_decode_out: {[mlp_decode_out]}", flush=True)

# check graphs
graphs = glob("graphs/mlp/20_*.ir")
if len(graphs) != 1:
    raise ValueError()
