import vllm_mindspore  # Add this line on the top of script.

from glob import glob

from mindspore import Tensor,  mint,  set_context
from mindspore.common import dtype as mstype
from mindspore.nn.utils import no_init_parameters
from vllm.distributed import (init_distributed_environment,
                              initialize_model_parallel)
from vllm.utils import get_distributed_init_method, get_ip, get_open_port

from vllm_mindspore.model_executor.layers.vocab_parallel_embedding import \
    VocabParallelEmbedding

# init mindspore context
set_context(mode=1, jit_config={"jit_level": "O0", "infer_boost": "on"},
            save_graphs=True, save_graphs_path='graphs/vocab_embedding')


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
    vocab_emb = VocabParallelEmbedding(
        num_embeddings=152064,
        embedding_dim=3584,
        params_dtype=mstype.bfloat16,
    )

# test set_inputs
vocab_emb.set_inputs(
    Tensor(shape=[None, None], dtype=mstype.int32),  # input_ids
)

# test profile_run
vocab_emb_profile_out = vocab_emb(
    mint.zeros((1, 32768), dtype=mstype.int32),
)
print(f"vocab_emb_profile_out: {[vocab_emb_profile_out]}", flush=True)


# test prefill
vocab_emb_prefill_out = vocab_emb(
    mint.zeros((1, 17), dtype=mstype.int32),
)
print(f"vocab_emb_prefill_out: {[vocab_emb_prefill_out]}", flush=True)


# test decode
vocab_emb_decode_out = vocab_emb(
    mint.zeros((3, 1), dtype=mstype.int32),
)
print(f"vocab_emb_decode_out: {[vocab_emb_decode_out]}", flush=True)

# check graphs
graphs = glob("graphs/vocab_embedding/20_*.ir")
if len(graphs) != 1:
    raise ValueError()
