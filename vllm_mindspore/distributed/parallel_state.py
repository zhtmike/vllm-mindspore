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

# 该文件需要实现组网管理初始化， 这部分接口可以只支持pynative， 因为都可以在init阶段确定的。
import subprocess
from typing import List, Optional

from pathlib import Path

from mindspore.communication import (
    get_rank,
    init,
    get_group_size,
    create_group,
    GlobalComm,
)
from .ms_communicate import init_ms_distributed

logger = None


class _Tmp:
    def __init__(self):
        self.sched_p = None

    def set_sched_process(self, p):
        self.sched_p = p

    def __del__(self):
        if self.sched_p:
            self.sched_p.kill()


_tmp = _Tmp()

_TP = None
_IS_FIRST_TP_RANK = False

_PP = None
_IS_LAST_PP_RANK: bool = False
_IS_FIRST_PP_RANK: bool = False
_NEXT_PP_RANK = -1

_PP_GROUP_RANKS: List
_TP_GROUP_RANKS: List

world = GlobalComm.WORLD_COMM_GROUP


def get_tp_group():
    return _TP


def get_pp_group():
    return _PP


def get_tensor_model_parallel_rank():
    return get_rank(group=get_tp_group())


def get_tensor_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_group_size(group=get_tp_group())


def get_pipeline_model_parallel_rank():
    return get_rank(group=get_pp_group())


def get_pipeline_model_parallel_world_size():
    """Return world size for the tensor model parallel group."""
    return get_group_size(group=get_pp_group())


def init_distributed_environment(
    world_size: int = -1,
    rank: int = -1,
    distributed_init_method: str = "env://",
    local_rank: int = -1,
    backend: str = "nccl",
):
    global logger
    if logger is None:
        from vllm.logger import init_logger

        logger = init_logger(__name__)

    logger.debug(
        "world_size=%d rank=%d local_rank=%d distributed_init_method=%s backend=%s",
        world_size,
        rank,
        local_rank,
        distributed_init_method,
        backend,
    )

    if local_rank == -1:
        import vllm.envs as envs

        if distributed_init_method == "env://":
            local_rank = envs.LOCAL_RANK
        else:
            local_rank = rank

    # TODO(tronzhang): ms api such as get_group_size should communicate first...
    if rank == 0:
        with open(str(Path() / "schedule.log"), "w") as scedule_f:
            scipt = Path(__file__).parent / "ms_communicate.py"
            sched_p = subprocess.Popen(
                [
                    "python",
                    str(scipt),
                    "--role",
                    "MS_SCHED",
                    "--rank_id",
                    str(rank),
                    "--local_rank_id",
                    str(local_rank),
                    "--rank_size",
                    str(world_size),
                    "--distributed_init_method",
                    distributed_init_method,
                ],
                shell=False,
                stdout=scedule_f,
                stderr=subprocess.STDOUT,
            )
            _tmp.set_sched_process(sched_p)

    init_ms_distributed(
        "MS_WORKER", rank, local_rank, world_size, distributed_init_method
    )


def init_model_parallel_group(
    group_ranks: List[List[int]],
    local_rank: int,
    backend: str,
    use_custom_allreduce: Optional[bool] = None,
    use_message_queue_broadcaster: bool = False,
    group_name: Optional[str] = None,
    is_pp_init=False,
):
    global _IS_FIRST_PP_RANK
    global _IS_LAST_PP_RANK
    global _NEXT_PP_RANK
    global _PREV_PP_RANK
    global _IS_FIRST_TP_RANK
    global _PP_GROUP_RANKS
    global _TP_GROUP_RANKS
    group_name: str
    for i, ranks in enumerate(group_ranks):
        if local_rank in ranks:
            create_group(f"{group_name}_{i}", ranks)
            if not is_pp_init:
                pos = ranks.index(local_rank)
                if pos == 0:
                    _IS_FIRST_TP_RANK = True
                _TP_GROUP_RANKS = ranks
            if is_pp_init:
                pos = ranks.index(local_rank)
                if pos == 0:
                    _IS_FIRST_PP_RANK = True
                if pos == len(ranks) - 1:
                    _IS_LAST_PP_RANK = True
                _NEXT_PP_RANK = ranks[(pos + 1) % len(ranks)]
                _PREV_PP_RANK = ranks[(pos + len(ranks) - 1) % len(ranks)]
            group_name = f"{group_name}_{i}"
            break
    return group_name


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    world_size = get_group_size()
    if world_size != tensor_model_parallel_size * pipeline_model_parallel_size:
        raise RuntimeError(
            f"world_size ({world_size}) is not equal to "
            f"tensor_model_parallel_size ({tensor_model_parallel_size}) x "
            f"pipeline_model_parallel_size ({pipeline_model_parallel_size})"
        )

    this_rank = get_rank()

    # 每个PP stage 有一个tp group， 此处计算有多少个PP stage， 即多少个tp group
    global _TP
    assert _TP is None, "tensor model parallel group is already initialized"
    group_ranks = []
    num_tensor_model_parallel_groups: int = world_size // tensor_model_parallel_size

    for i in range(num_tensor_model_parallel_groups):
        ranks = list(
            range(i * tensor_model_parallel_size, (i + 1) * tensor_model_parallel_size)
        )
        group_ranks.append(ranks)

    # message queue broadcaster is only used in tensor model parallel group
    _TP = init_model_parallel_group(
        group_ranks,
        this_rank,
        None,
        use_message_queue_broadcaster=True,
        group_name="tp",
    )

    global _PP
    assert _PP is None, "pipeline model parallel group is already initialized"
    num_pipeline_model_parallel_groups: int = world_size // pipeline_model_parallel_size
    group_ranks = []
    for i in range(num_pipeline_model_parallel_groups):
        ranks = list(range(i, world_size, num_pipeline_model_parallel_groups))
        group_ranks.append(ranks)
    # pipeline parallel does not need custom allreduce
    _PP = init_model_parallel_group(
        group_ranks,
        this_rank,
        backend,
        use_custom_allreduce=False,
        group_name="pp",
        is_pp_init=True,
    )


def is_first_tp_rank():
    return _IS_FIRST_TP_RANK


def is_last_pp_rank():
    return _IS_LAST_PP_RANK


def is_first_pp_rank():
    return _IS_FIRST_PP_RANK


def next_pp_rank():
    return _NEXT_PP_RANK


def prev_pp_rank():
    return _PREV_PP_RANK


def get_pp_group_size():
    return get_group_size(get_pp_group())


def get_pp_rank_in_group():
    return get_rank(get_pp_group())


def get_world_group():
    return world


def get_world_rank_from_tp_group_rank(group_rank):
    return _TP_GROUP_RANKS[group_rank]


def ensure_kv_transfer_initialized(vllm_config: "VllmConfig") -> None: ...


def model_parallel_is_initialized():
    return _TP is not None and _PP is not None


def ensure_model_parallel_initialized(
    tensor_model_parallel_size: int,
    pipeline_model_parallel_size: int,
    backend: Optional[str] = None,
) -> None:
    if not model_parallel_is_initialized():
        initialize_model_parallel(
            tensor_model_parallel_size, pipeline_model_parallel_size, backend
        )
