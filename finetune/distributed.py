import logging
import os
from functools import lru_cache
from typing import List, Union

import torch
import torch.distributed as dist

logger = logging.getLogger("distributed")

BACKEND = "nccl"


@lru_cache()
def get_rank() -> int:
    return dist.get_rank()


@lru_cache()
def get_world_size() -> int:
    return dist.get_world_size()


def visible_devices() -> List[int]:
    # PBS may set CUDA_VISIBLE_DEVICES to GPU UUIDs (e.g. "GPU-5e648ed8-...")
    # instead of integer indices. Fall back to range(device_count) in that case.
    raw = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not raw:
        return list(range(torch.cuda.device_count()))
    parts = raw.split(",")
    try:
        return [int(d) for d in parts]
    except ValueError:
        return list(range(len(parts)))


def set_device():
    logger.info(f"torch.cuda.device_count: {torch.cuda.device_count()}")
    logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ['CUDA_VISIBLE_DEVICES']}")
    logger.info(f"local rank: {int(os.environ['LOCAL_RANK'])}")

    assert torch.cuda.is_available()

    assert len(visible_devices()) == torch.cuda.device_count()

    if torch.cuda.device_count() == 1:
        # gpus-per-task set to 1
        torch.cuda.set_device(0)
        return

    local_rank = int(os.environ["LOCAL_RANK"])
    logger.info(f"Set cuda device to {local_rank}")

    assert 0 <= local_rank < torch.cuda.device_count(), (
        local_rank,
        torch.cuda.device_count(),
    )
    torch.cuda.set_device(local_rank)


def avg_aggregate(metric: Union[float, int]) -> Union[float, int]:
    buffer = torch.tensor([metric], dtype=torch.float32, device="cuda")
    dist.all_reduce(buffer, op=dist.ReduceOp.SUM)
    return buffer[0].item() / get_world_size()


def is_torchrun() -> bool:
    return "TORCHELASTIC_RESTART_COUNT" in os.environ
