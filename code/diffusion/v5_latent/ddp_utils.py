"""
Minimal Distributed Data Parallel helpers.

No heroics, no external deps beyond torch. The functions here let a
single script run either (a) as a plain single-GPU job or (b) under
torchrun with N ranks and essentially the same logic.

Usage pattern inside a training script:

    from ab_ld4lg.ddp_utils import ddp_setup, ddp_teardown, is_main, \
        all_reduce_mean

    state = ddp_setup()
    device = state.device
    model = MyModel().to(device)
    if state.is_distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[state.local_rank]
        )
    ...
    if is_main(state):
        torch.save(ckpt, "...")
    ddp_teardown(state)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import torch
import torch.distributed as dist


@dataclass
class DDPState:
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def ddp_setup() -> DDPState:
    """Initialize NCCL process group if torchrun env vars are present.

    Falls back to single-process mode otherwise. Safe to call from any
    script; the caller uses `state.is_distributed` to branch on things
    like sampler choice.
    """
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank % max(1, torch.cuda.device_count())))
        if torch.cuda.is_available():
            torch.cuda.set_device(local_rank)
            backend = "nccl"
            device = torch.device(f"cuda:{local_rank}")
        else:
            backend = "gloo"
            device = torch.device("cpu")
        dist.init_process_group(backend=backend, rank=rank, world_size=world)
        return DDPState(True, rank, world, local_rank, device)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return DDPState(False, 0, 1, 0, device)


def ddp_teardown(state: DDPState) -> None:
    if state.is_distributed and dist.is_initialized():
        dist.destroy_process_group()


def is_main(state: DDPState) -> bool:
    return state.rank == 0


def all_reduce_mean(x: torch.Tensor, state: DDPState) -> torch.Tensor:
    """Reduce a scalar tensor across all ranks to a running mean."""
    if not state.is_distributed:
        return x
    x = x.detach().clone()
    dist.all_reduce(x, op=dist.ReduceOp.SUM)
    x /= state.world_size
    return x


def unwrap(model: torch.nn.Module) -> torch.nn.Module:
    """Return the underlying module whether model is DDP-wrapped or not."""
    if hasattr(model, "module"):
        return model.module
    return model
