"""Utility helpers for distributed setup and validation."""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist


def get_rank(group: Optional[dist.ProcessGroup] = None) -> int:
    if not dist.is_initialized():
        return 0
    return dist.get_rank(group)


def get_world_size(group: Optional[dist.ProcessGroup] = None) -> int:
    if not dist.is_initialized():
        return 1
    return dist.get_world_size(group)


def get_local_rank() -> int:
    if "LOCAL_RANK" in os.environ:
        return int(os.environ["LOCAL_RANK"])
    if "RANK" in os.environ and torch.cuda.is_available():
        return int(os.environ["RANK"]) % torch.cuda.device_count()
    return 0


def ensure_distributed(
    backend: Optional[str] = None, device_type: str = "cuda"
) -> None:
    if dist.is_initialized():
        return
    if backend is None:
        backend = "nccl" if device_type == "cuda" else "gloo"
    dist.init_process_group(backend=backend, init_method="env://")
    if device_type == "cuda" and torch.cuda.is_available():
        torch.cuda.set_device(get_local_rank())


def require_distributed() -> None:
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed is not initialized")
