"""Checkpoint save/load helpers."""

from __future__ import annotations

from typing import Optional

import torch

from ..dist.state_dict import get_state_dict, load_state_dict
from ..utils import get_rank


def save_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    fsdp_backend: str,
) -> None:
    if get_rank() != 0:
        return
    state = {"model": get_state_dict(model, fsdp_backend)}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)


def load_checkpoint(
    path: str,
    model: torch.nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    fsdp_backend: str,
    map_location: str = "cpu",
) -> None:
    state = torch.load(path, map_location=map_location)
    load_state_dict(model, state["model"], fsdp_backend)
    if optimizer is not None and "optimizer" in state:
        optimizer.load_state_dict(state["optimizer"])
