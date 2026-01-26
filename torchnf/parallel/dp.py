"""Data parallel wrapper (FSDP/FSDP2)."""

from __future__ import annotations

import torch

from ..dist.fsdp import wrap_fsdp
from ..dist.fsdp2 import wrap_fsdp2


def apply_dp(model: torch.nn.Module, config, mesh):
    if config.dp <= 1 or config.fsdp_backend == "none":
        return model
    dp_group = None
    if mesh is not None:
        try:
            dp_group = mesh.get_group("dp")
        except Exception:
            dp_group = None
    if config.fsdp_backend == "fsdp2":
        return wrap_fsdp2(model, process_group=dp_group)
    return wrap_fsdp(model, process_group=dp_group)
