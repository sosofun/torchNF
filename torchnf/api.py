"""Public API for distribution and training."""

from __future__ import annotations

import torch.nn as nn

from .config import ParallelConfig
from .dist.mesh import build_device_mesh
from .parallel.mixed import apply_mixed
from .runtime.engine import Trainer
from .utils import ensure_distributed


def distribute(model: nn.Module, config: ParallelConfig) -> nn.Module:
    ensure_distributed(device_type=config.device_type)
    mesh = build_device_mesh(config.dp, config.tp, config.ep, config.device_type)
    model = apply_mixed(model, config, mesh)
    model._device_mesh = mesh
    return model


__all__ = ["Trainer", "distribute"]
