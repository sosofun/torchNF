"""Tensor parallel logic."""

from __future__ import annotations

import torch.nn as nn

from .modules import ParallelLinear


def apply_tp(model: nn.Module, config, mesh):
    if config.tp <= 1:
        return model
    _replace_linear(model, config, mesh)
    return model


def _replace_linear(module: nn.Module, config, mesh) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.Linear):
            setattr(
                module,
                name,
                ParallelLinear.from_linear(
                    child,
                    mesh=mesh,
                    tp_style=config.tp_style,
                    gather_output=config.tp_gather_output,
                ),
            )
        else:
            _replace_linear(child, config, mesh)
