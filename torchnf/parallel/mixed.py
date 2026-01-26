"""Compose DP/TP/EP strategies."""

from __future__ import annotations

import torch.nn as nn

from .dp import apply_dp
from .ep import apply_ep
from .tp import apply_tp


def apply_mixed(model: nn.Module, config, mesh):
    model = apply_tp(model, config, mesh)
    model = apply_ep(model, config, mesh)
    model = apply_dp(model, config, mesh)
    return model
