"""Expert parallel logic for MoE layers."""

from __future__ import annotations

import torch.nn as nn

from .modules import MoELayer


def apply_ep(model: nn.Module, config, mesh):
    if config.ep <= 1:
        return model
    ep_group = None
    if mesh is not None:
        try:
            ep_group = mesh.get_group("ep")
        except Exception:
            ep_group = None
    for module in model.modules():
        if isinstance(module, MoELayer):
            module.set_ep_group(ep_group)
    return model
