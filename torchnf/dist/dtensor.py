"""DTensor conversion helpers."""

from __future__ import annotations

from typing import Sequence

import torch

try:
    from torch.distributed.tensor import DTensor, distribute_tensor
except Exception:  # pragma: no cover - optional import
    DTensor = None
    distribute_tensor = None


def is_dtensor(value) -> bool:
    if DTensor is None:
        return False
    return isinstance(value, DTensor)


def to_dtensor(tensor: torch.Tensor, mesh, placements: Sequence):
    if is_dtensor(tensor):
        return tensor
    if distribute_tensor is None:
        raise RuntimeError("DTensor is not available in this PyTorch version")
    return distribute_tensor(tensor, mesh, placements)


def from_dtensor(dtensor: torch.Tensor) -> torch.Tensor:
    if is_dtensor(dtensor):
        return dtensor.to_local()
    return dtensor
