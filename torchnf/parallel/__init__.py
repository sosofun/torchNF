"""Parallel strategy modules."""

from .dp import apply_dp
from .ep import apply_ep
from .mixed import apply_mixed
from .tp import apply_tp

__all__ = ["apply_dp", "apply_ep", "apply_mixed", "apply_tp"]
