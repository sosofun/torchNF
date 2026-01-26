"""DTensor placement helpers."""

from __future__ import annotations

from typing import List

from torch.distributed.tensor import Partial, Replicate, Shard


def replicate_placements(mesh) -> List[Replicate]:
    return [Replicate() for _ in range(mesh.ndim)]


def tp_placements(mesh, shard_dim: int, tp_dim: str = "tp"):
    placements = replicate_placements(mesh)
    try:
        tp_index = mesh.mesh_dim_names.index(tp_dim)
    except Exception as exc:
        raise ValueError("mesh does not contain tp dim") from exc
    placements[tp_index] = Shard(shard_dim)
    return placements


def partial_placements(mesh, tp_dim: str = "tp"):
    placements = replicate_placements(mesh)
    tp_index = mesh.mesh_dim_names.index(tp_dim)
    placements[tp_index] = Partial()
    return placements
