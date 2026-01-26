"""FSDP2 adapter (composable)."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def wrap_fsdp2(
    model: torch.nn.Module, process_group: Optional[dist.ProcessGroup] = None
):
    try:
        from torch.distributed._composable.fsdp import fully_shard
    except Exception as exc:  # pragma: no cover - optional
        raise RuntimeError("FSDP2 is not available") from exc
    fully_shard(model, process_group=process_group)
    return model
