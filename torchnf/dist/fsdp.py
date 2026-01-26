"""FSDP adapter."""

from __future__ import annotations

from typing import Optional

import torch
import torch.distributed as dist


def wrap_fsdp(model: torch.nn.Module, process_group: Optional[dist.ProcessGroup] = None):
    try:
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    except Exception as exc:  # pragma: no cover - import guarded
        raise RuntimeError("FSDP is not available") from exc
    return FSDP(
        model,
        process_group=process_group,
        use_orig_params=True,
        sync_module_states=True,
    )
