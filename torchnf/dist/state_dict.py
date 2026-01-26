"""state_dict helpers for FSDP/FSDP2."""

from __future__ import annotations

from typing import Any, Dict

import torch


def get_state_dict(model: torch.nn.Module, fsdp_backend: str) -> Dict[str, Any]:
    if fsdp_backend == "fsdp":
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    return model.state_dict()
        except Exception:
            pass
    return model.state_dict()


def load_state_dict(
    model: torch.nn.Module, state_dict: Dict[str, Any], fsdp_backend: str
) -> None:
    if fsdp_backend == "fsdp":
        try:
            from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
            from torch.distributed.fsdp import StateDictType

            if isinstance(model, FSDP):
                with FSDP.state_dict_type(model, StateDictType.FULL_STATE_DICT):
                    model.load_state_dict(state_dict)
                    return
        except Exception:
            pass
    model.load_state_dict(state_dict)
