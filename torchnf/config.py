"""Configuration for DP/TP/EP parallelism."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import torch.distributed as dist


def _parse_mesh(mesh: str) -> Dict[str, int]:
    parts = [p.strip() for p in mesh.split(",") if p.strip()]
    out: Dict[str, int] = {}
    for part in parts:
        if "=" not in part:
            raise ValueError(f"Invalid mesh item: {part}")
        key, value = part.split("=", 1)
        key = key.strip().lower()
        out[key] = int(value.strip())
    return out


@dataclass
class ParallelConfig:
    dp: int = 1
    tp: int = 1
    ep: int = 1
    mesh: Optional[str] = None
    fsdp_backend: str = "fsdp"  # "fsdp" | "fsdp2" | "none"
    tp_style: str = "column"  # "column" | "row"
    tp_gather_output: bool = True
    moe_top_k: int = 1
    device_type: str = "cuda"
    validate_world_size: bool = True

    def __post_init__(self) -> None:
        if self.mesh:
            parsed = _parse_mesh(self.mesh)
            self.dp = parsed.get("dp", self.dp)
            self.tp = parsed.get("tp", self.tp)
            self.ep = parsed.get("ep", self.ep)
        self._validate()

    def _validate(self) -> None:
        for name, value in {"dp": self.dp, "tp": self.tp, "ep": self.ep}.items():
            if value < 1:
                raise ValueError(f"{name} must be >= 1")
        if self.fsdp_backend not in {"fsdp", "fsdp2", "none"}:
            raise ValueError("fsdp_backend must be fsdp|fsdp2|none")
        if self.tp_style not in {"column", "row"}:
            raise ValueError("tp_style must be column|row")
        if self.moe_top_k != 1:
            raise ValueError("MVP only supports moe_top_k=1")
        if self.validate_world_size and dist.is_initialized():
            world_size = dist.get_world_size()
            expected = self.dp * self.tp * self.ep
            if world_size != expected:
                raise ValueError(
                    f"world_size {world_size} != dp*tp*ep {expected}"
                )

    @property
    def mesh_shape(self) -> tuple[int, int, int]:
        return (self.dp, self.tp, self.ep)

    def to_dict(self) -> Dict[str, int | str | bool]:
        return {
            "dp": self.dp,
            "tp": self.tp,
            "ep": self.ep,
            "mesh": self.mesh or "",
            "fsdp_backend": self.fsdp_backend,
            "tp_style": self.tp_style,
            "tp_gather_output": self.tp_gather_output,
            "moe_top_k": self.moe_top_k,
            "device_type": self.device_type,
        }
