"""Parallel modules built on DTensor."""

from __future__ import annotations

from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from ..dist.dtensor import from_dtensor, is_dtensor, to_dtensor
from ..dist.layout import replicate_placements, tp_placements
from .moe import MoERouter, dispatch_to_experts, gather_from_experts


def _mesh_dim_size(mesh, name: str) -> int:
    if mesh is None:
        return 1
    try:
        idx = mesh.mesh_dim_names.index(name)
    except Exception:
        return 1
    if hasattr(mesh, "shape"):
        return int(mesh.shape[idx])
    if hasattr(mesh, "mesh") and hasattr(mesh.mesh, "shape"):
        return int(mesh.mesh.shape[idx])
    return 1


class ParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        mesh=None,
        tp_style: str = "column",
        gather_output: bool = True,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.mesh = mesh
        self.tp_style = tp_style
        self.gather_output = gather_output
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.bias = nn.Parameter(torch.empty(out_features)) if bias else None
        self.reset_parameters()
        self._shard_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=5**0.5)
        if self.bias is not None:
            fan_in = self.weight.size(1)
            bound = 1 / fan_in**0.5
            nn.init.uniform_(self.bias, -bound, bound)

    def _shard_parameters(self) -> None:
        if self.mesh is None:
            return
        if _mesh_dim_size(self.mesh, "tp") <= 1:
            return
        shard_dim = 0 if self.tp_style == "column" else 1
        placements = tp_placements(self.mesh, shard_dim=shard_dim)
        self.weight = nn.Parameter(to_dtensor(self.weight, self.mesh, placements))
        if self.bias is not None:
            if self.tp_style == "column":
                bias_placements = tp_placements(self.mesh, shard_dim=0)
            else:
                bias_placements = replicate_placements(self.mesh)
            self.bias = nn.Parameter(
                to_dtensor(self.bias, self.mesh, bias_placements)
            )

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        mesh=None,
        tp_style: str = "column",
        gather_output: bool = True,
    ) -> "ParallelLinear":
        new = cls(
            linear.in_features,
            linear.out_features,
            bias=linear.bias is not None,
            mesh=None,
            tp_style=tp_style,
            gather_output=gather_output,
        )
        new.weight.data.copy_(linear.weight.data)
        if linear.bias is not None and new.bias is not None:
            new.bias.data.copy_(linear.bias.data)
        new.mesh = mesh
        new._shard_parameters()
        return new

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mesh is None or _mesh_dim_size(self.mesh, "tp") <= 1:
            return F.linear(x, self.weight, self.bias)
        if not is_dtensor(x):
            x = to_dtensor(x, self.mesh, replicate_placements(self.mesh))
        y = F.linear(x, self.weight, self.bias)
        if self.gather_output and is_dtensor(y):
            y = y.redistribute(replicate_placements(self.mesh))
            y = from_dtensor(y)
        return y


class ParallelAttention(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
        raise NotImplementedError("MVP does not include ParallelAttention")


class MoELayer(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_experts: int,
        top_k: int = 1,
        expert_hidden: Optional[int] = None,
        expert_factory: Optional[Callable[[int, int], nn.Module]] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = MoERouter(hidden_size, num_experts, top_k=top_k)
        self.expert_hidden = expert_hidden or hidden_size * 4
        self.expert_factory = expert_factory or self._default_expert
        self.ep_group = None
        self.experts = nn.ModuleList()
        self._local_expert_offset = 0
        self._ep_world_size = 1
        self.last_aux_loss = None
        self._experts_built = False

    def _default_expert(self, hidden: int, hidden_ffn: int) -> nn.Module:
        return nn.Sequential(
            nn.Linear(hidden, hidden_ffn),
            nn.GELU(),
            nn.Linear(hidden_ffn, hidden),
        )

    def set_ep_group(self, ep_group) -> None:
        self.ep_group = ep_group
        if ep_group is not None:
            self._ep_world_size = dist.get_world_size(ep_group)
            rank = dist.get_rank(ep_group)
        else:
            self._ep_world_size = 1
            rank = 0
        if self.num_experts % self._ep_world_size != 0:
            raise ValueError("num_experts must be divisible by ep world_size")
        local_experts = self.num_experts // self._ep_world_size
        self._local_expert_offset = rank * local_experts
        if self._experts_built and len(self.experts) != local_experts:
            raise RuntimeError(
                "MoELayer experts already built with different ep size"
            )
        if not self._experts_built:
            for _ in range(local_experts):
                self.experts.append(
                    self.expert_factory(self.hidden_size, self.expert_hidden)
                )
            self._experts_built = True

    def _ensure_experts(self) -> None:
        if self._experts_built:
            return
        local_experts = self.num_experts // self._ep_world_size
        for _ in range(local_experts):
            self.experts.append(
                self.expert_factory(self.hidden_size, self.expert_hidden)
            )
        self._experts_built = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x = x.reshape(-1, self.hidden_size)
        self._ensure_experts()
        expert_id, aux_loss = self.router(x)
        self.last_aux_loss = aux_loss
        if self.ep_group is None or self._ep_world_size <= 1:
            out = self._forward_local(x, expert_id)
        else:
            out = self._forward_ep(x, expert_id)
        return out.reshape(original_shape)

    def _forward_local(
        self, tokens: torch.Tensor, expert_id: torch.Tensor
    ) -> torch.Tensor:
        output = torch.zeros_like(tokens)
        for idx, expert in enumerate(self.experts):
            mask = expert_id == idx
            if mask.any():
                output[mask] = expert(tokens[mask])
        return output

    def _forward_ep(
        self, tokens: torch.Tensor, expert_id: torch.Tensor
    ) -> torch.Tensor:
        if self.ep_group is None:
            raise RuntimeError("ep_group is not set for MoELayer")
        recv_tokens, recv_expert_id, recv_pos, _, recv_splits = (
            dispatch_to_experts(tokens, expert_id, self.num_experts, self.ep_group)
        )
        local_expert_id = recv_expert_id - self._local_expert_offset
        output = torch.zeros_like(recv_tokens)
        for idx, expert in enumerate(self.experts):
            mask = local_expert_id == idx
            if mask.any():
                output[mask] = expert(recv_tokens[mask])
        return gather_from_experts(
            output,
            recv_pos,
            recv_splits,
            self.ep_group,
            tokens.shape[0],
        )
