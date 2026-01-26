"""DeviceMesh helpers."""

from __future__ import annotations

import torch
import torch.distributed as dist


def build_device_mesh(
    dp: int, tp: int, ep: int, device_type: str = "cuda"
):
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before mesh")
    mesh_shape = (dp, tp, ep)
    dim_names = ["dp", "tp", "ep"]
    try:
        from torch.distributed.device_mesh import init_device_mesh

        return init_device_mesh(device_type, mesh_shape, mesh_dim_names=dim_names)
    except Exception:
        from torch.distributed.device_mesh import DeviceMesh

        world_size = dist.get_world_size()
        expected = dp * tp * ep
        if world_size != expected:
            raise ValueError(f"world_size {world_size} != dp*tp*ep {expected}")
        mesh = torch.arange(world_size).reshape(mesh_shape)
        return DeviceMesh(device_type, mesh, mesh_dim_names=dim_names)
