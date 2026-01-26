"""MoE router and all-to-all helpers."""

from __future__ import annotations

from typing import List, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn


class MoERouter(nn.Module):
    def __init__(self, hidden_size: int, num_experts: int, top_k: int = 1):
        super().__init__()
        if top_k != 1:
            raise ValueError("MVP only supports top_k=1")
        self.num_experts = num_experts
        self.top_k = top_k
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits = self.gate(x)
        expert_id = torch.argmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        load = probs.mean(dim=0)
        aux_loss = (load * load).sum() * self.num_experts
        return expert_id, aux_loss


def _group_info(group) -> Tuple[int, int]:
    return dist.get_rank(group), dist.get_world_size(group)


def _compute_recv_splits(
    send_splits: List[int], group, device
) -> List[int]:
    rank, world_size = _group_info(group)
    send = torch.tensor(send_splits, device=device, dtype=torch.int64)
    gathered = [torch.empty_like(send) for _ in range(world_size)]
    dist.all_gather(gathered, send, group=group)
    return [int(g[rank].item()) for g in gathered]


def _all_to_all_single_var(
    tensor: torch.Tensor,
    send_splits: List[int],
    recv_splits: List[int],
    group,
) -> torch.Tensor:
    out_shape = (sum(recv_splits),) + tuple(tensor.shape[1:])
    output = tensor.new_empty(out_shape)
    dist.all_to_all_single(
        output,
        tensor,
        input_split_sizes=send_splits,
        output_split_sizes=recv_splits,
        group=group,
    )
    return output


def dispatch_to_experts(
    tokens: torch.Tensor,
    expert_id: torch.Tensor,
    num_experts: int,
    group,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], List[int]]:
    rank, world_size = _group_info(group)
    if num_experts % world_size != 0:
        raise ValueError("num_experts must be divisible by ep world_size")
    experts_per_rank = num_experts // world_size
    expert_rank = expert_id // experts_per_rank

    send_splits = (
        torch.bincount(expert_rank, minlength=world_size)
        .to(dtype=torch.int64)
        .tolist()
    )
    order = torch.argsort(expert_rank)
    send_tokens = tokens[order]
    send_expert_id = expert_id[order]
    send_pos = torch.arange(
        tokens.shape[0], device=tokens.device, dtype=torch.int64
    )[order]

    recv_splits = _compute_recv_splits(send_splits, group, tokens.device)
    recv_tokens = _all_to_all_single_var(
        send_tokens, send_splits, recv_splits, group
    )
    recv_expert_id = _all_to_all_single_var(
        send_expert_id, send_splits, recv_splits, group
    )
    recv_pos = _all_to_all_single_var(
        send_pos, send_splits, recv_splits, group
    )
    return recv_tokens, recv_expert_id, recv_pos, send_splits, recv_splits


def gather_from_experts(
    outputs: torch.Tensor,
    recv_pos: torch.Tensor,
    recv_splits: List[int],
    group,
    original_tokens: int,
) -> torch.Tensor:
    send_splits = recv_splits
    recv_splits_back = _compute_recv_splits(send_splits, group, outputs.device)
    out_tokens = _all_to_all_single_var(
        outputs, send_splits, recv_splits_back, group
    )
    out_pos = _all_to_all_single_var(
        recv_pos, send_splits, recv_splits_back, group
    )
    result = outputs.new_empty((original_tokens,) + outputs.shape[1:])
    result[out_pos] = out_tokens
    return result
