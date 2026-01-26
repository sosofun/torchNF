"""Training engine for MVP."""

from __future__ import annotations

from typing import Callable, Iterable, Optional

import torch

from .logging import get_logger, log_rank0


class Trainer:
    def __init__(self, config, log_interval: int = 10) -> None:
        self.config = config
        self.log_interval = log_interval
        self.logger = get_logger()

    def fit(
        self,
        model: torch.nn.Module,
        dataloader: Iterable,
        optimizer: torch.optim.Optimizer,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        epochs: int = 1,
    ) -> None:
        if optimizer is None:
            raise ValueError("optimizer is required")
        model.train()
        step = 0
        for epoch in range(epochs):
            for batch in dataloader:
                loss = self._compute_loss(model, batch, loss_fn)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if step % self.log_interval == 0:
                    log_rank0(
                        self.logger,
                        f"epoch={epoch} step={step} loss={loss.item():.6f}",
                    )
                step += 1

    def _compute_loss(
        self,
        model: torch.nn.Module,
        batch,
        loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    ) -> torch.Tensor:
        if isinstance(batch, (list, tuple)) and len(batch) == 2:
            inputs, targets = batch
            outputs = model(inputs)
            if loss_fn is None:
                raise ValueError("loss_fn is required for (inputs, targets)")
            return loss_fn(outputs, targets)
        outputs = model(batch)
        if not torch.is_tensor(outputs):
            raise ValueError("model must return a loss tensor when loss_fn is None")
        return outputs
