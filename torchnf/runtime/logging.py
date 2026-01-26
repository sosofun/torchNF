"""Logging helpers."""

from __future__ import annotations

import logging
from typing import Optional

from ..utils import get_rank


def get_logger(name: str = "torchnf") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s][%(levelname)s] %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def log_rank0(logger: logging.Logger, message: str) -> None:
    if get_rank() == 0:
        logger.info(message)
