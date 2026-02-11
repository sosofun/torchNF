"""
High-level convenience API for fast checkpoint save / load.

These functions provide a simple ``save_checkpoint`` / ``load_checkpoint``
interface that hides the complexity of the PingPongBuffer and
AsyncCheckpointSaver.

For full control (e.g. reusing a saver across epochs) use
:class:`AsyncCheckpointSaver` directly.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch


# ---------------------------------------------------------------------------
# Module-level singleton saver (lazily created)
# ---------------------------------------------------------------------------

_global_saver = None
_global_saver_lock = __import__("threading").Lock()


def _get_global_saver():
    """Return (and lazily create) the module-level AsyncCheckpointSaver."""
    global _global_saver
    if _global_saver is None:
        with _global_saver_lock:
            if _global_saver is None:
                from torchNF.checkpoint.async_checkpoint_saver import (
                    AsyncCheckpointSaver,
                )

                _global_saver = AsyncCheckpointSaver()
    return _global_saver


# ---------------------------------------------------------------------------
# save_checkpoint
# ---------------------------------------------------------------------------


def save_checkpoint(
    state_dict: Dict[str, Any],
    path: Union[str, Path],
    *,
    extra_state: Optional[Dict[str, Any]] = None,
    async_save: bool = True,
    saver: Optional[Any] = None,
) -> None:
    """Save a checkpoint using pinned-memory pools and Ping-Pong buffering.

    Parameters
    ----------
    state_dict : dict
        Model (and optionally optimizer) state dict.  GPU tensors will be
        asynchronously copied to pinned CPU memory before writing to disk.
    path : str or Path
        Where to write the checkpoint file.
    extra_state : dict, optional
        Non-tensor metadata to include (epoch, global step, …).
    async_save : bool
        If *True* (default), the disk write happens in a background thread.
        The function returns once the D2H copy and buffer swap are done.
        If *False*, the function blocks until the file is fully written.
    saver : AsyncCheckpointSaver, optional
        Pass an existing saver for full lifecycle control.  If *None*,
        a module-level singleton is used.

    Examples
    --------
    >>> save_checkpoint(model.state_dict(), "ckpt/step_1000.pt",
    ...                 extra_state={"epoch": 5, "step": 1000})
    """
    if saver is None:
        saver = _get_global_saver()

    future = saver.save(state_dict, path, extra_state=extra_state)
    if not async_save:
        future.result()  # block


# ---------------------------------------------------------------------------
# load_checkpoint
# ---------------------------------------------------------------------------


def load_checkpoint(
    path: Union[str, Path],
    *,
    map_location: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load a checkpoint from disk.

    This is a thin wrapper around ``torch.load`` with a sensible default for
    ``map_location`` and validation of the file path.

    Parameters
    ----------
    path : str or Path
        The checkpoint file path.
    map_location : optional
        Forwarded to ``torch.load``.  Defaults to ``'cpu'``.

    Returns
    -------
    dict
        The loaded state dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    if map_location is None:
        map_location = "cpu"

    return torch.load(str(path), map_location=map_location, weights_only=False)


# ---------------------------------------------------------------------------
# Cleanup helper
# ---------------------------------------------------------------------------


def shutdown_saver() -> None:
    """Shut down the global checkpoint saver, waiting for pending I/O."""
    global _global_saver
    with _global_saver_lock:
        if _global_saver is not None:
            _global_saver.close()
            _global_saver = None
