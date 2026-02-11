"""
Ping-Pong Buffering on top of Pinned Memory Pools.

In high-frequency checkpoint scenarios, a single pinned memory pool would
require the saver to *wait* until the I/O worker finishes writing the
previous checkpoint before the pool can be recycled for the next D2H copy.
This synchronous wait becomes the bottleneck.

The **Ping-Pong Buffer** solves this by maintaining **two independent pinned
memory pools** (called "ping" and "pong").  They alternate roles each
checkpoint cycle:

    Cycle N  :  Pool-A = *write buffer* (GPU → pinned)   |  Pool-B = *read buffer* (pinned → disk by I/O worker)
    Cycle N+1:  Pool-B = *write buffer*                   |  Pool-A = *read buffer*

Because the two pools are independent, D2H copies into the write buffer can
proceed concurrently with I/O from the read buffer, forming a pipeline.

Lifecycle
---------
1. ``get_write_pool()`` – returns the current write pool for D2H copies.
2. ``swap()``           – flips the roles.  The caller must ensure that the
   previous I/O from the *other* pool has completed (or will not conflict).
3. ``get_read_pool()``  – returns the pool that was just written and is now
   ready for the I/O worker.
"""

from __future__ import annotations

import threading
from typing import Dict, Optional, Tuple

import torch

from torchNF.checkpoint.pinned_memory_pool import PinnedMemoryPool


class PingPongBuffer:
    """Double-buffered pinned memory for pipelined D2H + I/O.

    Parameters
    ----------
    prealloc_spec : dict, optional
        ``{name: (shape, dtype)}`` – forwarded to both internal pools for
        pre-allocation.
    """

    def __init__(
        self,
        prealloc_spec: Optional[Dict[str, Tuple[torch.Size, torch.dtype]]] = None,
    ):
        # Initialise _closed first so __del__ never sees a missing attribute.
        self._closed = False
        self._pool_a = PinnedMemoryPool(prealloc_spec=prealloc_spec)
        self._pool_b = PinnedMemoryPool(prealloc_spec=prealloc_spec)

        # Index into (_pool_a, _pool_b).
        # _write_idx points to the pool currently used for D2H.
        self._write_idx: int = 0
        self._lock = threading.Lock()
        self._swap_count: int = 0

    # ------------------------------------------------------------------
    # Pool accessors
    # ------------------------------------------------------------------

    @property
    def write_pool(self) -> PinnedMemoryPool:
        """The pool that should receive D2H copies right now."""
        return self._pools[self._write_idx]

    @property
    def read_pool(self) -> PinnedMemoryPool:
        """The pool that the I/O worker should drain right now.

        This is the pool that was the *write* pool in the previous cycle.
        """
        return self._pools[1 - self._write_idx]

    @property
    def _pools(self):
        return (self._pool_a, self._pool_b)

    # ------------------------------------------------------------------
    # Swap
    # ------------------------------------------------------------------

    def swap(self) -> None:
        """Swap write and read roles.

        Call this **after** all D2H copies for the current cycle are done and
        you have handed off the filled buffers to the I/O worker.
        """
        with self._lock:
            self._write_idx = 1 - self._write_idx
            self._swap_count += 1

    # ------------------------------------------------------------------
    # Convenience wrappers
    # ------------------------------------------------------------------

    def acquire_write(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Acquire a pinned buffer from the **write** pool."""
        return self.write_pool.acquire(shape, dtype)

    def release_read(self, tensor: torch.Tensor) -> None:
        """Release a buffer back to the **read** pool after I/O completes."""
        self.read_pool.release(tensor)

    def release_read_all(self, tensors) -> None:
        """Batch-release buffers back to the **read** pool."""
        self.read_pool.release_all(tensors)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor]) -> "PingPongBuffer":
        """Create a PingPongBuffer pre-sized for the given state_dict."""
        spec: Dict[str, Tuple[torch.Size, torch.dtype]] = {}
        for name, tensor in state_dict.items():
            spec[name] = (tensor.shape, tensor.dtype)
        return cls(prealloc_spec=spec)

    def close(self) -> None:
        """Release all resources from both pools."""
        if self._closed:
            return
        self._closed = True
        self._pool_a.close()
        self._pool_b.close()

    @property
    def swap_count(self) -> int:
        return self._swap_count

    def __repr__(self) -> str:
        return (
            f"PingPongBuffer(write_idx={self._write_idx}, "
            f"swaps={self._swap_count}, "
            f"pool_a={self._pool_a!r}, pool_b={self._pool_b!r})"
        )

    def __del__(self) -> None:
        self.close()
