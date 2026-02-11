"""
Pinned Memory Pool for fast GPU-to-CPU (D2H) tensor copy.

In standard PyTorch checkpoint saving, every call to `tensor.cpu()` or
`tensor.to('cpu')` triggers a fresh `cudaHostAlloc` / `cudaMallocHost` under
the hood (when non-blocking copy is used with pinned memory).  The repeated
allocation & deallocation of page-locked memory is expensive.

This module implements a **reusable pinned memory pool** that:
1. Pre-allocates a set of pinned-memory tensors matching the shapes/dtypes
   needed by the model's state_dict.
2. Allows zero-copy, asynchronous D2H transfers via CUDA streams.
3. Returns buffers to the pool after I/O workers have finished writing,
   so they can be reused in the next checkpoint cycle.
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import torch


# A shape+dtype key that uniquely identifies a buffer "slot".
_BufferKey = Tuple[torch.Size, torch.dtype]


class PinnedMemoryPool:
    """A pool of pre-allocated pinned-memory CPU tensors.

    Each buffer is identified by (shape, dtype).  The pool maintains a free-list
    for every unique (shape, dtype) combination.  When a buffer is requested,
    it is popped from the free-list; when the caller is done, it is returned.

    If the free-list is empty, a new pinned tensor is allocated on-the-fly and
    added to the pool (so the pool grows lazily to match demand).

    Parameters
    ----------
    prealloc_spec : dict, optional
        A mapping ``{name: (shape, dtype)}`` used to pre-allocate buffers.
        Typically built from ``model.state_dict()`` before the first save.
    """

    def __init__(self, prealloc_spec: Optional[Dict[str, Tuple[torch.Size, torch.dtype]]] = None):
        self._lock = threading.Lock()
        # free-list: key -> [tensor, tensor, …]
        self._free: Dict[_BufferKey, List[torch.Tensor]] = defaultdict(list)
        # Track all allocated tensors so we can free them on shutdown.
        self._all_buffers: List[torch.Tensor] = []
        self._closed = False

        if prealloc_spec is not None:
            self._preallocate(prealloc_spec)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def acquire(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Get a pinned-memory buffer of the requested shape and dtype.

        If a matching buffer is available in the free-list it is reused;
        otherwise a new pinned tensor is allocated.

        Returns
        -------
        torch.Tensor
            A CPU tensor backed by pinned (page-locked) memory.
        """
        if self._closed:
            raise RuntimeError("PinnedMemoryPool has been closed.")

        key: _BufferKey = (shape, dtype)
        with self._lock:
            free_list = self._free[key]
            if free_list:
                return free_list.pop()

        # Allocate outside the lock to avoid holding it during CUDA calls.
        return self._allocate(shape, dtype)

    def release(self, tensor: torch.Tensor) -> None:
        """Return a buffer to the pool for future reuse.

        Parameters
        ----------
        tensor : torch.Tensor
            Must be a pinned-memory CPU tensor previously obtained from
            :meth:`acquire`.
        """
        if self._closed:
            return
        key: _BufferKey = (tensor.shape, tensor.dtype)
        with self._lock:
            self._free[key].append(tensor)

    def release_all(self, tensors: List[torch.Tensor]) -> None:
        """Batch-release multiple tensors back to the pool."""
        if self._closed:
            return
        with self._lock:
            for t in tensors:
                key: _BufferKey = (t.shape, t.dtype)
                self._free[key].append(t)

    def close(self) -> None:
        """Release all resources held by the pool.

        After calling this, the pool can no longer be used.
        """
        with self._lock:
            self._closed = True
            self._free.clear()
            self._all_buffers.clear()

    @property
    def num_free(self) -> int:
        """Total number of free buffers across all keys."""
        with self._lock:
            return sum(len(v) for v in self._free.values())

    @property
    def num_allocated(self) -> int:
        """Total number of buffers ever allocated by this pool."""
        with self._lock:
            return len(self._all_buffers)

    # ------------------------------------------------------------------
    # Pre-allocation helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_state_dict(cls, state_dict: Dict[str, torch.Tensor]) -> "PinnedMemoryPool":
        """Create a pool pre-sized for the given state_dict.

        This inspects every tensor in the state_dict and pre-allocates a
        matching pinned buffer so that the first checkpoint save incurs no
        allocation overhead.
        """
        spec: Dict[str, Tuple[torch.Size, torch.dtype]] = {}
        for name, tensor in state_dict.items():
            spec[name] = (tensor.shape, tensor.dtype)
        return cls(prealloc_spec=spec)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _preallocate(self, spec: Dict[str, Tuple[torch.Size, torch.dtype]]) -> None:
        """Pre-allocate one pinned buffer per entry in *spec*."""
        for _name, (shape, dtype) in spec.items():
            buf = self._allocate(shape, dtype)
            key: _BufferKey = (shape, dtype)
            with self._lock:
                self._free[key].append(buf)

    def _allocate(self, shape: torch.Size, dtype: torch.dtype) -> torch.Tensor:
        """Allocate a single pinned-memory tensor and register it.

        Falls back to regular CPU memory when CUDA / pinned-memory allocation
        is not available (e.g. CPU-only machines).
        """
        try:
            buf = torch.empty(shape, dtype=dtype, pin_memory=True)
        except RuntimeError:
            # pin_memory requires a working CUDA driver.  Gracefully degrade.
            buf = torch.empty(shape, dtype=dtype)
        with self._lock:
            self._all_buffers.append(buf)
        return buf

    def __repr__(self) -> str:
        return (
            f"PinnedMemoryPool(allocated={self.num_allocated}, "
            f"free={self.num_free}, closed={self._closed})"
        )

    def __del__(self) -> None:
        self.close()
