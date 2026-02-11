"""
Asynchronous Checkpoint Saver – the orchestrator.

This module ties together:
- **PingPongBuffer** for double-buffered pinned memory,
- **CUDA streams** for asynchronous D2H copies,
- **I/O worker threads** for background disk writes.

Save Pipeline (per cycle)
-------------------------
1. *D2H stage* – iterate over the model's state_dict, acquire pinned buffers
   from the **write pool** of the PingPongBuffer, and launch non-blocking
   ``tensor.copy_()`` on a dedicated CUDA stream.
2. *Stream sync* – synchronize the CUDA stream so that all D2H transfers are
   complete before handing the data to the I/O worker.
3. *Swap* – flip the PingPongBuffer so the just-filled pool becomes the
   **read pool**.
4. *I/O stage* – submit the filled buffers to a background thread (or thread
   pool) that serialises and writes them to disk.  Meanwhile, the next
   ``save()`` call can immediately start D2H into the now-available write pool.
5. *Recycle* – when the I/O worker finishes, it returns all buffers to the
   read pool, making them available for the next swap cycle.

Thread Safety
-------------
- The D2H stage and swap are always called from the **training thread**.
- The I/O stage runs in a separate daemon thread.
- Buffer acquire/release use the pool's internal lock.
"""

from __future__ import annotations

import io
import os
import logging
import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

from torchNF.checkpoint.ping_pong_buffer import PingPongBuffer

logger = logging.getLogger(__name__)


class AsyncCheckpointSaver:
    """High-performance asynchronous checkpoint saver.

    Parameters
    ----------
    num_io_workers : int
        Number of background threads for I/O.  ``1`` is usually enough since
        the bottleneck is typically the filesystem.
    ping_pong : PingPongBuffer, optional
        An existing PingPongBuffer to use.  If *None*, one will be lazily
        created on the first ``save()`` call based on the state_dict.
    cuda_stream : torch.cuda.Stream, optional
        A dedicated CUDA stream for D2H copies.  If *None*, a new one is
        created on the first ``save()``.
    serialize_fn : callable, optional
        Custom serialisation function ``(state_dict, file_path) -> None``.
        Defaults to ``torch.save``.
    """

    def __init__(
        self,
        num_io_workers: int = 1,
        ping_pong: Optional[PingPongBuffer] = None,
        cuda_stream: Optional[torch.cuda.Stream] = None,
        serialize_fn: Optional[Callable] = None,
    ):
        self._num_io_workers = num_io_workers
        self._ping_pong = ping_pong
        self._cuda_stream = cuda_stream
        self._serialize_fn = serialize_fn or _default_serialize
        self._io_pool: Optional[ThreadPoolExecutor] = None

        # Track the latest I/O future so we can optionally wait for it.
        self._latest_io_future: Optional[Future] = None
        self._save_count: int = 0
        self._lock = threading.Lock()
        self._closed = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save(
        self,
        state_dict: Dict[str, Any],
        path: Union[str, Path],
        *,
        extra_state: Optional[Dict[str, Any]] = None,
        wait_for_prev: bool = True,
    ) -> Future:
        """Save a checkpoint asynchronously.

        Parameters
        ----------
        state_dict : dict
            Typically ``model.state_dict()``.  Tensors on GPU will be copied
            to pinned CPU memory asynchronously.
        path : str or Path
            Destination file path.
        extra_state : dict, optional
            Additional non-tensor metadata (e.g. optimizer state on CPU,
            epoch number) that will be included in the saved file.
        wait_for_prev : bool
            If *True* (default), block until the previous I/O job finishes
            before starting D2H.  This prevents the write pool from being
            exhausted when saves are issued faster than disk can write.

        Returns
        -------
        concurrent.futures.Future
            A future that resolves when the I/O write is complete.
        """
        if self._closed:
            raise RuntimeError("AsyncCheckpointSaver has been closed.")

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Lazy init --------------------------------------------------------
        if self._ping_pong is None:
            self._ping_pong = PingPongBuffer.from_state_dict(
                {k: v for k, v in state_dict.items() if isinstance(v, torch.Tensor)}
            )
        if self._io_pool is None:
            self._io_pool = ThreadPoolExecutor(
                max_workers=self._num_io_workers,
                thread_name_prefix="ckpt_io",
            )
        if self._cuda_stream is None and torch.cuda.is_available():
            self._cuda_stream = torch.cuda.Stream()

        # Wait for previous I/O to free the read pool ---------------------
        if wait_for_prev and self._latest_io_future is not None:
            self._latest_io_future.result()  # blocks

        # 1) D2H copy into the write pool ---------------------------------
        pinned_state, cpu_state = self._d2h_copy(state_dict)

        # 2) Synchronize the CUDA stream ----------------------------------
        if self._cuda_stream is not None:
            self._cuda_stream.synchronize()

        # 3) Swap buffers -------------------------------------------------
        self._ping_pong.swap()

        # Merge extra state -----------------------------------------------
        full_state: Dict[str, Any] = {}
        full_state.update(pinned_state)
        full_state.update(cpu_state)
        if extra_state:
            full_state.update(extra_state)

        # 4) Submit I/O to background thread ------------------------------
        # We capture a reference to the buffers so the I/O worker can
        # release them back to the (now) read pool when done.
        pinned_buffers = list(pinned_state.values())
        read_pool = self._ping_pong.read_pool  # snapshot reference

        future = self._io_pool.submit(
            self._io_worker,
            full_state,
            path,
            pinned_buffers,
            read_pool,
        )
        self._latest_io_future = future
        self._save_count += 1
        return future

    def wait(self) -> None:
        """Block until the latest I/O job completes."""
        if self._latest_io_future is not None:
            self._latest_io_future.result()

    def close(self) -> None:
        """Shut down the I/O thread pool and release all buffers."""
        if self._closed:
            return
        self._closed = True
        self.wait()
        if self._io_pool is not None:
            self._io_pool.shutdown(wait=True)
        if self._ping_pong is not None:
            self._ping_pong.close()

    @property
    def save_count(self) -> int:
        return self._save_count

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _d2h_copy(
        self, state_dict: Dict[str, Any]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """Copy GPU tensors to pinned CPU buffers; leave CPU data as-is.

        Returns
        -------
        pinned_state : dict
            Keys whose values were GPU tensors, now residing in pinned CPU
            memory obtained from the write pool.
        cpu_state : dict
            Keys whose values were already on CPU or are non-tensor objects.
        """
        assert self._ping_pong is not None
        pinned_state: Dict[str, torch.Tensor] = {}
        cpu_state: Dict[str, Any] = {}

        stream = self._cuda_stream

        for key, value in state_dict.items():
            if isinstance(value, torch.Tensor) and value.is_cuda:
                buf = self._ping_pong.acquire_write(value.shape, value.dtype)
                if stream is not None:
                    with torch.cuda.stream(stream):
                        buf.copy_(value, non_blocking=True)
                else:
                    buf.copy_(value)
                pinned_state[key] = buf
            elif isinstance(value, torch.Tensor):
                # Already on CPU – clone to avoid mutation during async I/O.
                cpu_state[key] = value.clone()
            else:
                cpu_state[key] = value

        return pinned_state, cpu_state

    def _io_worker(
        self,
        state_dict: Dict[str, Any],
        path: Path,
        pinned_buffers: List[torch.Tensor],
        read_pool,
    ) -> None:
        """Background I/O worker: serialise *state_dict* to *path*, then
        return pinned buffers to the pool."""
        try:
            self._serialize_fn(state_dict, str(path))
            logger.debug("Checkpoint written to %s", path)
        finally:
            # 5) Recycle buffers back to the pool.
            read_pool.release_all(pinned_buffers)

    def __repr__(self) -> str:
        return (
            f"AsyncCheckpointSaver("
            f"saves={self._save_count}, "
            f"io_workers={self._num_io_workers}, "
            f"ping_pong={self._ping_pong!r})"
        )

    def __del__(self) -> None:
        self.close()


# ------------------------------------------------------------------
# Default serialisation helper
# ------------------------------------------------------------------


def _default_serialize(state_dict: Dict[str, Any], path: str) -> None:
    """Serialize using torch.save with an intermediate buffer to avoid
    partial writes on crash."""
    tmp_path = path + ".tmp"
    torch.save(state_dict, tmp_path)
    os.replace(tmp_path, path)  # atomic on POSIX
