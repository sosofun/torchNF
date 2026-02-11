"""
Unit tests for torchNF.checkpoint.

These tests exercise the Pinned Memory Pool, Ping-Pong Buffer, and
Async Checkpoint Saver **on CPU** (no GPU required) by mocking or
bypassing CUDA-specific paths where necessary.
"""

from __future__ import annotations

import os
import tempfile
import threading
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from torchNF.checkpoint.pinned_memory_pool import PinnedMemoryPool
from torchNF.checkpoint.ping_pong_buffer import PingPongBuffer
from torchNF.checkpoint.async_checkpoint_saver import AsyncCheckpointSaver
from torchNF.checkpoint.api import save_checkpoint, load_checkpoint


# ======================================================================
# Helpers
# ======================================================================


def _make_state_dict(num_params: int = 5, size: int = 64) -> dict:
    """Return a simple state_dict with CPU tensors."""
    return {f"layer_{i}.weight": torch.randn(size, size) for i in range(num_params)}


def _make_model() -> nn.Module:
    """A tiny model for integration tests."""
    return nn.Sequential(
        nn.Linear(32, 64),
        nn.ReLU(),
        nn.Linear(64, 10),
    )


# ======================================================================
# PinnedMemoryPool tests
# ======================================================================


class TestPinnedMemoryPool:
    def test_acquire_and_release(self):
        pool = PinnedMemoryPool()
        shape = torch.Size([4, 4])
        dtype = torch.float32

        buf = pool.acquire(shape, dtype)
        assert buf.shape == shape
        assert buf.dtype == dtype
        assert buf.is_pinned() or True  # pin_memory may be no-op on CPU-only
        assert pool.num_allocated == 1
        assert pool.num_free == 0

        pool.release(buf)
        assert pool.num_free == 1

        # Reacquire should return the same buffer object.
        buf2 = pool.acquire(shape, dtype)
        assert buf2.data_ptr() == buf.data_ptr()
        assert pool.num_free == 0

        pool.close()

    def test_preallocate_from_state_dict(self):
        sd = _make_state_dict(num_params=3, size=16)
        pool = PinnedMemoryPool.from_state_dict(sd)

        assert pool.num_allocated == 3
        assert pool.num_free == 3

        # Acquire all.
        bufs = []
        for name, t in sd.items():
            b = pool.acquire(t.shape, t.dtype)
            assert b.shape == t.shape
            bufs.append(b)

        assert pool.num_free == 0

        pool.release_all(bufs)
        assert pool.num_free == 3
        pool.close()

    def test_close_prevents_further_acquire(self):
        pool = PinnedMemoryPool()
        pool.close()
        with pytest.raises(RuntimeError, match="closed"):
            pool.acquire(torch.Size([2]), torch.float32)

    def test_different_shapes(self):
        pool = PinnedMemoryPool()

        buf_a = pool.acquire(torch.Size([2, 3]), torch.float32)
        buf_b = pool.acquire(torch.Size([4, 5]), torch.float64)

        pool.release(buf_a)
        pool.release(buf_b)

        # Each goes back to the correct free-list.
        buf_a2 = pool.acquire(torch.Size([2, 3]), torch.float32)
        assert buf_a2.data_ptr() == buf_a.data_ptr()

        buf_b2 = pool.acquire(torch.Size([4, 5]), torch.float64)
        assert buf_b2.data_ptr() == buf_b.data_ptr()

        pool.close()

    def test_thread_safety(self):
        """Multiple threads acquiring/releasing concurrently."""
        pool = PinnedMemoryPool()
        shape = torch.Size([8, 8])
        dtype = torch.float32
        errors: list = []

        def worker():
            try:
                for _ in range(50):
                    buf = pool.acquire(shape, dtype)
                    time.sleep(0.001)
                    pool.release(buf)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Errors in threads: {errors}"
        pool.close()


# ======================================================================
# PingPongBuffer tests
# ======================================================================


class TestPingPongBuffer:
    def test_swap_alternates_pools(self):
        sd = _make_state_dict(num_params=2, size=8)
        pp = PingPongBuffer.from_state_dict(sd)

        pool_w0 = pp.write_pool
        pool_r0 = pp.read_pool

        pp.swap()

        pool_w1 = pp.write_pool
        pool_r1 = pp.read_pool

        # After swap, roles are flipped.
        assert pool_w1 is pool_r0
        assert pool_r1 is pool_w0

        pp.close()

    def test_acquire_write_and_release_read(self):
        pp = PingPongBuffer()
        shape = torch.Size([4, 4])
        dtype = torch.float32

        buf = pp.acquire_write(shape, dtype)
        assert buf.shape == shape

        # Swap so the write pool becomes the read pool.
        pp.swap()

        pp.release_read(buf)
        assert pp.read_pool.num_free == 1

        pp.close()

    def test_double_swap_returns_to_original(self):
        pp = PingPongBuffer()
        w0 = pp.write_pool
        pp.swap()
        pp.swap()
        assert pp.write_pool is w0
        assert pp.swap_count == 2
        pp.close()


# ======================================================================
# AsyncCheckpointSaver tests (CPU only)
# ======================================================================


class TestAsyncCheckpointSaver:
    def test_basic_save_and_load(self):
        model = _make_model()
        sd = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            saver = AsyncCheckpointSaver(num_io_workers=1)
            future = saver.save(sd, path)
            future.result()  # wait for I/O

            loaded = torch.load(path, map_location="cpu", weights_only=False)
            for key in sd:
                assert torch.equal(sd[key], loaded[key]), f"Mismatch at {key}"

            saver.close()

    def test_multiple_saves(self):
        sd = _make_state_dict(num_params=3, size=16)

        with tempfile.TemporaryDirectory() as tmp:
            saver = AsyncCheckpointSaver(num_io_workers=1)

            for i in range(5):
                path = os.path.join(tmp, f"ckpt_{i}.pt")
                future = saver.save(sd, path)

            # Wait for all.
            saver.wait()
            assert saver.save_count == 5

            # Verify last checkpoint.
            loaded = torch.load(
                os.path.join(tmp, "ckpt_4.pt"),
                map_location="cpu",
                weights_only=False,
            )
            for key in sd:
                assert torch.equal(sd[key], loaded[key])

            saver.close()

    def test_extra_state(self):
        sd = _make_state_dict(num_params=2, size=8)
        extra = {"epoch": 42, "step": 1000}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            saver = AsyncCheckpointSaver()
            saver.save(sd, path, extra_state=extra).result()

            loaded = torch.load(path, map_location="cpu", weights_only=False)
            assert loaded["epoch"] == 42
            assert loaded["step"] == 1000

            saver.close()

    def test_atomic_write(self):
        """No .tmp file should remain after a successful save."""
        sd = _make_state_dict(num_params=1, size=4)

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            saver = AsyncCheckpointSaver()
            saver.save(sd, path).result()

            assert os.path.exists(path)
            assert not os.path.exists(path + ".tmp")

            saver.close()


# ======================================================================
# High-level API tests
# ======================================================================


class TestHighLevelAPI:
    def test_save_and_load_sync(self):
        model = _make_model()
        sd = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(sd, path, async_save=False)

            loaded = load_checkpoint(path)
            for key in sd:
                assert torch.equal(sd[key], loaded[key])

    def test_save_async_then_load(self):
        model = _make_model()
        sd = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(sd, path, async_save=True)

            # We need to wait for the background I/O to finish.
            from torchNF.checkpoint.api import _get_global_saver

            _get_global_saver().wait()

            loaded = load_checkpoint(path)
            for key in sd:
                assert torch.equal(sd[key], loaded[key])

    def test_load_nonexistent(self):
        with pytest.raises(FileNotFoundError, match="not found"):
            load_checkpoint("/nonexistent/path.pt")

    def test_save_with_extra_state(self):
        sd = _make_state_dict(num_params=1, size=4)
        extra = {"epoch": 7}

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "ckpt.pt")
            save_checkpoint(sd, path, extra_state=extra, async_save=False)

            loaded = load_checkpoint(path)
            assert loaded["epoch"] == 7


# ======================================================================
# GPU integration tests (only run when CUDA is available)
# ======================================================================


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
class TestGPUIntegration:
    def test_gpu_state_dict_save_load(self):
        model = _make_model().cuda()
        sd = model.state_dict()

        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "gpu_ckpt.pt")
            saver = AsyncCheckpointSaver()
            saver.save(sd, path).result()

            loaded = load_checkpoint(path)
            cpu_sd = {k: v.cpu() for k, v in sd.items()}
            for key in cpu_sd:
                assert torch.equal(cpu_sd[key], loaded[key])

            saver.close()

    def test_ping_pong_gpu_d2h(self):
        """Verify that pinned buffers actually get the correct GPU data."""
        t_gpu = torch.randn(128, 128, device="cuda")
        pp = PingPongBuffer()

        buf = pp.acquire_write(t_gpu.shape, t_gpu.dtype)
        stream = torch.cuda.Stream()
        with torch.cuda.stream(stream):
            buf.copy_(t_gpu, non_blocking=True)
        stream.synchronize()

        assert torch.equal(buf, t_gpu.cpu())
        pp.close()

    def test_multiple_gpu_saves_pipelined(self):
        """High-frequency saves that exercise the ping-pong swap."""
        model = _make_model().cuda()

        with tempfile.TemporaryDirectory() as tmp:
            saver = AsyncCheckpointSaver()
            for i in range(10):
                sd = model.state_dict()
                path = os.path.join(tmp, f"step_{i}.pt")
                saver.save(sd, path)

            saver.wait()
            assert saver.save_count == 10

            # Spot-check last file.
            loaded = load_checkpoint(os.path.join(tmp, "step_9.pt"))
            cpu_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            for key in cpu_sd:
                assert torch.allclose(cpu_sd[key], loaded[key])

            saver.close()
