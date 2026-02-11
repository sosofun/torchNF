# torchNF - PyTorch 快速 Checkpoint 方案

基于 **固定内存池（Pinned Memory Pool）** 和 **Ping-Pong 双缓冲机制** 的高性能异步 Checkpoint 保存方案。

---

## 设计背景

在 PyTorch 训练中，保存 Checkpoint 的标准流程是：

```
GPU Tensor → CPU Tensor（D2H copy） → 序列化 → 写磁盘
```

每次 D2H 拷贝时，PyTorch 默认会通过 `cudaHostAlloc` 分配一块临时的 page-locked（pinned）内存，
拷贝完成后再通过 `cudaHostFree` 释放。**反复分配/释放 pinned memory 的开销在高频存储场景下非常显著。**

本方案通过两层优化解决这一问题：

### 1. Pinned Memory Pool（固定内存池）

预分配一组与模型 `state_dict` 形状匹配的 pinned memory buffer，
在整个训练过程中**复用**这些 buffer，避免每次 checkpoint 都触发内存分配。

### 2. Ping-Pong Buffering（双缓冲交替）

在固定内存池基础上，维护 **两个独立的内存池**（Pool-A 和 Pool-B），
它们交替扮演 **写 buffer** 和 **读 buffer** 的角色：

```
Cycle N  :  Pool-A ← GPU (D2H write)   |  Pool-B → Disk (I/O read)
Cycle N+1:  Pool-B ← GPU (D2H write)   |  Pool-A → Disk (I/O read)
```

- **写 buffer**：接收来自 GPU 的异步 D2H 拷贝
- **读 buffer**：被后台 I/O worker 消费，数据写入磁盘

由于两个池互不干扰，D2H 拷贝和磁盘 I/O 形成**流水线**，
不再需要同步等待上一次 I/O 完成后才能开始下一次 D2H。

---

## 架构总览

```
┌──────────────────────────────────────────────────────────┐
│                   AsyncCheckpointSaver                   │
│                                                          │
│  ┌─────────────┐    ┌───────────────────┐    ┌────────┐  │
│  │ CUDA Stream │───▶│  PingPongBuffer   │───▶│  I/O   │  │
│  │ (async D2H) │    │ ┌───────┬───────┐ │    │ Worker │  │
│  └─────────────┘    │ │Pool-A │Pool-B │ │    │ Thread │  │
│                     │ │(write)│(read) │ │    └───┬────┘  │
│                     │ └───────┴───────┘ │        │       │
│                     │    swap() ↻       │        ▼       │
│                     └───────────────────┘     [Disk]     │
└──────────────────────────────────────────────────────────┘
```

---

## 快速开始

### 安装依赖

```bash
pip install torch
```

### 基本使用

```python
import torch
import torch.nn as nn
from torchNF.checkpoint import save_checkpoint, load_checkpoint

# 定义模型
model = nn.Linear(1024, 1024).cuda()
optimizer = torch.optim.Adam(model.parameters())

# ---- 保存 Checkpoint（异步） ----
save_checkpoint(
    model.state_dict(),
    "checkpoints/step_1000.pt",
    extra_state={"epoch": 5, "step": 1000},
    async_save=True,   # 后台 I/O，不阻塞训练
)

# ---- 加载 Checkpoint ----
ckpt = load_checkpoint("checkpoints/step_1000.pt")
model.load_state_dict({k: v for k, v in ckpt.items() if k != "epoch" and k != "step"})
print(f"Resumed from epoch {ckpt['epoch']}, step {ckpt['step']}")
```

### 高级用法：显式管理 Saver 生命周期

```python
from torchNF.checkpoint import AsyncCheckpointSaver

# 创建 saver，可自定义 I/O 线程数
saver = AsyncCheckpointSaver(num_io_workers=2)

for step in range(10000):
    # ... 训练 ...

    if step % 500 == 0:
        future = saver.save(
            model.state_dict(),
            f"checkpoints/step_{step}.pt",
            extra_state={"step": step},
        )
        # future.result()  # 如需同步等待，可调用此行

# 训练结束，等待所有 I/O 完成
saver.close()
```

### 直接操作内存池

```python
from torchNF.checkpoint import PinnedMemoryPool, PingPongBuffer

# 从 state_dict 预分配
pool = PinnedMemoryPool.from_state_dict(model.state_dict())
buf = pool.acquire(torch.Size([1024, 1024]), torch.float32)
buf.copy_(gpu_tensor, non_blocking=True)
# ... 使用 buf ...
pool.release(buf)

# Ping-Pong 双缓冲
pp = PingPongBuffer.from_state_dict(model.state_dict())
write_buf = pp.acquire_write(shape, dtype)  # 从写池获取
# ... D2H copy ...
pp.swap()                                    # 角色交换
pp.release_read(write_buf)                   # I/O 完成后归还读池
```

---

## 模块说明

| 模块 | 说明 |
|------|------|
| `PinnedMemoryPool` | 固定内存池，管理 pinned memory buffer 的分配、复用和回收 |
| `PingPongBuffer` | 双缓冲机制，两个内存池交替充当 D2H 写入端和 I/O 读取端 |
| `AsyncCheckpointSaver` | 异步 Checkpoint 保存器，编排 CUDA Stream + PingPong + I/O Worker |
| `save_checkpoint` | 高层 API，一行代码完成异步保存 |
| `load_checkpoint` | 高层 API，加载 checkpoint 文件 |

---

## 运行测试

```bash
pip install pytest
python -m pytest tests/test_checkpoint.py -v
```

GPU 集成测试会在 CUDA 可用时自动运行，否则跳过。

---

## 性能优势

| 场景 | 标准 `torch.save` | 本方案 |
|------|-------------------|--------|
| 内存分配 | 每次 D2H 都 `cudaHostAlloc` | 预分配 + 复用，零分配开销 |
| D2H 与 I/O 串行 | 必须等 I/O 完成才能下次 D2H | Ping-Pong 流水线并行 |
| 训练阻塞 | D2H + I/O 全程阻塞 | 仅 D2H 阻塞，I/O 异步 |
| 高频保存 | 内存池回收等待开销大 | 双缓冲消除同步等待 |

---

## License

MIT
