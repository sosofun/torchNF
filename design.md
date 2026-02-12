## 固定内存 N 路缓冲区 + 背压机制：4+1 架构视图

本设计在现有 `memory_checkpoint` 架构上，引入 **N 路固定内存缓冲池（Ring Buffer Pool）+ 背压机制**，用于进一步提升高频 checkpoint 场景下的吞吐与稳定性。下面采用 4+1 视图描述。

---

### 一、逻辑视图（Logical View）

关注“有哪些角色与核心组件、它们之间的关系”。

```mermaid
flowchart LR
    subgraph TrainingSide[训练侧（每个 rank）]
        A[训练循环<br/>Megatron Engine]
        B[Checkpoint API<br/>save_checkpoint/load_checkpoint]
        C[MemoryCheckpoint<br/>(singleton per process)]
        D[SharedMemoryHandler<br/>+ N 路 BufferPool 适配层]
    end

    subgraph AsyncSaver[异步落盘侧（每 node）]
        E[AsyncCheckpointSaver<br/>(Agent 内)]
        F[RingBufferPool&lt;N&gt;<br/>固定内存缓冲池]
        G[IO Workers<br/>多线程/多进程]
    end

    subgraph ReplicaLayer[副本层（跨 rank）]
        H[ReplicaCheckpointManager]
    end

    subgraph Storage[持久化存储]
        I[(FS / Object Store)]
    end

    A --> B --> C --> D
    C <--> H
    D <-->|Ping| F
    F <-->|Pong| G --> I
    C <--> E
```

要点：
- 训练侧通过 `Checkpoint API → MemoryCheckpoint` 写入共享内存 / 固定内存缓冲池。
- `RingBufferPool<N>` 在每个 rank 或每 node 内负责管理 N 个固定大小 buffer 的生命周期。
- `AsyncCheckpointSaver` 中的 IO workers 从 `RingBufferPool` 拉取 READY/FLUSHING 状态的 buffer，执行真正 I/O。
- `ReplicaCheckpointManager` 按原有机制（all_gather + backup shm）做跨 rank 副本，不与 N 路缓冲机制直接耦合。

---

### 二、开发视图（Development / Module View）

关注“代码层面的模块划分与依赖关系”。

```mermaid
flowchart TB
    subgraph PKG_memory_checkpoint[memory_checkpoint 包]
        subgraph Core[核心模块]
            MP[api.py<br/>save_checkpoint/load_checkpoint]
            MC[checkpoint.py<br/>MemoryCheckpoint<br/>ReplicaCheckpointManager]
            SH[shared_memory_handler.py<br/>SharedMemoryHandler]
            RB[buffer_pool.py (新增)<br/>RingBufferPool&lt;N&gt; + 背压接口]
            SA[saver.py<br/>AsyncCheckpointSaver + IOWorkers]
        end

        subgraph Common[公共模块]
            CT[common/constants.py]
            CF[common/config.py]
            MT[common/meta.py]
            CM[common/communicator.py]
            UT[common/util.py]
        end

        subgraph Elastic[弹性训练]
            AG[agent.py<br/>CloudmlTrainingAgent<br/>Rendezvous Backend]
            RN[run.py<br/>入口脚本]
        end
    end

    MP --> MC --> SH
    SH --> RB
    MC --> SA
    SA --> RB
    MC --> HLP[ReplicaCheckpointManager]

    RB --> CT
    SA --> CM
    MC --> CF
    SH --> MT
    AG --> SA
    RN --> AG
```

建议的新增 / 改动点：
- **新增 `buffer_pool.py`（或整合到 `shared_memory_handler.py`）**：
  - 定义 `BufferState/BufferSlot/RingBufferPool` 与背压策略接口（阻塞/丢弃/降级）。
  - 提供通用接口：`acquire_for_write/mark_ready/acquire_for_flush/mark_flushed`。
- **在 `SharedMemoryHandler` 中增加对 `RingBufferPool` 的适配层**：
  - 写路径从“单一 shm buffer”升级为“从池中选一个可写 buffer”。
  - 读/落盘路径从“读固定 shm”升级为“从池中取 READY 槽位”。
- **在 `AsyncCheckpointSaver` 中使用 `RingBufferPool`**：
  - 与当前消息队列机制组合：SAVE 信号不再隐含唯一 buffer，而是携带或绑定到一个 buffer slot。

---

### 三、进程视图（Process / Concurrency View）

关注“线程/进程之间的并发关系与背压路径”。

```mermaid
sequenceDiagram
    participant Train as 训练进程(rank i)
    participant MC as MemoryCheckpoint
    participant SH as SharedMemoryHandler<br/>+ BufferPool 适配
    participant Pool as RingBufferPool&lt;N&gt;
    participant Saver as AsyncCheckpointSaver
    participant IO as IO Worker(k)
    participant Store as 存储后端

    loop 每隔若干 step 触发一次 ckpt
        Train->>MC: save_checkpoint(...)
        MC->>SH: 序列化 state_dict
        SH->>Pool: acquire_for_write(wait / timeout)
        alt 有 FREE buffer
            Pool-->>SH: 返回 slot(j, state=FILLING)
            SH->>Slot: 写入数据 + meta(iter, path)
            SH->>Pool: mark_ready(slot j)
            MC->>Saver: QueueCommunicator.send(SAVE)
        else 无 FREE buffer
            opt 背压策略 A：阻塞等待
                Pool-->>SH: 阻塞直至有 FREE
            end
            opt 背压策略 B：降级 / 丢弃
                SH-->>MC: 返回失败（跳过本次 ckpt 或只保存轻量信息）
            end
        end
    end

    loop Async saver 后台线程
        Saver->>Saver: 统计收到的 SAVE 信号
        Saver->>IO: 通知可以从 Pool 取 READY buffer
    end

    loop IO worker(k)
        IO->>Pool: acquire_for_flush(wait)
        Pool-->>IO: 返回 READY slot(m) 或 None
        alt 得到 slot(m)
            IO->>Store: flush(slot m.data, slot m.meta.save_path)
            IO->>Pool: mark_flushed(slot m)
        else 无 READY
            IO-->>IO: 短暂休眠或结束
        end
    end
```

要点：
- 背压行为在 `acquire_for_write(wait, timeout)` 上实现：
  - `wait=True` + 合理 timeout：训练侧可容忍有限阻塞。
  - `wait=False` 或 timeout 到达：上层可选择跳过当前 checkpoint、仅写轻量 ckpt、或记录告警。
- I/O worker 与训练侧完全解耦，通过 `RingBufferPool` 状态与队列信号协作。

---

### 四、物理视图（Physical / Deployment View）

关注“在实际硬件 / 进程拓扑上的部署方式”。

```mermaid
flowchart TB
    subgraph Node0[Node 0]
        subgraph GPU0[GPU 设备组]
            T0[训练进程 rank0]
            T1[训练进程 rank1]
            T2[...]
        end

        subgraph Agent0[CloudmlTrainingAgent + AsyncCheckpointSaver]
            Q0[QueueCommunicator Server]
            P0[RingBufferPool&lt;N&gt; per node 或 per rank]
            W0[IO Worker 0..M]
        end
    end

    subgraph NodeK[Node k]
        subgraph GPUk[GPU 设备组]
            Tk0[训练进程 rank(k*8)]
            Tk1[...]
        end

        subgraph AgentK[CloudmlTrainingAgent + AsyncCheckpointSaver]
            Qk[QueueCommunicator Server]
            Pk[RingBufferPool&lt;N&gt;]
            Wk[IO Worker 0..M]
        end
    end

    subgraph Cluster[集群存储]
        S[(共享文件系统 / 对象存储)]
    end

    W0 --> S
    Wk --> S
```

可选实现策略：
- **每 rank 一个 RingBufferPool + shm**（保持与现有设计一致）：简单直接，隔离性好。
- **每 node 共用一个 RingBufferPool（按 global_rank 分片）**：
  - 更利于做跨 rank 合并写（batch I/O），但需要在缓冲区 meta 中记录 rank/offset。

---

### 五、“+1” 场景视图（Scenarios）

下面以一个典型高频 checkpoint 场景为例，说明 N 路缓冲与背压的价值。

#### 场景：高频全量 checkpoint（含 optimizer）+ 后端存储偶发抖动

1. 训练配置为 **每 N step 保存一次全量 checkpoint**，数据量较大。
2. 后端存储（例如远程文件系统 / 对象存储）在某个时间段性能抖动，单次写入耗时变长。
3. 在 **原有单 buffer** 或 **2 路 Ping-Pong** 设计中：
   - 很容易出现：两个 buffer 都处于 FLUSHING 状态，训练侧被迫阻塞等待，形成「长尾」。
4. 在 **N 路 RingBufferPool + 背压** 设计中：
   - 训练侧可提前配置：
     - `num_slots = 4`；
     - `acquire_for_write(wait=True, timeout=短)`；
   - 在短暂抖动期间，大部分 checkpoint 仍能快速写入新的 FREE 槽位并返回；
   - 当所有 4 个槽位都处于 FLUSHING/READY 且超过 timeout 时：
     - 背压策略生效：例如跳过一个 checkpoint 或仅写轻量 ckpt（不含 optimizer）。
   - 抖动结束后，I/O worker 逐步清理 backlog，缓冲池恢复到正常水位。

通过这种设计，系统在保持 **训练主循环尽量平滑** 的同时，为用户提供了可配置的 **一致性/完整性 vs 吞吐/延迟** 的折中点。

