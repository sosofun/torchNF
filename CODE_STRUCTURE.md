# 代码结构规划（MVP：DP/TP/EP + DTensor + FSDP/FSDP2）

本文件用于规划基于 PyTorch DTensor 的分布式训练框架代码结构。范围仅覆盖 **数据并行（DP）**、**张量并行（TP）**、**专家并行（EP）** 及其混合，并要求 **适配 FSDP 与 FSDP2**。

## 1. 设计原则
1. **DTensor 原生优先**：以 DeviceMesh + DTensor Layout 为唯一分布式张量抽象。
2. **MVP 优先**：只做 DP/TP/EP 相关模块，避免扩展到 PP/CP。
3. **清晰分层**：API → Planner → Runtime → Communication/Adapter。
4. **最小侵入**：对 torch.nn.Module 的改造尽量轻量。
5. **可替换**：FSDP/FSDP2 与 MoE Router/All-to-All 为可插拔适配层。

## 2. 建议目录结构
```
torchnf/
  __init__.py
  config/
    __init__.py
    parallel_config.py        # ParallelConfig 与校验
  api/
    __init__.py
    distribute.py             # distribute(model, config)
    trainer.py                # Trainer/训练循环
  core/
    __init__.py
    mesh.py                   # DeviceMesh 构建与拓扑
    layout.py                 # DTensor 布局/Placement 工具
    dtensor_utils.py          # DTensor 转换与辅助
  parallel/
    __init__.py
    dp.py                     # DP 逻辑（与 FSDP/FSDP2 适配）
    tp.py                     # TP 逻辑（并行线性/注意力等）
    ep.py                     # EP 逻辑（MoE 路由/All-to-All）
    mixed.py                  # DP+TP+EP 组合策略
  fsdp/
    __init__.py
    fsdp_adapter.py           # FSDP 适配层
    fsdp2_adapter.py          # FSDP2 适配层
    state_dict.py             # state_dict/load_state_dict 兼容
  moe/
    __init__.py
    router.py                 # Top-1/Top-2 路由
    experts.py                # Expert 容量与负载均衡
    all_to_all.py             # DTensor + NCCL All-to-All 包装
  modules/
    __init__.py
    parallel_linear.py        # TP 线性层
    parallel_attention.py     # TP 注意力（MVP 版）
    moe_layer.py              # MoE 层
  runtime/
    __init__.py
    engine.py                 # 训练引擎/Step 封装
    checkpoint.py             # 保存/恢复（兼容 FSDP/FSDP2）
    logging.py                # 统一日志与指标
  utils/
    __init__.py
    dist_init.py              # 分布式初始化
    env.py                    # 环境变量/设备检测
    validation.py             # 配置/布局合法性校验
  tests/
    unit/                     # 单元测试
    integration/              # DP/TP/EP 组合测试
    fsdp/                     # FSDP/FSDP2 兼容测试
  examples/
    mvp_moe_train.py           # MVP 训练脚本示例
  docs/
    DESIGN_REQUIREMENTS.md     # 设计需求说明书（可链接）
```

## 3. 关键模块职责

### 3.1 config/
- 负责并行配置建模与合法性校验。
- 输出结构化并行拓扑信息（dp/tp/ep 维度）。

### 3.2 core/
- **mesh.py**：构建 DeviceMesh，并暴露 mesh 维度命名（dp/tp/ep）。
- **layout.py**：封装 DTensor Placement（Shard/Replicate）。
- **dtensor_utils.py**：参数/激活/梯度的 DTensor 转换与检查。

### 3.3 parallel/
- **dp.py**：面向 DP 的训练包装与参数同步接口，内部使用 fsdp_adapter。
- **tp.py**：TP 切分逻辑与并行线性/注意力模块替换。
- **ep.py**：MoE 的路由/All-to-All/专家分配。
- **mixed.py**：DP+TP+EP 组合策略编排与冲突检测。

### 3.4 fsdp/
- **fsdp_adapter.py / fsdp2_adapter.py**：
  - 统一包装接口 `wrap_fsdp(model, config)`；
  - 提供 DTensor 布局与 FSDP/FSDP2 参数分片的一致性映射；
  - 处理 state_dict 兼容。

### 3.5 moe/
- **router.py**：Top-1/Top-2 路由策略；
- **experts.py**：专家容量、负载均衡损失；
- **all_to_all.py**：分布式 All-to-All，复用 DTensor 语义。

### 3.6 modules/
- 提供并行化模块替换，如 `ParallelLinear`、`ParallelAttention`、`MoELayer`。

### 3.7 runtime/
- **engine.py**：Trainer 训练流程的核心；
- **checkpoint.py**：与 FSDP/FSDP2 兼容的保存/恢复；
- **logging.py**：统一日志、吞吐、显存与通信统计。

## 4. 代码执行流程（MVP）
1. **dist_init**：初始化分布式与 DeviceMesh。
2. **ParallelConfig**：解析 dp/tp/ep 配置。
3. **distribute(model, config)**：
   - TP：替换模块为并行版本，生成 DTensor 布局。
   - EP：插入 MoE 层（路由 + All-to-All）。
   - DP：使用 FSDP/FSDP2 包装参数与优化器状态。
4. **Trainer.fit**：执行训练循环与检查点保存。

## 5. 文件命名规范
- 以功能为单位拆分文件，避免过大文件。
- 公开 API 放在 `api/` 与 `__init__.py`。
- Internal helper 放在 `utils/` 与 `core/`。

## 6. 最小可用模块清单（MVP 必须实现）
- `config/parallel_config.py`
- `core/mesh.py`, `core/layout.py`
- `parallel/dp.py`, `parallel/tp.py`, `parallel/ep.py`, `parallel/mixed.py`
- `fsdp/fsdp_adapter.py`, `fsdp/fsdp2_adapter.py`
- `moe/router.py`, `moe/all_to_all.py`
- `api/distribute.py`, `api/trainer.py`
- `runtime/engine.py`, `runtime/checkpoint.py`

## 7. 未来扩展（不在 MVP 范围）
- 流水线并行（PP）
- 上下文并行（CP）
- 自动并行搜索与自动切分

