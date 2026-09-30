# 实验进度表

对照总计划 [`RESEARCH_EXPERIMENT_PLAN.md`](./RESEARCH_EXPERIMENT_PLAN.md)。  
各步骤含义见 [`PHASE_STEP_GUIDE.md`](./PHASE_STEP_GUIDE.md)。

**状态约定：** `未开始` · `进行中` · `已完成` · `阻塞` · `跳过`

最后更新：2026-09-30

---

## 总览

| 阶段 | CLI | 目标一句话 | 状态 |
|------|-----|------------|------|
| 工程预检 | `go4cl smoke` | 实现可跑通，不计科学结论 | 已完成 |
| 阶段一 | `go4cl phase1 …` | 单任务学会什么、何时 grok | 进行中 |
| 阶段二 | `go4cl phase2 …` | 联合 vs 顺序、任务关系→行为 | 未开始 |
| 阶段三 | `go4cl phase3 …` | 迁移/遗忘的因果机理 | 未开始 |

---

## 工程预检

| 步骤 | 内容 | CLI / 产物 | 状态 | 备注 |
|------|------|------------|------|------|
| E0 | 单 batch 过拟合、划分无交、短训、joint/sequential 接线 | `uv run go4cl smoke` | 已完成 | 含 GPU smoke |

---

## 阶段一：单任务学习 / grokking / 算法形成

| ID | 步骤 | CLI | 产物目录 | 状态 | 锁定配置 / 结论摘要 | 日期 |
|----|------|-----|----------|------|---------------------|------|
| 1A-0 | Grokking regime 校准（中等模数） | `go4cl phase1 calibrate` | `runs/phase1/calibrate/` | 进行中 | 输入已改为 0–63；需用新词表重跑后锁定 `train_frac` / `wd` / `steps` | 2026-09-30 |
| 1A | 单操作 × 八模数扫描 | `go4cl phase1 scan-moduli` | `runs/phase1/scan_moduli/` | 未开始 | 依赖 1A-0 锁定配置 | |
| 1A-mech | **单模数机理分析**（Fourier / 探针 / patching） | `go4cl phase1 mech-single` | `runs/phase1/mech_single/` | 未开始 | 依赖 1A checkpoint；先确认单操作电路 | |
| 1B | 多操作单任务 / 同模数促进 | `go4cl phase1 multi-op` | `runs/phase1/multi_op/` | 未开始 | CLI 占位；建议在 1A-mech 之后 | |
| 1C | 多操作对照机理（共享计算 / 移植） | `go4cl phase1 mechanisms` | `runs/phase1/mechanisms/` | 未开始 | CLI 占位；依赖 1B（+ 1A-mech 基线） | |

**阶段一出门条件（进入阶段二前须满足）：**

- [ ] 单任务在多数 seeds 上 held-out 准确率可靠
- [ ] 统一训练配置已锁定（不按模数挑参）
- [ ] （推荐）1A-mech 至少在一个代表性模数上找到与行为一致的机理证据

---

## 阶段二：双任务行为动力学

| ID | 步骤 | CLI | 产物目录 | 状态 | 锁定配置 / 结论摘要 | 日期 |
|----|------|-----|----------|------|---------------------|------|
| 2A | 训练协议对照 | `go4cl phase2 protocols` | `runs/phase2/protocols/` | 未开始 | CLI 占位 | |
| 2B | 任务关系矩阵（8 → 27） | `go4cl phase2 relation-matrix` | `runs/phase2/relation_matrix/` | 未开始 | CLI 占位 | |
| 2C | 全程行为指标（非独立入口） | 随 2A/2B 记录 | 同上 | 未开始 | 见说明文档；无单独 CLI | |
| 2D | 容量消融 | `go4cl phase2 capacity` | `runs/phase2/capacity/` | 未开始 | CLI 占位 | |

**阶段二出门条件（进入阶段三前须满足）：**

- [ ] Joint 能同时解决 A、B（兼容解存在）
- [ ] Sequential 中 B 能学会；A 是否遗忘作为结果记录
- [ ] 已选出阶段三要解释的代表性条件

---

## 阶段三：持续学习机理

| ID | 步骤 | CLI | 产物目录 | 状态 | 锁定配置 / 结论摘要 | 日期 |
|----|------|-----|----------|------|---------------------|------|
| 3A | 优化层（梯度冲突等） | `go4cl phase3 optimize` | `runs/phase3/optimize/` | 未开始 | CLI 占位 | |
| 3B | 路由–计算–读出分解 | `go4cl phase3 route-compute-readout` | `runs/phase3/route_compute_readout/` | 未开始 | CLI 占位 | |
| 3C | 遗忘类型因果区分 | `go4cl phase3 forget-types` | `runs/phase3/forget_types/` | 未开始 | CLI 占位 | |

**阶段三成功标准（摘要）：**

- [ ] 能预测阶段二哪些条件迁移/遗忘
- [ ] 因果干预可恢复或复现行为
- [ ] 跨 seed / task pair 可复现
- [ ] 区分参数变化、表征变化与功能变化

---

## 变更日志

| 日期 | 变更 |
|------|------|
| 2026-09-30 | 建立本表；工程预检完成；1A-0 进行中；输入词表改为 0–63 |
| 2026-09-30 | 1A 后增加 **1A-mech**（`mech-single`）；原 1C 改为多操作对照机理 |
