# 分阶段步骤说明

本文说明 [`EXPERIMENT_PROGRESS.md`](./EXPERIMENT_PROGRESS.md) 中每一项**具体要做什么、产出什么、依赖什么**。  
科学细节与公式以 [`RESEARCH_EXPERIMENT_PLAN.md`](./RESEARCH_EXPERIMENT_PLAN.md) 为准。

---

## 0. 工程预检（E0）

**目的：** 只验证实现接线，不产生科学结论。

**内容：**

1. 单 batch 过拟合到接近 100%（前向、loss、标签一致）
2. residue-pair 划分无交集检查
3. 短程单任务 / joint / sequential 跑通

**入口：** `uv run go4cl smoke`  
**通过标准：** smoke 报告全 pass。

---

## 阶段一：单任务学习、grokking 与算法形成

**总目标：** 先理解模型如何学会结构化模加法，再研究持续学习。  
**明确不做：** A→B 遗忘实验。

### 1A-0 · `phase1 calibrate` — Grokking regime 校准

**何时做：** 扫完全部模数之前（必须先做）。

**内容：**

- 固定**单个操作**、中等模数（默认 \(p=31\)）
- 网格扫描至少包括：
  - 训练余数对比例 `train_frac`（默认扫 `0.4 0.6 0.8`；也可用可选 `n_train_pairs`）
  - `weight_decay`
  - （可选）训练步数 `steps`
- 有关位置：`train_frac`（或 `n_train_pairs`）+ `n_aliases=16`；无关 6 个数字位直接从词表 **0–63** 均匀采样一次（不做 `n_nuisance` 展开）
- 训练：`batch_size=2048`，**有放回采样**固定每步 2048（即使某模数训练集 < 2048）；eval 仍全量无放回
- 曲线打到 W&B

**要回答的问题：** 在什么数据比例与正则下，模型能记忆训练集并（若出现）泛化到 held-out 余数对？

**产出：**

- 锁定一套**全阶段一共用**的训练配置（不得按模数再挑参）
- 本地：`runs/phase1/calibrate/<stamp>/` + summary CSV
- W&B group：`p1_calibrate_…`

**入口：**

```bash
bash scripts/phase1/calibrate.sh --gpus 3,4,5 --workers-per-gpu 3
# 或
uv run go4cl phase1 calibrate --help
```

---

### 1A · `phase1 scan-moduli` — 单操作模数扫描

**依赖：** 1A-0 已锁定配置。

**内容：**

- 对 \(p\in\{19,23,29,31,37,41,43,47\}\) 各训一个**单有效 query** 模型
- 模型、优化器、有关位置暴露量（默认 `train_frac` × `n_aliases`；可选 `n_train_pairs`）全相同
- 数据构造（仅 1A）：只对有关操作数位置的无序余数对严格划分；无关位 ~\(U\{0,\ldots,63\}\)
- 训练：`batch_size=2048` + 有放回采样（跨模数每步算力对齐）；eval 全量无放回
- 记录完整 train/val/test 曲线，不只最终准确率
- 汇总 \(t_{\mathrm{mem}}\)、\(t_{\mathrm{gen}}\)、grokking delay、跃迁锐度等

**要回答的问题：** 难度如何随模数变化？哪些模数 grok、哪些只是平滑泛化？

**产出：** `runs/phase1/scan_moduli/`；跨模数学习曲线与时间表。

**入口：**

```bash
bash scripts/phase1/scan_moduli.sh \
  --train-frac 0.8 --weight-decay 0.3 --steps 100000 \
  --gpus 0,1 --workers-per-gpu 4
# optional fixed-count split:
#   ... --n-train-pairs 150 --weight-decay 0.3 --steps 100000 ...
```

---

### 1A-mech · `phase1 mech-single` — 单模数机理分析

**依赖：** 1A（`scan-moduli`）完成，并有可用 checkpoint（记忆点 / 泛化跃迁 / 最终）。

**何时做：** **紧接 1A 之后、1B 之前。** 先在「只有一个模加法操作」的最简设定下找电路，再进入多操作促进。

**内容（聚焦单操作 / 单模数）：**

- 对代表性模数（建议先 \(p=31\)，再扩到 1A 中学得最好/最差的模数）做：
  - 模 \(p\) Fourier：token embedding、query residual、unembedding
  - 探针：能否解码 \(x_i\bmod p\)、两操作数、部分和、最终结果
  - Attention：query 是否稳定选中正确的两个输入位置
  - 因果：Fourier 子空间消融、head/MLP ablation、正确/错误操作数位置 patching
- **不做**跨操作电路移植（那属于 1C）

**要回答的问题：** 单模数下模型是否形成可干预的「模加法算法」？不同模数是否共用同类机制？

**产出：** `runs/phase1/mech_single/`；单模数机理报告（与 1A 行为曲线对照）。  
**状态：** CLI 占位，实现待补。

**入口：**

```bash
uv run go4cl phase1 mech-single \
  --ckpt-root runs/phase1/scan_moduli/<stamp> \
  --moduli 31
```

---

### 1B · `phase1 multi-op` — 多操作单任务与同模数促进

**依赖：** 1A 配置可用；**建议 1A-mech 已给出单操作机理基线**后再做，便于对比「促进」是否来自共享计算电路。

**内容（对照设计）：**

1. 一个操作（与 1A 衔接）
2. 四个操作、四个不同模数
3. 四个操作中有一对同模数（位置与槽不同）
4. （可选）四个操作全同模数

约束：每操作 query 概率与样本暴露匹配；同模数操作共用 residue-pair split。

**要回答的问题：** 第二个同模数操作是否缩短 \(t_{\mathrm{gen}}\)？计算是否共享、路由是否分离？

**产出：** `runs/phase1/multi_op/`；促进效应的行为证据。  
**状态：** CLI 占位，实现待补。

---

### 1C · `phase1 mechanisms` — 多操作对照机理

**依赖：** 1B checkpoint；并最好已完成 1A-mech（单操作基线）。

**内容：**

- 同模数两操作之间：计算子电路是否共享、attention 路由是否分离
- 跨操作 circuit transplant / faithfulness
- 与 1A-mech 结论对照：促进是否来自复用同一套 Fourier/计算特征

**要回答的问题：** 多操作促进在机制上是「复制电路」还是「共享计算 + 分路由」？

**产出：** `runs/phase1/mechanisms/`。  
**状态：** CLI 占位，实现待补。

---

## 阶段二：双任务行为动力学

**总目标：** 系统比较联合学习与顺序学习，把任务结构关系映射到迁移/遗忘等行为。

### 2A · `phase2 protocols` — 训练协议

**内容：** 同一固定 A/B 数据集上比较：

| 协议 | 作用 |
|------|------|
| A-only / B-only | 单任务基线难度 |
| joint A+B（50/50） | 兼容解是否存在 |
| interleaved | 防遗忘行为上界之一 |
| sequential A→B / B→A | 核心持续学习条件（无 replay） |
| A-only continued | 控制自然参数漂移 |

样本暴露在协议间匹配；主顺序实验在 A 已泛化且阶段一算法指标稳定后切 B；另做切换时机消融。

**产出：** `runs/phase2/protocols/`。  
**状态：** CLI 占位。

---

### 2B · `phase2 relation-matrix` — 任务关系矩阵

**内容：**

1. 先跑 \(\rho_{\mathrm{slot}},\rho_{\mathrm{operand}},\rho_{\mathrm{mod}}\in\{0,1\}^3\) 共 8 个极端条件
2. 再跑 \(\{0,0.5,1\}^3\) 共 27 条件
3. 每条件：多 task pair、多 model seed、A→B 与 B→A；同一 manifest 上比 joint vs sequential

**要回答的问题：** 三类重叠如何分别影响遗忘与迁移？部分重叠是否比全同/全异更糟？

**产出：** `runs/phase2/relation_matrix/`。  
**状态：** CLI 占位。

---

### 2C · 全程行为指标（无独立 CLI）

**不是单独命令**，而是 2A/2B 运行时必须记录的规范，包括但不限于：

- 各任务 train/test loss、acc、margin；分槽位/模数
- B 开始后 A 的瞬时 jump、遗忘速率、长期保留
- 相对 B-from-scratch 的前向迁移
- joint 下同时/先后 grok 或互相阻碍
- 顺序训练中表征指标是否先于行为下降

进度表中单独一行，便于勾选「指标管线是否齐」。

---

### 2D · `phase2 capacity` — 容量消融

**内容：** \(d_{\mathrm{model}}\in\{32,64,128\}\)、\(L\in\{2,3,4\}\)。  
先在代表性关系条件上跑，再决定是否扩到全 27 格。

**要回答的问题：** 失败是「无兼容解」还是「容量不够」？

**产出：** `runs/phase2/capacity/`。  
**状态：** CLI 占位。

---

## 阶段三：持续学习机理分析

**总目标：** 解释阶段二的**代表性**现象，而非对所有 run 做全量电路分析。  
预先圈定例如：明显正迁移、明显遗忘、非单调部分重叠、joint 成但 sequential 败等。

### 3A · `phase3 optimize` — 优化层机制

**内容：** 整体/逐层/逐组件梯度余弦与 \(g_A^\top g_B\)；一阶预测 vs 真实 \(\Delta L_A\)；定位负干扰首发模块；joint 与 sequential 是否走到不同解。

**产出：** `runs/phase3/optimize/`。  
**状态：** CLI 占位。

---

### 3B · `phase3 route-compute-readout` — 路由–计算–读出

**内容（与重叠因素对齐）：**

- 操作数重叠 → attention 路由保留/重映射
- 模数重叠 → Fourier/MLP 计算特征复用或覆盖
- 槽位重叠 → query-conditioned residual 与读出

用组件级与路径级 patching 验证，不只看 attention map。

**产出：** `runs/phase3/route_compute_readout/`。  
**状态：** CLI 占位。

---

### 3C · `phase3 forget-types` — 三类遗忘机制

**内容：** 用干预区分：

1. **表征删除：** A 特征消失，探针与因果都失败  
2. **读出失配：** 特征可探，恢复旧 readout/下游可恢复  
3. **路由失配：** 计算仍在，query 读错操作数或子电路  

手段包括：从 \(\theta_A\) patch 回 B 后模型、activation 替换、分别恢复 routing / computation / readout、跨任务电路移植。

**产出：** `runs/phase3/forget_types/`。  
**状态：** CLI 占位。

---

## 文档与代码对应关系

| 进度表 ID | CLI | 代码位置 |
|-----------|-----|----------|
| E0 | `go4cl smoke` | `src/go4cl/scripts/smoke.py` |
| 1A-0 | `go4cl phase1 calibrate` | `src/go4cl/phases/phase1/calibrate.py` |
| 1A | `go4cl phase1 scan-moduli` | `src/go4cl/phases/phase1/modulus_scan.py` |
| 1A-mech | `go4cl phase1 mech-single` | `src/go4cl/phases/phase1/mech_single.py`（stub） |
| 1B | `go4cl phase1 multi-op` | `src/go4cl/phases/phase1/multi_op.py`（stub） |
| 1C | `go4cl phase1 mechanisms` | `src/go4cl/phases/phase1/mechanisms.py`（stub） |
| 2A | `go4cl phase2 protocols` | `src/go4cl/phases/phase2/`（stub） |
| 2B | `go4cl phase2 relation-matrix` | 同上 |
| 2D | `go4cl phase2 capacity` | 同上 |
| 3A–3C | `go4cl phase3 …` | `src/go4cl/phases/phase3/`（stub） |

更新进度时只改 [`EXPERIMENT_PROGRESS.md`](./EXPERIMENT_PROGRESS.md)；本说明仅在步骤定义或 CLI 变更时同步修改。
