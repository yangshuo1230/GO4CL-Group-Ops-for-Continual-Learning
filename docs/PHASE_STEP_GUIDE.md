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
- 按 **val 准确率**保存 best checkpoint（`*_best.pt` / `best.pt`）；汇总同时写 final 与 best 指标
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

- 对 \(p\in\{23,29,31,37,41,43,47,53\}\) 各训一个**单有效 query** 模型（暂：19→53）
- 模型、优化器、有关位置暴露量（默认 `train_frac` × `n_aliases`；可选 `n_train_pairs`）全相同
- 数据构造（仅 1A）：只对有关操作数位置的无序余数对严格划分；无关位 ~\(U\{0,\ldots,63\}\)
- 训练：`batch_size=2048` + 有放回采样（跨模数每步算力对齐）；eval 全量无放回
- 按 val 选 best checkpoint；报告/CSV 同时含 final 与 `best_A_val_acc` / `best_A_test_acc` / `best_step`
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

**依赖：** 1A（`scan-moduli`）完成，并有可用 checkpoint（best-by-val / 最终；可选 step ckpt）。

**何时做：** 可与 1B 并行；当前已实现 MVP，建议先跑 \(p=31\)。

**内容（MVP，聚焦单操作 / 单模数）：**

- 默认输入：`runs/phase1/scan_moduli/20260930_223737`，模数 \(p=31\)，`final` + `best`
- 模 \(p\) Fourier：digit token embedding、query residual（按 \((x_i+x_j)\bmod p\) 平均）
- 线性探针 / 转向：默认在 **layer 0、1** 的 `resid_post` query 位（不是最终 pre-head）；含随机初始化探针对照，以及中间层转向后继续前向
- Attention：query 位对各层的质量是否落在正确操作数位置 \(\{i,j\}\)
- 因果：digit embedding 上 Fourier 频率消融（默认扫掉 \(k=1..n\) 个能量最高 / 最低的共轭频率对，对比 acc 曲线；`--ablation-ks` 可指定子集）；中间层类均值转向 \(h+\alpha(\mu_{s+\delta}-\mu_s)\) 后跑完剩余层（含 shuffled-mean 对照）
- **组合位点（composition locus）**：全层 `resid_mid` / `resid_post` 信息阶梯探针（\(x_i,x_j,\mathrm{sum}\)）；逐层 zero-attn / zero-MLP 写回消融；**逐 head 消融**；top Fourier 频率上 operand vs sum 谐波 \(R^2\)。默认开启；`--skip-composition` 可关
- **不做**跨操作电路移植 / 完整 head·MLP 扫描（属 1C 或后续）

**要回答的问题：** 单模数下模型是否形成可干预的「模加法算法」？组合发生在哪一层 / 哪个组件？不同模数是否共用同类机制？

**产出：** `runs/phase1/mech_single/<stamp>/`

- `p{m}_{final|best}_report.json`
- `phase1_mech-single_report.json` / `phase1_mech-single_summary.csv`  
  （列含：`probe_sum_acc` / `probe_sum_acc_random` / `steered_acc_target` / `shuffled_steered_acc_target` / `ablation_delta_acc` / `compose_layer_guess` 等）
- `phase1_mech-single_ablation_curve.csv` — 重要 vs 不重要频率对数量 \(k\) 的 test acc / \(\Delta\)acc
- `phase1_mech-single_composition.csv` — ladder / knockout / **head_knockout** / harmonic 长表
- 规范汇总（结论+图）：`runs/phase1/mech_single/p31_summary/`；stamp 索引见同目录上级 `README.md`

**状态：** p=31 MVP 已跑通并整理；多模数 / 轨迹 progress 待补。

**入口：**

```bash
bash scripts/phase1/mech_single.sh \
  --ckpt-root runs/phase1/scan_moduli/20260930_223737 \
  --moduli 31 \
  --ckpt-kinds final best
# 或指定消融扫的 k（默认全扫）
uv run go4cl phase1 mech-single --moduli 31 --ablation-ks 1 2 4 8
# 跳过组合位点
uv run go4cl phase1 mech-single --moduli 31 --skip-composition
# 重画汇总图
bash scripts/phase1/plot_mech_figures.sh
```

---

### 1B · `phase1 multi-op` — 多操作单任务与同模数促进

**依赖：** 1A 配置可用。1B 锁定训练：`train_frac=0.8`、`wd=0.3`、`steps=100k`、`aliases=16`（仅 val/test 展开）、`bs=8192` query 样本/步。**训练为 packed online**：每步从各 op 的 train residue 池有放回各抽一对，拼成一条 8 位上下文，再对 4 个 query 各出一条（等权暴露）。旧 concat-train stamp 作废，数据 tag 含 `_pack1`。1A-mech 可后补。

**内容（对照设计，同一 `task_seed` 共享 operand matching + slots）：**

| 变体 | CLI 名 | 模数分配 |
|------|--------|----------|
| 全同模数（4 query） | `all_same` | \((p,p,p,p)\) |
| 四操作异模数 | `four_diff` | \((p,q,p_2,p_3)\)，其中 \((p,q)\) 为近邻模数对 |
| 一对同模数 | `pair_same` | \((p,p,p_2,p_3)\) |
| （遗留）真单操作 | `one` | \((p)\) — **默认不用**；暴露量与四操作不对齐 |

同模数操作共用同一套 residue-pair train/val/test split。默认跑 `all_same four_diff pair_same` × 多个 `task_seed`（三者均为 4 query，每步 context 数相同）。

**要回答的问题：** 第二个同模数操作是否缩短 \(t_{\mathrm{gen}}\) / 提高 held-out？相对近邻异模数对照，促进是否存在？

**产出：** `runs/phase1/multi_op/<stamp>/`；W&B 除总体 `A_val_acc` 外，还记录各模数 `A_val_acc/p{m}` 曲线。nuisance、margin、NCE 和按模数合并表只留在本地 metrics，不上传 W&B。  
**状态：** 已实现训练入口。

**入口：**

```bash
bash scripts/phase1/multi_op.sh --gpus 0,1,2,3,4,5 --workers-per-gpu 1
# 或
uv run go4cl phase1 multi-op \
  --variants all_same four_diff pair_same \
  --task-seeds 0 1 \
  --train-frac 0.8 --weight-decay 0.3 --batch-size 8192 --steps 100000 \
  --gpus 0,1,2,3,4,5
```

---

### 1C · `phase1 mechanisms` — 多操作对照机理

**依赖：** 1B checkpoint；并最好已完成 1A-mech（单操作基线）。

**内容：**

- 按 op（模数）过滤 val/test，复用 1A-mech：composition / attention / Fourier 消融
- L0 路由分离：对本 op 操作数质量 vs 其他 op 操作数位置
- **跨模 Fourier 选择性**：消融某一 \(p\) 的 top-1 频率对，测对所有 op 的 \(\Delta\)acc
- **Unembedding 模 \(p\) Fourier**（相对 digit-emb / query residual）：`scripts/phase1/unembed_fourier.py`

**要回答的问题：** 多操作促进在机制上是「复制电路」还是「共享计算 + 分路由」？

**默认目标（MVP）：** concat-1B `four_diff`[47,43,37,23] best  
`runs/phase1/multi_op/20260930_235556/runs/multi_four_diff_m47-43-37-23_…`

```bash
bash scripts/phase1/mechanisms.sh
# 或
uv run go4cl phase1 mechanisms --ckpt-kind best
```

**产出：** `runs/phase1/mechanisms/<stamp>/`（`CONCLUSIONS.md` + CSV/JSON）。  
**状态：** packed 1B 对照已跑（含 `pair_same`/`all_same`）。电路移植 / operand patching：`scripts/phase1/circuit_transplant.py`，产物 `mechanisms/20261004_1b_132221/*/transplant/`。

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

**产出：** `runs/phase2/protocols/<stamp>/`（`jobs.json`、summary CSV、`phase2_protocols_transfer.csv`、各 run 的 `eval_history.jsonl` / `metrics.json`）。  
**状态：** 已实现。默认超参：`train_frac=0.8`、`wd=0.3`、`bs=8192`、`steps=100000`、packed online，主评 `packed_id`。默认任务关系为全重叠 `(1,1,1)`。`--switch-on t_mem|t_gen` 在第一阶段事件触发后提前切换，第二阶段仍跑满 `--steps`。

**发现（全重叠 seed 0）：** 四模数 `{41,23,37,53}` 训练分布无误。`wd=0.5/0.8` 时 p=23 可长期停在 \(\approx 1/23\)（train/iid 同样低）：模型用 query token 做 23 类均匀猜测，L0 不看操作数，对该 op 的 trunk 梯度塌掉。`wd=0.3` 约 20k 步四 op 均可 grok。详见 `docs/IMPLEMENTATION_NOTES.md`。

```bash
bash scripts/phase2/protocols.sh --gpus 0,1,2,3 --workers-per-gpu 1
# 只列作业、不训练
bash scripts/phase2/protocols.sh --dry-run
```

---

### 2B · `phase2 relation-matrix` — 任务关系矩阵

**内容：**

1. 先跑 \(\rho_{\mathrm{slot}},\rho_{\mathrm{operand}},\rho_{\mathrm{mod}}\in\{0,1\}^3\) 共 8 个极端条件
2. 再跑 \(\{0,0.5,1\}^3\) 共 27 条件
3. 每条件：多 task pair、多 model seed、A→B 与 B→A；同一 manifest 上比 joint vs sequential

**要回答的问题：** 三类重叠如何分别影响遗忘与迁移？部分重叠是否比全同/全异更糟？

**产出：** `runs/phase2/relation_matrix/<stamp>/`，以及按 \(\rho\) 平均的 `phase2_relation-matrix_by_rho.csv`。  
**状态：** 已实现。默认 `--grid extreme`（8 格）× `a_only b_only joint sequential_ab sequential_ba`。同一 manifest 上比较 joint 与两种顺序。`--directions swap` 交换任务 token。`--grid full` 为 27 格。

```bash
bash scripts/phase2/relation_matrix.sh --gpus 0,1,2,3,4,5
bash scripts/phase2/relation_matrix.sh --grid full --dry-run
```

---

### 2C · 全程行为指标（无独立 CLI）

**不是单独命令**，而是 2A/2B 运行时必须记录的规范，包括但不限于：

- 各任务 train/test loss、acc、margin；分槽位/模数
- B 开始后 A 的瞬时 jump、遗忘速率、长期保留
- 相对 B-from-scratch 的前向迁移
- joint 下同时/先后 grok 或互相阻碍
- 顺序训练中表征指标是否先于行为下降

进度表中单独一行，便于勾选「指标管线是否齐」。

**已接入 2A/2B/2D 的量：** 各任务 val/test 的 loss、acc；按槽位与模数的 acc。margin、NCE、nuisance 对照写在本地 `metrics.json` / `eval_history.jsonl`，不上传 W&B。B 开始后 A 的 jump、最陡遗忘速率、长期保留（`forgetting_A = max Acc_A - final`）；相对 B-only 的暴露对齐 AUC 与到达 0.9 的步数差（`phase2_*_transfer.csv`）；joint/顺序下谁先跨过 0.9（`grok_order`）。顺序训练中 Fourier/探针是否先于行为下降留在阶段三，不在每个 phase-2 step 里重跑电路分析。

---

### 2D · `phase2 capacity` — 容量消融

**内容：** \(d_{\mathrm{model}}\in\{32,64,128\}\)、\(L\in\{2,3,4\}\)。  
先在代表性关系条件上跑，再决定是否扩到全 27 格。

**要回答的问题：** 失败是「无兼容解」还是「容量不够」？

**产出：** `runs/phase2/capacity/<stamp>/`。  
**状态：** 已实现。默认六种关系（全同、仅槽不同、仅操作数不同、仅模数不同、全不同、部分重叠 0.5³）× \(d\in\{32,64,128\}\) × \(L\in\{2,3,4\}\) × `a_only b_only joint sequential_ab`。

```bash
bash scripts/phase2/capacity.sh --dry-run
bash scripts/phase2/capacity.sh --d-models 64 --n-layers-list 3 --gpus 0,1
```

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
| 1A-mech | `go4cl phase1 mech-single` | `src/go4cl/phases/phase1/mech_single.py` |
| 1B | `go4cl phase1 multi-op` | `src/go4cl/phases/phase1/multi_op.py` |
| 1C | `go4cl phase1 mechanisms` | `src/go4cl/phases/phase1/mechanisms.py`（stub） |
| 2A | `go4cl phase2 protocols` | `src/go4cl/phases/phase2/protocols.py` |
| 2B | `go4cl phase2 relation-matrix` | `src/go4cl/phases/phase2/relation.py` |
| 2D | `go4cl phase2 capacity` | `src/go4cl/phases/phase2/capacity.py` |
| 3A–3C | `go4cl phase3 …` | `src/go4cl/phases/phase3/`（stub） |

更新进度时只改 [`EXPERIMENT_PROGRESS.md`](./EXPERIMENT_PROGRESS.md)；本说明仅在步骤定义或 CLI 变更时同步修改。
