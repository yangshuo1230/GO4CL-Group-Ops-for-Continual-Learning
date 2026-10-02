# 实验进度表

对照总计划 [`RESEARCH_EXPERIMENT_PLAN.md`](./RESEARCH_EXPERIMENT_PLAN.md)。  
各步骤含义见 [`PHASE_STEP_GUIDE.md`](./PHASE_STEP_GUIDE.md)。

**状态约定：** `未开始` · `进行中` · `已完成` · `阻塞` · `跳过`

最后更新：2026-10-02

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
| 1A-0 | Grokking regime 校准（中等模数） | `go4cl phase1 calibrate` | `runs/phase1/calibrate/` | 已完成 | 锁定试跑：`train_frac=0.8` / `wd=0.3` / `steps` 见 1A；`bs=2048` 有放回 | 2026-09-30 |
| 1A | 单操作 × 八模数扫描 | `go4cl phase1 scan-moduli` | `runs/phase1/scan_moduli/` | 已完成 | 比例版试跑完成；大模数可学到高 held-out；**暂不挡 1B** | 2026-09-30 |
| 1A-mech | **单模数机理分析**（Fourier / 探针 / attention / 消融 / 组合位点） | `go4cl phase1 mech-single` | `runs/phase1/mech_single/` | 进行中 | **p=31 MVP 结论已齐**（组合在 L0；频率必要；head 因果不均）。汇总：[`runs/phase1/mech_single/p31_summary/`](../runs/phase1/mech_single/p31_summary/)；索引 [`README`](../runs/phase1/mech_single/README.md)。待补：多模数复现、训练轨迹 | 2026-10-02 |
| 1B | 多操作单任务 / 同模数促进 | `go4cl phase1 multi-op` | `runs/phase1/multi_op/` | 进行中 | 锁定：`train_frac=0.8` / `wd=0.5` / `bs=8192` query/步 / `steps=100k`；**packed online 等权暴露**（`_pack1`）；旧 concat train 作废 | 2026-10-02 |
| 1C | 多操作对照机理（共享计算 / 移植） | `go4cl phase1 mechanisms` | `runs/phase1/mechanisms/` | 未开始 | CLI 占位；依赖 1B | |


**阶段一出门条件（进入阶段二前须满足）：**

- [ ] 单任务在多数 seeds 上 held-out 准确率可靠
- [ ] 统一训练配置已锁定（不按模数挑参）
- [x] （推荐）1A-mech 至少在一个代表性模数上找到与行为一致的机理证据（p=31；见 `mech_single/p31_summary/`）
- [ ] 1B：同模数促进 vs 近邻异模数对照有可复现行为结论

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
| 2026-09-30 | **1A 数据/训练协议修订（仅阶段一单操作）**：只对有关操作数位置按 `train_frac` 严格划分 train/val/test，并用 `n_aliases` 展开；无关 6 位从 0–63 均匀采样（取消 `n_nuisance`）；训练固定 `batch_size=2048` **有放回采样**；eval 仍全量无放回。试跑锁定 `train_frac=0.8` / `wd=0.3` / `steps=100k`，启动 `scan-moduli` |
| 2026-09-30 | **模数集合更新**：\(\mathcal P=\{19,23,29,31,37,41,43,47\}\)；输出头改为 47 类；校准默认模数改为 \(p=31\) |
| 2026-09-30 | **1A 划分双路径**：默认比例 `--train-frac` / `--train-fracs`；可选固定对数 `--n-train-pairs` |
| 2026-09-30 | **比例版 scan-moduli 试跑** `20260930_213357`：`train_frac=0.8` / `wd=0.3` / `steps=20k` / `n_aliases=16` / `bs=2048`。\(p\ge 29\) final A_test=1.0；\(p=23\)≈0.96；\(p=19\)≈0.63。当时评估仅为 **final** 权重 |
| 2026-09-30 | **Best-by-val checkpoint**：训练中按 val 准确率（多 val 取均值）存 `*_best.pt` + 别名 `best.pt`；协议结束后同时报告 final（`A_*_acc`）与 best（`best_A_*_acc` / `best_step`）；calibrate / scan-moduli CSV 增加 best 列。旧 run 无 best，需重训 |
| 2026-09-30 | **跳过 1A-mech，进入 1B**：实现 `phase1 multi-op`（`one` / `four_diff` / `pair_same` / 可选 `all_same`）；同 seed 共享 matching+slots，只改模数分配；训练超参对齐 1A |
| 2026-10-02 | **W&B 按模数准确率**：训练 eval 记录 `A_val_acc/p{m}` 等标量，并上传合并曲线 `charts/val_acc_by_modulus`；final/best 同样写入各模数 test/val 准确率（重跑 1B 后可见） |
| 2026-10-02 | **1B 锁定训练超参**：`train_frac=0.8`，`weight_decay=0.5`，`batch_size=8192`（有放回），`steps=100000`，`n_aliases=16`；已写入 CLI 默认值与脚本注释 |
| 2026-10-02 | **补做 1A-mech MVP**：`phase1 mech-single` — Fourier / 线性探针 / attention 路由 / digit-embedding Fourier 消融；默认分析 `scan_moduli/20260930_223737` 的 \(p=31\)（final+best） |
| 2026-10-02 | **1A-mech 探针对照 + residual 转向**：随机初始化模型同协议探针；query residual 类均值转向 \((s+\delta)\bmod p\)（含 shuffled-mean 对照） |
| 2026-10-02 | **1A-mech 改为前两层**：探针/转向默认 `resid_post` @ layer 0/1（继续后层+head）；避免最终 pre-head 同义反复 |
| 2026-10-02 | **1A-mech Fourier 消融扫 k**：digit emb 上按能量排序的共轭频率对，分别消融 top-\(k\)（重要）与 bottom-\(k\)（不重要），输出 `phase1_mech-single_ablation_curve.csv`；CLI `--ablation-ks` |
| 2026-10-02 | **1A-mech 组合位点**：全层 mid/post 信息阶梯、逐层 attn/MLP knockout、top-freq 谐波 \(R^2\)；产出 `phase1_mech-single_composition.csv`；`--skip-composition` |
| 2026-10-02 | **1A-mech 逐 head 消融**：每层每个 attention head 置零后测 \(\Delta\)acc；写入 composition CSV（`kind=head_knockout`） |
| 2026-10-02 | **1A-mech 结果整理**：规范目录 `runs/phase1/mech_single/p31_summary/`（结论+图）；出图脚本 `scripts/phase1/plot_mech_figures.sh`；stamp 索引见 `mech_single/README.md` |
| 2026-10-02 | **1B packed 多 query 训练**：不再预生成/拼接各模数 train 集；每步各 op 有放回抽一对拼成一条上下文，对四个 query 各训一次。val/test 仍落盘。数据 tag `_pack1` |
