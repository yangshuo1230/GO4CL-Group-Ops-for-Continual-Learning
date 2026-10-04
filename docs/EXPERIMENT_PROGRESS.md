# 实验进度表

对照总计划 [`RESEARCH_EXPERIMENT_PLAN.md`](./RESEARCH_EXPERIMENT_PLAN.md)。  
各步骤含义见 [`PHASE_STEP_GUIDE.md`](./PHASE_STEP_GUIDE.md)。

**状态约定：** `未开始` · `进行中` · `已完成` · `阻塞` · `跳过`

最后更新：2026-10-04

---

## 总览

| 阶段 | CLI | 目标一句话 | 状态 |
|------|-----|------------|------|
| 工程预检 | `go4cl smoke` | 实现可跑通，不计科学结论 | 已完成 |
| 阶段一 | `go4cl phase1 …` | 单任务学会什么、何时 grok | 已完成 |
| 阶段二 | `go4cl phase2 …` | 联合 vs 顺序、任务关系→行为 | 进行中 |
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
| 1A-mech | **单模数机理分析**（Fourier / 探针 / attention / 消融 / 组合位点） | `go4cl phase1 mech-single` | `runs/phase1/mech_single/` | 已完成 | **八模数 final 复现 p=31**：组合在 L0；频率必要；L0 head 因果不均。跨模：[`20261002_cross/`](../runs/phase1/mech_single/20261002_cross/)；p31 细图：[`p31_summary/`](../runs/phase1/mech_single/p31_summary/)。可选：训练轨迹 | 2026-10-02 |
| 1B | 多操作单任务 / 同模数促进 | `go4cl phase1 multi-op` | `runs/phase1/multi_op/` | 已完成 | 锁定：`train_frac=0.8` / `wd=0.3` / `bs=8192` / `steps=100k` packed。多种子 [`20261004_170554/`](../runs/phase1/multi_op/20261004_170554/) 3×3×3：**pair_same** best test 全 ≥0.997（\(t_{\mathrm{gen}}\) 中位 11k）优于 **four_diff**（5/9 ≥0.99，中位 19k）。**暂通过**；four_diff ts0/ms0 小模未齐（p23 final≈0.55）。旧网格 `wd=0.5` 仅归档 | 2026-10-04 |
| 1C | 多操作对照机理（共享计算 / 移植） | `go4cl phase1 mechanisms` + `circuit_transplant.py` | `runs/phase1/mechanisms/` | 已完成 | packed 1B ×6：Fourier 同模 `non_selective`、异模 `inconclusive`。深化：**L0 操作数 residual 因果必要**；L0 attn-write 不可单独移植；L2 MLP write 可被共享头读出。综述 [`TRANSPLANT_COMPARISON.md`](../runs/phase1/mechanisms/20261004_1b_132221/TRANSPLANT_COMPARISON.md) | 2026-10-04 |


**阶段一出门条件（进入阶段二前须满足）：**

- [x] 单任务在多数 seeds 上 held-out 准确率可靠（旧 𝒫 1A 已验证；新 𝒫/53 以 1B 网格为准，旧 47 类/含 19 仅归档）
- [x] 训练配置已锁定（1A 与 1B **两套**：1A `wd=0.3/bs=2048`；1B `wd=0.3/bs=8192/steps=100k/packed`）
- [x] （推荐）1A-mech 至少在一个代表性模数上找到与行为一致的机理证据（p=31；见 `mech_single/p31_summary/`）；**八模数 final 已复现**（`mech_single/20261002_cross/`，旧 𝒫）
- [x] 1B：同模数促进 vs 近邻异模数对照 — **暂通过** [`20261004_170554/`](../runs/phase1/multi_op/20261004_170554/)（task×model=3×3）。`pair_same` 9/9 best≥0.997；`four_diff` 更慢且 1/9 小模失败（不挡阶段二）
- [x] 1C：在新 1B packed_id ckpt 上重做并补 `pair_same`/`all_same`（[`20261004_1b_132221/`](../runs/phase1/mechanisms/20261004_1b_132221/)）；旧 concat MVP 仅归档；patching/移植见 `TRANSPLANT_COMPARISON.md`；unembed Fourier 见 `UNEMBED_FOURIER.md`

---

## 阶段二：双任务行为动力学

| ID | 步骤 | CLI | 产物目录 | 状态 | 锁定配置 / 结论摘要 | 日期 |
|----|------|-----|----------|------|---------------------|------|
| 2A | 训练协议对照 | `go4cl phase2 protocols` | `runs/phase2/protocols/` | 进行中 | 默认 `wd=0.3/bs=8192/steps=100k` packed，全重叠。p=23 在 `wd≥0.5` 易 query-only 盆地，见 IMPLEMENTATION_NOTES | 2026-10-04 |
| 2B | 任务关系矩阵（8 → 27） | `go4cl phase2 relation-matrix` | `runs/phase2/relation_matrix/` | 进行中 | 默认 extreme 8 格；`--grid full` 为 27。实验未跑 | 2026-10-04 |
| 2C | 全程行为指标（非独立入口） | 随 2A/2B 记录 | 同上 | 进行中 | eval history → forgetting / jump / exposure AUC / transfer CSV。Fourier 先于行为下降留阶段三 | 2026-10-04 |
| 2D | 容量消融 | `go4cl phase2 capacity` | `runs/phase2/capacity/` | 进行中 | 六种代表关系 × \{32,64,128\} × \{2,3,4\}。实验未跑 | 2026-10-04 |

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
| 2026-10-02 | **1A-mech 跨模数复现**：对 \(\mathcal P\) 八模数跑完整 mech（final+best，同 `223737` ckpt）；p=31 五条结论在 final 上 8/8（top-1 频率崩塌 7/8，p=37 仍大跌至 0.59）。见 `mech_single/20261002_cross/` |
| 2026-10-02 | **1C MVP**：`phase1 mechanisms` 对 multi-op 按 op 做 composition/attention/Fourier，并测跨模频率消融选择性。默认分析 concat-1B `four_diff`[47,43,37,23] best；结论：非独立 L0 复制电路，频率特征共享/串扰。`runs/phase1/mechanisms/20261002_four_diff_47433723/` |
| 2026-10-02 | **暂换模数**：\(\mathcal P\) 中 \(19\to 53\)；近邻对改为 \((23,29),(31,37),(41,43),(47,53)\)；输出头 **53 类**。原因：packed 1B 中 four_diff 仍难训好 p=19 |
| 2026-10-02 | **R1 协议修复**（数值以当前 53 类为准）：OperationKey / packed_id+nuisance eval / macro checkpoint / analysis dataset / steering 分 split / sampler_seed；见 `docs/REFACTOR_R1_SPEC.md`、`docs/IMPLEMENTATION_NOTES.md`。目录搬家与 Phase2 optimizer 推迟 |
| 2026-10-02 | **R1+ 补齐可改项**：packed_id 接进 `_task_loaders` 主评；t_mem/t_gen/t_iid + `first_stable_threshold.pt`；`train_segment(preserve)`；Fourier random/norm-matched/magnitude 对照；CSV DictWriter；RESEARCH §2.2/2.3 对齐。**仍差实验重跑**（见出门条件） |
| 2026-10-02 | **1B 默认变体去掉 `one`**：改为 `all_same four_diff pair_same`（均为 4 query，每步 packs 对齐）。`one` 仅作遗留可选 |
| 2026-10-04 | **1B 网格** `multi_op/20261003_132221`：`all_same/four_diff/pair_same` × ts0/ts1 全部 held-out≈1.0（best） |
| 2026-10-04 | **1C 重跑** 基于上述 6 ckpt → `mechanisms/20261004_1b_132221/`。同模 `all_same` 跨 op 消融几乎无选择性；异模有部分选择性但不稳。修复 selectivity CSV 对 packed op-key（`lat*/slot*/p*`）的写入 |
| 2026-10-04 | **阶段二 CLI**：`phase2 protocols` / `relation-matrix` / `capacity`。Packed 50/50 joint 与交替 interleaved；顺序训练保留优化器；行为汇总写入 `eval_history.jsonl` 与 transfer CSV。科学实验尚未开跑 |
| 2026-10-04 | **阶段二默认步数**改为 `50000`（2A/2B/2D 共用；joint / interleaved 总步数为 `2×50000`） |
| 2026-10-04 | **阶段二默认步数**改回 `100000`（joint / interleaved 总步数为 `2×100000`） |
| 2026-10-04 | **阶段二默认 wd** 改为 `0.3`（2A/2B/2D 共用）。`0.5/0.8` 下全重叠 seed0 的 p=23 易停在不看操作数的解；`0.3` 约 20k 步即可 grok |
| 2026-10-04 | **p=23 query-only 盆地**：数据/标签无误。卡住时 iid=train=val≈1/23；init 时 p=23 梯度最大；他 op grok 后 query 对操作数注意力仍≈0.18，\(\lVert\partial L/\partial\mathrm{emb}\rVert\approx 0\)，p=23 占总梯度约 0.4%。`wd=0.3` 于 ~20k 四 op 全 grok。记录见 `IMPLEMENTATION_NOTES` Phase 2 |
| 2026-10-04 | **1B 锁定 wd** 改为 `0.3`（与阶段二对齐；旧 1B 网格仍为 `wd=0.5`） |
| 2026-10-04 | **逐任务详细因果消融**：`scripts/phase1/causal_detail.py` → 各 variant `causal_detail/`（head/组件/Fourier k=1..6 + 跨 op 矩阵）。同模 head 共享、异模 head 部分分工 |
| 2026-10-04 | **多操作 residual steering**：`scripts/phase1/multi_op_steering.py` → 各 variant `steering/`。L1/L2 定向改预测≈1.0（shuf 对照低）；同模跨 op 可移植方向；个别异模 op（如 p29）要到 L2 才可 steer |
| 2026-10-04 | **Attention 位置互换**：`scripts/phase1/attn_pos_swap.py` → 各 variant `attn_swap/`。L0 质量成功挪到目标位置，但 acc_alt 仍≈随机（~0.07）；只改 routing 不足以定向改操作数 |
| 2026-10-04 | **TASK token 编辑**：`task_token_edit.py` → `*/task_token_edit/`。TASK_A→B 准确率不变；换成 digit 才偶发掉点（phase1 几乎不依赖 TASK 身份） |
| 2026-10-04 | **1C 电路移植 / activation patching**：`scripts/phase1/circuit_transplant.py`。L0 换自己的操作数 residual → 预测跟 donor；L0 只移植 attn write 双崩；L2 MLP write 可跨 op（甚至跨模）被共享头读出。综述 `mechanisms/20261004_1b_132221/TRANSPLANT_COMPARISON.md` |
| 2026-10-04 | **Unembedding 模 p Fourier**：`scripts/phase1/unembed_fourier.py`。头上行谱能量分散（top-1 仅 3–6%），消融几乎不掉点；与 digit-emb 主频常不对齐；L2 query 与 unembed cos≈1。1C 六个 + 1A 八模。综述 `mechanisms/20261004_1b_132221/UNEMBED_FOURIER.md` |
| 2026-10-04 | **代码结构重构（R0–R6）**：`AnalysisContext` / `analysis.pipelines`；训练协议拆分；CLI 迁到 `cli.py`；默认值 `defaults.py`；见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) |
| 2026-10-04 | **1B 多种子暂通过** `multi_op/20261004_170554`：`wd=0.3` packed，all_same/pair_same/four_diff × ts0–2 × ms0–2。同模促进成立；four_diff 1/9（ts0/ms0）p=23/29 未 grok。阶段一门关闭，进阶段二 |
