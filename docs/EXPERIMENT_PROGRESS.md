# 实验进度表

对照总计划 [`RESEARCH_EXPERIMENT_PLAN.md`](./RESEARCH_EXPERIMENT_PLAN.md)。  
代码与锁定协议见 [`ARCHITECTURE.md`](./ARCHITECTURE.md)。 CLI 列即各步入口。

**状态约定：** `未开始` · `进行中` · `已完成` · `阻塞` · `跳过`

最后更新：2026-10-08

---

## 总览

| 阶段 | CLI | 目标一句话 | 状态 |
|------|-----|------------|------|
| 工程预检 | `go4cl smoke` | 实现可跑通，不计科学结论 | 已完成 |
| 阶段一 | `go4cl phase1 …` | 单任务学会什么、何时 grok | 已完成 |
| 阶段二 | `go4cl phase2 …` | 联合 vs 顺序、任务关系→行为 | 进行中 |
| 阶段三 | `go4cl phase3 …` | 迁移/遗忘的因果机理 | 未开始 |
| 迁移机理 | `go4cl transfer-mechanism` | 模数特异性与 checkpoint mixing | 框架已实现，正式实验未开始 |

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
| 2A | 训练协议对照 | `go4cl phase2 protocols` | `runs/phase2/protocols/` | 进行中 | 默认 `wd=0.3/bs=8192/steps=100k` packed，全重叠。p=23 在 `wd≥0.5` 易 query-only 盆地，见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) | 2026-10-04 |
| 2B | 任务关系矩阵（8 → 27） | `go4cl phase2 relation-matrix` | `runs/phase2/relation_matrix/` | 进行中 | **无负样本全网格** [`20261004_202208/`](../runs/phase2/relation_matrix/20261004_202208/)：27×四协议=108/108 ok。joint 23/27 双任务≥0.9；sequential 仅 `s1_o1_m1` 双任务≥0.9，其余 B 会 grok、A 大多遗忘。**负样本** `--null-task-tokens --null-task-ratio 0.15`：他机试点 [`20261005_112940/`](../runs/phase2/relation_matrix/20261005_112940/) 一格四协议；本机全网格 [`20261005_154626/`](../runs/phase2/relation_matrix/20261005_154626/) 中途停，留 22/108 ok（见 2026-10-06 日志）。**10% replay**：legacy A-随-ρ [`20261006_131444/`](../runs/phase2/relation_matrix/20261006_131444/) 27/27；失败集中 m=0.5（4 格 A≈0.45–0.75）。**共享 A + replay** [`20261008_184031/`](../runs/phase2/relation_matrix/20261008_184031/) 36/36 ok（2 共享 A + 17 格×ms1/2）；A&lt;0.9 仅 3/34，均为单模 p=53 死亡卡在 ≈0.75 | 2026-10-08 |
| 2C | 全程行为指标（非独立入口） | 随 2A/2B 记录 | 同上 | 进行中 | eval history → forgetting / jump / exposure AUC / transfer CSV。该格事后 digit/unembed Fourier：[`FOURIER_POSTHOC.md`](../runs/phase2/relation_matrix/20261005_112940/FOURIER_POSTHOC.md) | 2026-10-05 |
| 2D | 容量消融 | `go4cl phase2 capacity` | `runs/phase2/capacity/` | 进行中 | 六种代表关系 × \{32,64,128\} × \{2,3,4\}。实验未跑 | 2026-10-04 |

**阶段二出门条件（进入阶段三前须满足）：**

- [x] Joint 能同时解决 A、B（兼容解存在）— 试点格 `s0.5_o0.5_m1` seed0
- [x] Sequential 中 B 能学会；A 是否遗忘作为结果记录 — 同格：B 学会、A 几乎忘光
- [ ] 已选出阶段三要解释的代表性条件

---

## 阶段三：持续学习机理

| ID | 步骤 | CLI | 产物目录 | 状态 | 锁定配置 / 结论摘要 | 日期 |
|----|------|-----|----------|------|---------------------|------|
| 3A | 优化层（梯度冲突等） | `go4cl phase3 optimize` | `runs/phase3/optimize/` | 未开始 | CLI 占位 | |
| 3B | 路由–计算–读出分解 | `go4cl phase3 route-compute-readout` | `runs/phase3/route_compute_readout/` | 未开始 | CLI 占位 | |
| 3C | 遗忘类型因果区分 | `go4cl phase3 forget-types` | `runs/phase3/forget_types/` | 未开始 | CLI 占位 | |
| 3E | Forward transfer 来源定位（无 replay） | `go4cl transfer-mechanism` | `runs/transfer_mechanism/` | 框架已实现，正式实验未开始 | **Framework implemented; formal GPU experiments not started.** 代码、配置、dry-run 脚本和 CPU 测试已落地。正式 final 网格未跑：模数 90 任务，checkpoint mixing 195 任务（15 个 A 源 + 180 个 intervention）。Smoke 只验证管线，不是研究结果 | 2026-10-08 |

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
| 2026-10-02 | **R1 协议修复**（数值以当前 53 类为准）：OperationKey / packed_id+nuisance eval / macro checkpoint / analysis dataset / steering 分 split / sampler_seed。目录搬家与 Phase2 optimizer 推迟 |
| 2026-10-02 | **R1+ 补齐可改项**：packed_id 接进 `_task_loaders` 主评；t_mem/t_gen/t_iid + `first_stable_threshold.pt`；`train_segment(preserve)`；Fourier random/norm-matched/magnitude 对照；CSV DictWriter；RESEARCH §2.2/2.3 对齐。**仍差实验重跑**（见出门条件） |
| 2026-10-02 | **1B 默认变体去掉 `one`**：改为 `all_same four_diff pair_same`（均为 4 query，每步 packs 对齐）。`one` 仅作遗留可选 |
| 2026-10-04 | **1B 网格** `multi_op/20261003_132221`：`all_same/four_diff/pair_same` × ts0/ts1 全部 held-out≈1.0（best） |
| 2026-10-04 | **1C 重跑** 基于上述 6 ckpt → `mechanisms/20261004_1b_132221/`。同模 `all_same` 跨 op 消融几乎无选择性；异模有部分选择性但不稳。修复 selectivity CSV 对 packed op-key（`lat*/slot*/p*`）的写入 |
| 2026-10-04 | **阶段二 CLI**：`phase2 protocols` / `relation-matrix` / `capacity`。Packed 50/50 joint 与交替 interleaved；顺序训练保留优化器；行为汇总写入 `eval_history.jsonl` 与 transfer CSV。科学实验尚未开跑 |
| 2026-10-04 | **阶段二默认步数**改为 `50000`（2A/2B/2D 共用；joint / interleaved 总步数为 `2×50000`） |
| 2026-10-04 | **阶段二默认步数**改回 `100000`（joint / interleaved 总步数为 `2×100000`） |
| 2026-10-04 | **阶段二默认 wd** 改为 `0.3`（2A/2B/2D 共用）。`0.5/0.8` 下全重叠 seed0 的 p=23 易停在不看操作数的解；`0.3` 约 20k 步即可 grok |
| 2026-10-04 | **p=23 query-only 盆地**：数据/标签无误。卡住时 iid=train=val≈1/23；init 时 p=23 梯度最大；他 op grok 后 query 对操作数注意力仍≈0.18，\(\lVert\partial L/\partial\mathrm{emb}\rVert\approx 0\)，p=23 占总梯度约 0.4%。`wd=0.3` 于 ~20k 四 op 全 grok。锁定协议见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) |
| 2026-10-04 | **1B 锁定 wd** 改为 `0.3`（与阶段二对齐；旧 1B 网格仍为 `wd=0.5`） |
| 2026-10-04 | **逐任务详细因果消融**：`scripts/phase1/causal_detail.py` → 各 variant `causal_detail/`（head/组件/Fourier k=1..6 + 跨 op 矩阵）。同模 head 共享、异模 head 部分分工 |
| 2026-10-04 | **多操作 residual steering**：`scripts/phase1/multi_op_steering.py` → 各 variant `steering/`。L1/L2 定向改预测≈1.0（shuf 对照低）；同模跨 op 可移植方向；个别异模 op（如 p29）要到 L2 才可 steer |
| 2026-10-04 | **Attention 位置互换**：`scripts/phase1/attn_pos_swap.py` → 各 variant `attn_swap/`。L0 质量成功挪到目标位置，但 acc_alt 仍≈随机（~0.07）；只改 routing 不足以定向改操作数 |
| 2026-10-04 | **TASK token 编辑**：`task_token_edit.py` → `*/task_token_edit/`。TASK_A→B 准确率不变；换成 digit 才偶发掉点（phase1 几乎不依赖 TASK 身份） |
| 2026-10-04 | **1C 电路移植 / activation patching**：`scripts/phase1/circuit_transplant.py`。L0 换自己的操作数 residual → 预测跟 donor；L0 只移植 attn write 双崩；L2 MLP write 可跨 op（甚至跨模）被共享头读出。综述 `mechanisms/20261004_1b_132221/TRANSPLANT_COMPARISON.md` |
| 2026-10-04 | **Unembedding 模 p Fourier**：`scripts/phase1/unembed_fourier.py`。头上行谱能量分散（top-1 仅 3–6%），消融几乎不掉点；与 digit-emb 主频常不对齐；L2 query 与 unembed cos≈1。1C 六个 + 1A 八模。综述 `mechanisms/20261004_1b_132221/UNEMBED_FOURIER.md` |
| 2026-10-04 | **代码结构重构（R0–R6）**：`AnalysisContext` / `analysis.pipelines`；训练协议拆分；CLI 迁到 `cli.py`；默认值 `defaults.py`；见 [`ARCHITECTURE.md`](./ARCHITECTURE.md) |
| 2026-10-04 | **1B 多种子暂通过** `multi_op/20261004_170554`：`wd=0.3` packed，all_same/pair_same/four_diff × ts0–2 × ms0–2。同模促进成立；four_diff 1/9（ts0/ms0）p=23/29 未 grok。阶段一门关闭，进阶段二 |
| 2026-10-05 | **2B 试点** `relation_matrix/20261005_112940`：`s0.5_o0.5_m1`，`null-task-ratio=0.15`，四协议全 ok。joint 双任务 ≈0.986；sequential A→B 遗忘 A（test 0.022），B 5k 达泛化 |
| 2026-10-05 | **事后 Fourier**（该 stamp 的 digit-emb / unembed，非训练中记录）：顺序学 B 后 digit 主频四模数均不变，谱余弦 0.80–0.97；不像 b_only。unembed 仍分散；\(p=41,53\) 头主峰对齐 b_only。见 [`FOURIER_POSTHOC.md`](../runs/phase2/relation_matrix/20261005_112940/FOURIER_POSTHOC.md) |
| 2026-10-05 | **docs 精简**：只留计划 / 进度 / 结构三份。删除步骤指南、R1 施工单、重构基线；协议锁定与 p=23 盆地并入 [`ARCHITECTURE.md`](./ARCHITECTURE.md) |
| 2026-10-05 | **sequential 切 B 前 vs 训完 B**：digit-emb 对 **B 的模数** 做 Fourier（`theta_A` vs `final`）。切前已尖；主频与训完 B 一致 72/108。见 `relation_matrix/20261004_202208/fourier_B_after_A_vs_after_B.csv` |
| 2026-10-06 | **负样本全网格中止** `relation_matrix/20261005_154626`：`--null-task-tokens --null-task-ratio 0.15`，计划 108 job，中途停。删 36 个无 `job_result` 的目录；**22 ok 保留**（a_only 10 / b_only 9 / joint 1 / sequential_ab 2）。对照无负样本 `20261004_202208`：a_only 仍 A≈1、B chance；joint `s0_o0.5_m1` 双任务 0.996/0.994（原 1.0/1.0）；sequential `s0_o0.5_m0.5`、`s0_o0_m1` 切 B 前 A≥0.99，结束后 A≈0.02、forget 0.96/0.98，B 仍≈1——**没有减轻遗忘**。唯一大差：`b_only s0_o0_m1` B 0.967→0.855（p=23 test 0.49，best 0.57；其余三模仍>0.94）。他机试点 `s0.5_o0.5_m1` 曾更差（顺序 A 0.267→0.022），该格本 stamp 未跑完 |
| 2026-10-06 | **1A Fourier 消融加随机子空间对照**：与 top-\(k\) 相同 \(\lVert\Delta W\rVert_F\) 的 digit-emb 高斯扰动（5 seed）。p=31 final：\(k=1\) acc 0.83 vs 重要频率 0.16。图 `p31_summary/figures/fig3_fourier_ablation.png` |
| 2026-10-08 | **顺序±replay 权重差分（事后）**：`scripts/phase2/weight_delta_switch.py` 对比 `theta_A`→`final`。产物 [`20261006_131444/weight_delta_switch/`](../runs/phase2/relation_matrix/20261006_131444/weight_delta_switch/)。变化主要集中在 **digit emb + L0/L1 MLP + head**；replay 并未整体缩小 ‖ΔW‖，失败格与成功格总变化量接近 |
| 2026-10-08 | **`--fixed-a` / `--share-a`**：同 seed 不同 ρ 共用同一 Task A；`--share-a` 先训共享 `θ_A` 再只跑 B。无 replay seed 网格 [`20261008_135656/`](../runs/phase2/relation_matrix/20261008_135656/) 34/34 ok（legacy A-随-ρ，sequential 无 replay，A 几乎全忘） |
| 2026-10-08 | **共享 A + 10% replay** [`20261008_184031/`](../runs/phase2/relation_matrix/20261008_184031/)：`--fixed-a --share-a --protocols sequential_ab_replay`，ms1/2，17 格（与无 replay seed 网格相同），36/36 ok。共享 A 模数固定 `{23,31,41,53}`（A_test 切 B 前 ≈0.980/0.994）。34 个 B-only：mean A 按 m 为 0.926 / 0.969 / 0.987；B 均 ≈0.99。A&lt;0.9 仅 **3/34**，全是丢掉 **p=53**、宏观卡在 ≈0.75：`s0_o0_m0` ms1、`s0.5_o0_m0` ms1、`s1_o0_m0.5` ms2（同格另一 seed 均 ≥0.92）。对照 legacy replay [`20261006_131444/`](../runs/phase2/relation_matrix/20261006_131444/)：A 随 ρ 变，m=0.5 的 A 为 `{29,37,43,53}`，失败 4 格（`s0_o0_m0.5` / `s0_o0.5_m0.5` / `s1_o0_m0.5` / `s1_o0.5_m0.5`，A≈0.45–0.75，常见死模 p=43）。同名 4 格在共享 A 下 7/8 seed ≥0.97，仅 `s1_o0_m0.5` ms2 仍挂。失败形态两次相同（1/4 op 永久归零）；ρ 标签上的 m=0.5 vs m=0「反转」来自 **A/B 模数集合不同**，不是 m 本身固有难易 |
| 2026-10-08 | **Forward transfer 机理框架**：`src/go4cl/phases/transfer_mechanism/`。模数特异性（`fixed_a`，`b_only`/`sequential_ab`）与 checkpoint mixing（12 个 intervention，fresh optimizer，`replay_ratio=0`）。Framework implemented; formal GPU experiments not started. 未改已有实验结果，smoke 数值不作结论 |
| 2026-10-08 | **实验有效性修复（未重跑）**：拒绝 `share-a`+`swap`；sequential 记录 `optimizer_transition=fresh\|preserve`（旧 Phase 2 默认仍为 preserve）；B 学会改为 `B_t_gen` / `B_first_stable.pt`；机理分析共用 analysis bundle；探针使用局部 seed；checkpoint 必须显式指定 role；val 选、test 确认。原始 JSON/CSV 未改 |

## 结果有效性与当前解释边界

下面只重新表述已经写过的结论。原始 JSON、CSV 和 checkpoint 都没有改写。

### 可以继续保留的结果

1. 数据生成、余数对划分，以及 train/val/test 不交叠，是可靠的。
2. 单任务中不同模数存在不同的优化轨迹和 grokking 时间。
3. `wd=0.5` 下出现小模数 query-only basin，`wd=0.3` 可以离开该 basin。这是可靠的优化现象。
4. 已经观察到 A→B 相比 matched B-only 更快。这是可靠的行为观察。
5. replay 下仍可能出现促进作用，可以作为行为结果保留。
6. task-partition 的行为准确率、遗忘率和 retention，只要来自固定 eval loader，可以保留。

### 必须降低结论强度的结果

**A→B 促进作用。** 在保留 A 阶段模型参数和 AdamW 状态的 sequential protocol 下，B 的学习速度高于 matched B-only baseline，说明 A 阶段训练状态产生了正向迁移。权重贡献与优化器状态贡献尚未分离。只有完成 fresh-vs-preserve optimizer 实验后，才能进一步归因。

**overlap 网格。** 旧 108 网格没有使用 fixed A。单个 condition 内的 protocol 对比仍然有效。跨 ρ cell 的差异只能视为探索性结果，不能宣称只由 overlap 改变导致，因为 A 本身也随 ρ 变化。后续 fixed-A 网格才作为确认性实验。

**best.pt 机理结论。** 旧 sequential `best.pt` 是 A/B tradeoff checkpoint，不等同于 B 学会后的最终状态。基于它的机制分析标记为 exploratory，需要用明确 checkpoint role 复核。可用角色是 `theta_A`、`phase_b_final`、`B_first_stable`、`B_best_val`、`AB_tradeoff_best`。

**task-identity probe。** 高 task probe accuracy 只能表明 task identity 可线性解码，可能只是 task-token embedding 被保留，不能单独证明形成了独立任务计算空间。更强的证据应来自 task-token counterfactual、routing intervention、activation/weight patching，以及在 matched contexts 上改变 task token 后是否切换到对应算法输出。

**在 test 上选电路。** 已有在 test 上同时选 head/layer 并报告效果的结果标记为 exploratory。正式结果需要 val 上选择、test 上确认，并同时保存 val selection score、test confirmation score 和 selected component。Phase 2 动态曲线此后默认使用 validation；test 只用于最终 endpoint 和确认性分析。

### 后续最小重跑矩阵（只列命令，本轮不启动）

优先级 1，在核心条件 `s0.5_o0.5_m1` 上分离 optimizer 贡献。五次运行保持同一 fixed A、同一 B、同一初始化、同一 sampler seed、同一 B exposure 和同一 evaluation contexts：

```bash
# 1. B-only + fresh optimizer
uv run go4cl phase2 relation-matrix --conditions s0.5_o0.5_m1 \
  --protocols b_only --fixed-a --directions forward \
  --optimizer-transition fresh --task-seeds 0 --model-seeds 0 --data-seed 0 \
  --out runs/phase2/optimizer_split/b_only_fresh

# 2–3. A→B fresh 与 A→B preserve
uv run go4cl phase2 relation-matrix --conditions s0.5_o0.5_m1 \
  --protocols sequential_ab --fixed-a --directions forward \
  --optimizer-transition fresh --task-seeds 0 --model-seeds 0 --data-seed 0 \
  --out runs/phase2/optimizer_split/ab_fresh
uv run go4cl phase2 relation-matrix --conditions s0.5_o0.5_m1 \
  --protocols sequential_ab --fixed-a --directions forward \
  --optimizer-transition preserve --task-seeds 0 --model-seeds 0 --data-seed 0 \
  --out runs/phase2/optimizer_split/ab_preserve

# 4–5. A→B replay fresh 与 preserve
uv run go4cl phase2 relation-matrix --conditions s0.5_o0.5_m1 \
  --protocols sequential_ab_replay --fixed-a --directions forward \
  --optimizer-transition fresh --task-seeds 0 --model-seeds 0 --data-seed 0 \
  --out runs/phase2/optimizer_split/replay_fresh
uv run go4cl phase2 relation-matrix --conditions s0.5_o0.5_m1 \
  --protocols sequential_ab_replay --fixed-a --directions forward \
  --optimizer-transition preserve --task-seeds 0 --model-seeds 0 --data-seed 0 \
  --out runs/phase2/optimizer_split/replay_preserve
```

优先级 2，fixed-A overlap 确认网格。不要加 `--share-a`，也不要把 `swap` 和共享 A 放在一起。顺序协议建议显式使用 `fresh`，这样确认网格不再和旧的 preserve 协议混在一起：

```bash
uv run go4cl phase2 relation-matrix --grid full --fixed-a --directions forward \
  --protocols b_only sequential_ab sequential_ab_replay \
  --optimizer-transition fresh \
  --out runs/phase2/relation_matrix/fixedA_confirmatory
```

优先级 3，在 fresh optimizer 下做 checkpoint 参数组混合，看促进作用来自 digit embedding、control embedding、attention、MLP、output，以及各层 attention/MLP。默认 12 个 intervention 已配置为 fresh、无 replay。逐层 intervention 在 `reserved_interventions` 中，默认脚本不启动。单 head mixing 仍未实现。

```bash
bash scripts/transfer_mechanism/component_reset.sh --dry-run
```

真正开跑时才把 `--dry-run` 换成 `--execute`。本轮没有执行这些命令。
