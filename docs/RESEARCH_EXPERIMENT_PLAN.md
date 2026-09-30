# Transformer 持续学习机理：总体实验计划

## 1. 研究目标

研究小型 Transformer 在依次学习两个结构化模加法任务时，已有计算电路如何被新任务复用、重路由、扩展或覆盖，并解释这些变化如何产生正迁移、负迁移和灾难性遗忘。

核心因果链为：

$$
\text{任务结构关系}
\rightarrow
\text{梯度关系}
\rightarrow
\text{电路复用/重路由/覆盖}
\rightarrow
\text{迁移与遗忘}.
$$

主要研究问题：

1. 输出槽、操作数位置和模数三个层面的重叠，是否分别影响读出、路由和计算电路？
2. 任务相似度与遗忘是否单调，还是部分重叠比完全相同和完全不同产生更强干扰？
3. A 的性能下降来自 A 特征被删除，还是特征仍存在但下游读出失配？
4. 模型容量如何决定复用、复制与覆盖之间的选择？

## 2. 任务定义

### 2.1 输入与单槽查询

每个样本包含定长整数序列：

$$
x=(x_1,\ldots,x_8),\qquad x_i\in\{0,\ldots,255\}.
$$

每个任务包含四个操作：

$$
T=\{o_k=(i_k,j_k,p_k)\}_{k=1}^{4},
\qquad
y_k=(x_{i_k}+x_{j_k})\bmod p_k.
$$

模型每次只查询一个输出槽：

```text
x1 x2 x3 x4 x5 x6 x7 x8 <TASK_T> <Q_k>  ->  y_k
```

完整四维输出通过将四个 query 放入 batch 中并行得到，而不是自回归生成。这样不会把前一个正确结果暴露给后一个预测。

### 2.2 Token 与序列长度

- 数字 token：256 个，对应 0–255。
- 任务 token：`<TASK_A>`、`<TASK_B>`。
- 查询 token：`<Q_0>`、`<Q_1>`、`<Q_2>`、`<Q_3>`。
- 输入词表大小：262。
- 输出类别：0–30，共 31 类。
- 上下文长度：10。

输入 embedding 和输出分类头不绑定；输出分类头在所有任务、槽位和模数之间共享。不根据当前模数屏蔽无效输出类别。

### 2.3 固定训练集与严格无交集划分

正式实验不使用无限在线数据流，而是为每个实验条件和 data seed 预先生成固定的 train/validation/test 数据集。模型在固定训练集上重复训练，以便区分训练集记忆与测试集泛化，并观察 grokking。

仅保证完整 token 序列不重复是不够的，因为目标只依赖两个被查询的操作数。划分必须发生在当前模数的无序余数对层面：

$$
u_p(x_i,x_j)=
\left(\min(x_i\bmod p,x_j\bmod p),
\max(x_i\bmod p,x_j\bmod p)\right).
$$

同一个无序余数对的所有 raw-token aliases、交换顺序和 nuisance-position 变体必须进入同一个 split。对于同一模数的多个操作，必须共享同一套 residue-pair split，防止一个操作的测试余数对通过另一个操作进入训练集。

划分在每个输出类别 $y=(r_1+r_2)\bmod p$ 内部分层进行，使 train/validation/test 都覆盖全部输出类别。初始比例约为 60/20/20；对于较小模数，以“每个输出类别在 validation/test 中至少有一个无序余数对”为优先约束。

对每个 split，从允许的余数对中采样固定数量的 raw aliases 和 nuisance contexts，生成不可变数据文件或可由 manifest 完全重建的数据集。保存 modulus、residue-pair split、完整 TaskSpec、data seed、样本数量、数据哈希和 raw-token 生成规则。

### 2.4 模数集合

主实验使用八个素数：

$$
\mathcal P=\{7,11,13,17,19,23,29,31\}.
$$

主实验的近邻配对为：

$$
(7,11),\quad(13,17),\quad(19,23),\quad(29,31).
$$

八个模数不存在倍数关系，且任意两个不同模数互素。每个任务从四个模数对中各选一个，因此任务内部四个模数互不相同。

输入统一从 0–255 均匀采样。对于最大模数 31，每个余数对应 8 或 9 个输入 token；最坏频率比为 9/8。评估时同时报告原始分布指标和 residue-pair macro 指标。

## 3. A/B 任务关系的正交构造

为四个潜在操作赋予固定身份 $z_0,\ldots,z_3$。从 A 构造 B 时，依次改变三类属性。

### 3.1 输出槽重叠

通过置换 $\pi$ 将潜在操作分配到查询槽。定义：

$$
\rho_{\mathrm{slot}}
=\frac{1}{4}\sum_z\mathbf 1[\pi_B(z)=\pi_A(z)].
$$

主实验取 $\rho_{\mathrm{slot}}\in\{0,0.5,1\}$，分别对应 0、2、4 个不动点。

### 3.2 操作数位置重叠

每个任务的四个无序操作数对构成八个输入位置上的 perfect matching，使每个输入位置在任务内恰好被使用一次。

$$
\rho_{\mathrm{operand}}
=\frac{1}{4}\sum_z
\mathbf 1[\{i_z^A,j_z^A\}=\{i_z^B,j_z^B\}].
$$

主实验取 $\rho_{\mathrm{operand}}\in\{0,0.5,1\}$，并禁止意外重复边。

### 3.3 模数重叠

每个潜在操作绑定一个模数对。若该操作的模数保持不变，则 B 使用与 A 相同的模数；若改变，则 B 使用该模数对的另一个成员。

$$
\rho_{\mathrm{mod}}
=\frac{1}{4}\sum_z\mathbf 1[p_z^A=p_z^B].
$$

三个水平对应：

| $\rho_{\mathrm{mod}}$ | A、B 合计使用的不同模数数目 |
|---:|---:|
| 1 | 4 |
| 0.5 | 6 |
| 0 | 8 |

模数对、潜在操作和输出槽之间的对应关系跨 task pair 随机置换。每个任务顺序同时生成 A→B 与 B→A 版本，以抵消模数大小和任务难度影响。

## 4. 主模型

主模型的初始候选配置：

```text
architecture: decoder-only causal Transformer
layers: 3
d_model: 64
attention_heads: 4
d_head: 16
d_mlp: 256
activation: ReLU
normalization: Pre-LayerNorm
dropout: 0
position_encoding: learned absolute position embedding
context_length: 10
output: shared 31-class linear head on the final query position
```

绝对位置编码与任务定义一致，因为操作明确引用绝对输入位置，且训练、验证均为固定长度，不要求长度外推。

主模型通过以下预先规定的规则锁定：选择能够在多个随机种子上可靠完成单任务和 A+B 联合训练的最小模型。如果 3×64 无法可靠完成联合训练，则将主模型提高到 3×128，而不是把联合训练失败解释为持续学习现象。

## 5. 三阶段实验路线

在三个科学阶段之前，先完成工程预检：单 batch 过拟合、固定数据集无交集测试、单任务短训练、A+B 联合短训练和 A→B 顺序短训练。工程预检只验证实现，不计入科学结论。

### 阶段一：单任务学习、grokking 与算法形成

目标是先理解模型如何学会这些任务，再研究持续学习。该阶段不引入 A→B 遗忘。

#### 1A. 单个操作的模数扫描

对八个模数分别训练只包含一个有效 query 的模型：

$$
p\in\{7,11,13,17,19,23,29,31\}.
$$

所有模数使用相同模型、优化器、每个操作的 batch 暴露次数和 residue-pair 训练比例。记录完整训练与测试曲线，而不是只记录最终准确率。

在扫描全部模数前，先用一个中等模数做小型 grokking regime calibration，考察训练余数对比例、weight decay 和训练步数。锁定统一训练配置后再扫描八个模数，不能为每个模数单独挑选最容易出现 grokking 的超参数。若某些模数只表现为平滑泛化，也应作为结果报告，而不是把 grokking 当作必须出现的成功条件。

定义：

- $t_{\mathrm{mem}}$：训练准确率首次持续超过 99% 的 step；
- $t_{\mathrm{gen}}$：测试准确率首次持续超过预设阈值的 step；
- grokking delay：$t_{\mathrm{gen}}-t_{\mathrm{mem}}$；
- transition sharpness：测试准确率对 log-step 的最大变化率。

比较不同模数的记忆时间、泛化时间、grokking delay、最终 margin 和 Fourier 特征形成时间。学习时间同时用 optimizer step、样本暴露数和等效 epoch 表示。

#### 1B. 多操作单任务与同模数促进

比较下列单任务组成：

1. 一个操作；
2. 四个操作且四个模数不同；
3. 四个操作，其中两个操作共享同一模数，但操作数位置和输出槽不同；
4. 可选：四个操作全部使用同一模数。

每个操作获得相同的 query 采样概率和样本暴露数。两个同模数操作共享相同的 residue-pair train/test split，使测试对不会通过另一个操作泄漏。

核心比较应保持总操作数为四，只改变其中一对操作是“相同模数”还是“规模相近的不同模数”。跨条件匹配模型初始化、数据量、query 暴露数、operand matching 和槽位安排，并在多个模数与位置组合上重复，避免把某个特定模数或槽位的难度误认为促进效应。

主要问题：

- 第二个同模数操作是否缩短 $t_{\mathrm{gen}}$？
- 两个操作是否共享同一套模数特征，但使用不同 attention 路由？
- 促进发生在训练早期的表示形成、后期 cleanup，还是读出阶段？

#### 1C. 单任务机理验证

在训练轨迹上保存 log-spaced checkpoint，并在训练集记忆、测试集跃迁和最终收敛附近加密保存。

相关性分析：

- 对 residue-averaged token embedding、query residual 和 unembedding 做模 $p$ Fourier 分析；
- 探测各层是否能线性或圆周式解码 $x_i\bmod p$、两个被选操作数、部分和与最终结果；
- 观察 attention 是否因 query 而稳定选择正确的两个输入位置；
- 测量同余类内部方差和 Fourier 频率能量随训练的变化。

因果验证：

- 消融或投影掉候选 Fourier 频率子空间；
- head/MLP ablation；
- 对正确和错误操作数位置做 activation patching；
- 在同模数两个操作之间移植候选计算子电路，检验计算复用与路由分离。

只有当表征分析、探针和因果干预一致时，才将机制描述为具体算法。

### 阶段二：双任务行为动力学

目标是系统比较联合学习与顺序学习，并建立任务关系到行为现象的映射。

#### 2A. 训练协议

使用同一组固定 A/B 数据集比较：

1. A-only 和 B-only；
2. joint A+B：每个 batch 以 50/50 比例包含两个任务；
3. interleaved A/B：按 batch 交替任务；
4. sequential A→B：A 达到预定泛化标准后切换到 B，无 replay；
5. sequential B→A：顺序对照；
6. A-only continued：A 学完后继续训练相同步数，控制自然漂移。

联合与顺序训练应匹配每个任务接收的样本暴露数。若顺序训练为 A 训练 $S$ step、B 训练 $S$ step，则 joint 训练总计 $2S$ step，并使每个任务期望获得 $S$ step 的数据。

主顺序实验在 A 已经泛化并且阶段一识别的算法指标稳定后切换到 B。额外做切换时机消融：A 记忆完成但尚未 grok、A 正在跃迁、A 完成 grokking。

#### 2B. 任务关系矩阵

先运行三个重叠因素取 $\{0,1\}$ 的 $2^3=8$ 个极端条件，确认效应和实现正确；随后运行 $\{0,0.5,1\}^3$ 的完整 27 条件设计。

每个条件包含多个 task pair、多个模型 seed、A→B 与 B→A，并使用相同数据 manifest 比较 joint 与 sequential。

#### 2C. 不只比较最终结果

完整记录：

- A 和 B 的 train/test loss、accuracy 与 margin；
- 每个槽位和模数的学习时间；
- B 开始后 A 的瞬时 loss jump、最快遗忘速率和长期保留；
- B 相对 B-from-scratch 的学习速度与前向迁移；
- joint 条件下两个任务是否同时 grok、先后 grok 或互相阻碍；
- 顺序训练中 A 的 Fourier/probe 指标是否先于行为性能下降；
- 部分重叠是否比完全相同或完全不同产生更强干扰。

#### 2D. 容量消融

宽度取 $d_{\mathrm{model}}\in\{32,64,128\}$，深度取 $L\in\{2,3,4\}$。先在完全相同、仅槽不同、仅操作数不同、仅模数不同、完全不同和一个强部分重叠条件上运行，再决定是否扩展到全部 27 条件。

### 阶段三：持续学习机理分析

目标是解释阶段二发现的代表性现象，而不是对所有运行机械地做完整电路分析。预先选择：明显正迁移、明显遗忘、非单调部分重叠、joint 成功但 sequential 失败等代表性条件。

保存：

$$
\theta_0,\quad
\theta_A^{(t)},\quad
\theta_A,\quad
\theta_{A\rightarrow B}^{(t)},\quad
\theta_{AB}.
$$

#### 3A. 优化层机制

- 测量整体、逐层、逐 head/MLP 的 $g_A^\top g_B$ 和梯度余弦；
- 比较一阶预测 $\Delta L_A\approx-\eta g_A^\top g_B$ 与真实 A loss 变化；
- 定位负梯度干扰首先出现在哪些模块；
- 比较 joint 与 sequential 是否沿不同优化路径到达不同解。

#### 3B. 路由—计算—读出分解

- 操作数重叠主要检查 attention routing 的保留和重映射；
- 模数重叠主要检查 Fourier/MLP 计算特征的复用和覆盖；
- 槽位重叠主要检查 query-conditioned residual 与输出读出变化；
- 使用组件级和路径级 patching 验证该分解，而不只观察 attention map。

#### 3C. 区分三类遗忘机制

1. 表征删除：A 的模数特征本身消失，探针和因果干预都失败；
2. 表征仍在但读出失配：A 特征仍可探测，恢复旧 readout 或下游模块可恢复性能；
3. 路由失配：计算模块仍正常，但 query 不再读取正确操作数或正确子电路。

关键干预包括：

- 将 $\theta_A$ 的单层、head、MLP 或 readout patch 回 B 训练后的模型；
- 将 B 后模型的中间 activation 替换为 $\theta_A$ activation；
- 分别恢复 routing、Fourier computation 和 readout，比较对 A 的恢复量；
- 消融共享组件，检查是否同时损害 A、B；
- 进行跨任务 circuit transplant 与 cross-task faithfulness 测试。

#### 3D. 阶段三的成功标准

机理解释至少应满足：

1. 能预测阶段二中哪些条件产生迁移或遗忘；
2. 能通过因果干预恢复或复现相应行为；
3. 能跨多个 model seed 和 task pair 重现；
4. 不依赖单个 attention head 的主观可视化；
5. 能区分参数变化、表征变化和实际功能变化。

## 6. 行为指标

持续记录每个 task/slot/modulus 的 loss 和 accuracy。

遗忘：

$$
F_A=max_t\operatorname{Acc}_A(t)
-\operatorname{Acc}_A(\text{after B}).
$$

还需报告：

- B 相对 B-from-scratch 的学习曲线 AUC；
- 达到固定准确率所需步数；
- backward transfer；
- 每个模数和每个槽位的准确率；
- residue-pair macro accuracy；
- 按 $\log p$ 归一化的交叉熵；
- 无关输入位置重采样后的预测一致性。

## 7. 必要基线

1. A-only：A 学完后继续在 A 上训练相同步数，控制自然参数漂移。
2. B-only from scratch：控制 B 的固有难度。
3. A→B：核心持续学习条件，不使用 replay。
4. B→A：顺序对照。
5. Joint A+B：验证兼容解是否存在，以及当前容量能否同时容纳两个任务。
6. Interleaved A/B：作为避免遗忘的行为上界之一。
7. 位置编码稳健性：主实验完成后，用固定 sinusoidal encoding 重复代表性条件。
8. 激活函数稳健性：主实验完成后，用 GELU 重复代表性条件。

## 8. 数据划分与验证原则

除单 batch 过拟合这一工程测试外，所有科学实验都使用固定数据集。正式实验必须在 residue-pair 层面建立 train/validation/test split，使同一个余数对的不同 raw-token aliases、交换顺序和 nuisance contexts 不会跨越集合。划分还应近似平衡：

- 每个输入余数的出现次数；
- 每个输出类别的出现次数；
- 交换对 $(r_1,r_2)$ 与 $(r_2,r_1)$ 必须进入同一集合。

同一 data seed 下的所有训练协议必须复用完全相同的数据 manifest。对于共享模数的多个操作，也必须复用同一 residue-pair split。训练集只在不同 data seed 的独立重复中改变。

关键 checkpoint 上枚举所有 residue pairs，并对所有 raw operand pairs 或充足的 raw aliases 进行验证。其余六个无关输入位置需要多次重采样，以排除 shortcut。

## 9. 统计与复现

- task seed、model seed 和 data seed 分开记录。
- 保存完整 TaskSpec、配置文件、Git commit、环境信息和 checkpoint。
- task pair 与 model seed 作为不同随机效应处理。
- 先报告每个条件的完整分布，再进行因素主效应和交互分析。
- 不根据遗忘强弱选择主模型；模型选择只依据单任务和联合训练的可靠性。

## 10. 当前成功标准

基础设置只有在以下条件满足后才进入完整实验：

1. 单任务 A、B 均能在多数种子上可靠达到高 held-out accuracy。
2. Joint A+B 能同时解决两个任务，说明兼容解存在。
3. 顺序训练中 B 能成功学会；A 是否遗忘作为被测结果，而不是成功前提。
4. 结果能够在多个 task pair 和模型种子上复现。
5. 至少一种因果干预可以把行为变化定位到具体组件或路径，而不仅是观察参数距离或 attention map。
