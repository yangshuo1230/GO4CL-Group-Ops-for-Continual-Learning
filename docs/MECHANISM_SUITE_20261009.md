# 五问题机制实验：两个种子版本

本批新增独立入口，不修改旧训练协议，不启动自动训练。默认随机初始化种子为 0、1；数据划分、采样、评估种子跨模型种子固定。四个运算使用不同模数。23、41 是重点观察模数，不预设学习难度。

## 任务构造

| latent | 固定 B：模数 / 操作数 / 槽位 | A：模数 / 操作数 / 槽位 |
|---|---|---|
| 0 | 23 / (0,1) / 0 | 相同模数取23，否则29；相同位置取(0,1)，否则(0,2)；槽位1 |
| 1 | 41 / (2,3) / 1 | 相同模数取41，否则43；相同位置取(2,3)，否则(1,3)；槽位0 |
| 2 | 31 / (4,5) / 2 | 37 / (4,6) / 3，四类A固定 |
| 3 | 47 / (6,7) / 3 | 53 / (5,7) / 2，四类A固定 |

四类A是 same_mod_same_pos、same_mod_diff_pos、diff_mod_same_pos、diff_mod_diff_pos。相同模数或位置仅针对前两个重点运算，因此对应整体重叠度0.5；背景运算始终不共享。槽位重叠按同latent计算，所有条件为0；两任务都使用完整的0..3槽位。

比例实验固定 same_mod_diff_pos 的A以及固定B的操作数和槽位，分别设置B模数：

| 条件 | B模数 |
|---|---|
| s0_o0_m0 | 29,43,31,47 |
| s0_o0_m0.5 | 23,41,31,47，即主实验B |
| s0_o0_m1 | 23,41,37,53 |

这三个条件严格固定A，但改变B的模数也改变B的难度，所以均保留各自B-only对照。它们只能回答这三个具体任务关系下的差异，不能单凭两个种子推断普遍规律。

## 训练表

| 阶段 | 每个种子 | 两个种子 | 用途 |
|---|---:|---:|---|
| A来源：四类A，各100k步 | 4 | 8 | 所有对应分支共用固定终点theta_A |
| 固定B-only | 1 | 2 | 促进基线 |
| 四类A→B，无replay | 4 | 8 | 问题1、2、5 |
| 四类A→B，10%replay | 4 | 8 | 问题3、5 |
| 四类A × 五组件 × reset/keep | 40 | 80 | 问题1的组件训练干预 |
| 三种overlap × 0.5%、1%、2%、5% | 12 | 24 | 问题4 |
| m0和m1补0%、10%锚点 | 4 | 8 | m0.5锚点复用主实验 |
| m0和m1补B-only | 2 | 4 | m0.5基线复用主实验 |
| 合计 | 71 | 142 | 每段100k步 |

`--stage core` = 26段；`--stage components` = 88段（含8个来源）；`--stage overlap` = 44段（含2个来源）。阶段总数不能相加，因为共用来源和锚点。已完成任务再次运行会跳过。

训练设置：3层、宽64、4头、MLP256、ReLU；AdamW lr=1e-3、betas=(0.9,0.98)、weight decay=0.3、梯度裁剪1；batch8192个query，即2048个四运算context。全部固定100k步，没有按B成功提前停止。A→B切换统一使用新AdamW，checkpoint仍保存A优化器供复查，但不继承。

A来源必须在最后连续5次保存点上所有运算val准确率≥0.95，才启动其分支。失败来源和被阻止分支均记录，不自动换种子、不筛掉失败。门槛是验证集，不查看测试集决定训练。来源为固定100k终点，没有从不同来源阶段选择不同最佳checkpoint。

每步总query数固定。replay替换部分B样本，因此比例增加会降低B累计曝光；结果记录实际A/B曝光数，比较时同时画按步数和按B曝光的曲线。比例按整数context四舍五入，结果保存请求比例和实际比例。使用完整A训练划分在线采样，不是有限memory覆盖率实验；不声称控制了独特原始context覆盖率。不同batch组成不保证逐样本B顺序相同，但B划分、分布及独立采样种子一致。

## 保存与分析

保存点：0–2k每100步，2k–10k每500步，10k–100k每2k步，外加指定梯度点，共82个点。每点保存模型、AdamW状态、Python/NumPy/Torch/CUDA RNG、A/B采样RNG、曝光量和逐运算A/B验证指标。latest/final为硬链接，不重复占模型空间。异常退出可从最近完整保存点重跑；强制kill留下`.running`时，先确认原进程已经退出，再手动移除该锁。不要并发执行同一输出根目录的同一任务。

梯度点：0、100、500、1k、5k、10k、20k、50k、100k。完整A迁移和replay分支记录；组件与B-only不记录。使用固定训练划分诊断batch，保存A、B、8个逐运算原始梯度，以及10%/当前比例的精确加权混合梯度。通过克隆实际AdamW状态分别执行A/B/mix的单步更新，保存裁剪后实际参数delta、关weight decay对照、固定A验证损失变化。保存参数分组切片，离线汇总全局、分层、组件范数及cosine。delta定义为theta_after-theta_before，与梯度夹角符号应区分。诊断是有限batch估计，不是完整分布梯度。

离线分析入口提供：

- 同一probe数据在每个checkpoint复用；train/val/test目标操作数residue pair互斥；逐运算、逐层attention后/MLP后query表示的xi、xj、sum线性probe。
- 相应theta_A的固定probe和当前checkpoint重训probe并列。每次使用相同probe初始化、训练步数和数据。源A模型上的B探针是参照诊断，不代表它已经学会B。
- 逐层逐头operand attention mass和TASK attention mass。
- query位置的head zeroing、MLP zeroing，以及同输入theta_A的resid_post恢复（query-only和整流两种范围）；TASK token反事实。
- 全部任务行为汇总CSV/JSON和PNG；梯度统计JSON。

Zeroing只提供该干预下的必要性证据；组件reset变慢和keep有效都需考虑接口协同。跨checkpoint donor patch失败不证明知识被删除。该入口没有实现表征对齐/基底旋转检验，也没有穷尽替代路径；这部分应根据轨迹结果再定针对性实验。当前不把attention权重或probe可读性当作计算功能证明。

比例扫描输出“所有已采样保存点均保持A”和“最后5个保存点A/B所有运算均通过”两个标准。0.5%到5%没有成功而10%成功时，只能给阈值区间；不外推单调性。两个种子只能形成机制案例与初步重复性结果。

## 运行

```bash
cd /mnt/ningxuefei/yangshuo/GO4CL-Group-Ops-for-Continual-Learning

# 仅生成计划，无训练（脚本默认行为）
bash scripts/phase2/mechanism_suite.sh --dry-run

# 推荐先执行26段核心实验；替换GPU编号为当前空闲卡
GPUS=0 bash scripts/phase2/mechanism_suite.sh --execute --stage core

# 结果确认后补组件或比例扫描，自动跳过已经完成的来源/锚点
GPUS=0 bash scripts/phase2/mechanism_suite.sh --execute --stage components
GPUS=0 bash scripts/phase2/mechanism_suite.sh --execute --stage overlap

# 一次执行完整142段；多个GPU每卡一个串行worker
GPUS=0,1 bash scripts/phase2/mechanism_suite.sh --execute --stage all

# CPU离线行为图、梯度汇总，不启动训练
PYTHONPATH=src .venv/bin/python -m go4cl.phases.phase2.mechanism_suite_analysis

# 单个分支的probe和因果检查；默认CPU，可以显式指定cuda:0
PYTHONPATH=src .venv/bin/python -m go4cl.phases.phase2.mechanism_suite_analysis \
  --job seed0/main/same_mod_diff_pos/replay_0.1 \
  --steps 0,1000,10000,100000 --causal
```

输出默认在`runs/phase2/mechanism_suite_20261009`。可用`RUN_ROOT`换新批次根目录。修改配置后必须使用新的根目录，已有结果有配置指纹保护。每个GPU轮流执行任务，先完成本次所需A来源后再启动B分支；日志在每任务的`stdout.log`。

仅按已有100k约22–23分钟的训练段计：core约9.5–10 GPU hours，全部约52–55 GPU hours。密集评估、checkpoint写盘和梯度单步诊断会增加成本；当前没有启动GPU计时，所以这不是新入口的实测时间。probe/消融独立执行且不包含在这个估算中。建议先以一个完整分支的结果时间校准总预算。
