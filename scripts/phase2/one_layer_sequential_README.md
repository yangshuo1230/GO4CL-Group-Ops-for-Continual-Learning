# 单层、双任务顺序学习 sweep

本轮只准备并验证了代码与启动脚本，未启动 GPU 训练。

## 启动

服务器项目目录：

```bash
cd /mnt/ningxuefei/yangshuo/GO4CL-Group-Ops-for-Continual-Learning

# 生成/核对完整计划，不训练。
bash scripts/phase2/one_layer_sequential.sh --dry-run

# 真正启动：自行选择可用 GPU；每个 GPU 同时最多一个训练进程。
GPUS=0,1,2,3 bash scripts/phase2/one_layer_sequential.sh --execute
```

默认输出 `runs/phase2/one_layer_sequential_20261011`。修改配置后必须换输出目录：

```bash
GPUS=0,1 RUN_ROOT=runs/phase2/one_layer_sequential_custom \
  bash scripts/phase2/one_layer_sequential.sh --execute \
  --config configs/phase2/one_layer_sequential.yaml
```

同一配置重复执行会跳过已完成任务、从未完成任务的 `latest.pt` 恢复。计划或 checkpoint 的 fingerprint 不一致时拒绝覆盖。运行中的任务使用 `.running` 锁；异常终止遗留锁时需要先核实没有对应进程，再单独处理，脚本不会自动删除锁或旧数据。

## 完整矩阵

- 每任务运算数：1、2。
- 模型：一层、2 heads、ReLU、dropout=0、输入输出权重不绑定。
- 窄模型：d_model=64，d_head=32，d_mlp=256。
- 宽模型：d_model=128，d_head=64，d_mlp=512。
- 种子：0、1；数据/采样/评估种子与 mechanism_suite 相同。
- B阶段：无 replay、请求10% A replay；不包含 B-only、joint、组件干预或 replay 比例扫描。
- 每个运算数×宽度×种子只预训练一次固定 A，8份A checkpoint各分支8种B训练，共8个source+64个branch=72个任务。

## 运算定义

保留8个数字位置和原有TASK/QUERY词表；数字token范围仍为0..63，TASK在位置8，QUERY在位置9。每个运算的QUERY分别产生一条样本，不将两个输出改成一个新的多输出网络。

同模数/同位置的关系按 **latent_id** 比较；双运算额外交换输出槽位。因此双运算的“同模数同位置”仍有不同的槽位函数，而单运算该条件仅TASK不同、目标函数相同，是基准条件。

单运算固定 A：`lat0: (digit0+digit1) mod23 -> Q0`。

| B条件 | 模数 | 操作数位置 | 槽位 |
|---|---:|---|---|
| same_mod_same_pos | 23 | (0,1) | Q0 |
| same_mod_diff_pos | 23 | (0,2) | Q0 |
| diff_mod_same_pos | 29 | (0,1) | Q0 |
| diff_mod_diff_pos | 29 | (0,2) | Q0 |

双运算固定 A：`lat0: (0,1), p23 -> Q1`，`lat1: (2,3), p41 -> Q0`。

| B条件 | lat0（Q0） | lat1（Q1） |
|---|---|---|
| same_mod_same_pos | (0,1), p23 | (2,3), p41 |
| same_mod_diff_pos | (0,2), p23 | (1,3), p41 |
| diff_mod_same_pos | (0,1), p29 | (2,3), p43 |
| diff_mod_diff_pos | (0,2), p29 | (1,3), p43 |

“不同位置”沿用原 mechanism_suite 的交叉配对；单个对应运算保留一个共同操作数位置，指操作数**位置对不同**，并非位置集合完全不相交。

## 训练与数据设置

用户指定 A/B各50000个optimizer更新；其余沿用服务器实际 `configs/phase2/mechanism_suite.yaml`：

| 项目 | 设置 |
|---|---|
| batch | 8192个query样本/步；不会随n_ops减半 |
| 优化器 | AdamW，lr=0.001，weight_decay=0.3，betas=(0.9,0.98) |
| 梯度裁剪 | 1.0 |
| A→B切换 | 读取固定预算A-final，fresh AdamW，不继承A moments |
| 数据划分 | 每模数无序residue pair按类别分层，60/20/20；相同模数共享划分 |
| 数据种子 | 20261009 |
| 采样种子 | 1701；B RNG独立，沿用偏移10007 |
| 评估种子 | 2701；每运算512条，val用于门槛，最终另评估test |
| checkpoint/val | 0–2000步每100；2000–10000步每500；之后每2000；另含诊断步及最终50000 |
| 梯度诊断 | 0、100、500、1000、5000、10000、20000、50000；固定512条诊断batch |
| 编译 | 原有torch.compile路径，失败时沿用原trainer的eager回退 |
| A mastery门槛 | 最后5次val中每个A运算均≥0.95；保持原配置，不提前结束A阶段 |

如果A没通过原有mastery门槛，对应B分支标记 `blocked_source_not_mastered`，不会悄悄从未学会A的模型开始。最终launcher会汇总并报告被阻止的分支；不会自动放宽门槛或追加训练。

每阶段固定query预算：50000×8192=409600000条。单运算每运算得到全部该任务query；双运算每运算得到该任务query的一半。10% replay通过整数pack舍入：单运算每步A=819、B=7373；双运算每步A=820、B=7372。实际比例及A/B曝光量写入结果，不把它们误标为严格10.000%。无replay与replay同更新预算，B曝光量不同。

训练复用ContextBuilder：单运算的其余6位置随机，双运算的其余4位置为0；packed val/test也沿用该规则。因此两种运算数量之间不是只有运算数量一个变量的严格匹配比较。未改动数据生成器，也没有将训练数据改成独立均匀整数对。后续机理分析可另用已有nuisance_random评估检验对背景数字的依赖。

## 复用与验证

复用 `mechanism_suite.run_job`、原ModelConfig/ModularTransformer、ContextBuilder、residue划分、评估、AdamW、梯度诊断、checkpoint和续训代码。唯一对原代码的修改是Stream的pack数和曝光计数从固定4改为n_ops，并检查A/B运算数一致；四运算的采样序列逐项回归验证不变。

新文件：`one_layer_sequential.py`、对应YAML、启动shell、13项测试。另执行了4项旧回归测试，覆盖原四运算stream、source/branches、exact resume/mastery gate及诊断隔离。所有验证仅用CPU短步测试与dry-run，不启动正式训练。

目录结构：

```text
RUN_ROOT/plan.json
RUN_ROOT/ops1/d64/seed0/source/A/{resolved.json,history.json,result.json,checkpoints/}
RUN_ROOT/ops1/d64/seed0/main/same_mod_same_pos/no_replay/...
RUN_ROOT/ops1/d64/seed0/main/same_mod_same_pos/replay_0.1/...
RUN_ROOT/ops2/d128/seed1/...
```

每个任务保存完整A/B定义、真实模型配置、数据划分、per-operation行为记录、A/B曝光量、训练日志和checkpoint。其他历史实验目录不被改写。
