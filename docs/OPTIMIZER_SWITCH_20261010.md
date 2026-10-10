# 优化器切换对照

服务器项目：`/mnt/ningxuefei/yangshuo/GO4CL-Group-Ops-for-Continual-Learning`

入口：`scripts/phase2/optimizer_switch.sh`；实现：`scripts/phase2/optimizer_switch.py`。
现有训练代码未改动。不重训 A，不写 W&B。默认只生成计划，显式 `--execute` 才训练。

## 实验矩阵

沿用已完成的四种 A/B 任务关系和两个模型种子，每个源 A checkpoint 分六个分支：

| 训练目标 | fresh | inherit_A |
|---|---|---|
| A→A：仅继续 A | 清空 AdamW 状态 | 完整继承源 A AdamW 状态 |
| A→B：仅 B | 清空 AdamW 状态 | 完整继承源 A AdamW 状态 |
| A→B＋10% replay | 清空 AdamW 状态 | 完整继承源 A AdamW 状态 |

共 48 个新分支，默认各 100,000 步，与原套件预算一致。fresh B/replay 也重新运行，以获得严格匹配的早期观测。
两种策略初始权重完全相同；学习率 .001、betas (.9,.98)、weight decay .3、clip 1、batch 8192 保持不变。
inherit_A 同时继承 exp_avg、exp_avg_sq 和 step，不单独区分三者作用；若后续需要定位，可再增加选择性重置。

A→A 默认继续源 checkpoint 的 A 采样 RNG，并将新增曝光量计数归零；两种优化器策略使用完全相同的继续采样流。
B 和 replay 使用原套件的重启采样协议，策略之间批次一致。所有分支使用原来的任务定义、数据划分和固定评估集。

## 执行

```bash
cd /mnt/ningxuefei/yangshuo/GO4CL-Group-Ops-for-Continual-Learning
GPUS=0,1,2,3,4,5,6,7 bash scripts/phase2/optimizer_switch.sh --execute
```

先检查计划：去掉 `--execute` 或加 `--dry-run`。
限制 GPU：例如 `GPUS=0,1`，每张 GPU 一个顺序 worker。
先筛查 20,000 步：必须换输出目录，避免与默认预算混合：

```bash
RUN_ROOT=runs/phase2/optimizer_switch_20k_20261010 GPUS=0,1 \
  bash scripts/phase2/optimizer_switch.sh --max-steps 20000 --execute
```

可以用 `--kinds same_mod_diff_pos diff_mod_diff_pos` 只跑两个代表关系；也可用 `--objectives A B` 排除 replay。
默认种子 0、1；用 `--seeds 0 1` 显式指定。改变计划需另选 RUN_ROOT。已完成的分支自动跳过，未完成的从 latest.pt 恢复。
突然终止可能留下 `.running`；该锁不会自动清除。确认对应 worker 已退出后才能移除该分支锁并续跑。

## 保存和比较

输出：`runs/phase2/optimizer_switch_20261010`。每分支保存 resolved.json、data_manifest.json、history.json、result.json、stdout.log。
默认保存 88 个轨迹 checkpoint：原有 82 个加第 1、2、5、10、20、50 步；保存模型、优化器、采样/global RNG 和历史，支持续跑。
history 包含 A/B 逐运算准确率和损失、累计 A/B 曝光量，以及观测步的实际单步更新范数、裁剪前梯度范数与批次 hash。
注意 last_update 是观测点前一次更新，不是整个观测间隔的累计更新。
梯度诊断点：0、1、10、100、500、1000、5000、10000、20000、50000、100000；诊断复用冻结副本上的 A/B/mix 原始梯度和实际 AdamW 更新分析。

优先比较：

1. A→A fresh/inherit 的早期跌落，判断优化器重置本身造成的影响。
2. A→B fresh/inherit 的 A 遗忘和 B 学习速度，判断继承是否只是延迟 B 学习。
3. 同一优化器策略下 A→A 与 A→B 的差异，检查换任务带来的额外变化；不能把非线性准确率差直接当作可加的因果效应。
4. replay fresh/inherit 的最低点、恢复时间和稳定保留，检查先失效再恢复是否依赖优化器切换。
5. 后续机理分析应按关键行为事件和 B 达到相似性能的阶段比较，兼顾训练步数与数据曝光；本脚本不自动执行 probe/消融分析，也不自动绘图。

## 验证

服务器 dry-run 已验证 48 分支；6 个独立临时 CPU 小批量分支各运行 2 步，检查初始权重、批次匹配、动量/二阶矩/计数继承、梯度保存、续跑和完成跳过。
这些 CPU 小批量验证不是正式实验结果。没有启动正式训练或 GPU 任务。
