# GO4CL R1 修复规范（裁剪版）

相对附件 `CURSOR_REFACTOR_AND_FIX_SPEC` 的**可执行首轮**。数值以当前仓库为准；目录大搬家与 Phase 2 引擎不进本轮硬门槛。

## 0. 当前数值（真源：`constants.py`）

```text
𝒫 = {23, 29, 31, 37, 41, 43, 47, 53}
NUM_OUTPUT_CLASSES = 53
近邻对: (23,29), (31,37), (41,43), (47,53)
```

旧 19 / 47-class 产物一律 `legacy_v1`，只读；本轮协议 tag `protocol_version: 2`。

正式 1B 步数与 `EXPERIMENT_PROGRESS` 锁定一致：**100k**（验收后网格可先 smoke 再拉长，不以 50k 当正式口径）。

## 1. 强制约束（保留）

- 不杀进程、不改 `runs/`、不覆盖旧 manifest
- checkpoint / 组件选择只用 train/val；test 只在预选 ckpt 上评估
- 首轮只跑单测 + smoke，不启正式多 seed 网格

## 2. 本轮必须修的问题

| ID | 问题 | 本轮动作 |
|----|------|----------|
| 2.1 | packed train vs nuisance eval 混在一个 acc | `packed_id` 主评 + `nuisance_random` 对照 |
| 2.2 | `filter_by_modulus` 混同模数 op | `OperationKey` / `latent_id` 筛选 |
| 2.3 | micro best + 首次锁死 | macro_operation_accuracy + loss 平局打破 |
| 2.4 | packed 无 train → 1C 空 | `build_analysis_dataset` |
| 2.5 | steering 用 test 估 mean | train/val means → test intervene |
| 2.7 | model_seed 绑 sampler | 显式 `sampler_seed` |

**明确推迟：** §2.6 完整 Fourier 五对照、§2.8 Phase2 optimizer、§4 目录重命名 → R3/R4。

## 3. 定义钉死

### 3.1 OperationKey（报告主键）

```python
OperationKey(task_id, latent_id, slot)
```

- **报告键 = OperationKey**（序列化为 `task{task_id}/lat{latent_id}/slot{slot}`）
- `slot`、`modulus` 是属性，不是跨配置的全局主键
- 允许派生 `by_slot` / `by_modulus`；禁止用 modulus 识别具体操作

### 3.2 Evaluation contexts

```text
packed_id (primary)
  同 context 填入全部 op；
  target 用 target_split residue；
  distractors 默认 packed_distractor_split=train

nuisance_random (control)
  只固定 target op；其余 digit 位 ~U{0..63}
```

旧固定 eval（nuisance 风格）标 `legacy_v1`；新跑默认 primary=`packed_id`。

### 3.3 稳定事件

```text
t_mem: 固定 train-eval macro_operation_accuracy ≥ 0.99，连续 5 次 eval
t_gen: primary packed_id held-out macro ≥ 0.90，连续 5 次 eval
t_iid: 与 train 同分布的 replayable_online 重采样 macro ≥ 0.95，连续 5 次
       （context_mode=packed_id，全部 op 用 train pool；不是 nuisance_random）
first_stable_threshold.pt: 首次满足 t_gen 连续窗口结束时的权重
```

`t_mem` 禁止用随机 minibatch accuracy。

### 3.4 CheckpointSelector

```text
primary: maximize macro_operation_accuracy (val)
tie_break: minimize macro_operation_loss
min_delta: 0.0  # 平局走 loss；严格大于才换 accuracy
```

不得因首次 val micro=1 永久锁早期 ckpt。

### 3.5 种子

```yaml
task_seed / data_seed(=split_seed) / sampler_seed / model_seed / analysis_seed
```

paired `four_diff` vs `pair_same` 默认共享除任务组成外的全部种子。

## 4. 诊断模板数值（替换 SPEC §6）

```yaml
experiment: matched_count_scan
moduli: [23, 31, 47, 53]   # 覆盖小→大；勿写回 19
n_train_pairs: [64, 96, 128, 150]
model_seeds: [0, 1, 2]
split_seeds: [0, 1, 2]

# 输出头对照（不替换主 setting）
shared_53_way_head
p_way_head_or_valid_logit_mask
```

## 5. 首轮交付（相对原 §9 裁剪）

必交：

1. `docs/IMPLEMENTATION_NOTES.md`
2. `OperationKey` + Example/batch `latent_id`
3. `ContextBuilder` + packed_id / nuisance_random
4. per-op + macro metrics + CheckpointSelector（含 t_mem/t_gen；t_iid 接口就绪）
5. `build_analysis_dataset` + mechanisms 按 op 过滤
6. steering 分 split
7. `sampler_seed` 解耦
8. 对应单测；legacy manifest 只读加载

首轮不要求：目录搬家、Phase2 preserve/reset、完整 Fourier 五对照、正式重跑网格。

## 6. 验收（本轮）

- `runs/` 未改
- legacy manifest 可加载；新代码默认写 protocol v2 字段（若落盘）
- pair_same 两同模 op 有独立指标/筛选
- val macro + loss tie-break 选 ckpt
- model≠sampler seed 时可独立变化
- steering direction 不来自 test
- 新增测试通过
