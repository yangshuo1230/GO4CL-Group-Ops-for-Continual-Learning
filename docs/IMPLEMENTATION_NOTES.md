# IMPLEMENTATION_NOTES — protocol fixes (R1+)

Date: 2026-10-02  
Spec: `docs/REFACTOR_R1_SPEC.md`（数值：𝒫 含 53，53-class head）

## Done

| Area | Change |
|------|--------|
| OperationKey | `tasks/spec.py` |
| Example + disk | `latent_id` / `task_id`；npz `latent_ids` |
| ContextBuilder | `data/context.py` + `eval_contexts.py` + analysis dataset |
| Packed loader | ContextBuilder；batch 含 `latent_ids` |
| **packed_id 主评接线** | `_task_loaders`：packed 时 val/test=`packed_id`；`*_nuisance` 对照；`train_eval`/`iid` 供事件 |
| Metrics | macro / by_operation / margin / NCE |
| CheckpointSelector | macro acc + loss 平局；仅 `A_val`/`B_val` 选 ckpt |
| Events | t_mem / t_gen / t_iid 钩进 `train_steps`；`first_stable_threshold.pt` |
| Steering | train/val means → test |
| Mechanisms | 按 op 过滤；同模 routing；op 键矩阵；analysis 重建 train |
| Seeds | `sampler_seed` |
| Manifest | v2 字段 + `assert_manifest_compatible` |
| CSV | `write_report` 用 `csv.DictWriter` |
| Fourier controls | random / norm-matched / magnitude（digit-emb 空间） |
| Phase2 prep | `train_segment(preserve\|reset)`；sequential 默认 preserve + 连续 step |
| Docs | RESEARCH §2.2/2.3 对齐 53 类 + packed protocol |
| config | `config_resolved.json` 每 run 写出 |
| Tests | `tests/test_r1_fixes.py` |

## Phase 2 (2026-10-04)

| Area | Change |
|------|--------|
| Data | `phases/phase2/data.py` packed A/B manifests, optional `swap` |
| Protocols | packed 50/50 joint; interleaved via `train_steps`; sequential `switch_on=fixed\|t_mem\|t_gen` |
| Metrics | `metrics/continual.py` forgetting / jump / exposure AUC / grok order / forward transfer |
| CLI | `go4cl phase2 protocols\|relation-matrix\|capacity`（`--dry-run` 只落盘作业表） |

Fourier/probe-before-behavior is still phase 3.

### Finding: late-grokking modulus and weight decay (2026-10-04)

Full-overlap seed 0 uses moduli `{41, 23, 37, 53}`. Packed train labels and residue-pair splits are intact: every class is covered, labels match \((x_i+x_j)\bmod p\).

At `wd=0.5` / `0.8`, A-only or B-only can sit at train/iid/val/test \(\approx 1/23\) on **p=23** for 100k steps while the other three ops are already \(\approx 1\). Init gradients for p=23 are *largest*. After the other ops grok (~20k), query attention on p=23 stays near uniform (operand mass \(\approx 0.18\)), \(\lVert\partial L/\partial \mathrm{emb}_{\mathrm{operands}}\rVert\approx 0\), and p=23’s share of the mean-CE parameter gradient drops to \(\sim 0.4\%\). The head has learned “this query is 23-way chance”; the trunk no longer reads the two operands. The same seed at `wd=0.3` groks all four ops by ~20k (train acc 1.0, `t_gen` fires). B-only at `wd=0.8` later grokked p=23 around 60k, so the basin is path-dependent, not a broken backward pass.

**Lock:** 1B and phase 2 default `weight_decay=0.3`. Old 1B grid `multi_op/20261003_132221` remains `wd=0.5`.

Full 2A/2B/2D grids have not been trained.

## Still deferred（需实验时间 / 大重构）

- §4 目录重命名搬家
- mean/resample activation ablation（完整版）
- 正式 **新 𝒫 + packed_id + 100k** 1B 网格重跑
- 1C 电路移植实验（对照机理已在 `mechanisms/20261004_1b_132221/` 完成）
- 1A 对 p=53 的 scan（旧 1A 为 47 类 / 含 19）

## How to verify

```bash
uv run pytest tests/ -q
```
