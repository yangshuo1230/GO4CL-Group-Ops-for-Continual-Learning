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

## Still deferred（需实验时间 / 大重构）

- §4 目录重命名搬家
- mean/resample activation ablation（完整版）
- 正式 **新 𝒫 + packed_id + 100k** 1B 网格重跑
- 1C 在新 1B ckpt 上重做 + `pair_same` 对照 + 移植实验
- 1A 对 p=53 的 scan（旧 1A 为 47 类 / 含 19）

## How to verify

```bash
uv run pytest tests/ -q
```
