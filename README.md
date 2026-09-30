# GO4CL

Transformer continual-learning experiments on structured modular-addition tasks.
Plan:

- [`docs/RESEARCH_EXPERIMENT_PLAN.md`](docs/RESEARCH_EXPERIMENT_PLAN.md) — 总体科学计划
- [`docs/EXPERIMENT_PROGRESS.md`](docs/EXPERIMENT_PROGRESS.md) — **分阶段进度表**（标注状态）
- [`docs/PHASE_STEP_GUIDE.md`](docs/PHASE_STEP_GUIDE.md) — 每个阶段/步骤的具体内容

## Setup

```bash
cd GO4CL
uv sync
uv run wandb login
```

## Task gist

- Input digits: **0–63** (64 tokens); plus `<TASK_A/B>` and `<Q_0..3>` → vocab **70**.
- Sequence: 8 digits + task + query → predict \((x_i+x_j)\bmod p\) (47-way head).
- Phase 1A: residue-pair split **only on relevant operand positions** (default `train_frac`, or optional `n_train_pairs`) × `n_aliases`; other digits ~\(U\{0..63\}\); train with **batch 2048 + replacement**.
- Dual-task / general path still uses residue-pair splits with `n_aliases` × `n_nuisance`.
- Moduli: \(\mathcal P=\{19,23,29,31,37,41,43,47\}\).

## Layout

```text
scripts/phase1/          # launchers for stage-1 steps
configs/phase1/          # default hyperparams
src/go4cl/phases/        # phase1 / phase2 / phase3 code
runs/phase1/calibrate/   # outputs (gitignored)
runs/phase1/scan_moduli/
docs/
tests/
```

进度与步骤说明：

```text
docs/EXPERIMENT_PROGRESS.md   # 勾选/标注进度
docs/PHASE_STEP_GUIDE.md      # 各步骤做什么
docs/RESEARCH_EXPERIMENT_PLAN.md
```

## Phase CLI

```bash
# Phase 1A prelude — calibrate on p=31
bash scripts/phase1/calibrate.sh --gpus 4,5 --workers-per-gpu 2

# Phase 1A — modulus scan (after locking train config)
bash scripts/phase1/scan_moduli.sh \
  --train-frac 0.8 --weight-decay 0.3 --steps 100000 \
  --gpus 0,1 --workers-per-gpu 4
# optional: --n-train-pairs 150 ...

# Stubs: phase1 multi-op | mechanisms ; phase2 * ; phase3 *
uv run go4cl phase1 --help
```

Outputs land under `runs/phaseN/<step>/`.

## Engineering

```bash
uv run go4cl smoke --quick
uv run go4cl generate-data --out data/demo --rho-slot 1 --rho-operand 1 --rho-mod 1
uv run go4cl train --data data/demo --out runs/tmp --protocol a_only --steps 1000
uv run pytest -q
```
