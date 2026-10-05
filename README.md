# GO4CL

Transformer continual-learning experiments on structured modular-addition tasks.

- [`docs/RESEARCH_EXPERIMENT_PLAN.md`](docs/RESEARCH_EXPERIMENT_PLAN.md) — overall science plan
- [`docs/EXPERIMENT_PROGRESS.md`](docs/EXPERIMENT_PROGRESS.md) — **progress table**
- [`docs/PHASE_STEP_GUIDE.md`](docs/PHASE_STEP_GUIDE.md) — what each step does
- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — code layout and data/train/analysis flow
- [`docs/IMPLEMENTATION_NOTES.md`](docs/IMPLEMENTATION_NOTES.md) — locked protocol details

## Setup

Requires **Python 3.11** (see `.python-version`) and a **CUDA 12.4** driver/runtime.

Pinned GPU stack (also listed in `pyproject.toml`):

| Package | Version | Notes |
|---------|---------|--------|
| `torch` | `2.6.0+cu124` | from the PyTorch cu124 wheel index |
| `triton` | `3.2.0` | required by CUDA torch; wheels from PyPI |
| `setuptools` | `>=70` | **runtime** import of Triton (not just a build tool) |

```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
cd GO4CL-Group-Ops-for-Continual-Learning
uv sync
uv run python -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.is_available())"
uv run wandb login
```

If `import triton` raises `ModuleNotFoundError: setuptools`, the venv is stale — run `uv sync` again.

Scratch files go under **`.tmp/`** and workspace **`.cache/uv`**. `uv run go4cl …`
and `scripts/phase1/*.sh` set this automatically.

## Task gist

- Input digits: **0–63** (64 tokens); plus `<TASK_A/B>` and `<Q_0..3>` → vocab **70**.
- Sequence: 8 digits + task + query → predict \((x_i+x_j)\bmod p\) (**53-way** head).
- Phase 1A: residue-pair split **only on relevant operand positions** (default `train_frac`, or optional `n_train_pairs`) × `n_aliases`; other digits ~\(U\{0..63\}\); train with **batch 2048 + replacement**.
- Phase 1B / 2: packed multi-query train (`batch_size` counts **query examples**, not packed contexts).
- Moduli: \(\mathcal P=\{23,29,31,37,41,43,47,53\}\).

Locked CLI defaults: `src/go4cl/defaults.py`. Each run writes `config_resolved.json`.

## Layout

```text
src/go4cl/
  cli.py defaults.py constants.py
  tasks/ data/ model/ train/ analysis/ phases/
scripts/phase1/*.sh     # launchers
scripts/phase1/*.py     # thin wrappers around analysis.pipelines
runs/phaseN/<step>/     # outputs (gitignored)
docs/ tests/
```

## Phase CLI

```bash
# Phase 1A-0 — calibrate on p=31
bash scripts/phase1/calibrate.sh --gpus 4,5 --workers-per-gpu 2

# Phase 1A — modulus scan
bash scripts/phase1/scan_moduli.sh \
  --train-frac 0.8 --weight-decay 0.3 --steps 100000 \
  --gpus 0,1 --workers-per-gpu 4

# Phase 1B / 1C / 2
uv run go4cl phase1 multi-op --help
uv run go4cl phase1 mechanisms --help
uv run go4cl phase2 protocols --help

# Analysis (also: python scripts/phase1/<name>.py)
uv run go4cl analyze transplant --help
```

Phase 3 commands exist but are **not implemented**.

Outputs land under `runs/phaseN/<step>/`.

## Engineering

```bash
uv run go4cl smoke --quick
uv run go4cl generate-data --out data/demo --rho-slot 1 --rho-operand 1 --rho-mod 1
uv run go4cl train --data data/demo --out runs/tmp --protocol a_only --steps 1000
uv run pytest -q
```
