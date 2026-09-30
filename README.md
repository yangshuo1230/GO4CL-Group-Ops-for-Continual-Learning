# GO4CL

Transformer continual-learning experiments on structured modular-addition tasks.

This repository implements the initial engineering framework for the research plan:
task specs with controlled A/B overlaps, fixed residue-pair datasets, a small
decoder-only Transformer, and training protocols (A-only / B-only / joint /
interleaved / sequential).

## Setup (uv)

```bash
# install uv if needed: curl -LsSf https://astral.sh/uv/install.sh | sh
cd GO4CL
uv sync
```

Uses `torch==2.6.0+cu124` (matches driver CUDA 12.4 on this cluster).
## Quick start

```bash
# engineering smoke checks (data split, overfit, short train, joint/sequential wiring)
uv run go4cl smoke --out runs/smoke --quick

# generate a fixed A/B dataset
uv run go4cl generate-data \
  --out data/s1_o1_m1_seed0 \
  --rho-slot 1 --rho-operand 1 --rho-mod 1 \
  --task-seed 0 --data-seed 0

# train a protocol
uv run go4cl train \
  --data data/s1_o1_m1_seed0 \
  --out runs/a_only \
  --protocol a_only \
  --steps 1000
```

## Layout

```text
src/go4cl/
  constants.py          # vocab, moduli, sequence layout
  tasks/                # Operation, TaskSpec, A/B overlap construction
  data/                 # residue-pair splits, fixed datasets, loaders
  model/                # decoder-only causal Transformer (3×64 default)
  train/                # training loop + protocols
  metrics/              # loss / accuracy / forgetting
  analysis/             # (stub) Fourier / probes / patching
  scripts/              # CLI implementations
configs/                # default + smoke YAML
tests/                  # unit tests for splits, relations, model
```

## Design notes (aligned with the plan)

- Input: 8 digits + `<TASK>` + `<Q_k>` → predict `(x_i + x_j) mod p` (31-way head).
- Residues are split at the unordered pair level so aliases cannot leak across splits.
- Ops form a perfect matching on the 8 positions; A/B overlaps are controlled on
  slot / operand / modulus factors in `{0, 0.5, 1}`.
- Scientific experiments should only start after smoke checks pass.

## Tests

```bash
uv run pytest -q
```
