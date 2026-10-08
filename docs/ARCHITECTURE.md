# Architecture

One-page map of GO4CL after the structure refactor. Science plan: `RESEARCH_EXPERIMENT_PLAN.md`. Progress: `EXPERIMENT_PROGRESS.md`.

## Layers (allowed dependencies point downward)

```text
CLI (go4cl.cli) / scripts/*.sh
        │
phases/                 experiment grids, GPU pool, CLI adapters
        │
train/                  protocols, loaders, train loop
analysis/               Fourier, probes, interventions, pipelines
        │
data/ + tasks/ + model/
```

- `analysis` must not import `phases`.
- `scripts/phase1/*.py` are thin wrappers around `analysis.pipelines`.
- Each run's `config_resolved.json` is the record of what actually executed.
- Locked CLI defaults live in `go4cl.defaults`.

## Data flow

1. `tasks/` defines operations (`OperationKey` = task / latent / slot).
2. `data/` splits residue pairs, builds packed contexts, writes manifests.
3. Packed **train** is an online stream (query examples per step).
4. Packed **val/test** primary eval is `packed_id`; disk splits are nuisance controls.
5. Checkpoints are selected on **val** (macro op accuracy, loss tie-break). Test is report-only.

## Training flow

`train/protocols.run_protocol` initializes model + loaders, then dispatches:

- `run_single_task` (`a_only` / `b_only`)
- `run_joint` / `run_interleaved`
- `run_sequential_ab` / `run_sequential_ab_replay` / `run_sequential_ba` / `run_continued_control`

Sequential protocols **preserve** the optimizer and continue the global step.
`sequential_ab` phase B is B-only. `sequential_ab_replay` mixes
`PHASE2.sequential_ab_replay_ratio` (default 0.1) packed A queries into each
phase-B batch (`MixedPackedReplayLoader`); it is opt-in via `--protocols`.

`--fixed-a` makes Task A depend only on `task_seed` (same A across ρ cells;
dataset tags get `_fixedA`). `--share-a` (requires `--fixed-a`) trains one
shared A per `(task_seed, model_seed, size)`, then sequential jobs load that
`θ_A` and only run phase B.

## Analysis flow

`analysis.context.AnalysisContext` resolves `data_dir`, checkpoint, operations,
and per-op datasets (rebuilds packed train via `build_analysis_dataset` if empty).

Pipelines (also `go4cl analyze <name>`):

- `single-op` — 1A-mech
- `multi-op` — 1C comparative mechanisms
- `causal-detail`, `transplant`, `attention-swap`, `steering`,
  `task-token-edit`, `unembed-fourier`

`scripts/phase1/*.py` remain compatible wrappers around the same pipelines.

## Locked protocol

- Packed **val/test** primary metric is `packed_id`; disk splits are nuisance only.
- Checkpoint on **val** macro-op accuracy (loss tie-break). Sequential keeps the optimizer.
- 1B / phase 2 default `weight_decay=0.3`, `batch_size=8192`, `steps=100000`.
- At `wd≥0.5`, p=23 can sit at \(\approx 1/23\) (query-only basin: head guesses uniformly, trunk stops reading operands). Same seed at `wd=0.3` groks all four ops by ~20k. Data/labels were intact; this is path-dependent, not a broken split.
