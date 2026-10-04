# Architecture

One-page map of GO4CL after the structure refactor. Scientific protocol
details stay in `RESEARCH_EXPERIMENT_PLAN.md` and `IMPLEMENTATION_NOTES.md`.

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
- `run_sequential_ab` / `run_sequential_ba` / `run_continued_control`

Sequential protocols **preserve** the optimizer and continue the global step.

## Analysis flow

`analysis.context.AnalysisContext` resolves `data_dir`, checkpoint, operations,
and per-op datasets (rebuilds packed train via `build_analysis_dataset` if empty).

Pipelines (also `go4cl analyze <name>`):

- `single-op` — 1A-mech
- `multi-op` — 1C comparative mechanisms
- `causal-detail`, `transplant`, `attention-swap`, `steering`,
  `task-token-edit`, `unembed-fourier`

`scripts/phase1/*.py` remain compatible wrappers around the same pipelines.
