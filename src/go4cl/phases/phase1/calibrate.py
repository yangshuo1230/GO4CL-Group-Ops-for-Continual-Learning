"""Phase 1A prelude: grokking regime calibration on a medium modulus.

Sweeps split hyperparams × weight decay (and optional steps), with a fixed
single-op task. Lock one training config before the full 8-modulus scan.

Split modes (exactly one):
  --train-fracs                           fixed train/val/test ratios (default)
  --n-train-pairs / --n-train-pairs-grid  fixed train residue-pair count
Default: --train-fracs 0.4 0.6 0.8.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from go4cl.constants import PRIMES
from go4cl.phases.common import (
    TrainJob,
    assign_gpus,
    resolve_gpus,
    run_job_pool,
    stamp,
    write_report,
)
from go4cl.phases.phase1.data import DEFAULT_BATCH_SIZE, prepare_single_op_dataset


# Medium prime from the plan set {19,23,29,31,37,41,43,47}
DEFAULT_CALIB_MODULUS = 31
DEFAULT_TRAIN_FRACS: tuple[float, ...] = (0.4, 0.6, 0.8)
DEFAULT_WEIGHT_DECAYS: tuple[float, ...] = (0.1, 0.3, 1.0)


def _split_tag(meta: dict[str, Any]) -> str:
    if meta["split_mode"] == "n_train_pairs":
        return f"ntp{meta['n_train_pairs']}"
    return f"tr{meta['train_frac']:g}"


def run_calibrate(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    modulus = int(args.modulus)
    if modulus not in PRIMES:
        print(f"[warn] modulus {modulus} not in PRIMES={PRIMES}; continuing anyway")

    train_fracs = list(args.train_fracs) if args.train_fracs is not None else None
    if args.n_train_pairs_grid is not None:
        n_train_pairs_list = [int(x) for x in args.n_train_pairs_grid]
    elif args.n_train_pairs is not None:
        n_train_pairs_list = [int(args.n_train_pairs)]
    else:
        n_train_pairs_list = None

    if train_fracs is not None and n_train_pairs_list is not None:
        raise SystemExit(
            "pass only one split mode: --train-fracs  OR  "
            "--n-train-pairs / --n-train-pairs-grid"
        )
    if train_fracs is None and n_train_pairs_list is None:
        train_fracs = list(DEFAULT_TRAIN_FRACS)

    weight_decays = list(args.weight_decays)
    steps_grid = list(args.steps_grid)
    model_seeds = list(args.model_seeds)
    gpus = resolve_gpus(args.gpus)
    workers_per_gpu = max(int(args.workers_per_gpu), 1)
    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE))

    print(f"[phase1/calibrate] out={out_root}")
    print(f"[phase1/calibrate] modulus={modulus} (single op)")
    if n_train_pairs_list is not None:
        print(f"[phase1/calibrate] n_train_pairs={n_train_pairs_list}")
    else:
        print(f"[phase1/calibrate] train_fracs={train_fracs}")
    print(f"[phase1/calibrate] weight_decays={weight_decays}")
    print(f"[phase1/calibrate] steps_grid={steps_grid}")
    print(
        f"[phase1/calibrate] aliases={args.n_aliases} "
        f"lr={args.lr} batch_size={batch_size} "
        f"(relevant-only residue split; nuisance positions ~ U{{0..63}})"
    )

    metas: list[dict[str, Any]] = []
    if n_train_pairs_list is not None:
        for ntp in n_train_pairs_list:
            metas.append(
                prepare_single_op_dataset(
                    out_root,
                    modulus=modulus,
                    n_train_pairs=ntp,
                    n_aliases=args.n_aliases,
                    task_seed=args.task_seed,
                    data_seed=args.data_seed,
                )
            )
    else:
        assert train_fracs is not None
        for train_frac in train_fracs:
            metas.append(
                prepare_single_op_dataset(
                    out_root,
                    modulus=modulus,
                    train_frac=train_frac,
                    n_aliases=args.n_aliases,
                    task_seed=args.task_seed,
                    data_seed=args.data_seed,
                )
            )

    wandb_group = args.wandb_group or f"p1_calibrate_p{modulus}_{stamp()}"
    jobs: list[TrainJob] = []
    for meta in metas:
        for wd in weight_decays:
            for steps in steps_grid:
                for model_seed in model_seeds:
                    job_id = (
                        f"{meta['tag']}__wd{wd:g}_steps{steps}__ms{model_seed}"
                    )
                    jobs.append(
                        TrainJob(
                            job_id=job_id,
                            data_dir=meta["data_dir"],
                            out_dir=str(out_root / "runs" / job_id),
                            gpu=0,  # filled below
                            steps=int(steps),
                            lr=args.lr,
                            weight_decay=float(wd),
                            d_model=args.d_model,
                            n_layers=args.n_layers,
                            model_seed=int(model_seed),
                            wandb_project=args.wandb_project,
                            wandb_group=wandb_group,
                            wandb_mode=args.wandb_mode,
                            wandb_tags=(
                                "phase1",
                                "calibrate",
                                f"p{modulus}",
                                _split_tag(meta),
                                f"wd{wd:g}",
                            ),
                            wandb_config={
                                "phase": "phase1",
                                "step": "calibrate",
                                "modulus": modulus,
                                "split_mode": meta["split_mode"],
                                "train_frac": meta["train_frac"],
                                "n_train_pairs": meta["n_train_pairs"],
                                "n_train_pairs_realized": meta["n_train_pairs_realized"],
                                "ratios": meta["ratios"],
                                "n_aliases": meta["n_aliases"],
                                "n_train_a": meta["n_train_a"],
                                "weight_decay": float(wd),
                                "steps": int(steps),
                                "task_seed": args.task_seed,
                                "data_seed": args.data_seed,
                                "single_op": True,
                                "relevant_only_split": True,
                            },
                            batch_size=batch_size,
                        )
                    )

    gpu_assign = assign_gpus(len(jobs), gpus, workers_per_gpu)
    jobs = [
        TrainJob(**{**job.__dict__, "gpu": gpu})
        for job, gpu in zip(jobs, gpu_assign)
    ]

    print(f"[phase1/calibrate] launching {len(jobs)} jobs  group={wandb_group}")
    for meta in metas:
        print(
            f"  {_split_tag(meta):<10}  A_train={meta['n_train_a']:<6} "
            f"A_val={meta['n_val_a']:<5} A_test={meta['n_test_a']}"
        )

    results = run_job_pool(jobs, gpus=gpus, workers_per_gpu=workers_per_gpu)
    write_report(
        out_root,
        phase="phase1",
        step="calibrate",
        config={
            "modulus": modulus,
            "train_fracs": train_fracs,
            "n_train_pairs": n_train_pairs_list,
            "weight_decays": weight_decays,
            "steps_grid": steps_grid,
            "n_aliases": args.n_aliases,
            "lr": args.lr,
            "batch_size": batch_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "task_seed": args.task_seed,
            "data_seed": args.data_seed,
            "model_seeds": model_seeds,
            "gpus": gpus,
            "workers_per_gpu": workers_per_gpu,
            "wandb_project": args.wandb_project,
            "wandb_group": wandb_group,
            "full_batch": False,
            "relevant_only_split": True,
        },
        datasets=metas,
        results=results,
        csv_fields=[
            "split_mode",
            "train_frac",
            "n_train_pairs",
            "weight_decay",
            "steps",
            "n_train_a",
            "model_seed",
            "status",
            "A_val_acc",
            "A_test_acc",
            "elapsed_sec",
            "wandb_url",
            "job_id",
        ],
        csv_row_fn=lambda r: [
            (r.get("wandb_config") or {}).get("split_mode", ""),
            (r.get("wandb_config") or {}).get("train_frac", ""),
            (r.get("wandb_config") or {}).get("n_train_pairs", ""),
            r.get("weight_decay", ""),
            r.get("steps", ""),
            (r.get("wandb_config") or {}).get("n_train_a", ""),
            r.get("model_seed", ""),
            r.get("status", ""),
            (r.get("metrics") or {}).get("A_val_acc", ""),
            (r.get("metrics") or {}).get("A_test_acc", ""),
            r.get("elapsed_sec", ""),
            r.get("wandb_url", "") or "",
            r.get("job_id", ""),
        ],
    )


def add_calibrate_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=f"runs/phase1/calibrate/{stamp()}",
    )
    parser.add_argument(
        "--modulus",
        type=int,
        default=DEFAULT_CALIB_MODULUS,
        help=f"Medium modulus for calibration (default {DEFAULT_CALIB_MODULUS})",
    )
    parser.add_argument(
        "--train-fracs",
        type=float,
        nargs="+",
        default=None,
        help="Train residue-pair fractions to sweep (default 0.4 0.6 0.8 if "
        "neither split mode is set; mutually exclusive with --n-train-pairs*)",
    )
    parser.add_argument(
        "--n-train-pairs",
        type=int,
        default=None,
        help="Optional: fixed train residue-pair count",
    )
    parser.add_argument(
        "--n-train-pairs-grid",
        type=int,
        nargs="+",
        default=None,
        help="Optional: sweep multiple fixed train residue-pair counts",
    )
    parser.add_argument(
        "--weight-decays",
        type=float,
        nargs="+",
        default=list(DEFAULT_WEIGHT_DECAYS),
        help="AdamW weight decay values to sweep",
    )
    parser.add_argument(
        "--steps-grid",
        type=int,
        nargs="+",
        default=[100_000],
        help="Training step budgets to sweep (default: single 100k)",
    )
    parser.add_argument(
        "--n-aliases",
        type=int,
        default=16,
        help="Raw aliases per residue pair on relevant operand positions",
    )
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"Mini-batch size (default {DEFAULT_BATCH_SIZE})",
    )
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--task-seed", type=int, default=0)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument("--wandb-project", type=str, default="go4cl")
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
