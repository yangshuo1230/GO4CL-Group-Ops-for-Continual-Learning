"""Phase 1A: single-operation modulus scan across all primes.

Uses one locked training config (from calibration) for every modulus.

Split modes (exactly one):
  --train-frac F      fixed train/val/test ratios (default 0.8)
  --n-train-pairs N   fixed train residue-pair count (optional)
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

DEFAULT_TRAIN_FRAC = 0.8


def run_scan_moduli(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    moduli = list(args.moduli) if args.moduli else list(PRIMES)
    model_seeds = list(args.model_seeds)
    gpus = resolve_gpus(args.gpus)
    workers_per_gpu = max(int(args.workers_per_gpu), 1)
    weight_decay = float(args.weight_decay)
    steps = int(args.steps)
    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE))

    train_frac = args.train_frac
    n_train_pairs = args.n_train_pairs
    if train_frac is not None and n_train_pairs is not None:
        raise SystemExit("pass only one of --train-frac or --n-train-pairs")
    if train_frac is None and n_train_pairs is None:
        train_frac = DEFAULT_TRAIN_FRAC

    print(f"[phase1/scan-moduli] out={out_root}")
    print(f"[phase1/scan-moduli] moduli={moduli}")
    if n_train_pairs is not None:
        split_msg = f"n_train_pairs={n_train_pairs}"
    else:
        split_msg = f"train_frac={train_frac}"
    print(
        f"[phase1/scan-moduli] locked cfg: {split_msg} "
        f"wd={weight_decay} steps={steps} aliases={args.n_aliases} "
        f"lr={args.lr} batch_size={batch_size} "
        f"(relevant-only residue split; nuisance positions ~ U{{0..63}})"
    )

    if n_train_pairs is not None:
        metas = [
            prepare_single_op_dataset(
                out_root,
                modulus=p,
                n_train_pairs=int(n_train_pairs),
                n_aliases=args.n_aliases,
                task_seed=args.task_seed,
                data_seed=args.data_seed,
            )
            for p in moduli
        ]
    else:
        metas = [
            prepare_single_op_dataset(
                out_root,
                modulus=p,
                train_frac=float(train_frac),
                n_aliases=args.n_aliases,
                task_seed=args.task_seed,
                data_seed=args.data_seed,
            )
            for p in moduli
        ]

    wandb_group = args.wandb_group or f"p1_scan_moduli_{stamp()}"
    jobs: list[TrainJob] = []
    for meta in metas:
        for model_seed in model_seeds:
            job_id = f"{meta['tag']}__wd{weight_decay:g}_steps{steps}__ms{model_seed}"
            jobs.append(
                TrainJob(
                    job_id=job_id,
                    data_dir=meta["data_dir"],
                    out_dir=str(out_root / "runs" / job_id),
                    gpu=0,
                    steps=steps,
                    lr=args.lr,
                    weight_decay=weight_decay,
                    d_model=args.d_model,
                    n_layers=args.n_layers,
                    model_seed=int(model_seed),
                    wandb_project=args.wandb_project,
                    wandb_group=wandb_group,
                    wandb_mode=args.wandb_mode,
                    wandb_tags=(
                        "phase1",
                        "scan-moduli",
                        f"p{meta['modulus']}",
                    ),
                    wandb_config={
                        "phase": "phase1",
                        "step": "scan-moduli",
                        "modulus": meta["modulus"],
                        "split_mode": meta["split_mode"],
                        "train_frac": meta["train_frac"],
                        "n_train_pairs": meta["n_train_pairs"],
                        "n_train_pairs_realized": meta["n_train_pairs_realized"],
                        "ratios": meta["ratios"],
                        "n_aliases": meta["n_aliases"],
                        "n_train_a": meta["n_train_a"],
                        "weight_decay": weight_decay,
                        "steps": steps,
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

    print(f"[phase1/scan-moduli] launching {len(jobs)} jobs  group={wandb_group}")
    for meta in metas:
        print(
            f"  p={meta['modulus']:<3}  A_train={meta['n_train_a']:<6} "
            f"pairs={meta['n_train_pairs_realized']}/{meta['n_total_pairs']} "
            f"mode={meta['split_mode']}"
        )

    results = run_job_pool(jobs, gpus=gpus, workers_per_gpu=workers_per_gpu)
    write_report(
        out_root,
        phase="phase1",
        step="scan-moduli",
        config={
            "moduli": moduli,
            "train_frac": train_frac,
            "n_train_pairs": n_train_pairs,
            "weight_decay": weight_decay,
            "steps": steps,
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
            "modulus",
            "n_train_a",
            "n_train_pairs_realized",
            "model_seed",
            "status",
            "A_val_acc",
            "A_test_acc",
            "elapsed_sec",
            "wandb_url",
            "job_id",
        ],
        csv_row_fn=lambda r: [
            (r.get("wandb_config") or {}).get("modulus", ""),
            (r.get("wandb_config") or {}).get("n_train_a", ""),
            (r.get("wandb_config") or {}).get("n_train_pairs_realized", ""),
            r.get("model_seed", ""),
            r.get("status", ""),
            (r.get("metrics") or {}).get("A_val_acc", ""),
            (r.get("metrics") or {}).get("A_test_acc", ""),
            r.get("elapsed_sec", ""),
            r.get("wandb_url", "") or "",
            r.get("job_id", ""),
        ],
    )


def add_scan_moduli_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=f"runs/phase1/scan_moduli/{stamp()}",
    )
    parser.add_argument(
        "--moduli",
        type=int,
        nargs="+",
        default=None,
        help=f"Moduli to scan (default: all PRIMES={list(PRIMES)})",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=None,
        help=f"Locked train residue-pair fraction (default {DEFAULT_TRAIN_FRAC} "
        "if --n-train-pairs is not set)",
    )
    parser.add_argument(
        "--n-train-pairs",
        type=int,
        default=None,
        help="Optional: fixed train residue-pair count (mutually exclusive "
        "with --train-frac)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0,
        help="Locked weight decay from calibration",
    )
    parser.add_argument("--steps", type=int, default=100_000)
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
