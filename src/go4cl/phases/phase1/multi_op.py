"""Phase 1B: multi-operation single-task and same-modulus facilitation.

Compares (default):

1. ``all_same`` — four ops, all modulus p (same-mod baseline; 4 queries)
2. ``four_diff`` — four ops, four moduli (p, q nearby; plus p2, p3)
3. ``pair_same`` — four ops; first two share p; p2, p3 unchanged
4. optional ``one`` — legacy single-op (1 query); **not** used in default 1B

Locked training defaults (1B):
  train_frac=0.8, weight_decay=0.3, steps=100000, n_aliases=16, batch_size=8192
  with replacement.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from go4cl.defaults import MODEL, MULTI_OP
from go4cl.phases.common import (
    TrainJob,
    assign_gpus,
    resolve_gpus,
    run_job_pool,
    stamp,
    write_report,
)
from go4cl.phases.phase1.data import prepare_multi_op_dataset
from go4cl.tasks.multi_op import VARIANTS, choose_base_moduli, describe_variant

DEFAULT_TRAIN_FRAC = MULTI_OP.train_frac
DEFAULT_WEIGHT_DECAY = MULTI_OP.weight_decay
DEFAULT_STEPS = MULTI_OP.steps
DEFAULT_BATCH_SIZE = MULTI_OP.batch_size
DEFAULT_VARIANTS: tuple[str, ...] = MULTI_OP.variants


def run_multi_op(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    variants = list(args.variants)
    for v in variants:
        if v not in VARIANTS:
            raise SystemExit(f"unknown variant {v}; choose from {list(VARIANTS)}")

    task_seeds = list(args.task_seeds)
    model_seeds = list(args.model_seeds)
    gpus = resolve_gpus(args.gpus)
    workers_per_gpu = max(int(args.workers_per_gpu), 1)
    train_frac = float(args.train_frac)
    weight_decay = float(args.weight_decay)
    steps = int(args.steps)
    batch_size = int(getattr(args, "batch_size", DEFAULT_BATCH_SIZE))

    print(f"[phase1/multi-op] out={out_root}")
    print(f"[phase1/multi-op] variants={variants}")
    print(f"[phase1/multi-op] task_seeds={task_seeds} model_seeds={model_seeds}")
    print(
        f"[phase1/multi-op] locked cfg: train_frac={train_frac} wd={weight_decay} "
        f"steps={steps} aliases={args.n_aliases} lr={args.lr} "
        f"batch_size={batch_size} (packed online: {batch_size}//n_ops packs/step)"
    )

    metas: list[dict[str, Any]] = []
    for task_seed in task_seeds:
        base = choose_base_moduli(task_seed)
        print(f"[phase1/multi-op] task_seed={task_seed} base_moduli={list(base)}")
        for variant in variants:
            print(f"  {describe_variant(variant, base)}")  # type: ignore[arg-type]
            meta = prepare_multi_op_dataset(
                out_root,
                variant=variant,
                n_aliases=args.n_aliases,
                task_seed=task_seed,
                data_seed=args.data_seed,
                train_frac=train_frac,
            )
            meta["task_seed"] = int(task_seed)
            metas.append(meta)

    wandb_group = args.wandb_group or f"p1_multi_op_{stamp()}"
    jobs: list[TrainJob] = []
    for meta in metas:
        for model_seed in model_seeds:
            job_id = (
                f"{meta['tag']}__wd{weight_decay:g}_steps{steps}__ms{model_seed}"
            )
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
                        "multi-op",
                        str(meta["variant"]),
                        f"nops{meta['n_ops']}",
                    ),
                    wandb_config={
                        "phase": "phase1",
                        "step": "multi-op",
                        "variant": meta["variant"],
                        "moduli": meta["moduli"],
                        "base_moduli": meta["base_moduli"],
                        "n_ops": meta["n_ops"],
                        "train_frac": train_frac,
                        "n_aliases": meta["n_aliases"],
                        "n_train_a": meta["n_train_a"],
                        "train_mode": meta.get("train_mode", "packed_online"),
                        "weight_decay": weight_decay,
                        "steps": steps,
                        "task_seed": meta["task_seed"],
                        "data_seed": args.data_seed,
                        "ops": meta["ops"],
                    },
                    batch_size=batch_size,
                )
            )

    gpu_assign = assign_gpus(len(jobs), gpus, workers_per_gpu)
    jobs = [
        TrainJob(**{**job.__dict__, "gpu": gpu})
        for job, gpu in zip(jobs, gpu_assign)
    ]

    print(f"[phase1/multi-op] launching {len(jobs)} jobs  group={wandb_group}")
    for meta in metas:
        print(
            f"  {meta['variant']:<10} mods={meta['moduli']}  "
            f"A_val={meta['n_val_a']} train_mode={meta.get('train_mode')}"
        )

    results = run_job_pool(jobs, gpus=gpus, workers_per_gpu=workers_per_gpu)
    write_report(
        out_root,
        phase="phase1",
        step="multi-op",
        config={
            "variants": variants,
            "task_seeds": task_seeds,
            "model_seeds": model_seeds,
            "train_frac": train_frac,
            "weight_decay": weight_decay,
            "steps": steps,
            "n_aliases": args.n_aliases,
            "lr": args.lr,
            "batch_size": batch_size,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "data_seed": args.data_seed,
            "gpus": gpus,
            "workers_per_gpu": workers_per_gpu,
            "wandb_project": args.wandb_project,
            "wandb_group": wandb_group,
            "train_replacement": True,
            "train_mode": "packed_online",
        },
        datasets=metas,
        results=results,
        csv_fields=[
            "variant",
            "moduli",
            "n_ops",
            "n_train_a",
            "task_seed",
            "model_seed",
            "status",
            "A_val_acc",
            "A_test_acc",
            "best_A_val_acc",
            "best_A_test_acc",
            "best_step",
            "elapsed_sec",
            "wandb_url",
            "job_id",
        ],
        csv_row_fn=lambda r: [
            (r.get("wandb_config") or {}).get("variant", ""),
            (r.get("wandb_config") or {}).get("moduli", ""),
            (r.get("wandb_config") or {}).get("n_ops", ""),
            (r.get("wandb_config") or {}).get("n_train_a", ""),
            (r.get("wandb_config") or {}).get("task_seed", ""),
            r.get("model_seed", ""),
            r.get("status", ""),
            (r.get("metrics") or {}).get("A_val_acc", ""),
            (r.get("metrics") or {}).get("A_test_acc", ""),
            (r.get("metrics") or {}).get("best_A_val_acc", ""),
            (r.get("metrics") or {}).get("best_A_test_acc", ""),
            (r.get("metrics") or {}).get("best_step", ""),
            r.get("elapsed_sec", ""),
            r.get("wandb_url", "") or "",
            r.get("job_id", ""),
        ],
    )


def add_multi_op_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=f"runs/phase1/multi_op/{stamp()}",
    )
    parser.add_argument(
        "--variants",
        type=str,
        nargs="+",
        default=list(DEFAULT_VARIANTS),
        help=f"Variants to run (default {list(DEFAULT_VARIANTS)}; legacy: one)",
    )
    parser.add_argument("--task-seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--n-aliases", type=int, default=MODEL.n_aliases)
    parser.add_argument("--lr", type=float, default=MODEL.lr)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--d-model", type=int, default=MODEL.d_model)
    parser.add_argument("--n-layers", type=int, default=MODEL.n_layers)
    parser.add_argument("--data-seed", type=int, default=0)
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
