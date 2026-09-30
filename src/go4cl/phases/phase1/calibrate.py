"""Phase 1A prelude: grokking regime calibration on a medium modulus.

Sweeps train residue-pair fraction × weight decay (and optional steps),
with a fixed single-op task. Lock one training config before the full
8-modulus scan.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from go4cl.constants import PRIMES
from go4cl.phases.common import (
    TrainJob,
    assign_gpus,
    ratios_from_train_frac,
    resolve_gpus,
    run_job_pool,
    stamp,
    write_report,
)


# Medium prime from the plan set {7,11,13,17,19,23,29,31}
DEFAULT_CALIB_MODULUS = 17
DEFAULT_TRAIN_FRACS: tuple[float, ...] = (0.4, 0.6, 0.8)
DEFAULT_WEIGHT_DECAYS: tuple[float, ...] = (0.1, 0.3, 1.0)


def _prepare_single_op_dataset(
    out: Path,
    *,
    modulus: int,
    train_frac: float,
    n_aliases: int,
    n_nuisance: int,
    task_seed: int,
    data_seed: int,
) -> dict[str, Any]:
    from go4cl.data.generate import generate_task_datasets, save_datasets
    from go4cl.data.manifest import DataManifest
    from go4cl.data.residue_pairs import assert_disjoint
    from go4cl.tasks.single_op import build_single_op_pair

    ratios = ratios_from_train_frac(train_frac)
    tag = (
        f"single_p{modulus}_tr{train_frac:g}"
        f"_ts{task_seed}_ds{data_seed}_a{n_aliases}_n{n_nuisance}"
    )
    data_dir = out / "data" / tag
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = DataManifest.load(manifest_path)
        print(f"[data] reuse {data_dir}  A_train={manifest.samples_per_slot['A']['train']}")
    else:
        pair = build_single_op_pair(modulus, task_seed=task_seed)
        manifest, datasets = generate_task_datasets(
            pair,
            data_seed=data_seed,
            n_aliases_per_pair=n_aliases,
            n_nuisance_contexts=n_nuisance,
            ratios=ratios,
            experiment_id=tag,
        )
        for split in manifest.residue_splits.values():
            assert_disjoint(split)
        save_datasets(data_dir, manifest, datasets)
        print(
            f"[data] wrote {data_dir}  A_train={manifest.samples_per_slot['A']['train']}  "
            f"ratios={ratios}  hash={manifest.dataset_hash[:12]}"
        )
    return {
        "tag": tag,
        "data_dir": str(data_dir),
        "modulus": modulus,
        "train_frac": train_frac,
        "ratios": list(ratios),
        "n_aliases": n_aliases,
        "n_nuisance": n_nuisance,
        "n_train_a": int(manifest.samples_per_slot["A"]["train"]),
        "n_val_a": int(manifest.samples_per_slot["A"]["val"]),
        "n_test_a": int(manifest.samples_per_slot["A"]["test"]),
        "dataset_hash": manifest.dataset_hash,
        "pair_id": manifest.task_pair.pair_id,
        "op": manifest.task_pair.task_a.operations[0].to_dict(),
    }


def run_calibrate(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    modulus = int(args.modulus)
    if modulus not in PRIMES:
        print(f"[warn] modulus {modulus} not in PRIMES={PRIMES}; continuing anyway")

    train_fracs = list(args.train_fracs)
    weight_decays = list(args.weight_decays)
    steps_grid = list(args.steps_grid)
    model_seeds = list(args.model_seeds)
    gpus = resolve_gpus(args.gpus)
    workers_per_gpu = max(int(args.workers_per_gpu), 1)

    print(f"[phase1/calibrate] out={out_root}")
    print(f"[phase1/calibrate] modulus={modulus} (single op)")
    print(f"[phase1/calibrate] train_fracs={train_fracs}")
    print(f"[phase1/calibrate] weight_decays={weight_decays}")
    print(f"[phase1/calibrate] steps_grid={steps_grid}")
    print(
        f"[phase1/calibrate] aliases={args.n_aliases} nuisance={args.n_nuisance} "
        f"lr={args.lr} full-batch"
    )

    metas: list[dict[str, Any]] = []
    for train_frac in train_fracs:
        metas.append(
            _prepare_single_op_dataset(
                out_root,
                modulus=modulus,
                train_frac=train_frac,
                n_aliases=args.n_aliases,
                n_nuisance=args.n_nuisance,
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
                                f"tr{meta['train_frac']:g}",
                                f"wd{wd:g}",
                            ),
                            wandb_config={
                                "phase": "phase1",
                                "step": "calibrate",
                                "modulus": modulus,
                                "train_frac": meta["train_frac"],
                                "ratios": meta["ratios"],
                                "n_aliases": meta["n_aliases"],
                                "n_nuisance": meta["n_nuisance"],
                                "n_train_a": meta["n_train_a"],
                                "weight_decay": float(wd),
                                "steps": int(steps),
                                "task_seed": args.task_seed,
                                "data_seed": args.data_seed,
                                "single_op": True,
                            },
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
            f"  tr={meta['train_frac']:<4}  A_train={meta['n_train_a']:<6} "
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
            "weight_decays": weight_decays,
            "steps_grid": steps_grid,
            "n_aliases": args.n_aliases,
            "n_nuisance": args.n_nuisance,
            "lr": args.lr,
            "d_model": args.d_model,
            "n_layers": args.n_layers,
            "task_seed": args.task_seed,
            "data_seed": args.data_seed,
            "model_seeds": model_seeds,
            "gpus": gpus,
            "workers_per_gpu": workers_per_gpu,
            "wandb_project": args.wandb_project,
            "wandb_group": wandb_group,
            "full_batch": True,
        },
        datasets=metas,
        results=results,
        csv_fields=[
            "train_frac",
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
            (r.get("wandb_config") or {}).get("train_frac", ""),
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
        default=list(DEFAULT_TRAIN_FRACS),
        help="Train residue-pair fractions to sweep",
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
    parser.add_argument("--n-aliases", type=int, default=16)
    parser.add_argument("--n-nuisance", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
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
