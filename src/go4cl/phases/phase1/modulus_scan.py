"""Phase 1A: single-operation modulus scan across all primes.

Uses one locked training config (from calibration) for every modulus.
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


def _prepare_dataset(
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
            f"hash={manifest.dataset_hash[:12]}"
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


def run_scan_moduli(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    moduli = list(args.moduli) if args.moduli else list(PRIMES)
    model_seeds = list(args.model_seeds)
    gpus = resolve_gpus(args.gpus)
    workers_per_gpu = max(int(args.workers_per_gpu), 1)
    train_frac = float(args.train_frac)
    weight_decay = float(args.weight_decay)
    steps = int(args.steps)

    print(f"[phase1/scan-moduli] out={out_root}")
    print(f"[phase1/scan-moduli] moduli={moduli}")
    print(
        f"[phase1/scan-moduli] locked cfg: train_frac={train_frac} "
        f"wd={weight_decay} steps={steps} aliases={args.n_aliases} "
        f"nuisance={args.n_nuisance} lr={args.lr} full-batch"
    )

    metas = [
        _prepare_dataset(
            out_root,
            modulus=p,
            train_frac=train_frac,
            n_aliases=args.n_aliases,
            n_nuisance=args.n_nuisance,
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
                        "train_frac": train_frac,
                        "ratios": meta["ratios"],
                        "n_aliases": meta["n_aliases"],
                        "n_nuisance": meta["n_nuisance"],
                        "n_train_a": meta["n_train_a"],
                        "weight_decay": weight_decay,
                        "steps": steps,
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

    print(f"[phase1/scan-moduli] launching {len(jobs)} jobs  group={wandb_group}")
    for meta in metas:
        print(f"  p={meta['modulus']:<3}  A_train={meta['n_train_a']}")

    results = run_job_pool(jobs, gpus=gpus, workers_per_gpu=workers_per_gpu)
    write_report(
        out_root,
        phase="phase1",
        step="scan-moduli",
        config={
            "moduli": moduli,
            "train_frac": train_frac,
            "weight_decay": weight_decay,
            "steps": steps,
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
            "modulus",
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
            (r.get("wandb_config") or {}).get("modulus", ""),
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
        default=0.6,
        help="Locked train residue-pair fraction from calibration",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1.0,
        help="Locked weight decay from calibration",
    )
    parser.add_argument("--steps", type=int, default=100_000)
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
