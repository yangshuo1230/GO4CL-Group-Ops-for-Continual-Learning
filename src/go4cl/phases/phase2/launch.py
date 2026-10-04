"""Shared phase-2 job grid: datasets, GPU pool, behavioral CSVs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from go4cl.metrics.continual import GROUP_KEYS, forward_transfer_rows
from go4cl.phases.common import (
    TrainJob,
    assign_gpus,
    resolve_gpus,
    run_job_pool,
    stamp,
    utc_now,
    write_report,
)
from go4cl.phases.phase2.data import prepare_phase2_dataset
from go4cl.phases.phase2.grid import ALL_PROTOCOLS, Condition

# Packed data matches phase 1B; wd=0.3 so the late-grokking modulus (p=23
# on the default full-overlap seed) still leaves the query-only basin.
DEFAULT_TRAIN_FRAC = 0.8
DEFAULT_WEIGHT_DECAY = 0.3
DEFAULT_STEPS = 100_000
DEFAULT_BATCH_SIZE = 8192
DEFAULT_N_ALIASES = 16

CSV_FIELDS = [
    "protocol",
    "condition",
    "rho_slot",
    "rho_operand",
    "rho_mod",
    "direction",
    "task_seed",
    "model_seed",
    "d_model",
    "n_layers",
    "status",
    "A_test_acc",
    "B_test_acc",
    "A_val_acc",
    "B_val_acc",
    "best_A_test_acc",
    "best_B_test_acc",
    "forgetting_A",
    "forgetting_B",
    "forgetting_A_from_switch",
    "forgetting_B_from_switch",
    "jump_A",
    "forget_rate_A",
    "retention_A",
    "b_exposure_auc",
    "b_exposure_steps_to_gen",
    "grok_order",
    "switch_step",
    "elapsed_sec",
    "wandb_url",
    "job_id",
]


def add_shared_args(
    parser: argparse.ArgumentParser,
    *,
    default_out: str,
    include_model_size: bool = True,
) -> None:
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Query examples per step. Must be divisible by 8 (2 tasks x 4 ops).",
    )
    parser.add_argument("--n-aliases", type=int, default=DEFAULT_N_ALIASES)
    parser.add_argument("--lr", type=float, default=1e-3)
    if include_model_size:
        parser.add_argument("--d-model", type=int, default=64)
        parser.add_argument("--n-layers", type=int, default=3)
    parser.add_argument("--n-heads", type=int, default=4)
    parser.add_argument("--task-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--model-seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument(
        "--directions",
        type=str,
        nargs="+",
        default=["forward"],
        choices=["forward", "swap"],
        help="forward keeps the constructed A/B token assignment; swap exchanges it.",
    )
    parser.add_argument(
        "--switch-on",
        type=str,
        default="fixed",
        choices=["fixed", "t_mem", "t_gen"],
        help="When the first sequential phase ends. fixed uses --steps. "
        "t_mem / t_gen stop early once that event fires; the second phase still "
        "runs --steps.",
    )
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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Write datasets and the job list, then exit before training.",
    )


def _require_protocols(names: list[str]) -> list[str]:
    unknown = [name for name in names if name not in ALL_PROTOCOLS]
    if unknown:
        raise SystemExit(
            f"unknown protocol(s) {unknown}; choose from {list(ALL_PROTOCOLS)}"
        )
    return names


def _behavior(result: dict[str, Any], key: str) -> Any:
    behavior = (result.get("metrics") or {}).get("behavior") or {}
    value = behavior.get(key, "")
    return "" if value is None else value


def _csv_row(result: dict[str, Any]) -> list[Any]:
    cfg = result.get("wandb_config") or {}
    metrics = result.get("metrics") or {}
    return [
        cfg.get("protocol", result.get("protocol", "")),
        cfg.get("condition", ""),
        cfg.get("rho_slot", ""),
        cfg.get("rho_operand", ""),
        cfg.get("rho_mod", ""),
        cfg.get("direction", ""),
        cfg.get("task_seed", ""),
        result.get("model_seed", cfg.get("model_seed", "")),
        cfg.get("d_model", ""),
        cfg.get("n_layers", ""),
        result.get("status", ""),
        metrics.get("A_test_acc", ""),
        metrics.get("B_test_acc", ""),
        metrics.get("A_val_acc", ""),
        metrics.get("B_val_acc", ""),
        metrics.get("best_A_test_acc", ""),
        metrics.get("best_B_test_acc", ""),
        metrics.get("forgetting_A", ""),
        metrics.get("forgetting_B", ""),
        metrics.get("forgetting_A_from_switch", ""),
        metrics.get("forgetting_B_from_switch", ""),
        _behavior(result, "jump_A"),
        _behavior(result, "forget_rate_A"),
        _behavior(result, "retention_A"),
        _behavior(result, "b_exposure_auc"),
        _behavior(result, "b_exposure_steps_to_gen"),
        _behavior(result, "grok_order"),
        metrics.get("switch_step", ""),
        result.get("elapsed_sec", ""),
        result.get("wandb_url") or "",
        result.get("job_id", ""),
    ]


def _flat_record(result: dict[str, Any]) -> dict[str, Any]:
    cfg = result.get("wandb_config") or {}
    metrics = result.get("metrics") or {}
    behavior = metrics.get("behavior") or {}
    row: dict[str, Any] = {
        "status": result.get("status"),
        "job_id": result.get("job_id"),
        "protocol": cfg.get("protocol", result.get("protocol")),
        "condition": cfg.get("condition"),
        "rho_slot": cfg.get("rho_slot"),
        "rho_operand": cfg.get("rho_operand"),
        "rho_mod": cfg.get("rho_mod"),
        "direction": cfg.get("direction"),
        "task_seed": cfg.get("task_seed"),
        "model_seed": result.get("model_seed", cfg.get("model_seed")),
        "d_model": cfg.get("d_model"),
        "n_layers": cfg.get("n_layers"),
        "A_test_acc": metrics.get("A_test_acc"),
        "B_test_acc": metrics.get("B_test_acc"),
        "forgetting_A": metrics.get("forgetting_A"),
        "grok_order": behavior.get("grok_order"),
        "b_exposure_auc": behavior.get("b_exposure_auc"),
        "b_exposure_steps_to_gen": behavior.get("b_exposure_steps_to_gen"),
    }
    return row


def _write_table(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if row.get(key) is None else row.get(key) for key in fields})


def _write_transfer(out_root: Path, step: str, results: list[dict[str, Any]]) -> None:
    rows = forward_transfer_rows([_flat_record(r) for r in results])
    fields = [
        *GROUP_KEYS,
        "protocol",
        "job_id",
        "baseline_b_exposure_auc",
        "b_exposure_auc",
        "delta_b_exposure_auc",
        "baseline_b_exposure_steps_to_gen",
        "b_exposure_steps_to_gen",
        "delta_b_exposure_steps_to_gen",
    ]
    path = out_root / f"phase2_{step}_transfer.csv"
    _write_table(path, rows, fields)
    print(f"[phase2/{step}] forward-transfer rows={len(rows)} -> {path}")


def _write_by_rho(out_root: Path, step: str, results: list[dict[str, Any]]) -> None:
    buckets: dict[tuple, list[dict[str, Any]]] = {}
    for result in results:
        if result.get("status") != "ok":
            continue
        flat = _flat_record(result)
        key = (
            flat.get("protocol"),
            flat.get("rho_slot"),
            flat.get("rho_operand"),
            flat.get("rho_mod"),
            flat.get("d_model"),
            flat.get("n_layers"),
        )
        buckets.setdefault(key, []).append(flat)
    rows: list[dict[str, Any]] = []
    for key, group in sorted(buckets.items(), key=lambda kv: tuple(str(x) for x in kv[0])):
        def _mean(field: str) -> float | str:
            vals = [
                float(row[field])
                for row in group
                if isinstance(row.get(field), (int, float))
            ]
            if not vals:
                return ""
            return sum(vals) / len(vals)

        rows.append(
            {
                "protocol": key[0],
                "rho_slot": key[1],
                "rho_operand": key[2],
                "rho_mod": key[3],
                "d_model": key[4],
                "n_layers": key[5],
                "n": len(group),
                "mean_A_test_acc": _mean("A_test_acc"),
                "mean_B_test_acc": _mean("B_test_acc"),
                "mean_forgetting_A": _mean("forgetting_A"),
                "mean_b_exposure_auc": _mean("b_exposure_auc"),
            }
        )
    path = out_root / f"phase2_{step}_by_rho.csv"
    _write_table(
        path,
        rows,
        [
            "protocol",
            "rho_slot",
            "rho_operand",
            "rho_mod",
            "d_model",
            "n_layers",
            "n",
            "mean_A_test_acc",
            "mean_B_test_acc",
            "mean_forgetting_A",
            "mean_b_exposure_auc",
        ],
    )
    print(f"[phase2/{step}] rho means={len(rows)} -> {path}")


def launch_grid(
    args: argparse.Namespace,
    *,
    step: str,
    conditions: list[Condition],
    protocols: list[str],
    sizes: list[tuple[int, int]],
) -> None:
    """Materialize datasets and run every (condition, protocol, seed, size) job."""
    protocols = _require_protocols(list(protocols))
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    batch_size = int(args.batch_size)
    if batch_size % 8 != 0:
        raise SystemExit(
            f"batch_size={batch_size} must be divisible by 8 "
            "(4 ops, and joint needs a 50/50 split)"
        )
    directions = list(args.directions)
    task_seeds = [int(s) for s in args.task_seeds]
    model_seeds = [int(s) for s in args.model_seeds]
    gpus = resolve_gpus(args.gpus)
    workers_per_gpu = max(int(args.workers_per_gpu), 1)
    train_frac = float(args.train_frac)
    weight_decay = float(args.weight_decay)
    steps = int(args.steps)

    print(f"[phase2/{step}] out={out_root}")
    print(
        f"[phase2/{step}] protocols={protocols} conditions={len(conditions)} "
        f"directions={directions} task_seeds={task_seeds} model_seeds={model_seeds} "
        f"sizes={sizes}"
    )
    print(
        f"[phase2/{step}] cfg: train_frac={train_frac} wd={weight_decay} "
        f"steps={steps} batch_size={batch_size} aliases={args.n_aliases} "
        f"switch_on={args.switch_on}"
    )

    metas: list[dict[str, Any]] = []
    meta_by_key: dict[tuple, dict[str, Any]] = {}
    for cond in conditions:
        for task_seed in task_seeds:
            for direction in directions:
                meta = prepare_phase2_dataset(
                    out_root,
                    rho_slot=cond.rho_slot,
                    rho_operand=cond.rho_operand,
                    rho_mod=cond.rho_mod,
                    task_seed=task_seed,
                    data_seed=int(args.data_seed),
                    n_aliases=int(args.n_aliases),
                    train_frac=train_frac,
                    direction=direction,
                )
                meta = {**meta, "condition": cond.name}
                metas.append(meta)
                meta_by_key[(cond.name, task_seed, direction)] = meta

    wandb_group = args.wandb_group or f"p2_{step}_{stamp()}"
    jobs: list[TrainJob] = []
    for cond in conditions:
        for task_seed in task_seeds:
            for direction in directions:
                meta = meta_by_key[(cond.name, task_seed, direction)]
                for protocol in protocols:
                    for model_seed in model_seeds:
                        for d_model, n_layers in sizes:
                            job_id = (
                                f"{protocol}_{cond.name}_{direction}"
                                f"_ts{task_seed}_ms{model_seed}"
                                f"_d{d_model}_L{n_layers}"
                                f"_wd{weight_decay:g}_steps{steps}"
                            )
                            jobs.append(
                                TrainJob(
                                    job_id=job_id,
                                    data_dir=meta["data_dir"],
                                    out_dir=str(out_root / "runs" / job_id),
                                    gpu=0,
                                    steps=steps,
                                    lr=float(args.lr),
                                    weight_decay=weight_decay,
                                    d_model=int(d_model),
                                    n_layers=int(n_layers),
                                    model_seed=int(model_seed),
                                    wandb_project=args.wandb_project,
                                    wandb_group=wandb_group,
                                    wandb_mode=args.wandb_mode,
                                    wandb_tags=(
                                        "phase2",
                                        step,
                                        protocol,
                                        cond.name,
                                        direction,
                                    ),
                                    wandb_config={
                                        "phase": "phase2",
                                        "step": step,
                                        "protocol": protocol,
                                        "condition": cond.name,
                                        "rho_slot": meta["rho_slot"],
                                        "rho_operand": meta["rho_operand"],
                                        "rho_mod": meta["rho_mod"],
                                        "direction": direction,
                                        "task_seed": task_seed,
                                        "data_seed": int(args.data_seed),
                                        "model_seed": int(model_seed),
                                        "d_model": int(d_model),
                                        "n_layers": int(n_layers),
                                        "n_heads": int(args.n_heads),
                                        "train_frac": train_frac,
                                        "weight_decay": weight_decay,
                                        "steps": steps,
                                        "batch_size": batch_size,
                                        "switch_on": args.switch_on,
                                        "pair_id": meta["pair_id"],
                                        "dataset_hash": meta["dataset_hash"],
                                        "train_mode": "packed_online",
                                    },
                                    batch_size=batch_size,
                                    protocol=protocol,
                                    include_test=True,
                                    switch_on=str(args.switch_on),
                                    n_heads=int(args.n_heads),
                                )
                            )

    gpu_assign = assign_gpus(len(jobs), gpus, workers_per_gpu)
    jobs = [
        TrainJob(**{**job.__dict__, "gpu": gpu})
        for job, gpu in zip(jobs, gpu_assign)
    ]
    manifest = {
        "created_at": utc_now(),
        "phase": "phase2",
        "step": step,
        "n_jobs": len(jobs),
        "protocols": protocols,
        "conditions": [c.__dict__ for c in conditions],
        "directions": directions,
        "task_seeds": task_seeds,
        "model_seeds": model_seeds,
        "sizes": [{"d_model": d, "n_layers": n} for d, n in sizes],
        "wandb_group": wandb_group,
        "jobs": [
            {
                "job_id": job.job_id,
                "protocol": job.protocol,
                "gpu": job.gpu,
                "data_dir": job.data_dir,
                "out_dir": job.out_dir,
                "d_model": job.d_model,
                "n_layers": job.n_layers,
            }
            for job in jobs
        ],
    }
    (out_root / "jobs.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"[phase2/{step}] jobs={len(jobs)} group={wandb_group}")
    if args.dry_run:
        print(f"[phase2/{step}] dry-run; wrote {out_root / 'jobs.json'}")
        return

    results = run_job_pool(jobs, gpus=gpus, workers_per_gpu=workers_per_gpu)
    _write_transfer(out_root, step, results)
    _write_by_rho(out_root, step, results)
    write_report(
        out_root,
        phase="phase2",
        step=step,
        config={
            "protocols": protocols,
            "conditions": [c.__dict__ for c in conditions],
            "directions": directions,
            "task_seeds": task_seeds,
            "model_seeds": model_seeds,
            "sizes": [{"d_model": d, "n_layers": n} for d, n in sizes],
            "train_frac": train_frac,
            "weight_decay": weight_decay,
            "steps": steps,
            "batch_size": batch_size,
            "n_aliases": int(args.n_aliases),
            "lr": float(args.lr),
            "n_heads": int(args.n_heads),
            "data_seed": int(args.data_seed),
            "switch_on": args.switch_on,
            "gpus": gpus,
            "workers_per_gpu": workers_per_gpu,
            "wandb_project": args.wandb_project,
            "wandb_group": wandb_group,
            "train_mode": "packed_online",
            "include_test": True,
        },
        datasets=metas,
        results=results,
        csv_fields=CSV_FIELDS,
        csv_row_fn=_csv_row,
    )
