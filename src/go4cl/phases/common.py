"""Shared helpers for phase experiment runners (GPU pool, W&B jobs, reports)."""

from __future__ import annotations

import json
import os
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def list_gpus() -> list[int]:
    try:
        import torch

        if not torch.cuda.is_available():
            return []
        return list(range(torch.cuda.device_count()))
    except Exception:
        return []


def resolve_gpus(gpus_arg: str | None) -> list[int]:
    gpus = list_gpus()
    if gpus_arg:
        gpus = [int(x) for x in gpus_arg.split(",") if x.strip() != ""]
    if not gpus:
        print("[warn] no CUDA GPUs; falling back to device index 0")
        gpus = [0]
    return gpus


def ratios_from_train_frac(train_frac: float) -> tuple[float, float, float]:
    """Split remaining mass evenly between val and test."""
    if not (0.05 < train_frac < 0.95):
        raise ValueError(f"train_frac out of range: {train_frac}")
    rem = 1.0 - train_frac
    return (float(train_frac), rem / 2.0, rem / 2.0)


@dataclass(frozen=True)
class TrainJob:
    """One a_only training job on a fixed dataset directory."""

    job_id: str
    data_dir: str
    out_dir: str
    gpu: int
    steps: int
    lr: float
    weight_decay: float
    d_model: int
    n_layers: int
    model_seed: int
    wandb_project: str
    wandb_group: str
    wandb_mode: str | None
    wandb_tags: tuple[str, ...]
    wandb_config: dict[str, Any]
    batch_size: int = 2048
    sampler_seed: int | None = None


def execute_a_only_job(job: TrainJob) -> dict[str, Any]:
    """Worker entrypoint: pin one GPU and run ``a_only``."""
    from go4cl.runtime_paths import configure_scratch_dirs

    configure_scratch_dirs()

    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(job.gpu)

    t0 = time.time()
    result: dict[str, Any] = {
        "job_id": job.job_id,
        "gpu": job.gpu,
        "steps": job.steps,
        "lr": job.lr,
        "weight_decay": job.weight_decay,
        "model_seed": job.model_seed,
        "data_dir": job.data_dir,
        "status": "running",
        "started_at": utc_now(),
        "wandb_config": job.wandb_config,
    }
    try:
        import torch

        from go4cl.model.transformer import ModelConfig
        from go4cl.train.loop import TrainConfig
        from go4cl.train.protocols import run_protocol

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model_cfg = ModelConfig(d_model=job.d_model, n_layers=job.n_layers)
        model_cfg.d_mlp = 4 * model_cfg.d_model
        eval_every = max(job.steps // 100, 100)
        train_cfg = TrainConfig(
            lr=job.lr,
            weight_decay=job.weight_decay,
            batch_size=int(job.batch_size),
            train_replacement=True,
            max_steps=job.steps,
            eval_every=eval_every,
            ckpt_every=max(job.steps // 5, eval_every),
            device=device,
        )
        proto = run_protocol(
            "a_only",
            job.data_dir,
            job.out_dir,
            model_cfg=model_cfg,
            train_cfg=train_cfg,
            model_seed=job.model_seed,
            sampler_seed=(
                int(job.sampler_seed)
                if job.sampler_seed is not None
                else int(job.model_seed)
            ),
            phase_steps=job.steps,
            wandb_enabled=True,
            wandb_project=job.wandb_project,
            wandb_name=job.job_id,
            wandb_mode=job.wandb_mode,
            wandb_group=job.wandb_group,
            wandb_tags=list(job.wandb_tags),
            wandb_config={
                **job.wandb_config,
                "batch_size": int(job.batch_size),
                "train_replacement": True,
                "full_batch": False,
                "wandb_group": job.wandb_group,
                "gpu": job.gpu,
                "sampler_seed": (
                    int(job.sampler_seed)
                    if job.sampler_seed is not None
                    else int(job.model_seed)
                ),
            },
        )
        result["status"] = "ok"
        result["metrics"] = proto.metrics
        result["wandb_url"] = proto.wandb_url
    except Exception as exc:  # noqa: BLE001
        result["status"] = "error"
        result["error"] = f"{type(exc).__name__}: {exc}"
        result["traceback"] = traceback.format_exc()

    result["elapsed_sec"] = round(time.time() - t0, 2)
    result["finished_at"] = utc_now()
    out = Path(job.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "job_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    return result


def run_job_pool(
    jobs: list[TrainJob],
    *,
    gpus: list[int],
    workers_per_gpu: int = 1,
    execute_fn: Callable[[TrainJob], dict[str, Any]] = execute_a_only_job,
) -> list[dict[str, Any]]:
    workers_per_gpu = max(int(workers_per_gpu), 1)
    max_workers = len(gpus) * workers_per_gpu
    print(f"[pool] jobs={len(jobs)} gpus={gpus} workers_per_gpu={workers_per_gpu}")

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=__import__("multiprocessing").get_context("spawn"),
    ) as pool:
        futures = {pool.submit(execute_fn, job): job for job in jobs}
        done = 0
        for fut in as_completed(futures):
            job = futures[fut]
            try:
                res = fut.result()
            except Exception as exc:  # noqa: BLE001
                res = {
                    "job_id": job.job_id,
                    "status": "error",
                    "error": f"{type(exc).__name__}: {exc}",
                    "traceback": traceback.format_exc(),
                }
            results.append(res)
            done += 1
            print(
                f"[{done}/{len(jobs)}] gpu={res.get('gpu')} {job.job_id} "
                f"-> {res.get('status')} ({res.get('elapsed_sec', '?')}s) "
                f"{res.get('wandb_url', '')}"
            )
    return results


def write_report(
    out_root: Path,
    *,
    phase: str,
    step: str,
    config: dict[str, Any],
    datasets: list[dict[str, Any]],
    results: list[dict[str, Any]],
    csv_fields: list[str],
    csv_row_fn: Callable[[dict[str, Any]], list[Any]],
) -> tuple[Path, Path]:
    import csv

    ok = [r for r in results if r.get("status") == "ok"]
    err = [r for r in results if r.get("status") != "ok"]
    summary = {
        "phase": phase,
        "step": step,
        "n_total": len(results),
        "n_ok": len(ok),
        "n_error": len(err),
        "wandb_group": config.get("wandb_group"),
        "wandb_project": config.get("wandb_project"),
        "errors": [{"job_id": r.get("job_id"), "error": r.get("error")} for r in err],
    }
    report = {
        "created_at": utc_now(),
        "phase": phase,
        "step": step,
        "out": str(out_root),
        "config": config,
        "datasets": datasets,
        "summary": summary,
        "results": results,
    }
    report_path = out_root / f"{phase}_{step}_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )

    csv_path = out_root / f"{phase}_{step}_summary.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()
        for r in results:
            row_vals = csv_row_fn(r)
            writer.writerow(dict(zip(csv_fields, row_vals, strict=False)))

    print(f"\n======== {phase.upper()} / {step} SUMMARY ========")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"report: {report_path}")
    print(f"csv:    {csv_path}")
    if summary["n_error"]:
        raise SystemExit(f"{summary['n_error']} jobs failed")
    return report_path, csv_path


def assign_gpus(n_jobs: int, gpus: list[int], workers_per_gpu: int) -> list[int]:
    slots = [g for g in gpus for _ in range(max(workers_per_gpu, 1))]
    return [slots[i % len(slots)] for i in range(n_jobs)]
