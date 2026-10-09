"""CLI entry points for the next-stage Phase 2 grids.

Dry-run is the default. ``--execute`` is the formal training command and is
not invoked by the smoke tests.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from go4cl.defaults import MODEL, PHASE2
from go4cl.phases.common import TrainJob, assign_gpus, resolve_gpus, run_job_pool, utc_now
from go4cl.phases.phase2.causal_modulus import (
    MASTERY_THRESHOLD,
    N_MAIN_RUNS as CAUSAL_N,
    a_mastery_spec,
    build_causal_tasks,
    plan_causal_jobs,
    prepare_causal_datasets,
    write_dry_run as write_causal_dry_run,
)
from go4cl.phases.phase2.replay_coverage import (
    N_MAIN_RUNS as COVERAGE_N,
    plan_coverage_jobs,
    write_dry_run as write_coverage_dry_run,
)


def _job(
    *,
    job_id: str,
    protocol: str,
    data_dir: Path,
    out_dir: Path,
    model_seed: int,
    steps: int,
    batch_size: int,
    wandb_group: str,
    extra: dict,
    theta_a_ckpt: str | None = None,
) -> TrainJob:
    return TrainJob(
        job_id=job_id,
        data_dir=str(data_dir),
        out_dir=str(out_dir),
        gpu=0,
        steps=int(steps),
        lr=MODEL.lr,
        weight_decay=PHASE2.weight_decay,
        d_model=MODEL.d_model,
        n_layers=MODEL.n_layers,
        n_heads=MODEL.n_heads,
        model_seed=int(model_seed),
        batch_size=int(batch_size),
        wandb_project="go4cl",
        wandb_group=wandb_group,
        wandb_mode="disabled",
        wandb_tags=("phase2", protocol),
        wandb_config=extra,
        protocol=protocol,
        include_test=True,
        switch_on="fixed",
        theta_a_ckpt=theta_a_ckpt,
        optimizer_transition="fresh",
    )


def pin_jobs(
    jobs: list[TrainJob], gpus: list[int], workers_per_gpu: int
) -> list[TrainJob]:
    """Write a physical GPU id onto each job.

    ``run_job_pool`` only uses ``gpus`` to size the process pool. Each worker
    sets ``CUDA_VISIBLE_DEVICES`` from ``job.gpu``. Leaving that field at 0
    puts every worker on physical GPU 0.
    """
    if not gpus:
        raise ValueError("no GPUs to pin")
    assigned = assign_gpus(len(jobs), gpus, workers_per_gpu)
    return [
        TrainJob(**{**job.__dict__, "gpu": int(gpu)})
        for job, gpu in zip(jobs, assigned)
    ]


def run_causal_modulus(args: argparse.Namespace) -> None:
    out = Path(args.out)
    jobs = plan_causal_jobs()
    if not args.execute:
        payload = write_causal_dry_run(out, jobs, data_seed=int(args.data_seed))
        counts = payload["counts"]
        print(
            f"[causal-modulus] dry-run jobs={counts['n_main_runs']} "
            f"b_only={counts['n_b_only']} a_sources={counts['n_a_sources']} "
            f"wrote {out / 'jobs.json'}"
        )
        if counts["n_main_runs"] != CAUSAL_N:
            raise SystemExit(f"expected {CAUSAL_N} causal jobs")
        return
    print(f"[causal-modulus] executing {len(jobs)} jobs at {utc_now()}")
    datasets = prepare_causal_datasets(
        out,
        data_seed=int(args.data_seed),
        n_aliases=int(args.n_aliases),
        train_frac=float(args.train_frac),
    )
    group = f"causal_modulus_{utc_now()}"
    train_jobs = []
    for job in jobs:
        variant = "different" if job.protocol == "different_p_to_b" else "same"
        data_dir = datasets[(job.target_modulus, job.construction, variant)]
        built = build_causal_tasks(job.target_modulus, job.construction)
        pair = built.different if variant == "different" else built.same
        extra = {
            "experiment": "causal_modulus",
            "target_modulus": job.target_modulus,
            "construction": job.construction,
            "protocol": job.protocol,
            "b_task_hash": job.b_task_hash,
            "a_task_hash": job.a_task_hash,
            "optimizer_transition": "fresh",
            "replay_ratio": 0.0,
        }
        if job.protocol != "b_only":
            extra["a_mastery"] = a_mastery_spec(pair.task_a, MASTERY_THRESHOLD)
        protocol = "b_only" if job.protocol == "b_only" else "sequential_ab"
        train_jobs.append(
            _job(
                job_id=job.job_id,
                protocol=protocol,
                data_dir=data_dir,
                out_dir=out / "runs" / job.job_id,
                model_seed=job.model_seed,
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                wandb_group=group,
                extra=extra,
            )
        )
    gpus = resolve_gpus(args.gpus)
    train_jobs = pin_jobs(train_jobs, gpus, int(args.workers_per_gpu))
    print(f"[causal-modulus] gpu pin={[job.gpu for job in train_jobs]}")
    run_job_pool(
        train_jobs,
        gpus=gpus,
        workers_per_gpu=int(args.workers_per_gpu),
    )


def run_replay_coverage(args: argparse.Namespace) -> None:
    out = Path(args.out)
    jobs = plan_coverage_jobs(
        data_seed=int(args.data_seed), batch_size=int(args.batch_size)
    )
    if not args.execute:
        payload = write_coverage_dry_run(out, jobs)
        counts = payload["counts"]
        print(
            f"[replay-coverage] dry-run jobs={counts['n_main_runs']} "
            f"a_pretrain={counts['n_a_sources']} "
            f"a_replay_per_step={counts['a_replay_examples_per_step']} "
            f"wrote {out / 'jobs.json'}"
        )
        if counts["n_main_runs"] != COVERAGE_N:
            raise SystemExit(f"expected {COVERAGE_N} coverage jobs")
        return
    from go4cl.phases.phase2.data import prepare_phase2_dataset
    from go4cl.phases.phase2.replay_coverage import CONDITIONS

    print(f"[replay-coverage] executing {len(jobs)} jobs at {utc_now()}")
    data_dirs = {}
    for name, rho_slot, rho_operand, rho_mod in CONDITIONS:
        meta = prepare_phase2_dataset(
            out,
            rho_slot=rho_slot,
            rho_operand=rho_operand,
            rho_mod=rho_mod,
            task_seed=0,
            data_seed=int(args.data_seed),
            n_aliases=int(args.n_aliases),
            train_frac=float(args.train_frac),
            fixed_a=True,
        )
        data_dirs[name] = Path(meta["data_dir"])
    group = f"replay_coverage_{utc_now()}"
    # One theta_A per (task A, model seed). Coverages and both conditions reuse it.
    a_jobs = []
    a_ckpt: dict[str, str] = {}
    seen: set[str] = set()
    first_condition = CONDITIONS[0][0]
    for job in jobs:
        if job.a_source_key in seen:
            continue
        seen.add(job.a_source_key)
        job_id = f"a_only_{job.a_source_key[:12]}_ms{job.model_seed}"
        ckpt = str(out / "runs" / job_id / "ckpts" / "final.pt")
        a_ckpt[job.a_source_key] = ckpt
        a_jobs.append(
            _job(
                job_id=job_id,
                protocol="a_only",
                data_dir=data_dirs[first_condition],
                out_dir=out / "runs" / job_id,
                model_seed=job.model_seed,
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                wandb_group=group,
                extra={"experiment": "replay_coverage", "role": "theta_A"},
            )
        )
    main = []
    for job in jobs:
        spec = job.to_dict()
        main.append(
            _job(
                job_id=job.job_id,
                protocol="sequential_ab_replay",
                data_dir=data_dirs[job.condition],
                out_dir=out / "runs" / job.job_id,
                model_seed=job.model_seed,
                steps=int(args.steps),
                batch_size=int(args.batch_size),
                wandb_group=group,
                extra={
                    "experiment": "replay_coverage",
                    "coverage": job.coverage,
                    "condition": job.condition,
                    "replay_coverage": {
                        "buffer_seed": spec["buffer_seed"],
                        "train_pairs_by_latent": spec["train_pairs_by_latent"],
                    },
                    "replay_ratio": 0.1,
                    "optimizer_transition": "fresh",
                },
                theta_a_ckpt=a_ckpt[job.a_source_key],
            )
        )
    gpus = resolve_gpus(args.gpus)
    workers = int(args.workers_per_gpu)
    a_jobs = pin_jobs(a_jobs, gpus, workers)
    main = pin_jobs(main, gpus, workers)
    print(f"[replay-coverage] A gpu pin={[job.gpu for job in a_jobs]}")
    print(f"[replay-coverage] B gpu pin={[job.gpu for job in main]}")
    run_job_pool(a_jobs, gpus=gpus, workers_per_gpu=workers)
    run_job_pool(main, gpus=gpus, workers_per_gpu=workers)


def run_analyze_modulus(args: argparse.Namespace) -> None:
    from go4cl.phases.phase2.modulus_analysis import run_modulus_analysis

    result = run_modulus_analysis(args.stamp, args.out)
    print(f"[analyze-modulus] rows={result['n_rows']} out={result['out_dir']}")


def run_param_patch_smoke(args: argparse.Namespace) -> None:
    from go4cl.phases.phase2.param_patch import smoke_param_patch

    result = smoke_param_patch(Path(args.out))
    print(f"[param-patch] smoke patches={result['n_patches']} out={result['out_dir']}")


def _add_common(parser: argparse.ArgumentParser, default_out: str) -> None:
    parser.add_argument("--out", type=str, default=default_out)
    parser.add_argument("--steps", type=int, default=PHASE2.steps)
    parser.add_argument("--batch-size", type=int, default=PHASE2.batch_size)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--n-aliases", type=int, default=PHASE2.n_aliases)
    parser.add_argument("--train-frac", type=float, default=PHASE2.train_frac)
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--workers-per-gpu", type=int, default=1)
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Train the formal grid. Omit this flag to dry-run.",
    )


def add_causal_args(parser: argparse.ArgumentParser) -> None:
    _add_common(parser, "runs/phase2/causal_modulus/dry_run")


def add_coverage_args(parser: argparse.ArgumentParser) -> None:
    _add_common(parser, "runs/phase2/replay_coverage/dry_run")


def add_analyze_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--stamp",
        type=str,
        default="runs/phase2/next80_formal/01_core_fresh",
    )
    parser.add_argument("--out", type=str, default=None)


def add_patch_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase2/param_patch/smoke")
