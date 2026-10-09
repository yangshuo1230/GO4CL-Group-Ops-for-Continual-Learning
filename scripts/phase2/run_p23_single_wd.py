"""Single-task p=23 weight-decay check.

One task, four moduli (23, 29, 31, 37), the existing causal construction-0
B split. Two weight decays, four model seeds. Does not write into the formal
causal run directory.
"""

from __future__ import annotations

import json
from pathlib import Path

import os
import subprocess
import sys

from go4cl.defaults import MODEL, PHASE2
from go4cl.phases.common import TrainJob, execute_train_job, resolve_gpus, utc_now
from go4cl.phases.phase2.next_launch import pin_jobs

DATA_DIR = Path("runs/phase2/causal_modulus/formal/data/causal_p23_c0_same")
OUT = Path("runs/phase2/p23_single_wd")
WEIGHT_DECAYS = (0.1, 0.2)
SEEDS = (0, 1, 2, 3)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    group = f"p23_single_wd_{utc_now()}"
    jobs: list[TrainJob] = []
    for wd in WEIGHT_DECAYS:
        tag = f"wd{wd:.1f}".replace(".", "p")
        for seed in SEEDS:
            job_id = f"b_only_p23_c0_{tag}_ms{seed}"
            jobs.append(
                TrainJob(
                    job_id=job_id,
                    data_dir=str(DATA_DIR),
                    out_dir=str(OUT / "runs" / job_id),
                    gpu=0,
                    steps=PHASE2.steps,
                    lr=MODEL.lr,
                    weight_decay=float(wd),
                    d_model=MODEL.d_model,
                    n_layers=MODEL.n_layers,
                    n_heads=MODEL.n_heads,
                    model_seed=int(seed),
                    batch_size=PHASE2.batch_size,
                    wandb_project="go4cl",
                    wandb_group=group,
                    wandb_mode="disabled",
                    wandb_tags=("phase2", "b_only", "p23", tag),
                    wandb_config={
                        "experiment": "p23_single_task_wd",
                        "target_modulus": 23,
                        "construction": 0,
                        "protocol": "b_only",
                        "weight_decay": float(wd),
                        "moduli": [23, 29, 31, 37],
                    },
                    protocol="b_only",
                    include_test=True,
                    switch_on="fixed",
                    optimizer_transition="fresh",
                )
            )
    gpus = resolve_gpus("0,1,2,3,4,5,6,7")
    jobs = pin_jobs(jobs, gpus, 1)
    (OUT / "jobs.json").write_text(
        json.dumps(
            [
                {
                    "job_id": job.job_id,
                    "gpu": job.gpu,
                    "wd": job.weight_decay,
                    "seed": job.model_seed,
                    "out": job.out_dir,
                }
                for job in jobs
            ],
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    if len(sys.argv) > 1 and sys.argv[1] == "--worker":
        index = int(sys.argv[2])
        job = jobs[index]
        # Visible devices are fixed by the parent before this process starts.
        # Setting the id again after import would not move an existing context.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        result = execute_train_job(job)
        print(
            f"{job.job_id} -> {result.get('status')} ({result.get('elapsed_sec')}s)",
            flush=True,
        )
        if result.get("status") != "ok":
            print(result.get("error"), flush=True)
            raise SystemExit(1)
        return
    print(f"launching {len(jobs)} at {utc_now()}", flush=True)
    print("pin", [(job.job_id, job.gpu) for job in jobs], flush=True)
    children = []
    for index, job in enumerate(jobs):
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(job.gpu)
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        children.append(
            subprocess.Popen(
                [sys.executable, __file__, "--worker", str(index)],
                env=env,
            )
        )
    failed = 0
    for child, job in zip(children, jobs):
        code = child.wait()
        print(f"{job.job_id} exit={code}", flush=True)
        if code != 0:
            failed += 1
    print(f"done ok={len(jobs) - failed} bad={failed}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
