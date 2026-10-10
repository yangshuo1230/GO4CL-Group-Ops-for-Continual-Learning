#!/usr/bin/env bash
# Run inside the GO4CL project. Creates a NEW result directory; never overwrites runs.
# Example: bash run_single_op_suite.sh --gpus 0,1,2,3,4,5,6,7
# Preview: bash run_single_op_suite.sh --dry-run
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-/mnt/ningxuefei/yangshuo/GO4CL-Group-Ops-for-Continual-Learning}"
cd "$PROJECT_ROOT"
source scripts/env.sh
UV_BIN="$(command -v uv || true)"
if [[ -z "$UV_BIN" ]]; then
  UV_BIN="${HOME}/.local/bin/uv"
fi
"$UV_BIN" run python - "$@" <<'PY'
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import textwrap

from go4cl.constants import NUM_DIGITS, NUM_OUTPUT_CLASSES
from go4cl.data.generate import generate_task_datasets, save_datasets
from go4cl.data.manifest import hash_payload
from go4cl.data.residue_pairs import assert_disjoint
from go4cl.defaults import MODEL, PHASE2
from go4cl.tasks.relations import TaskPairSpec
from go4cl.tasks.spec import Operation, TaskSpec


# Each training subprocess sees ONE GPU before torch is imported.
# Reuse the existing model/data/training implementation, not a second trainer.
WORKER = textwrap.dedent('''
import json
from pathlib import Path
import sys
import traceback
import time
import torch
from go4cl.model.transformer import ModelConfig
from go4cl.train.loop import TrainConfig
from go4cl.train.protocols import run_protocol

job = json.loads(sys.argv[1])
out = Path(job["out_dir"])
out.mkdir(parents=True, exist_ok=False)
started = time.monotonic()
result = {"job_id": job["job_id"], "protocol": job["protocol"], "status": "running"}
try:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; refusing accidental CPU training")
    model_cfg = ModelConfig(
        d_model=job["d_model"], n_layers=job["n_layers"], n_heads=job["n_heads"],
        d_mlp=4 * job["d_model"], activation="relu",
    )
    train_cfg = TrainConfig(
        lr=job["lr"], weight_decay=job["weight_decay"],
        batch_size=job["batch_size"], train_replacement=True,
        max_steps=job["phase_steps"], eval_every=job["eval_every"],
        ckpt_every=job["ckpt_every"], device="cuda", null_task_tokens=False,
        log_to_wandb=False,
    )
    protocol = run_protocol(
        job["protocol"], job["data_dir"], str(out),
        model_cfg=model_cfg, train_cfg=train_cfg,
        model_seed=job["model_seed"], sampler_seed=job["sampler_seed"],
        phase_steps=job["phase_steps"], include_test=True,
        switch_on="fixed", theta_a_ckpt=job.get("theta_a_ckpt"),
        optimizer_transition="fresh", wandb_enabled=False,
        eval_n_per_operation=1024, wandb_config={"single_op_suite": job},
    )
    result.update(status="ok", metrics=protocol.metrics)
except Exception:
    result.update(status="error", traceback=traceback.format_exc())
result["elapsed_sec"] = time.monotonic() - started
(out / "job_result.json").write_text(json.dumps(result, indent=2) + "\\n")
print(json.dumps(result), flush=True)
sys.exit(0 if result["status"] == "ok" else 1)
''')


def main():
    parser = argparse.ArgumentParser(description="Single-operation A/B continual-learning suite")
    parser.add_argument("--gpus", default="0,1,2,3,4,5,6,7")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--p", type=int, default=31)
    parser.add_argument("--q", type=int, default=41)
    parser.add_argument("--a-steps", type=int, default=20000)
    parser.add_argument("--steps", type=int, default=10000, help="Actual updates for EVERY second-stage/baseline job, including joint")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--train-frac", type=float, default=0.8)
    parser.add_argument("--wd", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=MODEL.lr)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--eval-every", type=int, default=500)
    parser.add_argument("--ckpt-every", type=int, default=500)
    parser.add_argument("--out", default="runs/phase2/single_op/" + datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--dry-run", action="store_true", help="Build and audit data/jobs, but do not train")
    args = parser.parse_args()

    gpus = [int(g) for g in args.gpus.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        parser.error("GPU IDs must be unique nonnegative integers")
    if len(set(args.seeds)) != len(args.seeds) or min(args.seeds) < 0:
        parser.error("Seeds must be unique nonnegative integers")
    if args.p == args.q or min(args.p, args.q) < 2:
        parser.error("p and q must be distinct and >= 2")
    if max(args.p, args.q) > min(NUM_DIGITS, NUM_OUTPUT_CLASSES):
        parser.error("Moduli exceed the existing digit/output vocabulary")
    if args.steps < 2 or args.steps % 2 or args.a_steps < 1:
        parser.error("--steps must be positive and even (joint doubles phase_steps)")
    if args.batch_size < 2 or args.batch_size % 2:
        parser.error("--batch-size must be positive and even")
    if not 0.05 < args.train_frac < 0.95:
        parser.error("--train-frac must lie between 0.05 and 0.95")
    if min(args.eval_every, args.ckpt_every) < 1:
        parser.error("Evaluation/checkpoint intervals must be positive")
    if abs(PHASE2.sequential_ab_replay_ratio - 0.1) > 1e-12:
        parser.error("Existing trainer replay default is not 0.1; refusing mislabeled replay runs")

    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=False)
    (root / "logs").mkdir()
    remainder = (1 - args.train_frac) / 2
    task_a = TaskSpec("A", 0, (Operation(0, 0, 1, args.p, 0),))
    conditions = [
        ("C1_same_route_same_mod", 0, 1, args.p),
        ("C2_diff_route_same_mod", 2, 3, args.p),
        ("C3_same_route_diff_mod", 0, 1, args.q),
        ("C4_diff_route_diff_mod", 2, 3, args.q),
    ]
    datasets = {}
    reference_a_split = None
    for name, i, j, modulus in conditions:
        task_b = TaskSpec("B", 1, (Operation(0, i, j, modulus, 0),))
        pair = TaskPairSpec(task_a, task_b, 1., float(i == 0), float(modulus == args.p), 0, name)
        manifest, data = generate_task_datasets(
            pair, data_seed=args.data_seed, n_aliases_per_pair=16,
            n_nuisance_contexts=1, ratios=(args.train_frac, remainder, remainder),
            experiment_id=name, skip_train=True,
        )
        manifest.fixed_a = True
        manifest.task_a_hash = hash_payload(task_a.to_dict())
        for split in manifest.residue_splits.values():
            assert_disjoint(split)
        a_split = manifest.residue_splits[args.p].to_dict()
        if reference_a_split is not None and reference_a_split != a_split:
            raise RuntimeError("Task A residue split changed across conditions")
        reference_a_split = a_split
        data_dir = root / "data" / name
        save_datasets(data_dir, manifest, data)
        datasets[name] = str(data_dir)

    common = dict(
        lr=args.lr, weight_decay=args.wd, batch_size=args.batch_size,
        d_model=MODEL.d_model, n_layers=MODEL.n_layers, n_heads=MODEL.n_heads,
        eval_every=args.eval_every, ckpt_every=args.ckpt_every,
        train_frac=args.train_frac, data_seed=args.data_seed,
        optimizer_transition="fresh", replay_ratio_requested=0.1,
    )
    a_jobs, main_jobs = [], []
    protocols = ("b_only", "sequential_ab", "sequential_ab_replay", "joint")
    for seed in args.seeds:
        a_id = f"shared_A_ms{seed}"
        a_out = root / "runs" / a_id
        a_jobs.append(dict(
            common, job_id=a_id, protocol="a_only", condition="shared_A",
            model_seed=seed, sampler_seed=seed, data_dir=datasets[conditions[0][0]],
            out_dir=str(a_out), phase_steps=args.a_steps, actual_updates=args.a_steps,
        ))
        for name, _, _, _ in conditions:
            for protocol in protocols:
                job_id = f"{name}_{protocol}_ms{seed}"
                job = dict(
                    common, job_id=job_id, protocol=protocol, condition=name,
                    model_seed=seed, sampler_seed=seed, data_dir=datasets[name],
                    out_dir=str(root / "runs" / job_id), actual_updates=args.steps,
                    phase_steps=args.steps // 2 if protocol == "joint" else args.steps,
                )
                if protocol.startswith("sequential_ab"):
                    job["theta_a_ckpt"] = str(a_out / "ckpts" / "final.pt")
                main_jobs.append(job)

    # Fixed lanes: at most ONE process per selected GPU, even when jobs finish unevenly.
    for jobs in (a_jobs, main_jobs):
        for index, job in enumerate(jobs):
            job["gpu"] = gpus[index % len(gpus)]
    plan = dict(
        args=vars(args), root=str(root), n_jobs=len(a_jobs) + len(main_jobs),
        total_updates=sum(j["actual_updates"] for j in a_jobs + main_jobs),
        jobs=a_jobs + main_jobs,
        notes=[
            "No mastery gate, no pilot/calibration; shared A uses its fixed-budget FINAL checkpoint.",
            "Joint is exactly --steps updates, NOT twice --steps; compare B exposure separately.",
            "Training uses fixed residue pools with online raw aliases/nuisance digits, not a fixed list of complete sequences.",
            "All outputs are isolated under this new root; no existing experiment is overwritten.",
        ],
    )
    (root / "jobs.json").write_text(json.dumps(plan, indent=2) + "\n")
    print(f"OUT={root}\nJOBS={plan['n_jobs']} TOTAL_UPDATES={plan['total_updates']}", flush=True)
    if args.dry_run:
        print("Dry run complete: no training launched.", flush=True)
        return

    def run_stage(jobs):
        def lane(gpu):
            records = []
            for job in (j for j in jobs if j["gpu"] == gpu):
                env = os.environ.copy()
                env.update(CUDA_VISIBLE_DEVICES=str(gpu), CUDA_DEVICE_ORDER="PCI_BUS_ID", OMP_NUM_THREADS="1")
                # Separate per-device compile caches avoid concurrent cache writes.
                env["TORCHINDUCTOR_CACHE_DIR"] = str(root / "cache" / f"gpu{gpu}")
                print(f"START gpu={gpu} {job['job_id']}", flush=True)
                with (root / "logs" / f"{job['job_id']}.log").open("w") as log:
                    child = subprocess.run(
                        [sys.executable, "-u", "-c", WORKER, json.dumps(job)],
                        env=env, stdout=log, stderr=subprocess.STDOUT,
                    )
                record = dict(job_id=job["job_id"], gpu=gpu, exit_code=child.returncode)
                records.append(record)
                print(f"DONE gpu={gpu} {job['job_id']} exit={child.returncode}", flush=True)
            return records
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            return [record for records in pool.map(lane, gpus) for record in records]

    print("Stage 1: fixed-budget shared A (no accuracy gate)", flush=True)
    results = run_stage(a_jobs)
    (root / "launcher_results.json").write_text(json.dumps(results, indent=2) + "\n")
    if any(record["exit_code"] for record in results):
        raise SystemExit("Shared A execution failed; inspect logs. B branches were not launched.")
    for job in a_jobs:
        if not (Path(job["out_dir"]) / "ckpts" / "final.pt").is_file():
            raise SystemExit(f"Missing A checkpoint: {job['job_id']}")
    print("Stage 2: ALL four conditions x four protocols x seeds", flush=True)
    results += run_stage(main_jobs)
    (root / "launcher_results.json").write_text(json.dumps(results, indent=2) + "\n")
    failed = sum(record["exit_code"] != 0 for record in results)
    print(f"COMPLETE ok={len(results)-failed} failed={failed} OUT={root}", flush=True)
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
PY
