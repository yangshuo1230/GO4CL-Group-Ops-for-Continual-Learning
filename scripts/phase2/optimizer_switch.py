"""Matched optimizer-switch trajectories; planning is the default, W&B disabled."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch

from go4cl.data.generate import build_shared_residue_splits
from go4cl.metrics.behavioral import evaluate
from go4cl.phases.phase2.mechanism_suite import (
    KINDS, Stream, atomic_save, digest, eval_loaders, gradient_snapshot,
    restore_rng, rng_state, schedule,
)
from go4cl.tasks.spec import TaskSpec
from go4cl.train.loop import TrainConfig, _train_callable, build_optimizer
from go4cl.utils.checkpoint import load_checkpoint, write_json

VERSION = 1
OBJECTIVES = ('A', 'B', 'mix_0.1')
POLICIES = ('fresh', 'inherit_A')


def file_hash(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def batch_hash(batch):
    h = hashlib.sha256()
    for key in sorted(batch):
        h.update(key.encode())
        h.update(batch[key].numpy().tobytes())
    return h.hexdigest()


def build_plan(args):
    jobs = []
    source_root = Path(args.source_root).resolve()
    for seed in args.seeds:
        for kind in args.kinds:
            source = source_root / f'seed{seed}/source/{kind}'
            resolved = json.loads((source / 'resolved.json').read_text())
            result = json.loads((source / 'result.json').read_text())
            if result['status'] != 'complete' or not result['source_mastered']:
                raise RuntimeError(f'Source not complete/mastered: {source}')
            checkpoint = source / 'checkpoints/final.pt'
            identity = dict(path=str(checkpoint), sha256=file_hash(checkpoint),
                            fingerprint=result['fingerprint'])
            cfg = copy.deepcopy(resolved['config'])
            cfg['train']['max_steps'] = args.max_steps
            # Extra very early observations; main schedule remains unchanged.
            cfg['gradient_steps'] = sorted(set(cfg['gradient_steps']) | {1, 10})
            for objective in args.objectives:
                for policy in args.policies:
                    job = dict(id=f'seed{seed}/{kind}/{objective}/{policy}', seed=seed,
                               A=resolved['job']['A'], B=resolved['job']['B'],
                               objective=objective, optimizer_policy=policy,
                               replay_ratio=.1 if objective == 'mix_0.1' else 0.,
                               a_stream=args.a_stream, source=identity, config=cfg)
                    jobs.append(job)
    return jobs


def points_for(cfg):
    return sorted(set(schedule(cfg)) | {
        s for s in [1, 2, 5, 10, 20, 50] if s <= cfg['train']['max_steps']})


def setup(job, device):
    cfg = job['config']
    a, b = TaskSpec.from_dict(job['A']), TaskSpec.from_dict(job['B'])
    splits = build_shared_residue_splits([a, b], data_seed=cfg['data_seed'])
    model, donor = load_checkpoint(job['source']['path'], map_location='cpu')
    if donor['fingerprint'] != job['source']['fingerprint']:
        raise RuntimeError('Source checkpoint fingerprint differs from source result')
    model.to(device)
    tc = TrainConfig(**cfg['train'], device=str(device), log_to_wandb=False)
    optimizer = build_optimizer(model, tc)
    if job['optimizer_policy'] == 'inherit_A':
        optimizer.load_state_dict(copy.deepcopy(donor['optimizer_state']))
        for pg in optimizer.param_groups:
            if (pg['lr'], pg['weight_decay'], tuple(pg['betas'])) != (tc.lr, tc.weight_decay, (.9, .98)):
                raise RuntimeError('Inherited optimizer hyperparameters do not match requested training')
    stream = Stream(a, b, splits, tc.batch_size, job['replay_ratio'],
                    cfg['sampler_seed'], source=job['objective'] == 'A')
    if job['objective'] == 'A' and job['a_stream'] == 'continue':
        state = copy.deepcopy(donor['stream_state'])
        state['counts'] = {'A': 0, 'B': 0}
        stream.load_state_dict(state)
    # Optimizer-policy pairs also start with identical global RNG states.
    state = copy.deepcopy(donor['rng_state'])
    if device.type == 'cpu':
        state['cuda'] = None
    restore_rng(state)
    return model, optimizer, stream, donor, tc, a, b, splits


def run_job(job, root, device):
    torch.set_num_threads(1)
    device = torch.device(device)
    cfg = job['config']
    directory = root / job['id']
    directory.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(dict(version=VERSION, job=job))
    result_path = directory / 'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise RuntimeError('Existing result has different settings; choose a new output root')
        print(f"skip {job['id']}", flush=True)
        return
    lock = directory / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    started = time.time()
    try:
        if file_hash(Path(job['source']['path'])) != job['source']['sha256']:
            raise RuntimeError('Source checkpoint changed after planning')
        model, optimizer, stream, donor, tc, a, b, splits = setup(job, device)
        write_json(directory / 'resolved.json', dict(job=job, config=cfg, fingerprint=fingerprint))
        write_json(directory / 'data_manifest.json', dict(
            residue_splits={str(p): s.to_dict() for p, s in splits.items()},
            sampler_seed=cfg['sampler_seed'], eval_seed=cfg['eval_seed'],
            a_stream=job['a_stream'] if job['objective'] == 'A' else 'restart',
            definition='Same weights, batches, evaluation, LR, betas, decay and clipping within policy pairs'))
        loaders = eval_loaders(a, b, splits, cfg)
        cpdir = directory / 'checkpoints'
        cpdir.mkdir(exist_ok=True)
        start, stable, history, events = 0, 0, [], {}
        latest = cpdir / 'latest.pt'
        if latest.exists():
            payload = torch.load(latest, map_location='cpu', weights_only=False)
            if payload['fingerprint'] != fingerprint:
                raise RuntimeError('Resume checkpoint settings mismatch')
            model.load_state_dict(payload['model_state'])
            optimizer.load_state_dict(payload['optimizer_state'])
            stream.load_state_dict(payload['stream_state'])
            restore_rng(payload['rng_state'])
            start, stable = payload['step'], payload['stable']
            history, events = payload['history'], payload['events']
        train_model = _train_callable(model, device, compile_model=tc.compile_model)
        points = set(points_for(cfg))
        # A resumed checkpoint can precede its diagnostic save if interrupted.
        if start > 0 and start in cfg['gradient_steps']:
            path = directory / 'gradients' / f'step_{start:06d}.pt'
            if not path.exists():
                gradient_snapshot(model, optimizer, a, b, splits, loaders, cfg,
                                  job, start, path, device)
        for step in range(start, tc.max_steps + 1):
            update = None
            if step != start:
                model.train()
                batch = stream.next()
                # Record the actual one-step delta at each observation point.
                before = {n: p.detach().clone() for n, p in model.named_parameters()} if step in points else None
                optimizer.zero_grad(set_to_none=True)
                try:
                    out = train_model(batch['tokens'].to(device), batch['labels'].to(device))
                except Exception:
                    if train_model is model:
                        raise
                    object.__setattr__(model, '_go4cl_compiled', False)
                    train_model = model
                    out = model(batch['tokens'].to(device), batch['labels'].to(device))
                out['loss'].backward()
                norm = torch.nn.utils.clip_grad_norm_(model.parameters(), tc.grad_clip)
                optimizer.step()
                if before is not None:
                    delta2 = sum((p.detach() - before[n]).double().square().sum() for n, p in model.named_parameters())
                    update = dict(train_loss=float(out['loss'].detach()),
                                  raw_gradient_norm_before_clip=float(norm),
                                  actual_update_norm=float(delta2.sqrt()),
                                  batch_sha256=batch_hash(batch))
            if step not in points or (step == start and start > 0):
                continue
            model.eval()
            metrics = {name: evaluate(model, loader, device).to_dict() for name, loader in loaders.items()}
            row = dict(step=step, metrics=metrics, exposure=stream.counts.copy(),
                       last_update=update, elapsed_seconds=time.time() - started)
            history.append(row)
            mastered = all(v['accuracy'] >= cfg['mastery_accuracy'] for v in metrics['A']['by_operation'].values())
            stable = stable + 1 if mastered else 0
            for name in ('A', 'B'):
                if all(v['accuracy'] >= cfg['mastery_accuracy'] for v in metrics[name]['by_operation'].values()):
                    events.setdefault(name + '_first_pass', step)
            payload = dict(model_state=model.state_dict(), model_config=model.cfg.to_dict(),
                           optimizer_state=optimizer.state_dict(), step=step, fingerprint=fingerprint,
                           stream_state=stream.state_dict(), rng_state=rng_state(),
                           stable=stable, history=history, events=events,
                           meta=dict(job=job, task_A=job['A'], task_B=job['B'],
                                     optimizer_policy=job['optimizer_policy'], source_step=donor['step']))
            path = cpdir / f'step_{step:06d}.pt'
            atomic_save(payload, path)
            tmp = cpdir / 'latest.link'
            tmp.unlink(missing_ok=True)
            os.link(path, tmp)
            tmp.replace(latest)
            write_json(directory / 'history.json', history)
            print(f"{job['id']} step={step} A={metrics['A']['macro_operation_accuracy']:.4f} B={metrics['B']['macro_operation_accuracy']:.4f}", flush=True)
            if step in cfg['gradient_steps']:
                gradient_snapshot(model, optimizer, a, b, splits, loaders, cfg,
                                  job, step, directory / 'gradients' / f'step_{step:06d}.pt', device)
        final = cpdir / 'final.pt'
        if not final.exists():
            os.link(latest, final)
        tests = {name: evaluate(model, loader, device).to_dict() for name, loader in eval_loaders(a, b, splits, cfg, 'test').items()}
        write_json(result_path, dict(status='complete', fingerprint=fingerprint,
                   source_mastered=True, optimizer_policy=job['optimizer_policy'], objective=job['objective'],
                   A_final_window_passed=stable >= cfg['mastery_window'], events=events,
                   final_val=history[-1]['metrics'], final_test=tests, query_exposure=stream.counts,
                   effective_replay_ratio=stream.na / (stream.na + stream.nb),
                   training_and_analysis_seconds=time.time() - started))
    finally:
        lock.unlink(missing_ok=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-root', default='runs/phase2/mechanism_suite_20261009')
    p.add_argument('--root', default='runs/phase2/optimizer_switch_20261010')
    p.add_argument('--seeds', nargs='+', type=int, default=[0, 1])
    p.add_argument('--kinds', nargs='+', choices=KINDS, default=list(KINDS))
    p.add_argument('--objectives', nargs='+', choices=OBJECTIVES, default=list(OBJECTIVES))
    p.add_argument('--policies', nargs='+', choices=POLICIES, default=list(POLICIES))
    p.add_argument('--max-steps', type=int, default=100000)
    p.add_argument('--a-stream', choices=['continue', 'restart'], default='continue')
    p.add_argument('--gpus', default='0')
    p.add_argument('--execute', action='store_true')
    p.add_argument('--dry-run', action='store_true')
    p.add_argument('--run-job', help=argparse.SUPPRESS)
    p.add_argument('--device', default='cuda:0')
    args = p.parse_args()
    if args.max_steps < 1:
        p.error('--max-steps must be positive')
    root = Path(args.root).resolve()
    if args.run_job:
        if not args.execute or args.dry_run:
            p.error('--run-job requires --execute')
        plan = json.loads((root / 'plan.json').read_text())
        run_job(next(j for j in plan['jobs'] if j['id'] == args.run_job), root, args.device)
        return
    jobs = build_plan(args)
    if len({j['id'] for j in jobs}) != len(jobs):
        p.error('Duplicate seed, kind, objective or policy')
    plan = dict(version=VERSION, jobs=jobs, unique_jobs=len(jobs),
                checkpoint_steps=points_for(jobs[0]['config']),
                nominal_training_steps=len(jobs) * args.max_steps,
                definitions=dict(fresh='Reset all AdamW moments and step counter',
                                 inherit_A='Load all source AdamW moments and step counter',
                                 A='Continue A only; no task-token change',
                                 B='B only', mix_0_1='90% B + 10% A'))
    root.mkdir(parents=True, exist_ok=True)
    path = root / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError('Existing plan differs; choose a new output root')
    write_json(path, plan)
    print(json.dumps(dict(jobs=len(jobs), seeds=args.seeds, steps_per_job=args.max_steps,
                         checkpoints_per_job=len(plan['checkpoint_steps']), plan=str(path)), indent=2), flush=True)
    if not args.execute or args.dry_run:
        print('Plan only; no training started.')
        return
    if args.device == 'cpu':
        for job in jobs:
            run_job(job, root, 'cpu')
        return
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        p.error('--gpus must contain distinct integer IDs')

    def worker(gpu, subset):
        for job in subset:
            log = root / job['id'] / 'stdout.log'
            log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', WANDB_MODE='disabled')
            cmd = [sys.executable, str(Path(__file__).resolve()), '--root', str(root),
                   '--execute', '--run-job', job['id'], '--device', 'cuda:0']
            with log.open('a') as handle:
                print(f"GPU {gpu}: {job['id']} -> {log}", flush=True)
                subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures = [pool.submit(worker, gpu, jobs[i::len(gpus)]) for i, gpu in enumerate(gpus)]
        for future in futures:
            future.result()


if __name__ == '__main__':
    main()
