"""Paired causal suite. Planning is the default; GPU execution is explicit.

Independent entry point: existing phase-2 protocols and pending edits are untouched.
All branch optimizers are fresh AdamW. The final fixed-budget A checkpoint is
the common donor, conditional on five consecutive per-operation validation passes.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import yaml

from go4cl.data.context import ContextBuilder
from go4cl.data.eval_contexts import make_eval_context_loader
from go4cl.data.generate import build_shared_residue_splits
from go4cl.data.packed import replay_pack_counts
from go4cl.metrics.behavioral import evaluate
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.transfer_mechanism.checkpoint_mix import build_hybrid, registry_for_model
from go4cl.tasks.spec import Operation, TaskSpec
from go4cl.train.loop import TrainConfig, _train_callable, build_optimizer
from go4cl.utils.checkpoint import write_json

VERSION = 1
KINDS = ('same_mod_same_pos', 'same_mod_diff_pos', 'diff_mod_same_pos', 'diff_mod_diff_pos')


def task(name, moduli, edges, slots, task_id):
    return TaskSpec(name, task_id, tuple(Operation(z, *edges[z], moduli[z], slots[z]) for z in range(4)))


def tasks():
    b = task('B', (23, 41, 31, 47), ((0, 1), (2, 3), (4, 5), (6, 7)), (0, 1, 2, 3), 1)
    a = {}
    for kind in KINDS:
        moduli = (23, 41, 37, 53) if kind.startswith('same_mod') else (29, 43, 37, 53)
        edges = ((0, 1), (2, 3)) if kind.endswith('same_pos') else ((0, 2), (1, 3))
        a[kind] = task('A', moduli, edges + ((4, 6), (5, 7)), (1, 0, 3, 2), 0)
    overlap = {m: task('B', mods, tuple((o.i, o.j) for o in b.operations), (0, 1, 2, 3), 1)
               for m, mods in [('m0', (29, 43, 31, 47)), ('m0.5', (23, 41, 31, 47)),
                               ('m1', (23, 41, 37, 53))]}
    return a, b, overlap


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def build_plan(cfg):
    aa, fixed_b, overlap = tasks()
    jobs = {}

    def add(seed, label, a, b, *, stage, source=None, intervention='full_A', ratio=0.):
        jid = f'seed{seed}/{label}'
        if jid in jobs:
            if stage not in jobs[jid]['stages']:
                jobs[jid]['stages'].append(stage)
            return jid
        jobs[jid] = dict(id=jid, seed=seed, stages=[stage], source=source,
                         A=a.to_dict(), B=b.to_dict(), intervention=intervention,
                         replay_ratio=ratio, source_job=label.startswith('source/'),
                         gradients=(source is not None and intervention == 'full_A'
                                    and stage != 'components'))
        return jid

    for seed in cfg['seeds']:
        sources = {kind: add(seed, f'source/{kind}', a, fixed_b, stage='core') for kind, a in aa.items()}
        add(seed, 'main/B_only', aa[KINDS[1]], fixed_b, stage='core', intervention='full_fresh')
        for kind, a in aa.items():
            add(seed, f'main/{kind}/full_A', a, fixed_b, stage='core', source=sources[kind])
            add(seed, f'main/{kind}/replay_0.1', a, fixed_b, stage='core', source=sources[kind], ratio=.1)
            for group in cfg['component_groups']:
                for action in ('reset', 'keep'):
                    add(seed, f'main/{kind}/{action}_{group}', a, fixed_b, stage='components',
                        source=sources[kind], intervention=f'{action}_{group}')
        anchor = KINDS[1]
        for m, b in overlap.items():
            if m != 'm0.5':
                add(seed, f'overlap/{m}/B_only', aa[anchor], b, stage='overlap', intervention='full_fresh')
            else:
                add(seed, 'main/B_only', aa[anchor], b, stage='overlap', intervention='full_fresh')
            for r in [0., *cfg['replay_ratios'], .1]:
                label = (f'main/{anchor}/' + ('full_A' if r == 0 else 'replay_0.1')) if m == 'm0.5' and r in (0., .1) else f'overlap/{m}/replay_{r:g}'
                add(seed, label, aa[anchor], b, stage='overlap', source=sources[anchor], ratio=r)
    return list(jobs.values())


def selected_jobs(plan, stages):
    chosen = {j['id'] for j in plan if set(j['stages']) & set(stages)}
    index = {j['id']: j for j in plan}
    for jid in list(chosen):
        if index[jid]['source']:
            chosen.add(index[jid]['source'])
    return [j for j in plan if j['id'] in chosen]


class Stream:
    """Online packed sampling with independently restorable A/B RNG states."""
    def __init__(self, a, b, splits, batch_size, ratio, seed, source=False):
        if batch_size < 4 or batch_size % 4:
            raise ValueError('batch_size must be a positive multiple of four')
        self.builders = {'A': ContextBuilder(a, splits), 'B': ContextBuilder(b, splits)}
        packs = batch_size // 4
        self.na, self.nb = ((packs, 0) if source else (0, packs) if ratio == 0 else replay_pack_counts(packs, ratio))
        self.rng = {'A': np.random.default_rng(seed), 'B': np.random.default_rng(seed + 10007)}
        self.counts = {'A': 0, 'B': 0}

    def next(self):
        parts = []
        for name, count in [('A', self.na), ('B', self.nb)]:
            if count:
                parts.append(self.builders[name].sample_train_batch(self.rng[name], count))
                self.counts[name] += count * 4
        return {key: torch.from_numpy(np.ascontiguousarray(np.concatenate([p[key] for p in parts]))) for key in parts[0]}

    def state_dict(self):
        return dict(rng={k: v.bit_generator.state for k, v in self.rng.items()}, counts=self.counts.copy())

    def load_state_dict(self, state):
        for key, value in state['rng'].items():
            self.rng[key].bit_generator.state = value
        self.counts = state['counts'].copy()


def eval_loaders(a, b, splits, cfg, split='val'):
    return {name: make_eval_context_loader(t, splits, target_split=split,
            n_per_operation=cfg['eval_per_operation'], seed=cfg['eval_seed'] + t.task_id * 1009 + (split == 'test'), batch_size=2048)
            for name, t in [('A', a), ('B', b)]}


def schedule(cfg):
    n = cfg['train']['max_steps']
    out = {0, n}
    for start, end, stride in [(0, min(n, 2000), cfg['checkpoint_early_every']),
                                (2000, min(n, 10000), cfg['checkpoint_middle_every']),
                                (10000, n, cfg['checkpoint_late_every'])]:
        out.update(range(start + stride, end + 1, stride))
    out.update(s for s in cfg['gradient_steps'] if s <= n)
    return sorted(out)


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_initialized() else None)


def restore_rng(state):
    random.setstate(state['python']); np.random.set_state(state['numpy']); torch.set_rng_state(state['torch'].cpu())
    if state['cuda'] is not None:
        torch.cuda.set_rng_state_all(state['cuda'])


def atomic_save(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    torch.save(payload, tmp)
    tmp.replace(path)


def raw_gradient(model, batch, device):
    model.zero_grad(set_to_none=True)
    out = model(batch['tokens'].to(device), batch['labels'].to(device))
    out['loss'].backward()
    return {n: (p.grad.detach().cpu().clone() if p.grad is not None else torch.zeros_like(p, device='cpu'))
            for n, p in model.named_parameters()}, float(out['loss'].detach())


def gradient_snapshot(model, optimizer, a, b, splits, loaders, cfg, job, step, path, device):
    """Clone real optimizer state: hypothetical A/B/mixed AdamW steps never mutate training."""
    saved_rng = rng_state()
    try:
        dstream = Stream(a, b, splits, cfg['diagnostic_batch_size'], .5, cfg['diagnostic_seed'])
        combined = dstream.next()
        batches = {name: {k: v[v_task] for k, v in combined.items()}
                   for name, v_task in [('A', combined['tokens'][:, 8] == a.task_token),
                                        ('B', combined['tokens'][:, 8] == b.task_token)]}
        clone = ModularTransformer(model.cfg).to(device)
        clone.load_state_dict(model.state_dict()); clone.eval()
        gradients, losses = {}, {}
        for name, batch in batches.items():
            gradients[name], losses[name] = raw_gradient(clone, batch, device)
            t = a if name == 'A' else b
            for op in t.operations:
                mask = batch['tokens'][:, 9] == 66 + op.slot
                sub = {k: v[mask] for k, v in batch.items()}
                key = f'{name}/lat{op.latent_id}/p{op.modulus}'
                gradients[key], losses[key] = raw_gradient(clone, sub, device)
        ratios = sorted({.1, float(job['replay_ratio'])} - {0.})
        for ratio in ratios:
            gradients[f'mix_{ratio:g}'] = {n: ratio * gradients['A'][n] + (1 - ratio) * gradients['B'][n] for n in gradients['A']}
        deltas, effects = {}, {}
        baseline = evaluate(clone, loaders['A'], device).loss
        base = {n: p.detach().clone() for n, p in model.named_parameters()}
        for objective in ['A', 'B', *[f'mix_{r:g}' for r in ratios]]:
            for decay in [True, False]:
                clone.load_state_dict(model.state_dict())
                opt = build_optimizer(clone, TrainConfig(**cfg['train'], device=str(device), log_to_wandb=False))
                opt.load_state_dict(copy.deepcopy(optimizer.state_dict()))
                if not decay:
                    for group in opt.param_groups:
                        group['weight_decay'] = 0.
                opt.zero_grad(set_to_none=True)
                for n, p in clone.named_parameters():
                    p.grad = gradients[objective][n].to(device).clone()
                clip = cfg['train']['grad_clip']
                norm = float(torch.nn.utils.clip_grad_norm_(clone.parameters(), clip)) if clip is not None else None
                opt.step()
                key = objective + ('/adamw' if decay else '/adam_no_decay')
                deltas[key] = {n: (p.detach() - base[n]).cpu() for n, p in clone.named_parameters()}
                effects[key] = dict(A_val_loss_before=baseline, A_val_loss_after=evaluate(clone, loaders['A'], device).loss,
                                    raw_norm_before_clip=norm)
        atomic_save(dict(step=step, gradients=gradients, diagnostic_train_losses=losses, deltas=deltas,
                         effects=effects, optimizer_state=optimizer.state_dict(),
                         parameter_groups=[s.to_dict() for s in registry_for_model(model)],
                         definition='delta = theta_after - theta_before; gradients use fixed train-only pairs; effects use held-out A'), path)
    finally:
        restore_rng(saved_rng)


def run_job(job, cfg, root, device):
    directory = root / job['id']; directory.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(dict(version=VERSION, job=job, config=cfg))
    result_path = directory / 'result.json'
    if result_path.exists():
        old = json.loads(result_path.read_text())
        if old['fingerprint'] != fingerprint:
            raise RuntimeError(f'Configuration changed for existing job {directory}; use a new run root')
        print(f"skip {job['id']}: {old['status']}", flush=True); return
    lock = directory / '.running'
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    os.close(fd)
    started = time.time()
    try:
        write_json(directory / 'resolved.json', dict(job=job, config=cfg, fingerprint=fingerprint))
        a = TaskSpec.from_dict(job['A']); b = TaskSpec.from_dict(job['B'])
        splits = build_shared_residue_splits([a, b], data_seed=cfg['data_seed'])
        write_json(directory / 'data_manifest.json', dict(tasks={'A': job['A'], 'B': job['B']},
                   residue_splits={str(p): s.to_dict() for p, s in splits.items()},
                   B_distribution_hash=digest(dict(task=job['B'], splits={str(p): splits[p].to_dict() for p in b.moduli()},
                                                  sampler_seed=cfg['sampler_seed'], eval_seed=cfg['eval_seed'])),
                   mode='online_packed_full_train_split', split_ratios=[.6, .2, .2]))
        random.seed(job['seed']); np.random.seed(job['seed']); torch.manual_seed(job['seed'])
        model = ModularTransformer(ModelConfig(**cfg['model']))
        theta0 = {n: v.detach().clone() for n, v in model.state_dict().items()}
        if job['source']:
            source_dir = root / job['source']
            source_result = json.loads((source_dir / 'result.json').read_text())
            if cfg['require_source_mastery'] and not source_result['source_mastered']:
                write_json(result_path, dict(status='blocked_source_not_mastered', fingerprint=fingerprint,
                                             source=job['source'], source_mastered=False)); return
            donor_path = source_dir / 'checkpoints' / 'final.pt'
            donor = torch.load(donor_path, map_location='cpu', weights_only=False)
            hybrid, manifest = build_hybrid(theta0, donor['model_state'], job['intervention'], registry_for_model(model),
                          n_layers=model.cfg.n_layers, task_seed=0, model_seed=job['seed'], dataset_hash=fingerprint)
            model.load_state_dict(hybrid)
            write_json(directory / 'intervention.json', dict(manifest=manifest, source=str(donor_path),
                       donor_fingerprint=source_result['fingerprint'], optimizer_policy='fresh_AdamW'))
        model.to(device)
        traincfg = TrainConfig(**cfg['train'], device=str(device), log_to_wandb=False)
        optimizer = build_optimizer(model, traincfg)
        stream = Stream(a, b, splits, traincfg.batch_size, job['replay_ratio'], cfg['sampler_seed'], job['source_job'])
        loaders = eval_loaders(a, b, splits, cfg)
        cpdir = directory / 'checkpoints'
        cpdir.mkdir(exist_ok=True)
        latest = cpdir / 'latest.pt'
        start = 0; stable = 0; history = []; events = {}
        if latest.exists():
            payload = torch.load(latest, map_location='cpu', weights_only=False)
            if payload['fingerprint'] != fingerprint:
                raise RuntimeError('Resume checkpoint fingerprint mismatch')
            model.load_state_dict(payload['model_state']); optimizer.load_state_dict(payload['optimizer_state'])
            stream.load_state_dict(payload['stream_state']); restore_rng(payload['rng_state'])
            start = payload['step']; stable = payload['stable']; history = payload['history']; events = payload['events']
        train_model = _train_callable(model, torch.device(device), compile_model=traincfg.compile_model)
        points = set(schedule(cfg))
        if start > 0 and job['gradients'] and start in cfg['gradient_steps']:
            diagnostic_path = directory / 'gradients' / f'step_{start:06d}.pt'
            if not diagnostic_path.exists():
                gradient_snapshot(model, optimizer, a, b, splits, loaders, cfg, job, start, diagnostic_path, torch.device(device))
        for step in range(start, traincfg.max_steps + 1):
            if step != start:
                model.train(); batch = stream.next(); optimizer.zero_grad(set_to_none=True)
                try:
                    out = train_model(batch['tokens'].to(device), batch['labels'].to(device))
                except Exception:
                    if train_model is model:
                        raise
                    object.__setattr__(model, '_go4cl_compiled', False); train_model = model
                    out = model(batch['tokens'].to(device), batch['labels'].to(device))
                out['loss'].backward()
                if traincfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), traincfg.grad_clip)
                optimizer.step()
            if step not in points or (step == start and start > 0):
                continue
            model.eval()
            metrics = {name: evaluate(model, loader, torch.device(device)).to_dict() for name, loader in loaders.items()}
            row = dict(step=step, metrics=metrics, exposure=stream.counts.copy(), elapsed_seconds=time.time() - started)
            history.append(row)
            mastered = all(v['accuracy'] >= cfg['mastery_accuracy'] for v in metrics['A']['by_operation'].values())
            stable = stable + 1 if mastered else 0
            for name in ['A', 'B']:
                ok = all(v['accuracy'] >= cfg['mastery_accuracy'] for v in metrics[name]['by_operation'].values())
                if ok and name + '_first_pass' not in events:
                    events[name + '_first_pass'] = step
            payload = dict(model_state=model.state_dict(), model_config=model.cfg.to_dict(), optimizer_state=optimizer.state_dict(),
                           step=step, fingerprint=fingerprint, stream_state=stream.state_dict(), rng_state=rng_state(),
                           stable=stable, history=history, events=events,
                           meta=dict(job=job, task_A=job['A'], task_B=job['B'], optimizer_policy='fresh_at_switch'))
            path = cpdir / f'step_{step:06d}.pt'; atomic_save(payload, path)
            # latest is a small pointer on disk (hard link) to the same immutable checkpoint.
            tmp = cpdir / 'latest.link'
            if tmp.exists(): tmp.unlink()
            os.link(path, tmp); tmp.replace(latest)
            write_json(directory / 'history.json', history)
            print(f"{job['id']} step={step} A={metrics['A']['macro_operation_accuracy']:.4f} B={metrics['B']['macro_operation_accuracy']:.4f}", flush=True)
            if job['gradients'] and step in cfg['gradient_steps']:
                gradient_snapshot(model, optimizer, a, b, splits, loaders, cfg, job, step,
                                  directory / 'gradients' / f'step_{step:06d}.pt', torch.device(device))
        final = cpdir / 'final.pt'
        if not final.exists():
            os.link(latest, final)
        tests = {name: evaluate(model, loader, torch.device(device)).to_dict() for name, loader in eval_loaders(a, b, splits, cfg, 'test').items()}
        source_mastered = (stable >= cfg['mastery_window'] if job['source_job'] else
                           source_result['source_mastered'] if job['source'] else None)
        write_json(result_path, dict(status='complete', fingerprint=fingerprint, source_mastered=source_mastered,
                   A_final_window_passed=stable >= cfg['mastery_window'],
                   events=events, final_val=history[-1]['metrics'], final_test=tests,
                   query_exposure=stream.counts, requested_replay_ratio=job['replay_ratio'],
                   effective_replay_ratio=stream.na / (stream.na + stream.nb) if not job['source_job'] else None,
                   training_and_analysis_seconds=time.time() - started, peak_cuda_bytes=torch.cuda.max_memory_allocated() if str(device).startswith('cuda') else 0))
    finally:
        lock.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/phase2/mechanism_suite.yaml')
    parser.add_argument('--root', default='runs/phase2/mechanism_suite_20261009')
    parser.add_argument('--stage', choices=['all', 'core', 'components', 'overlap'], default='all')
    parser.add_argument('--gpus', default='0', help='Explicit physical GPU IDs, e.g. 0,1; one worker per GPU')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--run-job', help=argparse.SUPPRESS)
    parser.add_argument('--device', default='cuda:0')
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    plan = build_plan(cfg)
    stages = ['core', 'components', 'overlap'] if args.stage == 'all' else [args.stage]
    chosen = selected_jobs(plan, stages)
    root = Path(args.root).resolve(); root.mkdir(parents=True, exist_ok=True)
    if args.run_job:
        if not args.execute or args.dry_run:
            parser.error('--run-job requires --execute')
        run_job(next(j for j in plan if j['id'] == args.run_job), cfg, root, args.device); return
    write_json(root / f'plan_{args.stage}.json', dict(version=VERSION, config=cfg, jobs=chosen,
               unique_jobs=len(chosen), all_jobs=len(plan), checkpoint_steps=schedule(cfg),
               nominal_training_steps=sum(cfg['train']['max_steps'] for _ in chosen)))
    print(json.dumps(dict(seeds=cfg['seeds'], stage=args.stage, jobs=len(chosen), all_jobs=len(plan),
                         sources=sum(j['source_job'] for j in chosen), checkpoints_per_job=len(schedule(cfg)),
                         plan=str(root / f'plan_{args.stage}.json')), indent=2), flush=True)
    if not args.execute or args.dry_run:
        print('Dry run only; no training started.'); return
    if args.device == 'cpu':
        for job in sorted(chosen, key=lambda j: not j['source_job']):
            run_job(job, cfg, root, 'cpu')
        return
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        parser.error('--gpus must contain distinct integer GPU IDs')

    def worker(gpu, jobs):
        for job in jobs:
            log = root / job['id'] / 'stdout.log'; log.parent.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, OMP_NUM_THREADS='1', WANDB_MODE='disabled')
            cmd = [sys.executable, '-m', 'go4cl.phases.phase2.mechanism_suite', '--config', str(Path(args.config).resolve()), '--root', str(root),
                   '--execute', '--run-job', job['id'], '--device', 'cuda:0']
            with log.open('a') as handle:
                print(f"GPU {gpu}: {job['id']} -> {log}", flush=True)
                subprocess.run(cmd, env=env, stdout=handle, stderr=subprocess.STDOUT, check=True)

    # Barrier between sources and branches; each GPU has exactly one sequential worker.
    for source_phase in [True, False]:
        phase = [j for j in chosen if j['source_job'] == source_phase]
        with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
            futures = [pool.submit(worker, gpu, phase[i::len(gpus)]) for i, gpu in enumerate(gpus)]
            for future in futures: future.result()


if __name__ == '__main__':
    main()
