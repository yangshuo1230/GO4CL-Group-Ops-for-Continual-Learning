"""One-layer A->B sweep; reuse mechanism_suite's stream, trainer and checkpoints."""
from __future__ import annotations

import argparse
import copy
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess
import sys

import yaml

from go4cl.phases.phase2.mechanism_suite import KINDS, digest, run_job, schedule
from go4cl.tasks.spec import Operation, TaskSpec
from go4cl.utils.checkpoint import write_json


def task_pair(cfg, n_ops, kind):
    """Fixed A across all B conditions; preserve the earlier two-operation edges."""
    if n_ops not in (1, 2) or kind not in KINDS:
        raise ValueError('Expected 1/2 operations and one of the four task relations')
    a_moduli = cfg['a_moduli'][:n_ops]
    b_moduli = a_moduli if kind.startswith('same_mod') else cfg['different_b_moduli'][:n_ops]
    a_edges = ((0, 1), (2, 3))[:n_ops]
    b_edges = a_edges if kind.endswith('same_pos') else ((0, 2), (1, 3))[:n_ops]
    a_slots = (0,) if n_ops == 1 else (1, 0)
    b_slots = tuple(range(n_ops))
    def make(name, task_id, moduli, edges, slots):
        return TaskSpec(name, task_id, tuple(Operation(z, *edges[z], moduli[z], slots[z]) for z in range(n_ops)))
    return make('A', 0, a_moduli, a_edges, a_slots), make('B', 1, b_moduli, b_edges, b_slots)


def validate_config(cfg):
    if cfg['model']['n_layers'] != 1 or cfg['model']['n_heads'] != 2:
        raise ValueError('This sweep requires one layer and two heads')
    widths = [(w['d_model'], w['d_mlp']) for w in cfg['widths']]
    if len(set(widths)) != len(widths) or any(w not in [(64, 256), (128, 512)] for w in widths) or not widths:
        raise ValueError('Widths must be distinct selections from (64,256) and (128,512)')
    for key in ['seeds', 'operations_per_task']:
        if not cfg[key] or len(set(cfg[key])) != len(cfg[key]):
            raise ValueError(f'{key} must be nonempty and unique')
    if any(s < 0 for s in cfg['seeds']) or any(n not in (1, 2) for n in cfg['operations_per_task']):
        raise ValueError('Invalid seeds or operation counts')
    if cfg['train']['max_steps'] < 1 or cfg['train']['batch_size'] < 4 or cfg['train']['batch_size'] % 2:
        raise ValueError('Positive update budget and an even batch of at least four are required')
    mods = cfg['a_moduli'] + cfg['different_b_moduli']
    if len(cfg['a_moduli']) != 2 or len(cfg['different_b_moduli']) != 2 or len(set(mods)) != 4 or any(p < 2 or p > 64 for p in mods):
        raise ValueError('Use two distinct A moduli and two disjoint B moduli in 2..64')


def build_plan(cfg):
    validate_config(cfg)
    jobs = []
    for n_ops in cfg['operations_per_task']:
        for width in cfg['widths']:
            model = dict(cfg['model'], **width)
            for seed in cfg['seeds']:
                prefix = f'ops{n_ops}/d{model["d_model"]}/seed{seed}'
                a, canonical_b = task_pair(cfg, n_ops, KINDS[0])
                source_id = prefix + '/source/A'
                common = dict(seed=seed, stages=['core'], A=a.to_dict(), model=model,
                              operations_per_task=n_ops, intervention='full_A')
                jobs.append(dict(common, id=source_id, source=None, source_job=True,
                                 B=canonical_b.to_dict(), replay_ratio=0., gradients=False))
                for kind in KINDS:
                    paired_a, b = task_pair(cfg, n_ops, kind)
                    assert paired_a == a
                    for ratio in (0., .1):
                        label = 'no_replay' if ratio == 0 else 'replay_0.1'
                        jobs.append(dict(common, id=prefix + f'/main/{kind}/{label}',
                                         B=b.to_dict(), source=source_id, source_job=False,
                                         replay_ratio=ratio, gradients=True, relation=kind))
    return jobs


def config_for_job(cfg, job):
    result = copy.deepcopy(cfg)
    result['model'] = copy.deepcopy(job['model'])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default='configs/phase2/one_layer_sequential.yaml')
    parser.add_argument('--root', default='runs/phase2/one_layer_sequential_20261011')
    parser.add_argument('--gpus', default='0', help='One sequential worker per listed physical GPU')
    parser.add_argument('--execute', action='store_true', help='Explicitly start the training sweep')
    parser.add_argument('--dry-run', action='store_true', help='Write and audit the plan only')
    parser.add_argument('--run-job', help=argparse.SUPPRESS)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text())
    jobs = build_plan(cfg)
    gpu_ids = args.gpus.split(',')
    if len(set(gpu_ids)) != len(gpu_ids) or any(not g.isdigit() for g in gpu_ids):
        parser.error('--gpus must be distinct nonnegative integer GPU IDs')
    root = Path(args.root).resolve()
    if args.run_job:
        if not args.execute or args.dry_run:
            parser.error('--run-job requires --execute without --dry-run')
        job = next(j for j in jobs if j['id'] == args.run_job)
        run_job(job, config_for_job(cfg, job), root, 'cuda:0')
        return
    plan = dict(version=1, config=cfg, jobs=jobs, unique_jobs=len(jobs),
                sources=sum(j['source_job'] for j in jobs),
                branches=sum(not j['source_job'] for j in jobs),
                nominal_training_steps=len(jobs) * cfg['train']['max_steps'],
                checkpoint_steps=schedule(cfg),
                protocol='Fixed-budget A-final -> B, fresh AdamW at switch',
                relation_alignment='latent_id; output slots swapped only for two-op tasks',
                split_ratios=[.6, .2, .2],
                source_gate='All A operations >= mastery_accuracy at each of the last mastery_window validation checkpoints',
                query_batch_size=cfg['train']['batch_size'])
    plan['fingerprint'] = digest(plan)
    root.mkdir(parents=True, exist_ok=True)
    plan_path = root / 'plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise RuntimeError('Plan changed for this root; select a new --root')
    if not plan_path.exists():
        write_json(plan_path, plan)
    print(json.dumps({k: plan[k] for k in ['unique_jobs', 'sources', 'branches', 'nominal_training_steps']}, indent=2), flush=True)
    print(f'Plan: {plan_path}', flush=True)
    if not args.execute or args.dry_run:
        print('Plan only; no training started.', flush=True)
        return

    def worker(gpu, lane):
        for job in lane:
            directory = root / job['id']
            directory.mkdir(parents=True, exist_ok=True)
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu, CUDA_DEVICE_ORDER='PCI_BUS_ID',
                       OMP_NUM_THREADS='1', WANDB_MODE='disabled',
                       TORCHINDUCTOR_CACHE_DIR=str(root / 'compile_cache' / f'gpu{gpu}'))
            command = [sys.executable, '-u', '-m', 'go4cl.phases.phase2.one_layer_sequential',
                       '--config', str(Path(args.config).resolve()), '--root', str(root),
                       '--execute', '--run-job', job['id']]
            print(f'GPU {gpu}: {job["id"]}', flush=True)
            with (directory / 'stdout.log').open('a') as log:
                subprocess.run(command, env=env, stdout=log, stderr=subprocess.STDOUT, check=True)

    # Complete the common A checkpoints before starting any B branch.
    for source_phase in (True, False):
        phase = [j for j in jobs if j['source_job'] == source_phase]
        with ThreadPoolExecutor(max_workers=len(gpu_ids)) as pool:
            futures = [pool.submit(worker, gpu, phase[i::len(gpu_ids)]) for i, gpu in enumerate(gpu_ids)]
            for future in futures:
                future.result()
    counts = {}
    for job in jobs:
        status = json.loads((root / job['id'] / 'result.json').read_text())['status']
        counts[status] = counts.get(status, 0) + 1
    write_json(root / 'launcher_summary.json', counts)
    print(json.dumps(counts, indent=2), flush=True)
    if counts.get('blocked_source_not_mastered'):
        raise SystemExit('Some B branches were blocked because their fixed-budget A source was not mastered; see result.json')


if __name__ == '__main__':
    main()
