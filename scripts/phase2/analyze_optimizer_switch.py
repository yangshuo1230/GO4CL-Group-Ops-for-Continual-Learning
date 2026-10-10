"""Frozen trajectory analysis of matched optimizer-switch controls."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

import torch


def module(filename):
    p = Path(__file__).with_name(filename)
    spec = importlib.util.spec_from_file_location(p.stem, p)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def plans(root):
    jobs = json.loads((root / 'plan.json').read_text())['jobs']
    items = []
    for original in jobs:
        directory = root / original['id']
        result = json.loads((directory / 'result.json').read_text())
        if result['status'] != 'complete':
            raise RuntimeError(f"Incomplete training: {original['id']}")
        h = json.loads((directory / 'history.json').read_text())
        available = {r['step'] for r in h}
        steps = set([0,1,2,5,10,20,50,100,500,1000,5000,10000,20000,100000]) & available
        # Events are selected on training-time validation only, never probe test scores.
        steps.add(min(h, key=lambda r: r['metrics']['A']['macro_operation_accuracy'])['step'])
        early = [r for r in h if 0 < r['step'] <= 2000]
        nadir = min(early, key=lambda r: r['metrics']['A']['macro_operation_accuracy'])['step']
        steps.add(nadir)
        for z in range(4):
            hits = [r['step'] for r in h if r['step'] > nadir and
                    next(v['accuracy'] for k,v in r['metrics']['A']['by_operation'].items() if f'/lat{z}/' in k) >= .95]
            if hits: steps.add(hits[0])
        if original['objective'] != 'A':
            for threshold in [.5,.9,.95]:
                hits = [r['step'] for r in h if r['metrics']['B']['macro_operation_accuracy'] >= threshold]
                if hits: steps.add(hits[0])
            for z in range(4):
                hits = [r['step'] for r in h if next(v['accuracy'] for k,v in r['metrics']['B']['by_operation'].items() if f'/lat{z}/' in k) >= .95]
                if hits: steps.add(hits[0])
        job = dict(original, source=str(Path(original['source']['path']).parent.parent),
                   source_job=False, intervention='full_A')
        items.append(dict(job=job, kind='core', tasks=['A'] if original['objective']=='A' else ['A','B'],
                          steps=sorted(steps), probes=True))
    # Each fresh/inherit pair is examined at identical saved training steps.
    for item in items:
        pair = [i for i in items if i['job']['id'].rsplit('/',1)[0] == item['job']['id'].rsplit('/',1)[0]]
        shared = sorted({s for i in pair for s in i['steps']})
        for i in pair: i['steps'] = shared
    return items


def gradients(root, out, items, backend):
    rows = []
    for item in items:
        j = item['job']
        for p in sorted((root / j['id'] / 'gradients').glob('step_*.pt')):
            s = torch.load(p, map_location='cpu', weights_only=False)
            rows.append(dict(job=j['id'], seed=j['seed'], objective=j['objective'],
                             policy=j['optimizer_policy'], step=s['step'],
                             statistics=backend.vector_stats(s), effects=s['effects']))
    backend.atomic_json(out / 'gradient/all.json', rows)
    print('Gradient snapshots:', len(rows), flush=True)


def controls(root, out, item):
    for filename in ['mechanism_static_routing_20261010.py',
                     'mechanism_invocation_controls_20261010.py',
                     'mechanism_probe_audit_20261010.py']:
        m = module(filename)
        original_write = m.write_json
        def write(path, data, writer=original_write):
            relative = Path(path).relative_to(root / 'mechanism_followup_20261010')
            writer(out / relative, data)
        m.write_json = write
        m.run(root, item)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', default='runs/phase2/optimizer_switch_20261010')
    p.add_argument('--out')
    p.add_argument('--gpus', default='0,1,2,3,4,5,6,7')
    p.add_argument('--job')
    p.add_argument('--stage', choices=['trajectory','controls','gradient','all'], default='all')
    p.add_argument('--plan-only', action='store_true')
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    root = Path(args.root).resolve()
    out = Path(args.out).resolve() if args.out else root / 'mechanism_optimizer_analysis_20261010'
    out.mkdir(parents=True, exist_ok=True)
    backend = module('mechanism_followup_20261010.py')
    if args.job:
        items = json.loads((out / 'plan.json').read_text())['jobs']
        item = next(i for i in items if i['job']['id']==args.job)
        if args.stage in ['all','trajectory']:
            backend.analyze_job(root,out,item,'cuda:0',512,400,args.pilot)
        if not args.pilot and args.stage in ['all','controls']:
            marker = out / item['job']['id'] / 'controls_complete.json'
            if not marker.exists():
                controls(root,out,item)
                backend.atomic_json(marker,dict(complete=True))
        return
    items = plans(root)
    plan = dict(jobs=items,n=512,probe_steps=400,checkpoints=sum(len(i['steps']) for i in items),
                definition='Matched pair steps plus validation-selected events; common source A reference; frozen models')
    path = out / 'plan.json'
    if path.exists() and json.loads(path.read_text()) != plan:
        raise RuntimeError('Analysis plan changed; choose another output folder')
    backend.atomic_json(path,plan)
    print('Analysis jobs/checkpoints:',len(items),plan['checkpoints'],flush=True)
    if args.plan_only: return
    if args.stage in ['all','gradient']: gradients(root,out,items,backend)
    if args.stage == 'gradient': return
    gpus = args.gpus.split(',')
    if len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        p.error('GPU IDs must be distinct integers')
    def worker(gpu, subset):
        for item in subset:
            jid = item['job']['id']
            log = out / jid / 'stdout.log'
            log.parent.mkdir(parents=True,exist_ok=True)
            cmd = [sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--out',str(out),
                   '--job',jid,'--stage',args.stage]
            env = dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',WANDB_MODE='disabled')
            with log.open('a') as f:
                subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            print('Finished',jid,flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        futures=[pool.submit(worker,gpu,items[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in futures: f.result()


if __name__=='__main__':main()
