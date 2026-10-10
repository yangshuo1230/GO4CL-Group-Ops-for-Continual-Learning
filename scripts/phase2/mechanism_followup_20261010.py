"""Frozen-model causal follow-up; only probes are optimized. All figures PNG."""
from __future__ import annotations
import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import torch
import torch.nn.functional as F
from go4cl.analysis.probes import fit_linear_probe
from go4cl.data.generate import build_shared_residue_splits
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data, op_metrics, vector_stats
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint, write_json

NAME = 'mechanism_followup_20261010'


def atomic_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp'); write_json(tmp, data); tmp.replace(path)


def specifications(task, layers):
    out = []
    for op in task.operations:
        for layer in range(layers):
            for site in ['resid_mid', 'resid_post']:
                for position, target in [(p, 'sum') for p in range(10)] + [(9, 'xi'), (9, 'xj')]:
                    out.append(dict(latent=op.latent_id, modulus=op.modulus, i=op.i, j=op.j,
                                    layer=layer, site=site, position=position, target=target))
    return out


@torch.no_grad()
def cached(model, data, device):
    return {s: model.forward_with_cache(b['tokens'].to(device)) for s, b in data.items()}


@torch.no_grad()
def tensors(cache, data, specs, device):
    xs, ys = {}, {}
    for split, batch in data.items():
        xx, yy = [], []
        for sp in specs:
            mask = (batch['latent_ids'] == sp['latent']).to(device)
            xx.append(cache[split][sp['site']][sp['layer']][mask, sp['position']])
            raw = batch['tokens'][batch['latent_ids'] == sp['latent']]
            y = batch['labels'][batch['latent_ids'] == sp['latent']] if sp['target'] == 'sum' else raw[:, sp['i'] if sp['target'] == 'xi' else sp['j']] % sp['modulus']
            yy.append(y.to(device))
        xs[split] = torch.stack(xx); ys[split] = torch.stack(yy)
    return xs, ys


def batch_fit(xs, ys, specs, steps=400, seed=4401):
    """Independent classifiers batched in one optimizer, same initialization as fit_linear_probe."""
    device = xs['train'].device; count, n, d = xs['train'].shape; maxp = max(s['modulus'] for s in specs)
    ww = torch.zeros(count, maxp, d); bb = torch.zeros(count, maxp)
    for idx, sp in enumerate(specs):
        g = torch.Generator().manual_seed(seed); p = sp['modulus']; bound = 1 / math.sqrt(d)
        ww[idx, :p].uniform_(-bound, bound, generator=g); bb[idx, :p].uniform_(-bound, bound, generator=g)
    w = torch.nn.Parameter(ww.to(device)); b = torch.nn.Parameter(bb.to(device))
    mask = torch.arange(maxp, device=device)[None, :] < torch.tensor([s['modulus'] for s in specs], device=device)[:, None]
    opt = torch.optim.AdamW([w, b], lr=.05, weight_decay=.01)
    trace = []
    for step in range(1, steps + 1):
        logits = torch.bmm(xs['train'], w.transpose(1, 2)) + b[:, None]
        logits = logits.masked_fill(~mask[:, None], -1e9)
        loss = F.cross_entropy(logits.reshape(-1, maxp), ys['train'].flatten(), reduction='none').reshape(count, n).mean(1)
        opt.zero_grad(set_to_none=True); loss.sum().backward(); opt.step()
        if step in {100, 200, steps}: trace.append(dict(step=step, losses=loss.detach().cpu().tolist()))
    with torch.no_grad():
        scores = {}
        for split in xs:
            pred = (torch.bmm(xs[split], w.transpose(1, 2)) + b[:, None]).masked_fill(~mask[:, None], -1e9).argmax(-1)
            scores[split] = (pred == ys[split]).float().mean(1).cpu().tolist()
    return dict(weight=w.detach().cpu(), bias=b.detach().cpu(), scores=scores, loss_trace=trace, specs=specs)


@torch.no_grad()
def fixed_and_aligned(xs, ys, ref_x, fixed, specs):
    device = xs['test'].device
    w = fixed['weight'].to(device); b = fixed['bias'].to(device); c = w.shape[1]
    mask = torch.arange(c, device=device)[None, :] < torch.tensor([s['modulus'] for s in specs], device=device)[:, None]
    def score(x):
        pred = (torch.bmm(x, w.transpose(1, 2)) + b[:, None]).masked_fill(~mask[:, None], -1e9).argmax(-1)
        return (pred == ys['test']).float().mean(1).cpu().tolist()
    mx = xs['train'].mean(1, keepdim=True); mr = ref_x['train'].mean(1, keepdim=True)
    covariance = torch.bmm((xs['train'] - mx).transpose(1, 2), ref_x['train'] - mr)
    u, _, vh = torch.linalg.svd(covariance, full_matrices=False); rotation = u @ vh
    aligned = torch.bmm(xs['test'] - mx, rotation) + mr
    return score(xs['test']), score(aligned)


@torch.no_grad()
def causal_checks(model, reference, task, other_task, data, device):
    test = data['test']; tokens = test['tokens'].to(device); latent = test['latent_ids'].to(device)
    cur = model.forward_with_cache(tokens); old = reference.forward_with_cache(tokens)
    result = dict(baseline=op_metrics(cur['logits'], test), reference=op_metrics(old['logits'], test), routing=[], interventions={})
    means_head, means_mlp = {}, {}; handles = []
    for layer, block in enumerate(model.blocks):
        handles.append(block.attn.out.register_forward_pre_hook(lambda m, inp, l=layer: means_head.__setitem__(l, inp[0].detach().mean(0))))
        handles.append(block.mlp.register_forward_hook(lambda m, inp, out, l=layer: means_mlp.__setitem__(l, out.detach().mean(0))))
    try: model(data['train']['tokens'].to(device))
    finally:
        for h in handles: h.remove()
    def record(name, logits): result['interventions'][name] = op_metrics(logits, test)
    for layer, block in enumerate(model.blocks):
        att = cur['attn'][layer]
        for op in task.operations:
            m = latent == op.latent_id
            for h in range(model.cfg.n_heads):
                for q in [8, 9]:
                    result['routing'].append(dict(latent=op.latent_id, layer=layer, head=h, query_position=q,
                        operand_mass=float((att[m, h, q, op.i] + att[m, h, q, op.j]).mean()),
                        task_mass=float(att[m, h, q, 8].mean()), query_self_mass=float(att[m, h, q, q].mean())))
        for scope, positions in [('query', [9]), ('task', [8]), ('all', list(range(10)))]:
            for mode in ['zero', 'mean']:
                for h in range(model.cfg.n_heads):
                    start = h * model.cfg.d_head; end = start + model.cfg.d_head
                    def head_hook(module, inp, lo=start, hi=end, pos=positions, kind=mode, l=layer):
                        x = inp[0].clone()
                        x[:, pos, lo:hi] = 0 if kind == 'zero' else means_head[l][pos, lo:hi][None]
                        return (x,)
                    handle = block.attn.out.register_forward_pre_hook(head_hook)
                    try: logits = model(tokens)['logits']
                    finally: handle.remove()
                    record(f'{mode}_head/L{layer}/H{h}/{scope}', logits)
                def mlp_hook(module, inp, out, pos=positions, kind=mode, l=layer):
                    x = out.clone(); x[:, pos] = 0 if kind == 'zero' else means_mlp[l][pos][None]; return x
                handle = block.mlp.register_forward_hook(mlp_hook)
                try: logits = model(tokens)['logits']
                finally: handle.remove()
                record(f'{mode}_mlp/L{layer}/{scope}', logits)
        for scope in ['query', 'all']:
            patch = cur['resid_post'][layer].clone()
            if scope == 'query': patch[:, 9] = old['resid_post'][layer][:, 9]
            else: patch = old['resid_post'][layer]
            record(f'old_activation_new_tail/L{layer}/{scope}', model.continue_from_layer(patch, layer_idx=layer)['logits'])
            patch = old['resid_post'][layer].clone()
            if scope == 'query': patch[:, 9] = cur['resid_post'][layer][:, 9]
            else: patch = cur['resid_post'][layer]
            record(f'new_activation_old_tail/L{layer}/{scope}', reference.continue_from_layer(patch, layer_idx=layer)['logits'])
            shuffled = old['resid_post'][layer].clone()
            generator = torch.Generator().manual_seed(1401)
            for z in range(4):
                ids = (latent == z).nonzero().flatten(); perm = torch.randperm(len(ids), generator=generator).to(device)
                shuffled[ids] = old['resid_post'][layer][ids[perm]]
            patch = cur['resid_post'][layer].clone()
            if scope == 'query': patch[:, 9] = shuffled[:, 9]
            else: patch = shuffled
            record(f'shuffled_old_activation_new_tail/L{layer}/{scope}', model.continue_from_layer(patch, layer_idx=layer)['logits'])
            # Replace attention probabilities, keeping current V, output projection and all other modules.
            def routing_hook(module, inp, out, l=layer, sc=scope):
                x = inp[0]; n, t, d = x.shape
                v = module.qkv(x).reshape(n, t, 3, module.n_heads, module.d_head).permute(2, 0, 3, 1, 4)[2]
                values = (old['attn'][l] @ v).transpose(1, 2).contiguous().reshape(n, t, d)
                newout = module.resid_drop(module.out(values))
                if sc == 'query':
                    mixed = out.clone(); mixed[:, 9] = newout[:, 9]; return mixed
                return newout
            handle = block.attn.register_forward_hook(routing_hook)
            try: logits = model(tokens)['logits']
            finally: handle.remove()
            record(f'old_attention_current_values/L{layer}/{scope}', logits)
        record(f'self_continuation/L{layer}', model.continue_from_layer(cur['resid_post'][layer], layer_idx=layer)['logits'])
    record('old_finalLN_head_on_current_residual', reference.head(reference.ln_f(cur['resid_post'][-1])[:, 9]))
    record('new_finalLN_head_on_old_residual', model.head(model.ln_f(old['resid_post'][-1])[:, 9]))
    flipped = tokens.clone(); flipped[:, 8] = other_task.task_token
    flip_logits = model(flipped)['logits']; record('flip_task_original_labels', flip_logits)
    other_labels = torch.empty_like(test['labels'])
    slots = test['tokens'][:, 9] - 66
    for op in other_task.operations:
        m = slots == op.slot
        other_labels[m] = (test['tokens'][m, op.i] + test['tokens'][m, op.j]) % op.modulus
    other_batch = dict(test, labels=other_labels)
    result['task_counterfactual'] = dict(prediction_agreement=float((cur['logits'].argmax(-1) == flip_logits.argmax(-1)).float().mean()),
        original_outputs_scored_other_task=op_metrics(cur['logits'], other_batch), flipped_outputs_scored_other_task=op_metrics(flip_logits, other_batch),
        caveat='Counterfactual other task on the same contexts, not the primary held-out distribution of that other task.')
    # Swap the two operands: modular addition label is invariant (sanity/control for positional asymmetry).
    swapped = tokens.clone()
    for op in task.operations:
        m = latent == op.latent_id; swapped[m, op.i] = tokens[m, op.j]; swapped[m, op.j] = tokens[m, op.i]
    record('swap_operand_values_invariant_label', model(swapped)['logits'])
    return result


def chosen_steps(history, kind):
    available = {r['step'] for r in history}
    dense = [0,100,200,300,400,500,600,800,1000,1200,1500,2000,3000,5000,10000,20000,50000,100000]
    if kind == 'source': dense = [0,500,1000,2000,4000,10000,100000]
    if kind == 'overlap': dense = [0,500,2000,10000,50000,100000]
    if kind == 'component': dense = [0,100000]
    chosen = set(dense) & available
    if kind in ['core','overlap']:
        for name in ['A','B']:
            chosen.add(min(history,key=lambda r:r['metrics'][name]['macro_operation_accuracy'])['step'])
            for z in range(4):
                hits=[r['step'] for r in history if next(v['accuracy'] for k,v in r['metrics'][name]['by_operation'].items() if f'/lat{z}/' in k)>=.95 and r['step']>0]
                if hits: chosen.add(hits[0])
    return sorted(chosen)


def plans(root):
    jobs=json.loads((root/'plan_all.json').read_text())['jobs']; out=[]
    for j in jobs:
        if j['source_job']: kind='source'; tasknames=['A']
        elif j['intervention']=='full_fresh': kind='core' if '/main/' in j['id'] else 'overlap'; tasknames=['B']
        elif j['intervention']!='full_A': kind='component'; tasknames=['A','B']
        elif '/main/' in j['id']: kind='core'; tasknames=['A','B']
        else: kind='overlap'; tasknames=['A','B']
        h=json.loads((root/j['id']/'history.json').read_text())
        out.append(dict(job=j,kind=kind,tasks=tasknames,steps=chosen_steps(h,kind),probes=kind!='component'))
    return out


def analyze_job(root, outroot, item, device, n, probe_steps, pilot=False):
    torch.set_num_threads(1); device=torch.device(device); job=item['job']; out=outroot/job['id']; out.mkdir(parents=True,exist_ok=True)
    settings=dict(item=item,n_per_operation=n,probe_steps=probe_steps,probe_seed=4401)
    result=out/'complete.json'
    if result.exists():
        if json.loads(result.read_text())['settings']!=settings: raise RuntimeError('Analysis settings changed: use a new output folder')
        return
    train_dir=root/job['id']; cfg=json.loads((train_dir/'resolved.json').read_text())['config']
    a=TaskSpec.from_dict(job['A']); b=TaskSpec.from_dict(job['B']); ts={'A':a,'B':b}
    splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
    reference_dir=root/job['source'] if job['source'] else train_dir
    reference,_=load_checkpoint(reference_dir/'checkpoints/final.pt',map_location='cpu');reference.to(device);reference.eval()
    datas={name:probe_data(ts[name],splits,n,cfg['eval_seed']+50000+ts[name].task_id*1009) for name in item['tasks']}
    refs={};start=time.time()
    if item['probes']:
        for name in item['tasks']:
            specs=specifications(ts[name],reference.cfg.n_layers); rc=cached(reference,datas[name],device); rx,ry=tensors(rc,datas[name],specs,device)
            fixed=batch_fit(rx,ry,specs,probe_steps);refs[name]=(rx,fixed,specs)
            torch.save(fixed,out/f'reference_probes_{name}.pt')
            rows=[dict(sp,train=fixed['scores']['train'][i],val=fixed['scores']['val'][i],test=fixed['scores']['test'][i]) for i,sp in enumerate(specs)]
            atomic_json(out/f'reference_{name}.json',dict(checkpoint=str(reference_dir/'checkpoints/final.pt'),rows=rows))
            del rc
    for step in item['steps']:
        p=out/f'step_{step:06d}.json'
        if p.exists(): continue
        model,payload=load_checkpoint(train_dir/f'checkpoints/step_{step:06d}.pt',map_location='cpu'); model.to(device);model.eval()
        for param in model.parameters(): param.requires_grad_(False)
        records=[]
        before = {k:v.detach().clone() for k,v in model.state_dict().items()} if pilot else None
        for name in item['tasks']:
            data=datas[name]; t=ts[name]
            record=dict(step=step,task=name,causal=causal_checks(model,reference,t,ts['B' if name=='A' else 'A'],data,device))
            if item['probes']:
                rx,fixed,specs=refs[name];cc=cached(model,data,device);xs,ys=tensors(cc,data,specs,device)
                fit=batch_fit(xs,ys,specs,probe_steps);direct,aligned=fixed_and_aligned(xs,ys,rx,fixed,specs)
                record['probes']=[dict(sp,retrained_train=fit['scores']['train'][i],retrained_val=fit['scores']['val'][i],
                    retrained_test=fit['scores']['test'][i],fixed_test=direct[i],aligned_fixed_test=aligned[i],chance=1/sp['modulus']) for i,sp in enumerate(specs)]
                record['probe_loss_trace']=fit['loss_trace'];torch.save(fit,out/f'probes_{step:06d}_{name}.pt')
                del xs,ys,cc,fit
            records.append(record)
        atomic_json(p,dict(checkpoint_step=payload['step'],records=records))
        if pilot:
            assert all(torch.equal(before[k],v) for k,v in model.state_dict().items()), 'Causal analysis mutated model parameters'
            for record in records:
                c=record['causal']
                for name, values in c['interventions'].items():
                    if name.startswith('self_continuation'):
                        assert all(abs(v['accuracy']-c['baseline'][z]['accuracy'])<1e-7 for z,v in values.items())
        print(f"{job['id']} step {step} completed in {time.time()-start:.1f}s",flush=True)
        del model,payload
        if pilot:
            print('Pilot completed; full trajectory remains pending',flush=True);return
    atomic_json(result,dict(settings=settings,seconds=time.time()-start,main_model_training=False))


def gradients(root,out):
    torch.set_num_threads(1); rows=[]
    for j in json.loads((root/'plan_all.json').read_text())['jobs']:
        directory=root/j['id']; destination=out/'gradient'/j['id']/'summary.json'
        if destination.exists(): rows.extend(json.loads(destination.read_text()));continue
        recs=[];mods={o['modulus'] for o in j['A']['operations']}
        for p in sorted((directory/'gradients').glob('step_*.pt')):
            s=torch.load(p,map_location='cpu',weights_only=False);stats=vector_stats(s)
            a=torch.cat([v.flatten().double() for v in s['gradients']['A'].values()]);perop={}
            for name,g in s['gradients'].items():
                if not name.startswith('B/'):continue
                v=torch.cat([g[k].flatten().double() for k in s['gradients']['A']]);mod=int(name.rsplit('p',1)[1]);denom=float(a.norm()*v.norm())
                perop[name]=dict(shared_modulus=mod in mods,A_cosine=float(torch.dot(a,v))/denom if denom>0 else None,norm=float(v.norm()))
            recs.append(dict(job=j['id'],seed=j['seed'],replay=j['replay_ratio'],step=s['step'],statistics=stats,per_operation=perop,effects=s['effects']))
            del s
        if recs:atomic_json(destination,recs);rows.extend(recs)
    atomic_json(out/'gradient/all.json',rows);print('gradient snapshots',len(rows),flush=True)


def test_backend():
    torch.set_num_threads(1);g=torch.Generator().manual_seed(8);x=torch.randn(24,8,generator=g);y=torch.randint(0,3,(24,),generator=g)
    specs=[dict(modulus=3)];xs={s:x[None] for s in ['train','val','test']};ys={s:y[None] for s in xs}
    b=batch_fit(xs,ys,specs,40);a=fit_linear_probe(x,y,{'test':x},{'test':y},n_classes=3,steps=40,seed=4401)
    assert torch.allclose(a['weight'],b['weight'][0],atol=2e-6)
    assert torch.allclose(a['bias'],b['bias'][0],atol=2e-6)
    assert a['eval_acc']['test']==b['scores']['test'][0]
    fixed,aligned=fixed_and_aligned(xs,ys,xs,b,specs);assert fixed==aligned
    ym=torch.randint(0,5,(24,),generator=g);xx={s:torch.stack([x,x]) for s in xs};yy={s:torch.stack([y,ym]) for s in xs}
    mixed=batch_fit(xx,yy,[dict(modulus=3),dict(modulus=5)],40)
    exact=fit_linear_probe(x,ym,{'test':x},{'test':ym},n_classes=5,steps=40,seed=4401)
    assert torch.allclose(a['weight'],mixed['weight'][0,:3],atol=3e-6)
    assert torch.allclose(exact['weight'],mixed['weight'][1,:5],atol=3e-6)
    print('PASS: batched probe matches original single-probe weights, biases and accuracy; identity alignment matches',flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',default='runs/phase2/mechanism_suite_20261009');p.add_argument('--out')
    p.add_argument('--gpus',default='0,1,2,3,4,5,6,7');p.add_argument('--n',type=int,default=512);p.add_argument('--probe-steps',type=int,default=400)
    p.add_argument('--stage',choices=['all','gradient','core','overlap','component','source'],default='all');p.add_argument('--job');p.add_argument('--test',action='store_true');p.add_argument('--pilot',action='store_true')
    args=p.parse_args();torch.set_num_threads(1)
    if args.test:test_backend();return
    root=Path(args.root).resolve();out=Path(args.out).resolve() if args.out else root/NAME;out.mkdir(parents=True,exist_ok=True)
    plan=plans(root);atomic_json(out/'plan.json',dict(n=args.n,probe_steps=args.probe_steps,jobs=plan))
    if args.job:
        analyze_job(root,out,next(j for j in plan if j['job']['id']==args.job),'cuda:0',args.n,args.probe_steps,args.pilot);return
    if args.stage in ['all','gradient']:gradients(root,out)
    if args.stage=='gradient':return
    selected=[j for j in plan if args.stage=='all' or j['kind']==args.stage or (args.stage=='core' and j['kind']=='source')]
    gpus=args.gpus.split(',')
    def worker(gpu,items):
        for item in items:
            log=out/item['job']['id']/'stdout.log';log.parent.mkdir(parents=True,exist_ok=True)
            cmd=[sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--out',str(out),'--job',item['job']['id'],'--n',str(args.n),'--probe-steps',str(args.probe_steps)]
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu,OMP_NUM_THREADS='1',WANDB_MODE='disabled')
            with log.open('a') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
            print('finished',item['job']['id'],flush=True)
    with ThreadPoolExecutor(max_workers=len(gpus)) as pool:
        fs=[pool.submit(worker,gpu,selected[i::len(gpus)]) for i,gpu in enumerate(gpus)]
        for f in fs:f.result()


if __name__=='__main__':main()
