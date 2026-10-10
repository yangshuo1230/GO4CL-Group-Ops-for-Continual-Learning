"""Input-paired interchange and V-edge path tests; frozen models, no probes."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from go4cl.data.generate import build_shared_residue_splits
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint,write_json


def paired(task,splits,split,seed,n=128):
    raw=probe_data(task,splits,n,seed)[split];clean=raw['tokens'].numpy().copy();rng=np.random.default_rng(seed+91)
    variants={name:clean.copy() for name in ['i','j','same_sum','different_sum']}
    for op in task.operations:
        allowed=set(splits[op.modulus].get(split))
        pool=[(a,b) for a in range(64) for b in range(64) if tuple(sorted((a%op.modulus,b%op.modulus))) in allowed]
        for idx in np.where(raw['latent_ids'].numpy()==op.latent_id)[0]:
            a,b=clean[idx,op.i],clean[idx,op.j];p=op.modulus;y=(a+b)%p
            # Some held-out graph vertices have no alternative with one fixed operand.
            # Resample by input constraints only, before any model is evaluated.
            for attempt in range(10000):
                opts={side:[v for v in range(64) if (v+other)%p!=y and tuple(sorted((v%p,other%p))) in allowed]
                      for side,other in [('i',b),('j',a)]}
                same=[(u,v) for u,v in pool if (u+v)%p==y and u%p!=a%p and v%p!=b%p]
                diff=[(u,v) for u,v in pool if len({y,(u+b)%p,(a+v)%p,(u+v)%p})==4]
                if opts['i'] and opts['j'] and same and diff:break
                a,b=pool[rng.integers(len(pool))];y=(a+b)%p
            else:raise RuntimeError('No admissible paired held-out contexts')
            clean[idx,op.i]=a;clean[idx,op.j]=b
            for arr in variants.values():arr[idx]=clean[idx]
            for side in ['i','j']:variants[side][idx,op.i if side=='i' else op.j]=rng.choice(opts[side])
            for name,options in [('same_sum',same),('different_sum',diff)]:
                u,v=options[rng.integers(len(options))];variants[name][idx,op.i]=u;variants[name][idx,op.j]=v
    tensors={'clean':torch.from_numpy(clean),**{k:torch.from_numpy(v) for k,v in variants.items()}}
    labels={};hybrids={}
    for name,x in tensors.items():
        y=torch.empty(len(x),dtype=torch.long)
        for op in task.operations:
            m=raw['latent_ids']==op.latent_id;y[m]=(x[m,op.i]+x[m,op.j])%op.modulus
        labels[name]=y
    for name in ['donor_i_recipient_j','recipient_i_donor_j']:
        y=torch.empty(len(clean),dtype=torch.long)
        for op in task.operations:
            m=raw['latent_ids']==op.latent_id
            u=tensors['different_sum'][m,op.i] if name.startswith('donor') else tensors['clean'][m,op.i]
            v=tensors['clean'][m,op.j] if name.startswith('donor') else tensors['different_sum'][m,op.j]
            y[m]=(u+v)%op.modulus
        hybrids[name]=y
    assert torch.equal(labels['same_sum'],labels['clean'])
    assert all(torch.all(labels[name]!=labels['clean']) for name in ['i','j','different_sum'])
    return dict(tokens=tensors,labels=labels,hybrids=hybrids,latent=raw['latent_ids'])


@torch.no_grad()
def cache(model,tokens):
    heads={};mlps={};pre={};post={};attns={};handles=[]
    assert model.cfg.dropout==0, 'Cache reconstruction assumes deterministic attention'
    for l,b in enumerate(model.blocks):
        handles.append(b.register_forward_pre_hook(lambda m,i,l=l:pre.__setitem__(l,i[0].clone())))
        handles.append(b.register_forward_hook(lambda m,i,o,l=l:post.__setitem__(l,o.clone())))
        def attention(m,i,l=l):
            x=i[0];n,t,d=x.shape
            q,k,v=m.qkv(x).reshape(n,t,3,m.n_heads,m.d_head).permute(2,0,3,1,4)
            logits=(q@k.transpose(-2,-1))*m.d_head**-.5
            attns[l]=torch.softmax(logits.masked_fill(m.mask[:,:,:t,:t]==0,float('-inf')),-1)
        handles.append(b.attn.register_forward_pre_hook(attention))
        handles.append(b.attn.out.register_forward_pre_hook(lambda m,i,l=l:heads.__setitem__(l,i[0].clone())))
        handles.append(b.mlp.register_forward_hook(lambda m,i,o,l=l:mlps.__setitem__(l,o.clone())))
    try:c=model(tokens)
    finally:
        for h in handles:h.remove()
    c.update(heads=heads,mlps=mlps,resid_pre=pre,resid_post=post,attn=attns);return c


@torch.no_grad()
def patch(model,tokens,recipient,donor,node,return_cache=False):
    site,l,pos=node;b=model.blocks[l]
    def replace(m,i,o):
        x=o.clone();x[:,pos]=donor['resid_post' if site=='post' else 'mlps'][l][:,pos];return x
    h=(b if site=='post' else b.mlp).register_forward_hook(replace)
    try:return cache(model,tokens) if return_cache else model(tokens)['logits']
    finally:h.remove()


def frac(pred,label,mask):return float((pred[mask]==label[mask]).float().mean()) if mask.any() else None


@torch.no_grad()
def screen(model,data):
    tokens={k:v.cuda() for k,v in data['tokens'].items()};labels={k:v.cuda() for k,v in data['labels'].items()};latent=data['latent'].cuda()
    cc={k:cache(model,v) for k,v in tokens.items()};base={k:c['logits'].argmax(-1) for k,c in cc.items()}
    eligible=torch.ones(len(latent),device='cuda',dtype=torch.bool)
    for k in cc:eligible&=base[k]==labels[k]
    baselines={str(z):dict(accuracy={k:frac(v,labels[k],latent==z) for k,v in base.items()},eligible=int((eligible&(latent==z)).sum())) for z in range(4)}
    return tokens,labels,latent,cc,eligible,baselines


@torch.no_grad()
def node_check(model,state,data,node):
    tokens,labels,latent,cc,eligible,_=state
    preds={name:patch(model,tokens[recipient],cc[recipient],cc[donor],node).argmax(-1) for name,recipient,donor in
           [('restore_i','i','clean'),('restore_j','j','clean'),('same_sum_restore','i','same_sum'),('different_sum_transfer','clean','different_sum')]}
    out=[]
    for z in range(4):
        mask=latent==z;use=mask&eligible
        row=dict(latent=z,node=list(node),eligible=int(use.sum()),
                 **{k:frac(v,labels['different_sum'] if k=='different_sum_transfer' else labels['clean'],mask) for k,v in preds.items()})
        row['conditional']={k:frac(v,labels['different_sum'] if k=='different_sum_transfer' else labels['clean'],use) for k,v in preds.items()}
        row['hybrid_i']=frac(preds['different_sum_transfer'],data['hybrids']['donor_i_recipient_j'].cuda(),mask)
        row['hybrid_j']=frac(preds['different_sum_transfer'],data['hybrids']['recipient_i_donor_j'].cuda(),mask)
        row['sum_specificity']=min(row[k] for k in preds)-max(row['hybrid_i'],row['hybrid_j'])
        out.append(row)
    return out


@torch.no_grad()
def value_path(model,state,node,layer,head,donor_name='clean',mode='V'):
    tokens,labels,latent,cc,eligible,_=state;position=node[2]
    intervened=patch(model,tokens['i'],cc['i'],cc[donor_name],node,True)
    att=model.blocks[layer].attn
    def qkv(c):
        x=model.blocks[layer].ln1(c['resid_pre'][layer]);n,t,d=x.shape
        return att.qkv(x).reshape(n,t,3,att.n_heads,att.d_head).permute(2,0,3,1,4)
    qb,kb,vb=qkv(cc['i']);qd,kd,vd=qkv(intervened)
    q=qd[:,head,9] if mode=='Q' else qb[:,head,9]
    k=kb[:,head].clone();v=vb[:,head].clone()
    if mode in ['K','KV']:k[:,position]=kd[:,head,position]
    if mode in ['V','KV']:v[:,position]=vd[:,head,position]
    weights=torch.softmax(torch.einsum('bd,btd->bt',q,k)*att.d_head**-.5,-1)
    y=torch.einsum('bt,btd->bd',weights,v)
    if mode=='head':y=intervened['heads'][layer][:,9,head*att.d_head:(head+1)*att.d_head]
    def hook(m,inp):
        x=inp[0].clone();a=head*att.d_head;x[:,9,a:a+att.d_head]=y;return (x,)
    h=att.out.register_forward_pre_hook(hook)
    try:pred=model(tokens['i'])['logits'].argmax(-1)
    finally:h.remove()
    return [dict(latent=z,node=list(node),receiver_layer=layer,receiver_head=head,donor=donor_name,mode=mode,
                 accuracy=frac(pred,labels['clean'],latent==z),conditional_accuracy=frac(pred,labels['clean'],(latent==z)&eligible),
                 eligible=int(((latent==z)&eligible).sum())) for z in range(4)]


@torch.no_grad()
def task_routes(model,data,task,other):
    clean=data['tokens']['clean'].cuda();flipped=clean.clone();flipped[:,8]=other.task_token
    ca,cb=cache(model,clean),cache(model,flipped);ya=data['labels']['clean'].cuda();latent=data['latent'].cuda();yb=torch.empty_like(ya)
    for op in other.operations:
        m=clean[:,9]==66+op.slot;yb[m]=(clean[m,op.i]+clean[m,op.j])%op.modulus
    # Causality guarantees digit states are unchanged by the later TASK token.
    for l in range(3):assert torch.equal(ca['resid_post'][l][:,:8],cb['resid_post'][l][:,:8])
    rows=[]
    for l,block in enumerate(model.blocks):
        a=block.attn
        def qkv(c):
            x=block.ln1(c['resid_pre'][l]);n,t,d=x.shape
            return a.qkv(x).reshape(n,t,3,a.n_heads,a.d_head).permute(2,0,3,1,4)
        qa,ka,va=qkv(ca);qb,kb,vb=qkv(cb)
        for h in range(4):
            for mode in ['Q','K','V','QK','full_head']:
                q=qa[:,h,9] if mode in ['Q','QK'] else qb[:,h,9]
                k=ka[:,h] if mode in ['K','QK'] else kb[:,h]
                v=va[:,h] if mode=='V' else vb[:,h]
                weights=torch.softmax(torch.einsum('bd,btd->bt',q,k)*a.d_head**-.5,-1)
                y=torch.einsum('bt,btd->bd',weights,v)
                if mode=='full_head':y=ca['heads'][l][:,9,h*a.d_head:(h+1)*a.d_head]
                def hook(m,inp,y=y,h=h):
                    x=inp[0].clone();x[:,9,h*a.d_head:(h+1)*a.d_head]=y;return (x,)
                handle=a.out.register_forward_pre_hook(hook)
                try:pred=model(flipped)['logits'].argmax(-1)
                finally:handle.remove()
                rows.extend(dict(latent=z,layer=l,head=h,mode=mode,A_label_accuracy=frac(pred,ya,latent==z),B_label_accuracy=frac(pred,yb,latent==z)) for z in range(4))
    return dict(clean_accuracy=[frac(ca['logits'].argmax(-1),ya,latent==z) for z in range(4)],
                flipped_other_accuracy=[frac(cb['logits'].argmax(-1),yb,latent==z) for z in range(4)],tests=rows)


def run(root,out,job):
    torch.set_num_threads(1);directory=out/job['id'];directory.mkdir(parents=True,exist_ok=True)
    cfg=job['config'];tasks={t:TaskSpec.from_dict(job[t]) for t in ['A','B']};splits=build_shared_residue_splits(list(tasks.values()),data_seed=cfg['data_seed'])
    datasets={t:{s:paired(tasks[t],splits,s,cfg['eval_seed']+90000+tasks[t].task_id*1009,128) for s in ['val','test']} for t in tasks}
    if not (directory/'paired_inputs.pt').exists():torch.save(datasets,directory/'paired_inputs.pt')
    for step in [0,100,1000,10000,100000]:
        path=directory/f'step_{step:06d}.json'
        if path.exists():continue
        model,_=load_checkpoint(root/job['id']/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        before={k:v.clone() for k,v in model.state_dict().items()};records=[]
        for task in ['A','B']:
            discovery=screen(model,datasets[task]['val']);validation=screen(model,datasets[task]['test'])
            rec=dict(task=task,discovery_baselines=discovery[-1],test_baselines=validation[-1])
            # Poor native behavior cannot support interpreting this path as its correct computation.
            if max(v['eligible'] for v in discovery[-1].values())<32:
                rec.update(status='insufficient_correct_paired_examples',nodes=[],paths=[]);records.append(rec);continue
            nodes=[(site,l,pos) for site in ['post','mlp'] for l in range(3) for pos in range(10)]
            scores=[r for node in nodes for r in node_check(model,discovery,datasets[task]['val'],node)]
            chosen=set()
            for z in range(4):
                if discovery[-1][str(z)]['eligible']<32:continue
                for l in range(3):
                    for query in [True,False]:
                        options=[r for r in scores if r['latent']==z and r['node'][1]==l and (r['node'][2]==9)==query]
                        winner=max(options,key=lambda r:r['sum_specificity']);chosen.add(tuple(winner['node']))
            tests=[r for node in sorted(chosen) for r in node_check(model,validation,datasets[task]['test'],node)]
            # Select sender sites exclusively by discovery sum specificity; test all later receiver heads.
            senders=set()
            for z in range(4):
                opts=[r for r in scores if r['latent']==z and r['node'][0]=='post' and r['node'][1]<2 and r['node'][2]<8 and r['sum_specificity']>.4]
                if opts:senders.add(tuple(max(opts,key=lambda r:r['sum_specificity'])['node']))
            paths=[]
            for node in sorted(senders):
                for layer in range(node[1]+1,3):
                    for head in range(4):
                        for donor in ['clean','same_sum']:
                            paths.extend(value_path(model,validation,node,layer,head,donor))
            rec.update(status='tested',discovery_nodes=scores,nodes=tests,paths=paths,
                       task_routing=task_routes(model,datasets[task]['test'],tasks[task],tasks['B' if task=='A' else 'A']))
            records.append(rec)
        assert all(torch.equal(v,model.state_dict()[k]) for k,v in before.items())
        write_json(path,dict(step=step,job=job['id'],records=records,model_unchanged=True))
        print(job['id'],step,flush=True)
    write_json(directory/'complete.json',dict(complete=True))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/optimizer_switch_20261010');p.add_argument('--job');args=p.parse_args()
    root=Path(args.root).resolve();out=root/'causal_paths_20261010';jobs=[j for j in json.loads((root/'plan.json').read_text())['jobs'] if j['objective']=='mix_0.1']
    out.mkdir(exist_ok=True);write_json(out/'plan.json',dict(jobs=jobs,n=128,steps=[0,100,1000,10000,100000],
        selection='All node selection on val; confirmation on test; matched held-out residue pairs; all four operations; insufficient native performance explicitly excluded'))
    if args.job:run(root,out,next(j for j in jobs if j['id']==args.job));return
    def worker(gpu,subset):
        for j in subset:
            log=out/j['id']/'stdout.log';log.parent.mkdir(parents=True,exist_ok=True)
            with log.open('a') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',j['id']],
                env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1',WANDB_MODE='disabled'),stdout=f,stderr=subprocess.STDOUT,check=True)
            print('Finished',j['id'],flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(worker,g,jobs[g::8]) for g in range(8)]
        for f in fs:f.result()


if __name__=='__main__':main()
