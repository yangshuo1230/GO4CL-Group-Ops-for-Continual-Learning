"""Frozen no-replay time courses: train-only attention-template rescue and binding adapters."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data,op_metrics
from go4cl.data.generate import build_shared_residue_splits
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint,write_json

@torch.no_grad()
def run(root,out,j):
    torch.set_num_threads(1);directory=out/j['id'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'complete.json').exists():return
    tasks={t:TaskSpec.from_dict(j[t]) for t in ['A','B']};cfg=j['config'];splits=build_shared_residue_splits(list(tasks.values()),data_seed=cfg['data_seed'])
    data={t:probe_data(tasks[t],splits,128,cfg['eval_seed']+50000) for t in tasks}
    templates={};controls={}
    for t,step in [('A',0),('B',100000)]:
        reference,_=load_checkpoint(root/j['id']/f'checkpoints/step_{step:06d}.pt');reference.cuda().eval()
        c=reference.forward_with_cache(data[t]['train']['tokens'].cuda());latent=data[t]['test']['latent_ids'].cuda();templates[t]=[]
        for att in c['attn']:
            templates[t].append(torch.stack([att[(data[t]['train']['latent_ids']==z).cuda(),:,9].mean(0) for z in range(4)])[latent])
        controls[t]=dict(step=step,baseline=op_metrics(reference(data[t]['test']['tokens'].cuda())['logits'],data[t]['test']))
        del reference,c
    def intervene(model,t,layers,wrong=False):
        handles=[]
        for l in layers:
            att=templates[t][l].clone()
            if wrong:att[:,:,:8]=att[:,:,:8].roll(1,-1)
            def hook(module,inp,output,att=att):
                x=inp[0];n,length,d=x.shape
                v=module.qkv(x).reshape(n,length,3,module.n_heads,module.d_head).permute(2,0,3,1,4)[2]
                y=(att.unsqueeze(2)@v).transpose(1,2).reshape(n,d)
                result=output.clone();result[:,9]=module.out(y);return result
            handles.append(model.blocks[l].attn.register_forward_hook(hook))
        try:return op_metrics(model(data[t]['test']['tokens'].cuda())['logits'],data[t]['test'])
        finally:
            for h in handles:h.remove()
    history=json.loads((root/j['id']/'history.json').read_text());rows=[]
    for h in history:
        step=h['step'];model,_=load_checkpoint(root/j['id']/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        rows_step=[]
        for t in ['A','B']:
            d=data[t]['test'];tokens=d['tokens'].cuda();latent=d['latent_ids'].cuda();labels=d['labels'].cuda()
            baseline=op_metrics(model(tokens)['logits'],d);iv={}
            for layers in [(0,),(1,),(2,),(0,1,2)]:
                name='+'.join(str(l+1) for l in layers)
                for wrong in [False,True]:iv[('wrong' if wrong else 'correct')+'/'+name]=intervene(model,t,layers,wrong)
            row=dict(step=step,task=t,baseline=baseline,interventions=iv)
            if step==controls[t]['step']:controls[t]['interventions']=iv
            if t=='A':
                adapter=[]
                for op in tasks['A'].operations:
                    compatible=next((o for o in tasks['B'].operations if o.modulus==op.modulus),None)
                    if compatible is None:continue
                    source=tokens[latent==op.latent_id];target=labels[latent==op.latent_id]
                    permutation=[None]*8;permutation[compatible.i]=op.i;permutation[compatible.j]=op.j
                    for dest,src in zip([k for k in range(8) if permutation[k] is None],[k for k in range(8) if k not in [op.i,op.j]]):permutation[dest]=src
                    assert sorted(permutation)==list(range(8));rec=dict(latent=op.latent_id,modulus=op.modulus)
                    for mode in ['query_only','position_only','query_and_position']:
                        changed=source.clone()
                        if mode in ['position_only','query_and_position']:changed[:,:8]=source[:,permutation]
                        if mode in ['query_only','query_and_position']:changed[:,8]=tasks['B'].task_token;changed[:,9]=66+compatible.slot
                        rec[mode]=float((model(changed)['logits'].argmax(-1)==target).float().mean())
                    adapter.append(rec)
                row['shared_modulus_adapter']=adapter
            rows_step.append(row)
        rows.extend(rows_step);del model
    write_json(directory/'temporal.json',dict(job=j['id'],reference_controls=controls,records=rows,n_per_operation=128,
        definition='All saved training checkpoints. A-template from step0, B-template from final checkpoint; numeric values and every downstream parameter remain current. Templates averaged on train only, conditional on query, all heads jointly. Wrong control cycles only digit keys. B future-template is a rescue test, not proof native B routing formed early. Shared-modulus adapter uses task-defined bijective digit permutation, TASK and query, keeping A labels.'))
    write_json(directory/'complete.json',dict(complete=True,checkpoints=len(history)))
    print('DONE',j['id'],len(history),flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job');args=p.parse_args();root=Path('runs/phase2/optimizer_switch_20261010').resolve();out=root/'no_replay_temporal_20261010';out.mkdir(exist_ok=True)
    jobs=[j for j in json.loads((root/'plan.json').read_text())['jobs'] if j['objective']=='B']
    write_json(out/'plan.json',dict(jobs=jobs,n_per_operation=128,training=False))
    if args.job:run(root,out,next(j for j in jobs if j['id']==args.job));return
    def worker(g,subset):
        for j in subset:
            directory=out/j['id'];directory.mkdir(parents=True,exist_ok=True)
            with (directory/'stdout.log').open('a') as f:
                subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',j['id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1',WANDB_MODE='disabled'),stdout=f,stderr=subprocess.STDOUT,check=True)
            print('Finished',j['id'],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures=[pool.submit(worker,g,jobs[g::4]) for g in range(4)]
        for f in futures:f.result()

if __name__=='__main__':main()
