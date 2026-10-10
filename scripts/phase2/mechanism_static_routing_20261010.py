"""Train-input mean attention templates: no test-input donor attention is injected."""
import argparse,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data,op_metrics
from go4cl.data.generate import build_shared_residue_splits
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint,write_json


@torch.no_grad()
def run(root,item):
    torch.set_num_threads(1);job=item['job'];jdir=root/job['id'];base=root/'mechanism_followup_20261010';cfg=json.loads((jdir/'resolved.json').read_text())['config']
    a=TaskSpec.from_dict(job['A']);b=TaskSpec.from_dict(job['B']);splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
    data=probe_data(a,splits,512,cfg['eval_seed']+50000);reference,_=load_checkpoint(root/job['source']/'checkpoints/final.pt');reference.cuda().eval()
    train=reference.forward_with_cache(data['train']['tokens'].cuda());tokens=data['test']['tokens'].cuda();latent=data['test']['latent_ids'].cuda()
    template=[]
    for att in train['attn']:
        groups=[]
        for z in range(4):
            t=att.mean(0).clone();t[:,9]=att[(data['train']['latent_ids']==z).cuda(),:,9].mean(0);groups.append(t)
        template.append(torch.stack(groups)[latent])
    def patched(model,layer,wrong=False):
        att=template[layer].clone()
        if wrong:att[:,:,9,:8]=att[:,:,9,:8].roll(1,-1)
        def hook(module,inp,out):
            x=inp[0];n,t,d=x.shape;v=module.qkv(x).reshape(n,t,3,module.n_heads,module.d_head).permute(2,0,3,1,4)[2]
            values=(att@v).transpose(1,2).contiguous().reshape(n,t,d);new=module.out(values);mixed=out.clone();mixed[:,9]=new[:,9];return mixed
        h=model.blocks[layer].attn.register_forward_hook(hook)
        try:return op_metrics(model(tokens)['logits'],data['test'])
        finally:h.remove()
    source_baseline=op_metrics(reference(tokens)['logits'],data['test'])
    source_controls={str(l):patched(reference,l) for l in range(3)}
    history=json.loads((jdir/'history.json').read_text());nadir=min(history,key=lambda r:r['metrics']['A']['macro_operation_accuracy'])['step']
    records=[]
    for step in sorted({nadir,100000}):
        model,_=load_checkpoint(jdir/f'checkpoints/step_{step:06d}.pt');model.cuda().eval();row=dict(step=step,baseline=op_metrics(model(tokens)['logits'],data['test']),interventions={})
        for l in range(3):
            row['interventions'][f'correct_static_template/L{l}']=patched(model,l)
            row['interventions'][f'wrong_digit_keys_template/L{l}']=patched(model,l,True)
        records.append(row)
    write_json(base/'static_routing'/job['id']/'controls.json',dict(source_baseline=source_baseline,source_static_templates=source_controls,records=records,
               definition='Attention template fitted on train inputs only, conditioned by explicit query; test numeric values come solely from current model V; query-row digit key cyclic-shift negative control, preserving probabilities/entropy. Source-model template control checks whether averaging already breaks source computation.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/mechanism_suite_20261009');p.add_argument('--job');args=p.parse_args();root=Path(args.root).resolve()
    items=[i for i in json.loads((root/'mechanism_followup_20261010/plan.json').read_text())['jobs'] if i['job']['source'] and i['job']['intervention']=='full_A']
    if args.job:run(root,next(i for i in items if i['job']['id']==args.job));return
    def worker(gpu,items):
        for item in items:
            subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',item['job']['id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1'),check=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(worker,i,items[i::8]) for i in range(8)]
        for f in futures:f.result()
    print('Static routing controls finished for',len(items),'trajectories')


if __name__=='__main__':main()
