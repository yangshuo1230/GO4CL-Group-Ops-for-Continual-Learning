"""Single-step counterfactuals on copies of theta_A; no new training trajectory."""
from pathlib import Path
import argparse, copy, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.phases.phase2.mechanism_suite import Stream, eval_loaders
from go4cl.data.generate import build_shared_residue_splits
from go4cl.metrics.behavioral import evaluate
from go4cl.tasks.spec import TaskSpec
from go4cl.train.loop import TrainConfig, build_optimizer
from go4cl.utils.checkpoint import load_checkpoint, write_json


def run(root,j,device):
    torch.set_num_threads(1);device=torch.device(device);directory=root/j['id'];cfg=json.loads((directory/'resolved.json').read_text())['config']
    a=TaskSpec.from_dict(j['A']);b=TaskSpec.from_dict(j['B']);splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
    source,payload=load_checkpoint(directory/'checkpoints/final.pt',map_location='cpu');source.to(device);source.eval()
    tc=TrainConfig(**cfg['train'],device=str(device));loaders=eval_loaders(a,b,splits,cfg)
    baseline=evaluate(source,loaders['A'],device).to_dict();rows=[]
    for objective,ratio,isA in [('A',0.,True),('B',0.,False),('mix_0.1',.1,False)]:
        batch=Stream(a,b,splits,tc.batch_size,ratio,cfg['sampler_seed'],isA).next()
        for policy in ['fresh','inherit_A']:
            for decay in [True,False]:
                model,_=load_checkpoint(directory/'checkpoints/final.pt',map_location='cpu');model.to(device);model.eval()
                opt=build_optimizer(model,tc)
                if policy=='inherit_A':opt.load_state_dict(copy.deepcopy(payload['optimizer_state']))
                if not decay:
                    for pg in opt.param_groups:pg['weight_decay']=0.
                opt.zero_grad(set_to_none=True);loss=model(batch['tokens'].to(device),batch['labels'].to(device))['loss'];loss.backward()
                raw_norm=float(torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip))
                raw={n:p.grad.detach().clone() for n,p in model.named_parameters()}
                opt.step();model.eval();metrics=evaluate(model,loaders['A'],device).to_dict()
                delta=torch.cat([(p.detach()-dict(source.named_parameters())[n]).flatten().double() for n,p in model.named_parameters()]);g=torch.cat([v.flatten().double() for v in raw.values()]);denom=float(delta.norm()*g.norm())
                rows.append(dict(objective=objective,optimizer_policy=policy,weight_decay=decay,
                    train_loss=float(loss.detach()),raw_gradient_norm=raw_norm,delta_norm=float(delta.norm()),
                    delta_gradient_cosine=float(torch.dot(delta,g))/denom if denom>0 else None,A_before=baseline,A_after=metrics))
    out=root/'mechanism_followup_20261010/optimizer_controls'/j['id']/'single_step.json'
    write_json(out,dict(source=j['id'],seed=j['seed'],rows=rows,
        definition='Exact native 8192-query first batch, same weights and data; inherited source AdamW state versus fresh state; one step only; no checkpoint saved.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/mechanism_suite_20261009');p.add_argument('--job');args=p.parse_args();root=Path(args.root).resolve()
    jobs=[j for j in json.loads((root/'plan_all.json').read_text())['jobs'] if j['source_job']]
    if args.job:run(root,next(j for j in jobs if j['id']==args.job),'cuda:0');return
    def worker(i,j):
        env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(i),OMP_NUM_THREADS='1',WANDB_MODE='disabled')
        subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',j['id']],env=env,check=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(worker,i,j) for i,j in enumerate(jobs)]
        for f in fs:f.result()
    print('Completed 96 single-step counterfactuals on 8 sources',flush=True)


if __name__=='__main__':main()
