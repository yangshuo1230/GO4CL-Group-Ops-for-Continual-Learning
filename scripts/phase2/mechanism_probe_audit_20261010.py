"""Probe budget and shuffled-label checks at source, forgetting nadir and final endpoints."""
import argparse,json,os,subprocess,sys,importlib.util
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data
from go4cl.data.generate import build_shared_residue_splits
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint,write_json


def module():
    path=Path(__file__).with_name('mechanism_followup_20261010.py');s=importlib.util.spec_from_file_location('followup',path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m


def run(root,item):
    torch.set_num_threads(1);m=module();job=item['job'];jdir=root/job['id'];cfg=json.loads((jdir/'resolved.json').read_text())['config'];a=TaskSpec.from_dict(job['A']);b=TaskSpec.from_dict(job['B']);ts={'A':a,'B':b};splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
    tasks=['A'] if job['source_job'] else ['B'] if job['intervention']=='full_fresh' else ['A','B'];rows=[]
    history=json.loads((jdir/'history.json').read_text());nadir=min(history,key=lambda r:r['metrics']['A']['macro_operation_accuracy'])['step']
    for name in tasks:
        data=probe_data(ts[name],splits,512,cfg['eval_seed']+50000+ts[name].task_id*1009)
        steps={100000,nadir} if name=='A' and job['source'] else {100000}
        for step in sorted(steps):
            model,_=load_checkpoint(jdir/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
            specs=[s for s in m.specifications(ts[name],3) if s['target']=='sum' and s['position'] in [8,9]]
            xs,ys=m.tensors(m.cached(model,data,'cuda'),data,specs,'cuda');small=m.batch_fit(xs,ys,specs,400);large=m.batch_fit(xs,ys,specs,800)
            shuffled=dict(ys);shuffled['train']=ys['train'].clone()
            for idx,sp in enumerate(specs):
                g=torch.Generator().manual_seed(9101+sp['latent']);permutation=torch.randperm(xs['train'].shape[1],generator=g).cuda();shuffled['train'][idx]=ys['train'][idx][permutation]
            null=m.batch_fit(xs,shuffled,specs,400)
            rows.append(dict(task=name,step=step,probes=[dict(sp,test_400=small['scores']['test'][i],test_800=large['scores']['test'][i],train_400=small['scores']['train'][i],train_800=large['scores']['train'][i],shuffled_test=null['scores']['test'][i],chance=1/sp['modulus']) for i,sp in enumerate(specs)]))
    write_json(root/'mechanism_followup_20261010/probe_audit'/job['id']/'audit.json',rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/mechanism_suite_20261009');p.add_argument('--job');args=p.parse_args();root=Path(args.root).resolve();items=[i for i in json.loads((root/'mechanism_followup_20261010/plan.json').read_text())['jobs'] if i['kind'] in ['source','core']]
    if args.job:run(root,next(i for i in items if i['job']['id']==args.job));return
    def worker(gpu,items):
        for item in items:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',item['job']['id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1'),check=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(worker,i,items[i::8]) for i in range(8)]
        for f in fs:f.result()
    print('Probe budget / shuffled-label audit finished for',len(items),'jobs')


if __name__=='__main__':main()
