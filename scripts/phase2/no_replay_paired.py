"""Independent held-out paired routing sensitivity and sum interchange along no-replay trajectories."""
import argparse,importlib.util,json,os,subprocess,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.tasks.spec import TaskSpec
from go4cl.data.generate import build_shared_residue_splits
from go4cl.utils.checkpoint import load_checkpoint,write_json
spec=importlib.util.spec_from_file_location('paths',Path(__file__).with_name('causal_paths.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)

@torch.no_grad()
def run(root,out,j):
    torch.set_num_threads(1);directory=out/j['id'];directory.mkdir(parents=True,exist_ok=True)
    if (directory/'paired_complete.json').exists():return
    cfg=j['config'];tasks={t:TaskSpec.from_dict(j[t]) for t in ['A','B']};splits=build_shared_residue_splits(list(tasks.values()),data_seed=cfg['data_seed'])
    datasets={t:{s:m.paired(tasks[t],splits,s,cfg['eval_seed']+90000+tasks[t].task_id*1009,128) for s in ['val','test']} for t in tasks}
    torch.save(datasets,directory/'paired_inputs.pt')
    old=root/'mechanism_optimizer_analysis_20261010'/j['id'];selected={int(f.stem.split('_')[1]) for f in old.glob('step_*.json')}
    history=json.loads((root/j['id']/'history.json').read_text());steps=[h['step'] for h in history]
    for task in ['A','B']:
        for z in [0,1]:
            for threshold in [.2,.5,.8,.95]:
                idx=next((i for i,h in enumerate(history) if (next(v['accuracy'] for k,v in h['metrics'][task]['by_operation'].items() if f'/lat{z}/' in k)>=threshold)==(task=='B')),None)
                if idx is not None:selected.update(steps[max(0,idx-1):min(len(steps),idx+2)])
    records=[]
    for step in steps:
        model,_=load_checkpoint(root/j['id']/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        for task in ['A','B']:
            data=datasets[task]['test'];state=m.screen(model,data);tokens,labels,latent,cc,eligible,baselines=state
            background=tokens['clean'].clone()
            for op in tasks[task].operations:
                indices=(latent==op.latent_id).nonzero().flatten()
                other=next(o for o in tasks[task].operations if o.latent_id!=op.latent_id)
                # Shuffle an entire nontarget operand pair to retain its original train-pair support.
                donor=indices.roll(1)
                for pos in [other.i,other.j]:background[indices,pos]=tokens['clean'][donor,pos]
            bg=m.cache(model,background)
            sensitivity=[]
            for z in range(4):
                mask=latent==z
                for l in range(3):
                    clean=cc['clean']['heads'][l][:,9];scale=clean[mask].square().sum(-1).mean().sqrt().clamp_min(1e-8)
                    for donor in ['i','j','same_sum','different_sum','background']:
                        donor_c=bg if donor=='background' else cc[donor]
                        absolute=(donor_c['heads'][l][:,9]-clean)[mask].square().sum(-1).mean().sqrt()
                        sensitivity.append(dict(latent=z,layer=l,perturbation=donor,relative_query_attention_change=float(absolute/scale),absolute_query_attention_change=float(absolute),clean_query_attention_norm=float(scale)))
            rec=dict(step=step,task=task,baselines=baselines,sensitivity=sensitivity,nodes=[],node_status='not_selected_checkpoint')
            if step in selected:
                discovery=m.screen(model,datasets[task]['val'])
                if max(v['eligible'] for v in discovery[-1].values())>=32:
                    grid=[('post',l,pos) for l in range(3) for pos in range(10)]
                    scores=[r for node in grid for r in m.node_check(model,discovery,datasets[task]['val'],node)]
                    chosen=set()
                    for z in range(4):
                        if discovery[-1][str(z)]['eligible']<32:continue
                        for l in range(3):
                            for query in [True,False]:
                                candidates=[v for v in scores if v['latent']==z and v['node'][1]==l and (v['node'][2]==9)==query]
                                chosen.add(tuple(max(candidates,key=lambda v:v['sum_specificity'])['node']))
                    rec.update(nodes=[r for node in sorted(chosen) for r in m.node_check(model,state,data,node)],discovery_nodes=scores,node_status='tested')
                else:rec['node_status']='insufficient_correct_paired_examples'
            records.append(rec)
        del model
    write_json(directory/'paired_temporal.json',dict(job=j['id'],records=records,selected_node_steps=sorted(selected),definition='128 held-out paired inputs/op; all88 checkpoints sensitivity on query attention output, independently resampled target operand(s) vs shuffled complete background pair. Sensitivity is numerical dependence, not proof correct routing. Post-block node carriers selected on val and confirmed test at old analysis plus behavior-event points. Low native correctness excludes sum circuit interpretation; lack of confirmation is not loss proof.'))
    write_json(directory/'paired_complete.json',dict(complete=True,checkpoints=len(steps)));print('DONE paired',j['id'],flush=True)

def main():
    p=argparse.ArgumentParser();p.add_argument('--job');args=p.parse_args();root=Path('runs/phase2/optimizer_switch_20261010').resolve();out=root/'no_replay_temporal_20261010';jobs=json.loads((out/'plan.json').read_text())['jobs']
    if args.job:run(root,out,next(j for j in jobs if j['id']==args.job));return
    def worker(g,subset):
        for j in subset:
            with (out/j['id']/'paired_stdout.log').open('a') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',j['id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
            print('Finished paired',j['id'],flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        fs=[pool.submit(worker,g,jobs[g::4]) for g in range(4)]
        for f in fs:f.result()

if __name__=='__main__':main()
