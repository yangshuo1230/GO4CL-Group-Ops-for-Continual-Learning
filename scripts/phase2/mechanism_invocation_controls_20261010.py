"""Distinguish loss of the original A binding from reusable shared-modulus computation."""
from pathlib import Path
import argparse,json,os,subprocess,sys
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data,op_metrics
from go4cl.data.generate import build_shared_residue_splits
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint,write_json


@torch.no_grad()
def run(root,item):
    torch.set_num_threads(1);job=item['job'];directory=root/job['id'];cfg=json.loads((directory/'resolved.json').read_text())['config'];a=TaskSpec.from_dict(job['A']);b=TaskSpec.from_dict(job['B']);splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
    data=probe_data(a,splits,512,cfg['eval_seed']+50000)['test'];tokens=data['tokens'].cuda();latent=data['latent_ids'].cuda();labels=data['labels'].cuda()
    history=json.loads((directory/'history.json').read_text());nadir=min(history,key=lambda r:r['metrics']['A']['macro_operation_accuracy'])['step'];rows=[]
    for step in sorted({nadir,100000}):
        model,_=load_checkpoint(directory/f'checkpoints/step_{step:06d}.pt');model.cuda().eval();row=dict(step=step,baseline=op_metrics(model(tokens)['logits'],data),operations=[])
        for op in a.operations:
            m=latent==op.latent_id;original=tokens[m];target=labels[m]
            compatible=next((o for o in b.operations if o.modulus==op.modulus),None)
            rec=dict(latent=op.latent_id,modulus=op.modulus,compatible_B_operation=compatible.to_dict() if compatible else None,all_query_invocations=[])
            for task in [a,b]:
                for slot in range(4):
                    changed=original.clone();changed[:,8]=task.task_token;changed[:,9]=66+slot
                    rec['all_query_invocations'].append(dict(task=task.name,slot=slot,accuracy=float((model(changed)['logits'].argmax(-1)==target).float().mean())))
            if compatible:
                permutation=[None]*8;permutation[compatible.i]=op.i;permutation[compatible.j]=op.j
                rest=[k for k in range(8) if k not in [op.i,op.j]]
                for dest,src in zip([k for k in range(8) if permutation[k] is None],rest):permutation[dest]=src
                assert sorted(permutation)==list(range(8))
                rec['digit_permutation']=permutation
                for mode in ['query_only','position_only','query_and_position']:
                    changed=original.clone()
                    if mode in ['position_only','query_and_position']:changed[:,:8]=original[:,permutation]
                    if mode in ['query_only','query_and_position']:changed[:,8]=b.task_token;changed[:,9]=66+compatible.slot
                    rec[mode]=float((model(changed)['logits'].argmax(-1)==target).float().mean())
            row['operations'].append(rec)
        rows.append(row)
    write_json(root/'mechanism_followup_20261010/invocation_controls'/job['id']/'controls.json',dict(records=rows,
      interpretation='Labels remain original A labels. Shared-modulus B slot/operand mapping is specified solely by task definitions, not selected on test accuracy. Recovery demonstrates reusable current computation via an external adapter, not retention of the original A task binding or original A circuit. Counterfactual queries are not normal A task evaluation.'))


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/mechanism_suite_20261009');p.add_argument('--job');args=p.parse_args();root=Path(args.root).resolve()
    items=[i for i in json.loads((root/'mechanism_followup_20261010/plan.json').read_text())['jobs'] if i['job']['source'] and i['job']['intervention']=='full_A']
    if args.job:run(root,next(i for i in items if i['job']['id']==args.job));return
    def worker(gpu,items):
        for item in items:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--root',str(root),'--job',item['job']['id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(gpu),OMP_NUM_THREADS='1'),check=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(worker,i,items[i::8]) for i in range(8)]
        for f in fs:f.result()
    print('Task/query/operand invocation controls completed for',len(items),'trajectories')


if __name__=='__main__':main()
