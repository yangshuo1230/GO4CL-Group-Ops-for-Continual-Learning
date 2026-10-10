"""Exploratory token-specific first-layer MLP ablations on frozen replay models."""
import json
from pathlib import Path
import torch
from go4cl.data.generate import build_shared_residue_splits
from go4cl.phases.phase2.mechanism_suite_analysis import probe_data, op_metrics
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint, write_json


@torch.no_grad()
def main():
    torch.set_num_threads(1)
    root=Path('runs/phase2/optimizer_switch_20261010').resolve();out=root/'mechanism_optimizer_analysis_20261010/token_localization.json'
    jobs=json.loads((root/'plan.json').read_text())['jobs'];rows=[]
    for j in jobs:
        if '/same_mod_same_pos/' not in j['id'] or j['objective']!='mix_0.1':continue
        cfg=j['config'];a,b=TaskSpec.from_dict(j['A']),TaskSpec.from_dict(j['B']);splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
        data=probe_data(a,splits,512,cfg['eval_seed']+50000);tokens=data['test']['tokens'].cuda()
        for step in [0,10000,100000]:
            model,_=load_checkpoint(root/j['id']/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
            for p in model.parameters():p.requires_grad_(False)
            means=[];handle=model.blocks[0].mlp.register_forward_hook(lambda m,i,o:means.append(o.mean(0)))
            try:model(data['train']['tokens'].cuda())
            finally:handle.remove()
            baseline=op_metrics(model(tokens)['logits'],data['test']);tests={}
            for mode in ['zero','mean']:
                for position in range(10):
                    def patch(m,i,o,mode=mode,position=position):
                        z=o.clone();z[:,position]=0 if mode=='zero' else means[0][position];return z
                    handle=model.blocks[0].mlp.register_forward_hook(patch)
                    try:tests[f'{mode}/pos{position}']=op_metrics(model(tokens)['logits'],data['test'])
                    finally:handle.remove()
            rows.append(dict(job=j['id'],step=step,baseline=baseline,interventions=tests))
        print('Token localization',j['id'],flush=True)
    write_json(out,dict(records=rows,definition='Exploratory same_mod_same_pos replay cases, both seeds/policies, all first-layer MLP token positions; zero and train-mean controls; frozen main models; not selected for population inference.'))


if __name__=='__main__':main()
