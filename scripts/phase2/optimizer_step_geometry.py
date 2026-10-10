"""Rescale observed first-step deltas on frozen copies; CPU-only, no training."""
import argparse
import json
from pathlib import Path

import torch
from go4cl.data.generate import build_shared_residue_splits
from go4cl.metrics.behavioral import evaluate
from go4cl.phases.phase2.mechanism_suite import eval_loaders
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint, write_json


@torch.no_grad()
def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/optimizer_switch_20261010');args=p.parse_args()
    torch.set_num_threads(1);root=Path(args.root).resolve();out=root/'mechanism_optimizer_analysis_20261010/step_geometry.json'
    jobs=json.loads((root/'plan.json').read_text())['jobs'];groups={j['id'].rsplit('/',1)[0] for j in jobs};rows=[]
    for group in sorted(groups):
        fresh=next(j for j in jobs if j['id']==group+'/fresh');cfg=fresh['config']
        a,b=TaskSpec.from_dict(fresh['A']),TaskSpec.from_dict(fresh['B']);splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
        loaders=eval_loaders(a,b,splits,cfg)
        model,start=load_checkpoint(root/group/'fresh/checkpoints/step_000000.pt',map_location='cpu');model.eval()
        for p in model.parameters():p.requires_grad_(False)
        base={n:v.clone() for n,v in start['model_state'].items()};delta={};norm={}
        for policy in ['fresh','inherit_A']:
            payload=torch.load(root/group/policy/'checkpoints/step_000001.pt',map_location='cpu',weights_only=False)
            delta[policy]={n:payload['model_state'][n]-v for n,v in base.items()}
            norm[policy]=float(sum(v.double().square().sum() for v in delta[policy].values()).sqrt())
        vf,vi=[torch.cat([v.flatten().double() for v in delta[pol].values()]) for pol in ['fresh','inherit_A']]
        cosine=float(torch.dot(vf,vi)/(vf.norm()*vi.norm()))
        baseline={name:evaluate(model,loader,torch.device('cpu')).to_dict() for name,loader in loaders.items()}
        for direction in ['fresh','inherit_A']:
            for magnitude in ['fresh','inherit_A']:
                scale=norm[magnitude]/norm[direction]
                model.load_state_dict({n:v+scale*delta[direction][n] for n,v in base.items()})
                metrics={name:evaluate(model,loader,torch.device('cpu')).to_dict() for name,loader in loaders.items()}
                row=dict(group=group,objective=fresh['objective'],direction=direction,magnitude=magnitude,
                         scale=scale,observed_norms=norm,delta_direction_cosine=cosine,baseline=baseline,metrics=metrics)
                rows.append(row)
                # Unscaled endpoint must reproduce original training validation.
                if direction==magnitude:
                    original=json.loads((root/group/direction/'history.json').read_text())[1]['metrics']
                    assert all(abs(metrics[t]['macro_operation_accuracy']-original[t]['macro_operation_accuracy'])<1e-7 for t in ['A','B'])
        print('Geometry',group,flush=True)
    write_json(out,dict(records=rows,definition='Observed native 8192-query first updates, two directions × two observed norms; evaluation on identical fixed A/B validation; frozen CPU counterfactuals. Scaling includes the entire observed delta including decay. Not an optimizer training trajectory or selective moment reset.'))


if __name__=='__main__':main()
