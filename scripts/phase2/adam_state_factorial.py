"""Three-factor AdamW state counterfactuals from native first batches; no trajectories."""
import copy,importlib.util,json
from pathlib import Path
import torch
from go4cl.metrics.behavioral import evaluate
from go4cl.train.loop import build_optimizer,_train_callable
from go4cl.utils.checkpoint import write_json


def main():
    torch.set_num_threads(1);root=Path('runs/phase2/optimizer_switch_20261010').resolve();out=root/'causal_paths_20261010/adam_state_factorial.json'
    spec=importlib.util.spec_from_file_location('switch',Path(__file__).with_name('optimizer_switch.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
    jobs=[j for j in json.loads((root/'plan.json').read_text())['jobs'] if j['optimizer_policy']=='inherit_A'];rows=[]
    for j in jobs:
        device=torch.device('cuda:0')
        model,opt,stream,donor,tc,a,b,splits=m.setup(j,device);model.train()
        batch=stream.next();h=json.loads((root/j['id']/'history.json').read_text())[1]['last_update']['batch_sha256']
        assert m.batch_hash(batch)==h
        model.zero_grad(set_to_none=True)
        train_model=_train_callable(model,device,compile_model=tc.compile_model)
        loss=train_model(batch['tokens'].to(device),batch['labels'].to(device))['loss'];loss.backward()
        model.eval()
        torch.nn.utils.clip_grad_norm_(model.parameters(),tc.grad_clip)
        gradients={n:p.grad.clone() for n,p in model.named_parameters()};base=copy.deepcopy(model.state_dict());state=copy.deepcopy(opt.state_dict())
        loaders=m.eval_loaders(a,b,splits,j['config']);baseline={name:evaluate(model,loader,device).to_dict() for name,loader in loaders.items()}
        for keep_m in [False,True]:
            for keep_v in [False,True]:
                for keep_step in [False,True]:
                    model.load_state_dict(base);o=build_optimizer(model,tc);s=copy.deepcopy(state)
                    for v in s['state'].values():
                        if not keep_m:v['exp_avg'].zero_()
                        if not keep_v:v['exp_avg_sq'].zero_()
                        if not keep_step:v['step'].zero_()
                    o.load_state_dict(s)
                    for name,p in model.named_parameters():p.grad=gradients[name].clone()
                    o.step();model.eval()
                    metrics={name:evaluate(model,loader,device).to_dict() for name,loader in loaders.items()}
                    norm=float(sum((p.detach()-base[n]).double().square().sum() for n,p in model.named_parameters()).sqrt())
                    rows.append(dict(group=j['id'].rsplit('/',1)[0],objective=j['objective'],keep_m=keep_m,keep_v=keep_v,keep_step=keep_step,
                        update_norm=norm,batch_sha256=h,baseline=baseline,metrics=metrics))
                    if keep_m==keep_v==keep_step:
                        pol='inherit_A' if keep_m else 'fresh';native=json.loads((root/j['id'].rsplit('/',1)[0]/pol/'history.json').read_text())[1]['metrics']
                        assert all(abs(metrics[t]['macro_operation_accuracy']-native[t]['macro_operation_accuracy'])<1e-7 for t in ['A','B'])
        print('Adam factorial',j['id'],flush=True)
    write_json(out,dict(records=rows,definition='m/v/step independent retain/reset 2^3 on identical source weights, native 8192-query first batch and compiled CUDA gradient; exact global-state extremes reproduce formal first updates. Copied models, no new training trajectories. Partial reset is a counterfactual optimizer, not assumed natural or safe for long training; interactions retained.'))


if __name__=='__main__':main()
