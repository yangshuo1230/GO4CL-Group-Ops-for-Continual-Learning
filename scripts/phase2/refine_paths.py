"""Distinguish K/V/Q-mediated sender-to-head paths after narrow V tests."""
import importlib.util,json,os,subprocess,sys,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import torch
from go4cl.utils.checkpoint import load_checkpoint,write_json

spec=importlib.util.spec_from_file_location('paths',Path(__file__).with_name('causal_paths.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


def run(root,j):
    torch.set_num_threads(1);directory=root/'causal_paths_20261010'/j['id'];datasets=torch.load(directory/'paired_inputs.pt',weights_only=False)
    for step in [0,10000,100000]:
        out=directory/f'refined_{step:06d}.json'
        if out.exists() and json.loads(out.read_text()).get('version')==2:continue
        model,_=load_checkpoint(root/j['id']/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
        for p in model.parameters():p.requires_grad_(False)
        old=json.loads((directory/f'step_{step:06d}.json').read_text());records=[]
        for r in old['records']:
            state=m.screen(model,datasets[r['task']]['test']);senders=set();perop={}
            for z in range(4):
                selected=[]
                for query in [True,False]:
                    options=[n for n in r.get('discovery_nodes',[]) if n['latent']==z and n['node'][0]=='post' and n['node'][1]<2 and (n['node'][2]==9)==query and n['sum_specificity']>.4]
                    if options:
                        winner=tuple(max(options,key=lambda n:n['sum_specificity'])['node']);selected.append(list(winner));senders.add(winner)
                perop[str(z)]=selected
            tests=[]
            for node in sorted(senders):
                patched=m.patch(model,state[0]['i'],state[3]['i'],state[3]['clean'],node,True)
                plain=m.patch(model,state[0]['i'],state[3]['i'],state[3]['clean'],node)
                assert torch.allclose(patched['logits'],plain,atol=1e-6), 'Patched cache bypassed block intervention'
                for layer in range(node[1]+1,3):
                    for head in range(4):
                        for mode in ['Q','K','V','KV','head']:
                            for donor in ['clean','same_sum']:tests.extend(m.value_path(model,state,node,layer,head,donor,mode))
            records.append(dict(task=r['task'],senders_by_operation=perop,tests=tests))
        write_json(out,dict(version=2,job=j['id'],step=step,records=records,patched_cache_verified=True,definition='Sender selected on discovery val; only indicated receiver Q or sender-position K/V replaced; all other receiving head inputs and residual paths remain original corrupted run. Head mode is a broader single-head mediation control. Initial narrow V paths in step JSON used an incompatible cache traversal and are superseded by these version-2 results.'))
        print(j['id'],step,'refined',flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--job');args=p.parse_args();root=Path('runs/phase2/optimizer_switch_20261010').resolve();jobs=json.loads((root/'causal_paths_20261010/plan.json').read_text())['jobs']
    if args.job:run(root,next(j for j in jobs if j['id']==args.job));return
    def worker(g,subset):
        for j in subset:
            log=root/'causal_paths_20261010'/j['id']/'refine.log'
            with log.open('a') as f:subprocess.run([sys.executable,str(Path(__file__).resolve()),'--job',j['id']],env=dict(os.environ,CUDA_VISIBLE_DEVICES=str(g),OMP_NUM_THREADS='1'),stdout=f,stderr=subprocess.STDOUT,check=True)
            print('Refined',j['id'],flush=True)
    with ThreadPoolExecutor(max_workers=8) as pool:
        fs=[pool.submit(worker,g,jobs[g::8]) for g in range(8)]
        for f in fs:f.result()


if __name__=='__main__':main()
