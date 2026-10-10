"""Factor last-layer Q/K/V/skip routes carrying an independently selected sender."""
import importlib.util,json,itertools
from pathlib import Path
import torch
from go4cl.utils.checkpoint import load_checkpoint,write_json

spec=importlib.util.spec_from_file_location('paths',Path(__file__).with_name('causal_paths.py'));m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)


@torch.no_grad()
def main():
    torch.set_num_threads(1);root=Path('runs/phase2/optimizer_switch_20261010').resolve();base=root/'causal_paths_20261010';jobs=json.loads((base/'plan.json').read_text())['jobs'];rows=[]
    for j in jobs:
        data=torch.load(base/j['id']/'paired_inputs.pt',weights_only=False)
        for step in [0,10000,100000]:
            old=json.loads((base/j['id']/f'step_{step:06d}.json').read_text());model,_=load_checkpoint(root/j['id']/f'checkpoints/step_{step:06d}.pt');model.cuda().eval()
            for p in model.parameters():p.requires_grad_(False)
            block=model.blocks[2];att=block.attn
            def qkv(c):
                x=block.ln1(c['resid_pre'][2]);n,t,d=x.shape
                return att.qkv(x).reshape(n,t,3,att.n_heads,att.d_head).permute(2,0,3,1,4)
            for rec in old['records']:
                state=m.screen(model,data[rec['task']]['test']);tokens,labels,latent,cc,eligible,_=state
                selections={}
                for z in range(4):
                    candidates=[v for v in rec.get('discovery_nodes',[]) if v['latent']==z and v['node'][0]=='post' and v['node'][1]<2 and v['sum_specificity']>.8]
                    if candidates:selections[z]=tuple(max(candidates,key=lambda v:(v['sum_specificity'],-v['node'][1]))['node'])
                qb,kb,vb=qkv(cc['i'])
                for z,node in selections.items():
                    for donor in ['clean','same_sum']:
                        corrected=m.patch(model,tokens['i'],cc['i'],cc[donor],node,True)
                        qd,kd,vd=qkv(corrected)
                        for bits in itertools.product([False,True],repeat=4):
                            q,k,v=[(a if on else b) for on,a,b in zip(bits[:3],[qd,kd,vd],[qb,kb,vb])]
                            skip=(corrected if bits[3] else cc['i'])['resid_pre'][2][:,9]
                            weights=torch.softmax((q[:,:,9:10]@k.transpose(-2,-1))*att.d_head**-.5,-1)
                            y=(weights@v).transpose(1,2).reshape(len(latent),att.n_heads*att.d_head)
                            mid=skip+att.out(y);post=mid+block.mlp(block.ln2(mid));logits=model.head(model.ln_f(post));pred=logits.argmax(-1)
                            if all(bits):
                                assert torch.equal(pred,corrected['logits'].argmax(-1)), 'Last-layer factorization not faithful'
                            if not any(bits):assert torch.equal(pred,cc['i']['logits'].argmax(-1))
                            rows.append(dict(job=j['id'],step=step,task=rec['task'],latent=z,sender=list(node),donor=donor,
                                Q=bits[0],K=bits[1],V=bits[2],skip=bits[3],accuracy=m.frac(pred,labels['clean'],latent==z),
                                conditional_accuracy=m.frac(pred,labels['clean'],(latent==z)&eligible),eligible=int(((latent==z)&eligible).sum())))
        print('Bridge',j['id'],flush=True)
    write_json(base/'final_bridge.json',dict(records=rows,definition='Sender selection only on val sum specificity>0.8. Last-layer all-head Q/K/V and residual skip independently taken from original corrupted or sender-corrected run. No direct old-checkpoint interface substitution. Exact all-off/all-on predictions verified; interactions preserved; correct paired subset and unconditional results both saved.'))


if __name__=='__main__':main()
