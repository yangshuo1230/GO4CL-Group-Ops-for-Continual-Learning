"""Attach exploratory token localization and repack the completed PNG report."""
from pathlib import Path
import json
import zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

base=Path('runs/phase2/optimizer_switch_20261010/mechanism_optimizer_analysis_20261010').resolve()
out=base/'report';data=json.loads((base/'token_localization.json').read_text())
files=[]
for seed in [0,1]:
    fig,axes=plt.subplots(4,2,figsize=(13,10),layout='constrained')
    for ri,(policy,mode) in enumerate([('fresh','zero'),('fresh','mean'),('inherit_A','zero'),('inherit_A','mean')]):
        rr=[r for r in data['records'] if r['job']==f'seed{seed}/same_mod_same_pos/mix_0.1/{policy}']
        for z in [0,1]:
            mat=np.array([[r['baseline'][str(z)]['accuracy']-r['interventions'][f'{mode}/pos{p}'][str(z)]['accuracy'] for p in range(10)] for r in rr])
            ax=axes[ri,z];im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto')
            ax.set_xticks(range(10),[str(p) for p in range(8)]+['TASK','Q']);ax.set_yticks(range(3),[str(r['step']) for r in rr])
            ax.set_title(f'{policy} / {mode} / p={23 if z==0 else 41}');ax.set_xlabel('First-layer MLP token position');ax.set_ylabel('Switch step')
    fig.colorbar(im,ax=axes.flatten().tolist(),shrink=.7,label='Baseline minus ablated accuracy')
    fig.suptitle(f'seed {seed}: same modulus/position replay, token-specific first-layer contributions')
    name=f'token_localization_seed{seed}.png';fig.savefig(out/'figures'/name,dpi=150);plt.close(fig);files.append(name)
summary=json.loads((out/'summary.json').read_text());summary['token_localization']=data
summary['figures']=len(list((out/'figures').glob('*.png')))
invocation={}
for policy in ['fresh','inherit_A']:
    values=[]
    for jid,obj in summary['controls']['invocation_controls'].items():
        if '/B/'+policy not in jid:continue
        final=next(r for r in obj['records'] if r['step']==100000)
        values.extend(p['query_and_position'] for p in final['operations'] if p['compatible_B_operation'])
    invocation[policy]=dict(cases=len(values),at95=sum(v>=.95 for v in values),at99=sum(v>=.99 for v in values),minimum=min(values))
summary['counts']['shared_modulus_B_invocation']=invocation
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False),encoding='utf-8')
path=out/'机理分析报告.md';text=path.read_text().replace('生成74张PNG',f'生成{summary["figures"]}张PNG')
lines=['','## 补充：计算依赖的token位置与优化器策略','',
    '同模数同位置的两个种子、两种策略，在0/10k/100k逐token消融第一层MLP（zero和train-mean），共12个checkpoint的240次冻结模型干预。该小组为探索性定位，不表示所有关系都如此。',
    'seed0继承分支A模23在起点，query位置MLP消融后准确率为0.150/0.164（zero/mean）；终点为0.627/0.984。位置1从0.721/0.857变为0.207/0.219。seed1继承分支的query处终点消融准确率约0.918/0.912。这支持功能依赖位置改变，但不同消融强度及不同种子仍有差异。',
    'seed0重置分支终点query处消融准确率为0.031/0.025。这说明相同任务与replay比例也可能沿不同优化路径形成不同的功能依赖；不能统一规定第一层或第二层完成全部运算。',
    '这些干预检验的是MLP输出在某个token的功能贡献，尚不能单独证明该token执行完整模加法算法。',
    '', '共享模数的B调用适配：fresh的8/8案例≥99%；inherit_A的8/8≥95%，其中4/8≥99%，最低约95.9%。适配不是原始A保持，也不证明旧A电路保留。','']
lines += [f'- [{name}](figures/{name})' for name in files]
path.write_text(text+'\n'.join(lines)+'\n',encoding='utf-8')
with zipfile.ZipFile(base/'optimizer_mechanism_report.zip','w',zipfile.ZIP_DEFLATED) as z:
    for f in sorted(out.rglob('*')):
        if f.is_file():z.write(f,f.relative_to(out))
    for name in ['gradient/all.json','step_geometry.json','token_localization.json']:
        z.write(base/name,name.replace('/','_'))
print(json.dumps(dict(figures=summary['figures'],invocation=invocation),ensure_ascii=False))
