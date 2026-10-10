"""No-replay temporal evidence, explicit detection intervals and proxy limitations."""
import json,csv,zipfile,itertools
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

root=Path('runs/phase2/optimizer_switch_20261010').resolve();base=root/'no_replay_temporal_20261010';out=base/'report';out.mkdir(exist_ok=True)
old=root/'mechanism_optimizer_analysis_20261010';summary=json.loads((old/'report/summary.json').read_text());jobs=json.loads((base/'plan.json').read_text())['jobs'];rows=[];events=[]
def event(rr,key,predicate,n=3):
    idx=next((i for i in range(len(rr)-n+1) if all(predicate(v[key]) for v in rr[i:i+n])),None)
    return None if idx is None else dict(last_prior=rr[idx-1]['step'] if idx else None,first_observed=rr[idx]['step'],confirmed_at=rr[idx+n-1]['step'])
def number(x):return '-' if x is None else str(x['first_observed'])
for j in jobs:
    jid=j['id'];assert (base/jid/'complete.json').exists(),jid
    temporal=json.loads((base/jid/'temporal.json').read_text());native=json.loads((root/jid/'history.json').read_text())
    source=temporal['reference_controls'];controls={t:{k:v for k,v in source[t]['interventions'].items()} for t in ['A','B']}
    all_raw=[]
    for f in sorted((old/jid).glob('step_*.json')):all_raw.extend(json.loads(f.read_text())['records'])
    for t in ['A','B']:
        for z in range(4):
            native_rows=[dict(step=h['step'],accuracy=next(v['accuracy'] for k,v in h['metrics'][t]['by_operation'].items() if f'/lat{z}/' in k)) for h in native]
            ss=sorted([r for r in summary['records'] if r['job']==jid and r['task']==t and r['latent']==z],key=lambda r:r['step'])
            rr=[]
            for s in ss:
                raw=next(r for r in all_raw if r['step']==s['step'] and r['task']==t)
                probe={target:[next(v['retrained_test'] for v in raw['probes'] if v['target']==target and v['latent']==z and v['layer']==l and v['site']=='resid_post' and v['position']==9) for l in range(3)] for target in ['xi','xj']}
                tr=next(r for r in temporal['records'] if r['step']==s['step'] and r['task']==t)
                r=dict(job=jid,seed=j['seed'],kind=jid.split('/')[1],policy=j['optimizer_policy'],task=t,latent=z,modulus=s['modulus'],step=s['step'],accuracy=s['accuracy'],sum_query=s['sum_query'],sum_all_positions=s['sum_all_positions'],operand_i=probe['xi'],operand_j=probe['xj'],operand_readable=min(max(probe['xi']),max(probe['xj'])),sum_readable=max(s['sum_query']),sum_any_readable=max(max(v) for v in s['sum_all_positions']),attention_mass=max(v['operand_mass'] for v in s['routing'] if v['query_position']==9),old_activation=s['old_activation'],old_tail=s['old_tail'],dynamic_old_attention=s['dynamic_old_attention'],template_native=tr['baseline'][str(z)]['accuracy'],template={k:v[str(z)]['accuracy'] for k,v in tr['interventions'].items()},template_reference={k:v[str(z)]['accuracy'] for k,v in controls[t].items()})
                if t=='A':r['adapter']=next((v for v in tr.get('shared_modulus_adapter',[]) if v['latent']==z),None)
                rows.append(r);rr.append(r)
            ev=dict(job=jid,seed=j['seed'],kind=jid.split('/')[1],policy=j['optimizer_policy'],task=t,latent=z,modulus=ss[0]['modulus'],behavior95=event(native_rows,'accuracy',lambda v:v>=.95),behavior_below50=event(native_rows,'accuracy',lambda v:v<.5),operand80=event(rr,'operand_readable',lambda v:v>=.8),operand_below50=event(rr,'operand_readable',lambda v:v<.5),sum80=event(rr,'sum_readable',lambda v:v>=.8),sum_below50=event(rr,'sum_readable',lambda v:v<.5),sum_any_below50=event(rr,'sum_any_readable',lambda v:v<.5),attention70=event(rr,'attention_mass',lambda v:v>=.7))
            if rr[0]['operand_readable']<.5:ev['operand_below50']=None
            if rr[0]['sum_readable']<.5:ev['sum_below50']=None
            if rr[0]['sum_any_readable']<.5:ev['sum_any_below50']=None
            events.append(ev)
json.dump(dict(records=rows,events=events),open(out/'temporal_evidence.json','w'))
with (out/'event_times.csv').open('w',newline='') as f:
    fields=['job','task','latent','modulus','event','last_prior','first_observed','confirmed_at'];w=csv.DictWriter(f,fields);w.writeheader()
    for e in events:
        for key in ['behavior95','behavior_below50','operand80','operand_below50','sum80','sum_below50','sum_any_below50','attention70']:
            if e[key] is not None:w.writerow(dict(job=e['job'],task=e['task'],latent=e['latent'],modulus=e['modulus'],event=key,**e[key]))
figures=[]
for j in jobs:
    jid=j['id'];fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for row,t in enumerate(['A','B']):
        for col,z in enumerate([0,1]):
            ax=axes[row,col];rr=sorted([r for r in rows if r['job']==jid and r['task']==t and r['latent']==z],key=lambda r:r['step']);x=[r['step'] for r in rr]
            for key,label,color in [('accuracy','Behavior','black'),('operand_readable','Both operands, linear readout','#0072B2'),('sum_readable','Best query sum readout','#D55E00'),('attention_mass','Max operand attention mass','#009E73')]:ax.plot(x,[r[key] for r in rr],label=label,color=color)
            ax.set(xscale='symlog',ylim=(-.03,1.03),xlabel='Switch steps',title=f'{t}/lat{z}, p={rr[0]["modulus"]}');ax.axhline(.8,c='gray',ls=':',lw=.5)
            ax.legend(fontsize=7)
    fig.suptitle(jid+' | proxies are not independently proven circuits');name=jid.replace('/','_')+'_formation.png';fig.savefig(out/name,dpi=140);plt.close(fig);figures.append(name)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    tr=json.loads((base/jid/'temporal.json').read_text())
    for row,t in enumerate(['A','B']):
        for col,z in enumerate([0,1]):
            ax=axes[row,col];rr=[r for r in tr['records'] if r['task']==t];x=[r['step'] for r in rr]
            ax.plot(x,[r['baseline'][str(z)]['accuracy'] for r in rr],c='black',label='Native')
            for l,color in [('1','#0072B2'),('2','#D55E00'),('1+2+3','#009E73')]:
                ax.plot(x,[r['interventions']['correct/'+l][str(z)]['accuracy'] for r in rr],c=color,label='Template L'+l)
                ax.plot(x,[r['interventions']['wrong/'+l][str(z)]['accuracy'] for r in rr],c=color,ls=':',alpha=.5)
            if t=='A' and any(r.get('shared_modulus_adapter') for r in rr):
                ax.plot(x,[next(v['query_and_position'] for v in r['shared_modulus_adapter'] if v['latent']==z) for r in rr],c='#CC79A7',label='Invoke B binding')
            ax.set(xscale='symlog',ylim=(-.03,1.03),xlabel='Switch steps',title=f'{t}/lat{z}');ax.legend(fontsize=7)
    fig.suptitle(jid+' | dotted: wrong digit keys; A source/B final templates');name=jid.replace('/','_')+'_rescue.png';fig.savefig(out/name,dpi=140);plt.close(fig);figures.append(name)
lines=['# 无 replay：任务切换中的功能形成与失效','',
'分析范围：4 种模数/位置关系 × 2 种子 × fresh/inherit_A，共16条 B-only续训分支。逐运算分析，重点B的23/41及A的对应latent；A不共享模数时对应29/43。输出槽位在A/B中不同，所有关系都包含槽位重绑定；不同位置是重配对，不是完全不共享输入位置。背景运算始终不共享模数和配对。',
'','## 时间定义和证据边界',
'正式训练行为使用全部88个保存点。补充路由干预也覆盖全部88个点，合计1408个checkpoint、A/B共2816项任务评估。已有探针/组件检查只覆盖其中事件加密后的子集，各曲线时间分辨率不同。事件以连续3个已观测点达标定义，CSV保留前一观测点与确认点；不是精确事件步数，也不保证之后永不回落。分析阈值为描述性阈值，没有做统计显著性检验。',
'操作数80%指两个操作数各自在query三个层中最佳线性读出达到80%；可以来自不同层，不等价于同一计算模块收齐操作数。本批操作数线性probe即使在已学会A的源模型上也未通过该标准，不能用于判断路由何时形成或消失；图中保留原始低分，事件表下降项只在初始分数至少50%时计算。sum80%是query最佳层的重训线性探针，不等价于因果计算形成。attention70%是query某头对两操作数的总注意力，不能排除只看一个操作数或看而不用。所有探针使用已有独立训练/验证/测试划分。',
'','## 重点行为与表征事件：继承优化器',
'|seed|任务关系|B模数|B操作数可读80%|B模和可读80%|B行为95%|A对应操作数可读下降到50%以下|A模和可读下降到50%以下|A行为下降到50%以下|',
'|---|---|---:|---:|---:|---:|---:|---:|---:|']
for j in jobs:
    if j['optimizer_policy']!='inherit_A':continue
    for z in [0,1]:
        eb=next(e for e in events if e['job']==j['id'] and e['task']=='B' and e['latent']==z);ea=next(e for e in events if e['job']==j['id'] and e['task']=='A' and e['latent']==z)
        lines.append('|'+ '|'.join(map(str,[j['seed'],j['id'].split('/')[1],eb['modulus'],number(eb['operand80']),number(eb['sum80']),number(eb['behavior95']),number(ea['operand_below50']),number(ea['sum_below50']),number(ea['behavior_below50'])]))+'|')
lines+=['','## 路由干预如何解释',
'A路由模板来自切换前模型，B模板来自该分支终点；只用train输入按query平均attention概率。干预只替换query注意力权重，当前checkpoint的V、MLP及读出保持不变。错误对照循环移位数字key，保持概率和熵。单层与三层联合干预均保存。模板在其来源模型上先检查，来源模板本来就失败的运算不能用其救回失败定位遗忘。',
'A模板若在中间阶段显著救回并优于错误模板，说明当前网络在补足这一路由条件下仍可计算/利用A答案；不是旧算法完整保留的证明。B终点模板若提前救回B，说明该阶段已有可被这一路由调用的功能；由于模板来自未来，不代表当时自然路由已经形成。模板不能救回也不能证明计算删除，可能有模板失真、表示变化及缺失其他路径。',
'共享模数adapter把A输入按定义重排到B位置并切换TASK/query，仍用原A答案评估。成功说明当前模型可通过B绑定计算这些A数字；不证明原A绑定或原A电路保留。非共享模数没有兼容B运算，不能做同型adapter对比。',
'','## 文件','temporal_evidence.json保留四个运算全部记录、表征事件及路由干预；event_times.csv保留事件观测区间；32张PNG分别显示每条分支的形成/失效代理曲线和密集路由恢复轨迹。完整密集补查留在服务器no_replay_temporal_20261010/<job>/temporal.json。',
'','## 尚不能下的结论','注意力、探针、行为曲线的先后不能直接等价于路由、计算电路的因果先后。需要结合路由恢复窗口及后续输入配对节点干预；本报告明确区分代理信号和受控调用。仅两个种子，尤其非共享模数的p41学习顺序可能差异很大，不给出全局统一形成顺序。']
(out/'无replay时序分析.md').write_text('\n'.join(lines),encoding='utf-8')
with zipfile.ZipFile(base/'no_replay_report.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in out.iterdir():archive.write(p,'report/'+p.name)
print('DONE',len(rows),len(events),len(figures),(base/'no_replay_report.zip').stat().st_size)
for e in events:
    if e['policy']=='inherit_A' and e['latent']<2:print(e['job'],e['task'],e['latent'],{k:number(e[k]) for k in ['behavior95','behavior_below50','operand80','operand_below50','sum80','sum_below50','attention70']})
