"""Add confirmed carriers and behavior-independent input dependence to temporal report."""
import json,itertools,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path('runs/phase2/optimizer_switch_20261010').resolve();base=root/'no_replay_temporal_20261010';out=base/'report';jobs=json.loads((base/'plan.json').read_text())['jobs'];rows=[];events=[]
for j in jobs:
    assert (base/j['id']/'paired_complete.json').exists()
    d=json.loads((base/j['id']/'paired_temporal.json').read_text())
    for r in d['records']:
        for z in range(4):
            s={donor:[next(v['absolute_query_attention_change'] for v in r['sensitivity'] if v['latent']==z and v['layer']==l and v['perturbation']==donor) for l in range(3)] for donor in ['i','j','same_sum','different_sum','background']}
            strong=[v for v in r['nodes'] if v['latent']==z and v['eligible']>=64 and min(v[k] for k in ['restore_i','restore_j','same_sum_restore','different_sum_transfer'])>=.9 and max(v['hybrid_i'],v['hybrid_j'])<=.1]
            rows.append(dict(job=j['id'],step=r['step'],task=r['task'],latent=z,baseline=r['baselines'][str(z)],sensitivity=s,strong_nodes=strong,node_status=r['node_status']))
    for t in ['A','B']:
        for z in range(4):
            rr=[r for r in rows if r['job']==j['id'] and r['task']==t and r['latent']==z]
            strong=[r for r in rr if r['strong_nodes']]
            ratios=[min(r['sensitivity']['i'][0],r['sensitivity']['j'][0])/max(r['sensitivity']['background'][0],1e-8) for r in rr]
            onset={}
            for threshold in [1,2]:
                i=next((i for i in range(len(rr)-2) if min(ratios[i:i+3])>=threshold),None)
                onset[str(threshold)]=None if i is None else dict(first_observed=rr[i]['step'],last_prior=rr[i-1]['step'] if i else None,confirmed_at=rr[i+2]['step'])
            events.append(dict(job=j['id'],task=t,latent=z,first_confirmed_carrier=None if not strong else dict(step=strong[0]['step'],nodes=[v['node'] for v in strong[0]['strong_nodes']]),last_confirmed_carrier=None if not strong else dict(step=strong[-1]['step'],nodes=[v['node'] for v in strong[-1]['strong_nodes']]),selective_input_dependence_onset=onset,initial_dependence_ratio=ratios[0]))
json.dump(dict(records=rows,events=events),open(out/'paired_evidence.json','w'))
for j in jobs:
    if j['optimizer_policy']!='inherit_A':continue
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ri,t in enumerate(['A','B']):
        for ci,z in enumerate([0,1]):
            ax=axes[ri,ci];rr=[r for r in rows if r['job']==j['id'] and r['task']==t and r['latent']==z];x=[r['step'] for r in rr]
            for k,color in [('i','#0072B2'),('j','#D55E00'),('background','#009E73')]:ax.plot(x,[r['sensitivity'][k][0] for r in rr],label='L1 query attention response: '+k,c=color)
            ax.set(xscale='symlog',xlabel='Switch steps',ylabel='RMS activation change',title=f'{t}/lat{z}');ax.legend(fontsize=7)
    fig.suptitle(j['id']+' | matched operand changes vs nontarget pair shuffle');fig.savefig(out/(j['id'].replace('/','_')+'_input_dependence.png'),dpi=130);plt.close(fig)
    fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
    for ri,t in enumerate(['A','B']):
        for ci,z in enumerate([0,1]):
            ax=axes[ri,ci];rr=[r for r in rows if r['job']==j['id'] and r['task']==t and r['latent']==z and r['node_status']!='not_selected_checkpoint']
            mat=np.full((6,len(rr)),np.nan)
            for col,r in enumerate(rr):
                if r['node_status']=='tested':
                    for n in r['strong_nodes']:mat[2*n['node'][1]+int(n['node'][2]!=9),col]=n['sum_specificity']
            im=ax.imshow(mat,vmin=.8,vmax=1,cmap='viridis',aspect='auto');ax.set_xticks(range(len(rr)),[str(r['step']) for r in rr],rotation=90,fontsize=6);ax.set_yticks(range(6),[f'L{l+1} '+site for l in range(3) for site in ['query','prefix']]);ax.set_title(f'{t}/lat{z} | blanks are NOT proof of absence')
    fig.colorbar(im,ax=axes,label='Confirmed sum-specific interchange score');fig.suptitle(j['id']);fig.savefig(out/(j['id'].replace('/','_')+'_carriers.png'),dpi=130);plt.close(fig)
lines=['# 配对输入因果补查：无 replay 功能时序','',
'16条无replay分支，全部1408个checkpoint进行配对输入依赖检查。各运算128个测试配对，数字对保持其held-out支持；背景对照在同一query条件下打乱一整个非目标运算输入对，保留背景train-pair支持。第一层query attention输出的变化追踪输入通道依赖，不要求模型答对；该变化不等价于正确路由，也不能排除后续层路径。',
'', '模和节点测试在已有机制观测点和行为阈值前后点加密，候选post-block残差在val扫描三层全部token位置，按层与query/prefix分别选候选，再在独立test确认。严格标准：两个单操作数恢复、同和恢复、异和跟随均≥90%，两个混合操作数答案跟随≤10%，正确配对样本至少64。阴性或正确样本不足时不能解释计算消失。','',
'## 继承优化器下：B首次确认的模和载体与输入依赖',
'|seed|关系|B模数|第一层目标依赖超过背景2倍的首个连续窗口|首次确认模和载体|当时的位置（层从1计数）|',
'|---|---|---:|---:|---:|---|']
for j in jobs:
    if j['optimizer_policy']!='inherit_A':continue
    for z in [0,1]:
        e=next(e for e in events if e['job']==j['id'] and e['task']=='B' and e['latent']==z);carrier=e['first_confirmed_carrier'];onset=e['selective_input_dependence_onset']['2'];location='-' if carrier is None else ', '.join(f'L{n[1]+1}/pos{n[2]}' for n in carrier['nodes'])
        lines.append('|'+ '|'.join(map(str,[j['seed'],j['id'].split('/')[1],[23,41][z],'-' if onset is None else onset['first_observed'],'-' if carrier is None else carrier['step'],location]))+'|')
lines+=['','## 具体顺序和反例',
'seed0共享模数p23：A前2–5步行为与模和可读性已明显下降，第一层对两个目标操作数的敏感度仍接近源模型，之后在10–20步下降。这支持“原任务答案/表征失效早于该层输入依赖显著下降”，不等于所有路由完全保留、或所有模算法被覆盖。',
'seed0共享模数且相同位置的B p23：第一层双操作数敏感度在50步已显著高于背景；100步独立test确认第二/三层query具有模和传递功能。位置不同分支p23在200步确认第二/三层query载体，终点第一层query也通过确认，显示B内部实现会在功能学会之后继续变化。',
'密集静态路由恢复在校准有效的模板条件下没有发现“原本低于50%，补路由后达到80%”的重点运算窗口（精确统计见主报告补充）。不支持仅靠恢复这些query注意力概率即可修复A，或让B显著提前学会。失败不能排除动态路由、非query路径或表示接口因素。',
'', '## 比较限制',
'输入依赖比值受干预大小、表示尺度和背景运算影响，固定阈值1与2均保存，不能把阈值跨越叫作真实路由形成。首次确认节点是在所检查时点与候选中首次通过门槛，不能断言计算在此前不存在。无replay任务选择/槽位绑定变化与模数/位置因素共同作用；两个种子只能比较这批轨迹，不能宣称稳定总体规律。所有原始配对和分阶段节点结果留在服务器。']
(out/'配对因果时序补充.md').write_text('\n'.join(lines),encoding='utf-8')
# Count stringent template rescue windows using all dense records and calibrated references.
counts={};maxima={}
for task in ['A','B']:
    windows=[];best=[]
    for j in jobs:
        d=json.loads((base/j['id']/'temporal.json').read_text())
        for r in d['records']:
            if r['task']!=task:continue
            for z in [0,1]:
                native=r['baseline'][str(z)]['accuracy']
                for layers in ['1','2','3','1+2+3']:
                    a=r['interventions']['correct/'+layers][str(z)]['accuracy'];w=r['interventions']['wrong/'+layers][str(z)]['accuracy'];ref=d['reference_controls'][task]['interventions']['correct/'+layers][str(z)]['accuracy']
                    if ref>=.8 and native<.5:
                        best.append((a,j['id'],r['step'],z,layers,native,w))
                        if a>=.8 and a-w>=.3:windows.append((j['id'],r['step'],z,layers,native,a,w))
    counts[task]=len(windows);maxima[task]=sorted(best,reverse=True)[:5]
json.dump(dict(stringent_rescue_windows=counts,highest_rescue_at_native_below50=maxima),open(out/'template_rescue_summary.json','w'))
with (out/'无replay时序分析.md').open('a',encoding='utf-8') as f:f.write(f"\n\n补充：来源模板准确率≥80%、当前原生行为<50%、恢复后≥80%且高于错误模板至少30个百分点的重点运算窗口，A={counts['A']}，B={counts['B']}。全分支最大恢复记录见template_rescue_summary.json。\n")
with zipfile.ZipFile(base/'no_replay_report.zip','w',zipfile.ZIP_DEFLATED) as archive:
    for p in out.iterdir():archive.write(p,'report/'+p.name)
print('DONE paired report',len(rows),len(events),'template windows',counts,'archive', (base/'no_replay_report.zip').stat().st_size)
print('\n'.join(lines[6:26]))
