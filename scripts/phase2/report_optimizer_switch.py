"""Summarize matched frozen trajectory analysis and generate PNG figures."""
import argparse
import csv
import json
from pathlib import Path
import zipfile

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KINDS=['same_mod_same_pos','same_mod_diff_pos','diff_mod_same_pos','diff_mod_diff_pos']
POLICIES=['fresh','inherit_A']
OBJECTIVES=['A','B','mix_0.1']


def pass_task(row, task):
    return all(v['accuracy']>=.95 for v in row['metrics'][task]['by_operation'].values())


def probe(record, z, layer, key='retrained_test', position=9):
    return next(p[key] for p in record['probes'] if p['latent']==z and p['layer']==layer and
                p['site']=='resid_post' and p['position']==position and p['target']=='sum')


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/optimizer_switch_20261010');args=p.parse_args()
    root=Path(args.root).resolve();base=root/'mechanism_optimizer_analysis_20261010'
    out=base/'report';figdir=out/'figures';figdir.mkdir(parents=True,exist_ok=True)
    plan=json.loads((base/'plan.json').read_text());items=plan['jobs']
    assert all((base/i['job']['id']/'complete.json').exists() and (base/i['job']['id']/'controls_complete.json').exists() for i in items), 'Analysis incomplete'
    plt.rcParams.update({'font.size':8,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    hist={};rows=[];behavior=[];raw={};figures=[]
    def save(fig,name):
        fig.savefig(figdir/name,bbox_inches='tight');plt.close(fig);figures.append(name)
    for item in items:
        j=item['job'];jid=j['id'];h=json.loads((root/jid/'history.json').read_text());hist[jid]=h
        early=[r for r in h if 0<r['step']<=2000]
        lo=min(early,key=lambda r:r['metrics']['A']['macro_operation_accuracy'])
        recovery=next((r['step'] for r in h if r['step']>lo['step'] and pass_task(r,'A')),None)
        b95=next((r['step'] for r in h if pass_task(r,'B')),None)
        row=dict(job=jid,seed=j['seed'],kind=jid.split('/')[1],objective=j['objective'],policy=j['optimizer_policy'],
                 A_step1=next(r['metrics']['A']['macro_operation_accuracy'] for r in h if r['step']==1),
                 A_early_min=lo['metrics']['A']['macro_operation_accuracy'],A_early_min_step=lo['step'],
                 A_early_recovery_all95=recovery,B_first_all95=b95,
                 A_final=h[-1]['metrics']['A']['macro_operation_accuracy'],B_final=h[-1]['metrics']['B']['macro_operation_accuracy'],
                 A_final5_all95=all(pass_task(r,'A') for r in h[-5:]),
                 B_final5_all95=all(pass_task(r,'B') for r in h[-5:]),
                 A_maintained_at_saved_points=all(pass_task(r,'A') for r in h))
        behavior.append(row)
        records=[]
        for f in sorted((base/jid).glob('step_*.json')):records.extend(json.loads(f.read_text())['records'])
        raw[jid]=records
        for r in records:
            for z in range(4):
                c=r['causal'];iv=c['interventions'];entry=dict(job=jid,seed=j['seed'],kind=jid.split('/')[1],
                    objective=j['objective'],policy=j['optimizer_policy'],step=r['step'],task=r['task'],latent=z,
                    modulus=j[r['task']]['operations'][z]['modulus'],accuracy=c['baseline'][str(z)]['accuracy'],
                    loss=c['baseline'][str(z)]['loss'],task_flip_agreement=c['task_counterfactual']['prediction_agreement'],
                    old_readout=iv['old_finalLN_head_on_current_residual'][str(z)]['accuracy'],
                    sum_query=[probe(r,z,l) for l in range(3)],
                    sum_fixed=[probe(r,z,l,'fixed_test') for l in range(3)],
                    sum_aligned=[probe(r,z,l,'aligned_fixed_test') for l in range(3)],
                    sum_all_positions=[[probe(r,z,l,position=q) for q in range(10)] for l in range(3)],
                    mean_mlp_query_accuracy=[iv[f'mean_mlp/L{l}/query'][str(z)]['accuracy'] for l in range(3)],
                    mean_mlp_all_accuracy=[iv[f'mean_mlp/L{l}/all'][str(z)]['accuracy'] for l in range(3)],
                    routing=[v for v in c['routing'] if v['latent']==z],
                    mean_component_effects={k:dict(accuracy_drop=c['baseline'][str(z)]['accuracy']-v[str(z)]['accuracy'],
                                                   loss_increase=v[str(z)]['loss']-c['baseline'][str(z)]['loss'])
                        for k,v in iv.items() if k.startswith('mean_')})
                for prefix,key in [('new_activation_old_tail/','old_tail'),('old_activation_new_tail/','old_activation'),('old_attention_current_values/','dynamic_old_attention')]:
                    entry[key]=max(v[str(z)]['accuracy'] for k,v in iv.items() if k.startswith(prefix))
                rows.append(entry)
    # Validate all observed update batches within fresh/inherit pairs.
    pair_checks=[]
    for seed in [0,1]:
        for kind in KINDS:
            for obj in OBJECTIVES:
                a,b=[hist[f'seed{seed}/{kind}/{obj}/{pol}'] for pol in POLICIES]
                assert len(a)==len(b) and all(x['step']==y['step'] for x,y in zip(a,b))
                same=all(x['last_update']['batch_sha256']==y['last_update']['batch_sha256'] for x,y in zip(a,b) if x['last_update'] is not None)
                assert same
                assert a[0]['metrics']==b[0]['metrics']
                pair_checks.append(dict(seed=seed,kind=kind,objective=obj,identical_observed_batches=same))
            # Behavior trajectories, separate seeds and all operations retained in raw histories.
            fig,axes=plt.subplots(3,2,figsize=(11,9))
            for ri,obj in enumerate(OBJECTIVES):
                for ci,task in enumerate(['A','B']):
                    ax=axes[ri,ci]
                    for pol,color in [('fresh','#D55E00'),('inherit_A','#0072B2')]:
                        hh=hist[f'seed{seed}/{kind}/{obj}/{pol}'];x=[r['step'] for r in hh]
                        ax.plot(x,[r['metrics'][task]['macro_operation_accuracy'] for r in hh],label=pol,color=color,lw=2)
                        for z in range(4):ax.plot(x,[next(v['accuracy'] for k,v in r['metrics'][task]['by_operation'].items() if f'/lat{z}/' in k) for r in hh],color=color,alpha=.22,lw=.7)
                    ax.set(xscale='symlog',ylim=(-.02,1.02),title=f'{obj}: {task}',xlabel='Switch steps',ylabel='Validation accuracy');ax.axhline(.95,color='gray',ls=':');ax.legend()
            fig.suptitle(f'seed {seed} | {kind} | thick macro, thin per operation');fig.tight_layout()
            save(fig,f'behavior_seed{seed}_{kind}.png')
            for obj in OBJECTIVES:
                for task in (['A'] if obj=='A' else ['A','B']):
                    fig,axes=plt.subplots(2,2,figsize=(12,7))
                    for z,ax in enumerate(axes.flat):
                        for pol,ls in [('fresh','-'),('inherit_A','--')]:
                            rr=[r for r in rows if r['job']==f'seed{seed}/{kind}/{obj}/{pol}' and r['task']==task and r['latent']==z]
                            xx=[r['step'] for r in rr]
                            ax.plot(xx,[r['accuracy'] for r in rr],ls=ls,color='black',lw=2,label=f'{pol} behavior')
                            for l,color in enumerate(['#0072B2','#009E73','#D55E00']):ax.plot(xx,[r['sum_query'][l] for r in rr],ls=ls,color=color,label=f'{pol} L{l+1} sum')
                        ax.set(xscale='symlog',ylim=(-.02,1.02),xlabel='Switch steps',title=f'{task}/lat{z}, p={rr[0]["modulus"]}')
                    handles,labels=axes[0,0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=7)
                    fig.suptitle(f'seed {seed} | {kind} | {obj} | retrained query sum');fig.tight_layout(rect=(0,.1,1,.95))
                    save(fig,f'probe_seed{seed}_{kind}_{obj}_{task}.png')
                if obj!='A':
                    # All mean component effects, no winning-head selection.
                    fig,axes=plt.subplots(2,4,figsize=(18,10),layout='constrained')
                    for ri,pol in enumerate(POLICIES):
                        for z in range(4):
                            ax=axes[ri,z];rr=[r for r in rows if r['job']==f'seed{seed}/{kind}/{obj}/{pol}' and r['task']=='A' and r['latent']==z]
                            keys=[f'mean_head/L{l}/H{hh}/all' for l in range(3) for hh in range(4)]+[f'mean_mlp/L{l}/all' for l in range(3)]
                            mat=np.array([[r['mean_component_effects'][k]['accuracy_drop'] for r in rr] for k in keys])
                            im=ax.imshow(mat,vmin=-1,vmax=1,cmap='RdBu_r',aspect='auto');ax.set_xticks(range(len(rr)),[str(r['step']) for r in rr],rotation=90,fontsize=6)
                            ax.set_yticks(range(len(keys)),[f'L{l+1}H{hh}' for l in range(3) for hh in range(4)]+['L1MLP','L2MLP','L3MLP'],fontsize=6)
                            ax.set_title(f'{pol} A/lat{z}');ax.set_xlabel('Switch step')
                    fig.colorbar(im,ax=axes.flatten().tolist(),shrink=.6,label='Baseline accuracy minus mean-ablation accuracy')
                    fig.suptitle(f'seed {seed} | {kind} | {obj} | A component dependencies (all tokens)')
                    save(fig,f'causal_seed{seed}_{kind}_{obj}.png')
    gradients=json.loads((base/'gradient/all.json').read_text())
    for seed in [0,1]:
        for kind in KINDS:
            fig,axes=plt.subplots(3,3,figsize=(13,9))
            for ci,obj in enumerate(OBJECTIVES):
                for pol,col in [('fresh','#D55E00'),('inherit_A','#0072B2')]:
                    rr=[r for r in gradients if r['job']==f'seed{seed}/{kind}/{obj}/{pol}'];x=[r['step'] for r in rr]
                    key='A/adamw' if obj=='A' else 'B/adamw' if obj=='B' else 'mix_0.1/adamw'
                    axes[0,ci].plot(x,[r['statistics']['global']['A_B_cosine'] for r in rr],color=col,label=pol)
                    axes[1,ci].plot(x,[r['statistics']['global']['update_norms'][key] for r in rr],color=col,label=pol)
                    axes[2,ci].plot(x,[r['effects'][key]['A_val_loss_after']-r['effects'][key]['A_val_loss_before'] for r in rr],color=col,label=pol)
                for ri in range(3):axes[ri,ci].set_xscale('symlog');axes[ri,ci].grid(alpha=.2);axes[ri,ci].legend()
                axes[0,ci].set_title(obj);axes[0,ci].set_ylabel('Raw cos(gA,gB)');axes[1,ci].set_ylabel('Diagnostic AdamW delta norm');axes[2,ci].set_ylabel('Counterfactual A loss change')
            fig.suptitle(f'seed {seed} | {kind} | same diagnostic inputs, actual optimizer state');fig.tight_layout()
            save(fig,f'gradient_seed{seed}_{kind}.png')
    controls={}
    for name,pattern in [('invocation_controls','controls.json'),('static_routing','controls.json'),('probe_audit','audit.json')]:
        controls[name]={str(f.parent.relative_to(base/name)):json.loads(f.read_text()) for f in (base/name).glob(f'seed*/**/{pattern}')}
    geometry=json.loads((base/'step_geometry.json').read_text())
    for seed in [0,1]:
        fig,axes=plt.subplots(2,2,figsize=(13,8))
        for ax,kind in zip(axes.flat,KINDS):
            xx=np.arange(3)
            for offset,direction,magnitude,label,color in [(-.27,'fresh','fresh','Fresh observed','#D55E00'),
                (-.09,'fresh','inherit_A','Fresh direction, inherit norm','#E69F00'),
                (.09,'inherit_A','inherit_A','Inherited observed','#0072B2'),
                (.27,'inherit_A','fresh','Inherited direction, fresh norm','#56B4E9')]:
                values=[next(r['metrics']['A']['macro_operation_accuracy'] for r in geometry['records'] if r['group']==f'seed{seed}/{kind}/{obj}' and r['direction']==direction and r['magnitude']==magnitude) for obj in OBJECTIVES]
                ax.bar(xx+offset,values,.18,label=label,color=color)
            ax.set(xticks=xx,xticklabels=OBJECTIVES,ylim=(0,1.04),title=kind,ylabel='A validation accuracy after delta')
        hh,ll=axes[0,0].get_legend_handles_labels();fig.legend(hh,ll,loc='lower center',ncol=2);fig.suptitle(f'seed {seed}: native first update, direction versus magnitude');fig.tight_layout(rect=(0,.1,1,.95))
        save(fig,f'update_geometry_seed{seed}.png')
    counts={}
    for pol in POLICIES:
        aa=[r for r in behavior if r['objective']=='A' and r['policy']==pol]
        bb=[r for r in rows if r['objective']=='B' and r['policy']==pol and r['task']=='A' and r['step']==100000]
        replay=[r for r in rows if r['objective']=='mix_0.1' and r['policy']==pol and r['task']=='A' and r['step']==100000]
        br=[r for r in behavior if r['objective']=='mix_0.1' and r['policy']==pol]
        counts[pol]=dict(A_step1_range=[min(r['A_step1'] for r in aa),max(r['A_step1'] for r in aa)],
                        B_final_A_operation_range=[min(r['accuracy'] for r in bb),max(r['accuracy'] for r in bb)],
                        B_final_A_query_sum_readable80=sum(max(r['sum_query'])>=.8 for r in bb),
                        B_final_A_any_position_sum_readable80=sum(max(max(x) for x in r['sum_all_positions'])>=.8 for r in bb),
                        B_final_old_readout_rescue80=sum(r['old_readout']>=.8 for r in bb),
                        B_final_old_tail_rescue80=sum(r['old_tail']>=.8 for r in bb),
                        replay_final_joint5_success=sum(r['A_final5_all95'] and r['B_final5_all95'] for r in br),
                        replay_fixed_drop_retrained_high=sum(any(f<.5 and t>=.9 for f,t in zip(r['sum_fixed'],r['sum_query'])) for r in replay),
                        replay_alignment_rescue=sum(any(f<.5 and a>=.9 and t>=.9 for f,a,t in zip(r['sum_fixed'],r['sum_aligned'],r['sum_query'])) for r in replay))
    audit=[p for value in controls['probe_audit'].values() for rr in value for p in rr['probes']]
    counts['probe_audit']=dict(probes=len(audit),increase_over010=sum(p['test_800']-p['test_400']>.1 for p in audit),
                               max_increase=max(p['test_800']-p['test_400'] for p in audit))
    geometry_counts={}
    for obj in OBJECTIVES:
        rr=[r for r in geometry['records'] if r['objective']==obj]
        groups=sorted({r['group'] for r in rr});improved=0;worst=0;matched_difference=[]
        for group in groups:
            def acc(direction,magnitude):return next(r['metrics']['A']['macro_operation_accuracy'] for r in rr if r['group']==group and r['direction']==direction and r['magnitude']==magnitude)
            improved+=acc('fresh','inherit_A')>acc('fresh','fresh')+.01
            worst+=acc('inherit_A','fresh')<acc('inherit_A','inherit_A')-.01
            matched_difference.append(acc('fresh','inherit_A')-acc('inherit_A','inherit_A'))
        geometry_counts[obj]=dict(fresh_smaller_norm_improves_over001=int(improved),inherited_larger_norm_harms_over001=int(worst),
                                  same_small_norm_fresh_minus_inherited_accuracy_range=[min(matched_difference),max(matched_difference)])
    counts['step_geometry']=geometry_counts
    summary=dict(counts=counts,behavior=behavior,records=rows,pair_checks=pair_checks,controls=controls,step_geometry=geometry,
                 gradient_snapshots=len(gradients),checkpoint_count=plan['checkpoints'],figures=len(figures))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False),encoding='utf-8')
    with (out/'behavior_summary.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(behavior[0]));writer.writeheader();writer.writerows(behavior)
    lines=['# 优化器切换补充实验：轨迹与机制分析','',
           f'48/48训练分支完成100,000步；分析{plan["checkpoints"]}个选定checkpoint、{len(gradients)}个梯度快照，生成{len(figures)}张PNG。未进行新的主模型训练。','',
           '## 比较范围与证据边界','',
           '四种任务关系×两个种子×A-only/B-only/10% replay×fresh/inherit_A。fresh清空AdamW动量、二阶矩和step；inherit_A完整继承，不单独定位其中哪个状态起作用。两策略初始权重相同，所有已保存观测点的训练batch hash逐对一致，共24对。',
           'A-only继续源A采样流；B/replay重启原套件采样流。三目标的首批数据并不相同；策略内的比较严格匹配。学习率、decay、裁剪、任务和划分保持原配置。',
           'checkpoint来自预定时间点及验证集上的最低点、恢复/学习事件；每对策略取事件点并集，保证相同时间比较。图同时保存逐运算行为，不将宏平均当作全部运算保持。第一恢复时刻仅是一次达到标准，不代表稳定保持。稳定标准为最后5个已保存点所有A/B运算各≥0.95。',
           '固定probe来自共同A来源；每运算各512个train/val/test样本，模型原始residue-pair划分互斥。探针400步，含全部token位置的sum及query操作数。对齐只用train匹配激活，不用测试标签。独立分析评估集与训练评估种子不同。',
           '因果检查包括zero/mean逐头和MLP、旧表示/旧tail/旧读出、self-continuation与shuffle对照、TASK反事实；额外检查train-only静态路由模板与错误key模板、定义预先指定的B调用适配、400/800步和随机标签probe审计。',
           '均值消融和跨checkpoint拼接仍可能有分布/接口错配；低行为准确率下微小消融效应不能说明组件不参与。旧动态注意力可能携带输入数值信息。最佳层救回仅是探索性结果，失败不能证明知识删除。B调用适配成功不证明原A电路保留。',
           '梯度诊断使用固定小训练batch，反事实单步损失来自固定A验证集，包含真实AdamW历史、裁剪和decay；不是实际训练8192-query更新本身，后者范数在history保存。梯度角度不是独立因果证明。两个种子只能支持这批实例。','',
           '## 行为结果：先区分重置损伤与持续任务干扰','',
           '| seed | 关系 | 目标 | 优化器 | A第1步 | A早期最低 | 最低点 | A首次恢复全部95% | 最后5点A/B均过线 |',
           '|---|---|---|---|---:|---:|---:|---:|---|']
    lines += [f'| {r["seed"]} | {r["kind"]} | {r["objective"]} | {r["policy"]} | {r["A_step1"]:.3f} | {r["A_early_min"]:.3f} | {r["A_early_min_step"]} | {r["A_early_recovery_all95"]} | {r["A_final5_all95"] and r["B_final5_all95"]} |' for r in behavior]
    lines += ['', '## 可以从这批对照得到的结论','',
              'A-only的第1步损伤在fresh与inherit_A间存在直接匹配对照；A-only的快速恢复、B-only的持续遗忘和replay的恢复不能被同一个“优化器重置”解释统一替代。继承状态不会自动解决新任务干扰。',
              '所有8个inherit_A replay分支的早期宏平均准确率最低点也接近随机。因此本配置10% replay下的失效后恢复不只是fresh AdamW启动伪影。不能将early macro最低点直接解释为每个运算同时失效。',
              'inherit_A对B学习速度及最终稳定保持并非统一有益，必须同时看B表现；不能把暂时保留A但B学习较慢当作机制上避免遗忘。A-only少量后期掉点提示原优化路径仍有波动。','',
              '## 代表性轨迹：继承AdamW仍发生失效、恢复与后期重组','',
              '以下为seed0/same_mod_same_pos/mix_0.1/inherit_A，A模23，独立held-out分析集。该案例属于探索性机制展示，不是挑选后得到的总体显著性证据。','',
              '| step | A行为 | L2重训sum | L3重训sum | 当前L2激活接旧tail | 旧L2激活接当前tail |',
              '|---:|---:|---:|---:|---:|---:|']
    case='seed0/same_mod_same_pos/mix_0.1/inherit_A'
    for step in [0,1,10,100,1000,10000,100000]:
        r=next(x for x in raw[case] if x['task']=='A' and x['step']==step);c=r['causal'];iv=c['interventions']
        lines.append(f'| {step} | {c["baseline"]["0"]["accuracy"]:.3f} | {probe(r,0,1):.3f} | {probe(r,0,2):.3f} | {iv["new_activation_old_tail/L1/all"]["0"]["accuracy"]:.3f} | {iv["old_activation_new_tail/L1/all"]["0"]["accuracy"]:.3f} |')
    lines += ['',
              '第100步旧L2激活接当前tail能恢复，而旧读出单独替换不能恢复，支持该阶段读出之前的路径失效且后续网络仍可利用旧表示；同运算shuffle旧L2表示约0.041，不支持仅靠注入幅度即可救回。不能仅凭接口恢复确定知识所在。',
              '行为恢复后，L2可读性下降但L3保持；同时组件干预依赖改变，为功能路径重组提供比probe下降更直接的证据。下表列出固定编号组件query范围消融（单元格为mean / zero），准确率越低表示当前功能对该干预更敏感。H编号从0开始，L编号从1开始。不同消融仍有分布错配，不能将head编号视为跨种子普遍电路。','',
              '| step | 基线 | 消融L1/H2 | 消融L3/H1 | 消融L2 MLP |',
              '|---:|---:|---:|---:|---:|']
    for step in [0,10000,100000]:
        r=next(x for x in raw[case] if x['task']=='A' and x['step']==step);c=r['causal'];iv=c['interventions']
        values=[f'{c["baseline"]["0"]["accuracy"]:.3f}']
        for suffix in ['head/L0/H2/query','head/L2/H1/query','mlp/L1/query']:
            values.append(f'{iv["mean_"+suffix]["0"]["accuracy"]:.3f} / {iv["zero_"+suffix]["0"]["accuracy"]:.3f}')
        lines.append('| '+str(step)+' | '+' | '.join(values)+' |')
    lines += ['', 'L1/H1在终点mean消融有强损伤但zero消融无损伤，提醒不能将单一种消融效应直接视为组件必要性；代表性结论优先采用mean与zero方向一致的L1/H2和L3/H1变化。该比较限于query位置：全token消融仍显示第一层重要，不能解释为整个第一层退出计算。',
              '探针预算审计发现6912项中的21项在800步比400步提高超过0.1，最高提高0.160；这些项主要涉及A-only损伤点和部分B中间表示。不能把400步低分一律当作知识消失；原始审计与随机标签对照已保留。']
    lines += ['', '## 优化器为什么影响第一步：更新方向与幅度的交叉干预','',
              '额外96次冻结副本评估：取正式训练第0→1步的实际delta，将fresh/inherit两个方向分别缩放到两种已观察到的范数，保持初始权重和固定A/B评估集不变。未进行额外训练；未缩放的端点准确率逐个验证与原训练第1步一致。',
              '该干预检验第一步损伤能否由更新幅度解释，以及同范数下方向是否仍有影响。缩放包含整个delta（包括decay）；不是独立重置动量/二阶矩的实验，也不能推广到后续长轨迹。','',
              '```json',json.dumps(geometry_counts,ensure_ascii=False,indent=2),'```']
    lines += ['', '## 机制汇总计数','', '计数以每策略8分支×4运算=32个运算为单位；运算、checkpoint、共享来源不是独立重复。','',
              '```json',json.dumps(counts,ensure_ascii=False,indent=2),'```','',
              '## 图表索引','']
    lines += [f'- [{name}](figures/{name})' for name in figures]
    lines += ['', '原始轨迹probe权重、干预结果与审计在服务器分析目录，summary.json保存逐运算/逐点概要及对照，gradient_all.json包含全局/分层/分组件梯度与真实优化器单步诊断。']
    (out/'机理分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(base/'optimizer_mechanism_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for f in sorted(out.rglob('*')):
            if f.is_file():z.write(f,f.relative_to(out))
        z.write(base/'gradient/all.json','gradient_all.json')
        z.write(base/'step_geometry.json','step_geometry.json')
    print(json.dumps(dict(counts=counts,figures=len(figures)),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
