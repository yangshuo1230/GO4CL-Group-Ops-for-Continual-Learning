"""Read finished follow-up JSON, create mechanism figures and a Chinese report."""
from pathlib import Path
import argparse,json,statistics,zipfile
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

KINDS=['same_mod_same_pos','same_mod_diff_pos','diff_mod_same_pos','diff_mod_diff_pos']


def main():
    p=argparse.ArgumentParser();p.add_argument('--root',default='runs/phase2/mechanism_suite_20261009');args=p.parse_args();root=Path(args.root).resolve();base=root/'mechanism_followup_20261010';out=base/'report';out.mkdir(exist_ok=True);figdir=out/'figures';figdir.mkdir(exist_ok=True)
    plan=json.loads((base/'plan.json').read_text())['jobs'];done=[x for x in plan if (base/x['job']['id']/'complete.json').exists()]
    assert len(done)==142, f'Only {len(done)}/142 analysis jobs complete'
    plt.rcParams.update({'font.size':9,'savefig.dpi':150,'axes.spines.top':False,'axes.spines.right':False})
    summaries=[];figures=[];byjob={}
    def save(fig,name,label):
        fig.savefig(figdir/name,bbox_inches='tight');plt.close(fig);figures.append((name,label))
    def probe(row,z,layer,position=9,key='retrained_test'):
        return next(v[key] for v in row['probes'] if v['latent']==z and v['layer']==layer and v['site']=='resid_post' and v['position']==position and v['target']=='sum')
    def best_intervention(row,z,prefix):
        values=[(k,v[str(z)]['accuracy']) for k,v in row['causal']['interventions'].items() if k.startswith(prefix)]
        return max(values,key=lambda x:x[1]) if values else (None,None)
    for item in done:
        job=item['job'];jid=job['id'];records=[]
        for path in sorted((base/jid).glob('step_*.json')):
            records.extend(json.loads(path.read_text())['records'])
        byjob[jid]=records
        for row in records:
            for z in range(4):
                entry=dict(job=jid,kind=item['kind'],seed=job['seed'],replay=job['replay_ratio'],step=row['step'],task=row['task'],latent=z,
                           modulus=job[row['task']]['operations'][z]['modulus'],behavior=row['causal']['baseline'][str(z)]['accuracy'],
                           old_readout=row['causal']['interventions']['old_finalLN_head_on_current_residual'][str(z)]['accuracy'],
                           task_flip_agreement=row['causal']['task_counterfactual']['prediction_agreement'])
                for prefix,key in [('new_activation_old_tail/','old_tail'),('old_attention_current_values/','dynamic_old_routing'),('old_activation_new_tail/','old_activation'),('shuffled_old_activation_new_tail/','shuffled_old_activation')]:
                    k,v=best_intervention(row,z,prefix);entry[key]=v;entry[key+'_site']=k
                if 'probes' in row:
                    entry.update(sum_query=[probe(row,z,l) for l in range(3)],sum_query_fixed=[probe(row,z,l,key='fixed_test') for l in range(3)],
                                 sum_query_aligned=[probe(row,z,l,key='aligned_fixed_test') for l in range(3)],sum_task=[probe(row,z,l,8) for l in range(3)])
                    # Entire position map retained; no selecting a winning test location for claims.
                    entry['sum_position_maps']=[[probe(row,z,l,pos) for pos in range(10)] for l in range(3)]
                summaries.append(entry)
        if item['kind']=='core' and job['source']:
            for name in ['A','B']:
                rr=[r for r in records if r['task']==name];fig,axes=plt.subplots(2,2,figsize=(11,7))
                for z,ax in enumerate(axes.flatten()):
                    ax.plot([r['step'] for r in rr],[r['causal']['baseline'][str(z)]['accuracy'] for r in rr],color='black',lw=2,label='Behavior')
                    for l,c in enumerate(['#0072B2','#009E73','#D55E00']):
                        ax.plot([r['step'] for r in rr],[probe(r,z,l) for r in rr],color=c,label=f'L{l+1} retrained')
                        ax.plot([r['step'] for r in rr],[probe(r,z,l,key='fixed_test') for r in rr],color=c,ls='--',alpha=.75,label=f'L{l+1} fixed')
                    ax.set(xscale='symlog',ylim=(-.02,1.02),xlabel='Steps (symlog)',ylabel='Held-out accuracy',title=f"{name}/lat{z}, p={job[name]['operations'][z]['modulus']}");ax.grid(alpha=.2)
                h,labels=axes[0,0].get_legend_handles_labels();fig.legend(h,labels,loc='lower center',ncol=4,fontsize=8);fig.suptitle(jid+' | query sum probes');fig.tight_layout(rect=(0,.1,1,.95))
                save(fig,jid.replace('/','__')+f'__{name}_trajectory.png',jid+f'：{name}行为与固定/重训query sum probe')
            final=next(r for r in records if r['step']==100000 and r['task']=='A')
            fig,axes=plt.subplots(1,4,figsize=(15,4))
            for z,ax in enumerate(axes):
                mat=np.array([[probe(final,z,l,pos) for pos in range(10)] for l in range(3)])
                im=ax.imshow(mat,vmin=0,vmax=1,aspect='auto',cmap='viridis');ax.set_xticks(range(10),[str(x) for x in range(8)]+['TASK','Q'],rotation=45);ax.set_yticks(range(3),['L1','L2','L3']);ax.set_title(f'A/lat{z}: p={job["A"]["operations"][z]["modulus"]}')
            fig.colorbar(im,ax=axes.tolist(),shrink=.75,label='Retrained held-out sum accuracy');fig.suptitle(jid+' | final post-MLP sum readability at all positions');save(fig,jid.replace('/','__')+'__A_position_map.png',jid+'：终点A全token位置probe图')
    # Core cross-seed causal evidence: original computation -> current tail versus current -> original tail.
    for seed in [0,1]:
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for col,k in enumerate(KINDS):
            for row,branch in enumerate(['full_A','replay_0.1']):
                jid=f'seed{seed}/main/{k}/{branch}';ss=[s for s in summaries if s['job']==jid and s['task']=='A' and s['step']==100000];ax=axes[row,col]
                x=np.arange(4)
                for offset,key,label,c in [(-.27,'behavior','Behavior','#222222'),(-.09,'old_readout','Old LN/readout','#0072B2'),(.09,'old_tail','Best old tail','#009E73'),(.27,'dynamic_old_routing','Best donor attention','#D55E00')]:
                    ax.bar(x+offset,[s[key] for s in ss],.18,label=label,color=c)
                ax.set(ylim=(0,1.04),xticks=x,xticklabels=['lat0','lat1','lat2','lat3'],title=k.replace('_',' ')+'\n'+branch)
        h,ll=axes[0,0].get_legend_handles_labels();fig.legend(h,ll,loc='lower center',ncol=4);fig.suptitle(f'Causal recovery of A | seed {seed} | best sites exploratory');fig.tight_layout(rect=(0,.06,1,.95));save(fig,f'causal_recovery_seed{seed}.png',f'种子{seed}：读出、旧tail与动态注意力恢复')
    # Probe alignment for replay, showing all operation/layer cases without test-based selection.
    for seed in [0,1]:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for ax,k in zip(axes.flatten(),KINDS):
            ss=[s for s in summaries if s['job']==f'seed{seed}/main/{k}/replay_0.1' and s['task']=='A' and s['step']==100000]
            matrix=[]
            for s in ss:
                matrix.extend([[s['sum_query_fixed'][l],s['sum_query_aligned'][l],s['sum_query'][l]] for l in range(3)])
            im=ax.imshow(matrix,vmin=0,vmax=1,aspect='auto');ax.set_xticks(range(3),['Fixed','Aligned fixed','Retrained'],rotation=20);ax.set_yticks(range(12),[f'lat{z}/L{l+1}' for z in range(4) for l in range(3)]);ax.set_title(k)
        fig.suptitle(f'Replay final A: coordinate alignment versus relearning | seed {seed}');fig.tight_layout(rect=(0,0,1,.95));save(fig,f'alignment_seed{seed}.png',f'种子{seed}：固定/对齐/重训probe对比')
    gradient=json.loads((base/'gradient/all.json').read_text())
    for seed in [0,1]:
        fig,axes=plt.subplots(3,4,figsize=(15,10))
        for col,k in enumerate(KINDS):
            for branch,c in [('full_A','#D55E00'),('replay_0.1','#0072B2')]:
                rr=[r for r in gradient if r['job']==f'seed{seed}/main/{k}/{branch}'];x=[r['step'] for r in rr]
                axes[0,col].plot(x,[r['statistics']['global']['A_B_cosine'] for r in rr],color=c,label=branch)
                axes[1,col].plot(x,[r['statistics']['global']['update_A_gradient_cosines']['B/adamw'] for r in rr],color=c,label=branch)
                axes[2,col].plot(x,[r['effects']['B/adamw']['A_val_loss_after']-r['effects']['B/adamw']['A_val_loss_before'] for r in rr],color=c,label=branch+' B')
                axes[2,col].plot(x,[r['effects']['mix_0.1/adamw']['A_val_loss_after']-r['effects']['mix_0.1/adamw']['A_val_loss_before'] for r in rr],color=c,ls='--',label=branch+' mix')
            for row in range(3):axes[row,col].set_xscale('symlog',linthresh=100);axes[row,col].axhline(0,color='gray',lw=.7);axes[row,col].grid(alpha=.2)
            axes[0,col].set(ylim=(-1.05,1.05),title=k,ylabel='Raw cos(gA,gB)');axes[1,col].set(ylim=(-1.05,1.05),ylabel='cos(B AdamW delta, gA)');axes[2,col].set(ylabel='Actual A loss change',xlabel='Steps (symlog)');axes[2,col].legend(fontsize=6)
        fig.suptitle(f'Raw gradients versus actual AdamW updates | seed {seed}');fig.tight_layout(rect=(0,0,1,.96));save(fig,f'gradient_seed{seed}.png',f'种子{seed}：原始梯度、实际更新与A损失变化')
    early=[]
    for seed in [0,1]:
        fig,axes=plt.subplots(2,4,figsize=(15,7));baseline=[r for r in byjob[f'seed{seed}/main/B_only'] if r['task']=='B']
        for col,k in enumerate(KINDS):
            rr=[r for r in byjob[f'seed{seed}/main/{k}/full_A'] if r['task']=='B']
            at500=next(r for r in rr if r['step']==500);bb500=next(r for r in baseline if r['step']==500)
            for z in [0,1]:
                ax=axes[z,col]
                for records,label,c,ls in [(baseline,'B-only behavior','#222222','--'),(rr,'A->B behavior','#009E73','-')]:
                    ax.plot([r['step'] for r in records],[r['causal']['baseline'][str(z)]['accuracy'] for r in records],color=c,ls=ls,label=label)
                for records,label,c in [(baseline,'B-only L2 sum','#D55E00'),(rr,'A->B L2 sum','#0072B2')]:
                    ax.plot([r['step'] for r in records],[probe(r,z,1) for r in records],color=c,label=label)
                ax.set(xlim=(0,5000),ylim=(-.02,1.02),xlabel='B steps',title=k+f' / p={23 if z==0 else 41}');ax.grid(alpha=.2)
                early.append(dict(seed=seed,condition=k,modulus=23 if z==0 else 41,B_only_behavior=bb500['causal']['baseline'][str(z)]['accuracy'],B_only_L2_sum=probe(bb500,z,1),transfer_behavior=at500['causal']['baseline'][str(z)]['accuracy'],transfer_L2_sum=probe(at500,z,1)))
        h,ll=axes[0,0].get_legend_handles_labels();fig.legend(h,ll,loc='lower center',ncol=4);fig.suptitle(f'Early internal computation versus behavior | seed {seed}');fig.tight_layout(rect=(0,.06,1,.95));save(fig,f'early_computation_seed{seed}.png',f'种子{seed}：B早期内部计算与B-only比较')
    controls=[]
    for path in (base/'optimizer_controls').glob('seed*/**/single_step.json'):
        controls.append(json.loads(path.read_text()))
    if controls:
        fig,axes=plt.subplots(2,4,figsize=(15,7))
        for ax,control in zip(axes.flatten(),sorted(controls,key=lambda x:x['source'])):
            rr=[r for r in control['rows'] if r['weight_decay']];x=np.arange(3)
            for offset,policy,c in [(-.17,'fresh','#D55E00'),(.17,'inherit_A','#0072B2')]:
                ax.bar(x+offset,[next(r['A_after']['macro_operation_accuracy'] for r in rr if r['optimizer_policy']==policy and r['objective']==o) for o in ['A','B','mix_0.1']],.34,label=policy,color=c)
            ax.set(xticks=x,xticklabels=['A step','B step','mix step'],ylim=(0,1.05),title=control['source'].replace('source/',''));ax.legend(fontsize=8)
        fig.supylabel('A validation accuracy after one update');fig.suptitle('Optimizer-reset control: one native minibatch update from identical theta_A');fig.tight_layout(rect=(0,0,1,.96));save(fig,'optimizer_reset_control.png','优化器重置对照：同一A权重的单步更新')
    final_noreplay=[s for s in summaries if s['kind']=='core' and '/full_A' in s['job'] and s['task']=='A' and s['step']==100000]
    final_replay=[s for s in summaries if s['kind']=='core' and '/replay_0.1' in s['job'] and s['task']=='A' and s['step']==100000]
    counts=dict(no_replay_operations=len(final_noreplay),no_replay_query_sum_readable_080=sum(max(s['sum_query'])>=.8 for s in final_noreplay),
        no_replay_any_position_sum_readable_080=sum(max(max(l) for l in s['sum_position_maps'])>=.8 for s in final_noreplay),
        no_replay_old_readout_restore_080=sum(s['old_readout']>=.8 for s in final_noreplay),no_replay_best_old_tail_restore_080=sum(s['old_tail']>=.8 for s in final_noreplay),
        replay_operations=len(final_replay),replay_fixed_drop_retrained_high=sum(any(f<.5 and r>=.9 for f,r in zip(s['sum_query_fixed'],s['sum_query'])) for s in final_replay),
        replay_alignment_rescues=sum(any(f<.5 and a>=.9 and r>=.9 for f,a,r in zip(s['sum_query_fixed'],s['sum_query_aligned'],s['sum_query'])) for s in final_replay))
    static=[];audit=[]
    for path in (base/'static_routing').glob('seed*/**/controls.json'):
        obj=json.loads(path.read_text());jid=str(path.parent.relative_to(base/'static_routing'))
        for row in obj['records']:
            for z in range(4):
                candidates=[(l,row['interventions'][f'correct_static_template/L{l}'][str(z)]['accuracy'],row['interventions'][f'wrong_digit_keys_template/L{l}'][str(z)]['accuracy']) for l in range(3) if obj['source_static_templates'][str(l)][str(z)]['accuracy']>=.8]
                best=max(candidates,key=lambda x:x[1]) if candidates else (None,None,None)
                static.append(dict(job=jid,step=row['step'],latent=z,baseline=row['baseline'][str(z)]['accuracy'],best_source_valid_layer=best[0],correct_static_accuracy=best[1],wrong_static_accuracy=best[2]))
    for path in (base/'probe_audit').glob('seed*/**/audit.json'):
        jid=str(path.parent.relative_to(base/'probe_audit'))
        for row in json.loads(path.read_text()):
            audit.extend([dict(job=jid,step=row['step'],task=row['task'],**sp) for sp in row['probes']])
    pure=[s for s in static if '/main/' in s['job'] and s['job'].endswith('/full_A') and s['step']==100000]
    counts['no_replay_static_routing_rescue_080']=sum(s['correct_static_accuracy'] is not None and s['correct_static_accuracy']>=.8 for s in pure)
    counts['no_replay_static_routing_specific_080']=sum(s['correct_static_accuracy'] is not None and s['correct_static_accuracy']>=.8 and s['wrong_static_accuracy']<=.3 for s in pure)
    counts['probe_audit_rows']=len(audit);counts['probe_budget_increase_gt010']=sum(s['test_800']-s['test_400']>.1 for s in audit)
    invocation=[]
    for path in (base/'invocation_controls').glob('seed*/**/controls.json'):
        jid=str(path.parent.relative_to(base/'invocation_controls'))
        for row in json.loads(path.read_text())['records']:
            invocation.extend([dict(job=jid,step=row['step'],baseline=row['baseline'][str(op['latent'])]['accuracy'],**op) for op in row['operations']])
    matching=[s for s in invocation if '/main/' in s['job'] and s['job'].endswith('/full_A') and s['step']==100000 and s['compatible_B_operation']]
    counts['shared_modulus_invocation_recovery']=sum(s['query_and_position']>=.95 for s in matching);counts['shared_modulus_invocation_cases']=len(matching)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for ax,seed in zip(axes,[0,1]):
        cases=[s for s in matching if s['job'].startswith(f'seed{seed}/')];x=np.arange(len(cases))
        for offset,key,label,c in [(-.25,'baseline','Original A invocation','#333333'),(0,'query_only','B task/query only','#D55E00'),(.25,'query_and_position','B task/query + operands','#0072B2')]:ax.bar(x+offset,[s[key] for s in cases],.25,label=label,color=c)
        ax.set(ylim=(0,1.04),xticks=x,xticklabels=[s['job'].split('/')[2]+f'\np={s["modulus"]}' for s in cases],title=f'seed {seed}');ax.tick_params(axis='x',labelsize=7)
    h,ll=axes[0].get_legend_handles_labels();fig.legend(h,ll,loc='lower center',ncol=3,fontsize=8);fig.supylabel('Accuracy against original A labels');fig.suptitle('Shared-modulus computation remains callable via a predefined external adapter');fig.tight_layout(rect=(0,.12,1,.92));save(fig,'invocation_recovery.png','共享模数：任务/query/操作数适配后的功能恢复')
    atomic=dict(counts=counts,records=summaries,early_computation=early,optimizer_controls=controls,static_routing=static,invocation_controls=invocation,probe_audit=audit,gradient_snapshots=len(gradient),checkpoint_count=sum(len(i['steps']) for i in done))
    (out/'summary.json').write_text(json.dumps(atomic,ensure_ascii=False),encoding='utf-8')
    lines=['# 五问题机制分析：两个种子', '',f'完成142个模型分支、{atomic["checkpoint_count"]}个checkpoint分析，以及432个已保存梯度快照。所有主模型冻结，优化只发生在线性探针上；优化器对照仅在副本上执行一个minibatch更新，不保存新训练模型。','',
      '## 方法与解释边界','',
      '每个运算512个train、val、test样本，目标无序residue pair按原模型数据划分互斥。每个checkpoint复用同一批输入。固定探针来自共同A来源；B-only/来源轨迹固定探针来自自身终点。探针400步AdamW，lr0.05、weight decay0.01、固定初始化4401。分析验证/测试输入与训练时固定评估集使用不同种子，故准确率不必逐点相等。',
      'sum探针覆盖每层attention后与MLP后的全部10个token位置；xi/xj探针位于query。正交Procrustes只用匹配train激活，不用标签，测试于held-out激活。对齐恢复支持坐标变化解释，但不证明原电路保留；重训probe失败也不能排除非线性编码。',
      '消融包括query、TASK、全token范围的逐头与MLP zero/训练集均值替换。均值替换减少幅度错配，仍不是完美分布内干预。恢复包括旧激活→新tail、新激活→旧tail、旧LN/读出；带同运算shuffle激活负对照和self-continuation检查。',
      '旧注意力概率+当前V测试保留当前数值通路，但旧注意力概率本身依赖测试输入，可能带入计算信息，不能直接称为纯路由恢复。跨checkpoint组合存在接口错配，失败不能证明知识删除。最佳layer/head为探索性选择，所有原始结果保留，不作为校正后的统计显著性结论。',
      '梯度使用已保存固定诊断训练batch，实际AdamW delta包含裁剪、历史moment和weight decay；效果由固定A验证损失的真实单步差值衡量。原始cosine不是遗忘/促进的充分证据，有限batch和A梯度接近零时尤其应谨慎。两个种子不足以建立总体规律。','',
      '## 首要发现：优化器重置是早期骤降的重要影响因素','',
      '8个A checkpoint原本都已学会任务。用原生8192-query A minibatch更新一步，新建AdamW后A宏平均准确率降到10.5%–66.3%；继承A的AdamW状态则为99.1%–100%。因此，即使不输入B数据，优化器重置也足以造成显著初期损伤。B更新本身仍有干扰，继承状态后幅度因种子/配置不同；这不是长期遗忘只由重置造成的证明。',
      '后续严格迁移/遗忘研究应加入A→A新建/继承优化器对照，并匹配A→B状态策略。当前批次统一新建AdamW，不能将早期骤降全写成B覆盖A。','',
      '## 问题1：内部计算是否提早形成','',
      '500步时，共享模数两个条件在两个种子上已具有较高B行为与第二层query sum可读性；B-only的对应可读性明显较低。下面列出全部条件，避免只展示成功配置。探针与行为并行出现不证明探针方向被实际使用，因果组件结果需结合原始干预表解释。','',
      '| seed | A条件 | B模数 | B-only行为 | B-only L2 sum | A→B行为 | A→B L2 sum |','|---|---|---:|---:|---:|---:|---:|',
      *[f'| {r["seed"]} | {r["condition"]} | {r["modulus"]} | {r["B_only_behavior"]:.3f} | {r["B_only_L2_sum"]:.3f} | {r["transfer_behavior"]:.3f} | {r["transfer_L2_sum"]:.3f} |' for r in early],
      '', '## 问题2：无replay终点的知识与调用','',
      f'- 32个A运算中，query任一层sum probe≥0.8：{counts["no_replay_query_sum_readable_080"]}/32。全部token位置任一层达到0.8：{counts["no_replay_any_position_sum_readable_080"]}/32。后者仅表示存在可读信息，不等于模型实际使用。',
      f'- 替换旧LN/读出使准确率≥0.8：{counts["no_replay_old_readout_restore_080"]}/32；当前激活接旧tail的最佳层组合达到0.8：{counts["no_replay_best_old_tail_restore_080"]}/32。','',
      f'- 训练输入静态路由模板使当前模型恢复到≥0.8：{counts["no_replay_static_routing_rescue_080"]}/32；同时错误key模板≤0.3的案例：{counts["no_replay_static_routing_specific_080"]}/32。只计入静态模板在旧模型上本身仍可达到0.8的层；不向测试模型注入旧模型对当前测试输入的数值注意力。','',
      f'- 但在共享模数的{counts["shared_modulus_invocation_cases"]}个重点运算中，按任务定义将query和必要的操作数位置适配到B，{counts["shared_modulus_invocation_recovery"]}个全部恢复到≥0.95。原A调用路径的sum不可读不等于模型中不存在该模加法计算。这里证明当前算法可以复用，不证明原A电路的参数/路径保留；B-only也能学到相同函数。','',
      '## 问题3：replay下的坐标变化与重组','',
      f'- 32个A运算中，存在固定query sum probe<0.5、重训probe≥0.9的层：{counts["replay_fixed_drop_retrained_high"]}/32。',
      f'- 其中至少一层经正交对齐恢复到≥0.9，且重训仍≥0.9：{counts["replay_alignment_rescues"]}/32。不能把这些固定probe下降直接当成算法丢失。',
      '- 例：seed0/same_mod_same_pos的A模37运算，第二层固定sum probe为0.367，正交对齐后0.973，重训0.992。同一配置A模53运算较早层的重训sum仍接近随机，第三层约0.932；这不是所有层都可由同一种基底旋转解释。',
      '- 终点无replay模型在A输入上切换TASK后的预测一致率为99.8%–100%；10%replay分支约1.8%–2.3%。这是任务标记在当前调用中是否改变输出的直接证据，不能仅凭此定位是哪一层执行条件化。',
      '- 对较早层重训仍低、后层高的情况，需要结合token位置图与zero/mean消融判定功能位置。该报告保存全部位置矩阵，避免强迫不同种子使用同一头。','',
      '## 问题1、4、5的可复查数据','',
      '- 促进：逐运算早期probe/行为轨迹、TASK与query路由、逐头及MLP作用已保存；所有80个组件分支检查起点/终点，不把被混合checkpoint在起点造成的A破坏当作B遗忘。',
      '- replay比例：m0/m0.5/m1所有已有分支的选定checkpoint均分析A/B；原先定义的连续保持标准仍由完整行为历史判断。这里只解释具体种子/运算的恢复差异，不宣称比例单调或统一阈值。',
      '- 梯度：48条轨迹共432个快照的全局/分层/组件范数、夹角、逐B运算共享模数标记、实际A损失变化见gradient/all.json。',
      '- 优化器单步对照：8个共同A来源，A/B/mix目标×新建/继承AdamW×有/无decay，共96次更新。使用原配置8192 query的精确首批输入，控制权重与数据不变。','',
      '## 探针可靠性复查','',
      f'来源、核心分支终点与A行为最低点的query/TASK sum探针，共{len(audit)}项400/800步与随机标签检查；800步测试准确率比400步提高超过0.1的项数为{counts["probe_budget_increase_gt010"]}。逐项数值保存在summary.json，不用探针未拟合好来推断知识消失。','',
      '## 图表索引','']
    for name,label in figures:lines.append(f'- [{label}](figures/{name})')
    lines.extend(['','## 完整数据位置','','各模型分析在同级seed0/seed1目录；报告summary.json含全部逐运算概要，原始probe权重与checkpoint级JSON留在服务器。'])
    (out/'机理分析报告.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    with zipfile.ZipFile(base/'mechanism_report.zip','w',zipfile.ZIP_DEFLATED) as z:
        for path in sorted(out.rglob('*')):
            if path.is_file():z.write(path,path.relative_to(out))
        z.write(base/'gradient/all.json','gradient_all.json')
    print(json.dumps(dict(counts=counts,figures=len(figures),checkpoints=atomic['checkpoint_count']),ensure_ascii=False),flush=True)


if __name__=='__main__':main()
