import json,itertools,zipfile
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

base=Path('runs/phase2/optimizer_switch_20261010/causal_paths_20261010').resolve()
out=base/'report';out.mkdir(exist_ok=True)
jobs=json.loads((base/'plan.json').read_text())['jobs']
carrier=[];routing=[]
for j in jobs:
    for step in [0,100,1000,10000,100000]:
        d=json.loads((base/j['id']/f'step_{step:06d}.json').read_text())
        assert d['model_unchanged']
        for rec in d['records']:
            for z in range(4):
                ns=[v for v in rec.get('nodes',[]) if v['latent']==z and v['eligible']>=64]
                high=[v for v in ns if min(v[k] for k in ['restore_i','restore_j','same_sum_restore','different_sum_transfer'])>=.9 and max(v['hybrid_i'],v['hybrid_j'])<=.1]
                best=max(ns,key=lambda v:v['sum_specificity']) if ns else None
                carrier.append(dict(job=j['id'],step=step,task=rec['task'],latent=z,strong_nodes=high,best_node=best,baselines=rec.get('test_baselines')))
            route=rec.get('task_routing')
            if route:
                routing.append(dict(job=j['id'],step=step,task=rec['task'],**route))
json.dump(dict(carriers=carrier,routing=routing),open(out/'causal_summary.json','w'),ensure_ascii=False)
bridge=json.loads((base/'final_bridge.json').read_text());json.dump(bridge,open(out/'final_bridge.json','w'))
adam=json.loads((base/'adam_state_factorial.json').read_text());ar=adam['records']
compact=[dict(group=r['group'],objective=r['objective'],keep_m=r['keep_m'],keep_v=r['keep_v'],keep_step=r['keep_step'],update_norm=r['update_norm'],A=r['metrics']['A']['macro_operation_accuracy'],B=r['metrics']['B']['macro_operation_accuracy']) for r in ar]
json.dump(compact,open(out/'adam_factor_summary.json','w'))
bits=list(itertools.product([False,True],repeat=3))
fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
for ax,obj in zip(axes,['A','B','mix_0.1']):
    values=np.array([[r['A'] for r in compact if r['objective']==obj and (r['keep_m'],r['keep_v'],r['keep_step'])==b] for b in bits])
    assert values.shape==(8,8),(obj,values.shape)
    im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis');ax.set_title('First update: '+obj);ax.set_xlabel('Matched source (8)');ax.set_yticks(range(8),[''.join(str(int(x)) for x in b) for b in bits]);ax.set_ylabel('Retain m / v / step')
fig.colorbar(im,ax=axes,label='A macro accuracy');fig.savefig(out/'adam_first_step.png',dpi=160);plt.close(fig)
for pol in ['fresh','inherit_A']:
    job=f'seed0/same_mod_same_pos/mix_0.1/{pol}'
    fig,axes=plt.subplots(2,2,figsize=(13,7),layout='constrained')
    for row,z in enumerate([0,1]):
        for col,step in enumerate([0,100000]):
            ax=axes[row,col];x=[r for r in bridge['records'] if r['job']==job and r['task']=='A' and r['latent']==z and r['step']==step and r['donor']=='same_sum']
            if not x:ax.text(.2,.5,'No sender meets discovery criterion');ax.set_axis_off();continue
            labels=[''.join(k[0] for k in ['Q','K','V','skip'] if r[k]) or 'none' for r in x]
            ax.bar(range(len(x)),[r['accuracy'] for r in x]);ax.set_xticks(range(len(x)),labels,rotation=60);ax.set_ylim(0,1.05);ax.axhline(.9,c='gray',ls='--');ax.set_title(f'A p={[23,41][row]}, step {step}, sender {x[0]["sender"]}');ax.set_ylabel('Recovery accuracy')
    fig.savefig(out/f'bridge_{pol}.png',dpi=160);plt.close(fig)
    r=next(v for v in routing if v['job']==job and v['task']=='A' and v['step']==100000)
    fig,axes=plt.subplots(1,2,figsize=(10,6),layout='constrained');modes=['Q','K','V','QK','full_head']
    for ax,z in zip(axes,[0,1]):
        val=np.array([[next(t['A_label_accuracy'] for t in r['tests'] if t['latent']==z and t['layer']==l and t['head']==h and t['mode']==mode) for mode in modes] for l in range(3) for h in range(4)])
        im=ax.imshow(val,vmin=0,vmax=1,cmap='viridis');ax.set_xticks(range(5),modes);ax.set_yticks(range(12),[f'L{l+1}H{h}' for l in range(3) for h in range(4)]);ax.set_title(f'A p={[23,41][z]} TASK rescue')
    fig.colorbar(im,ax=axes,label='A-label accuracy in TASK-flipped context');fig.savefig(out/f'task_routes_{pol}.png',dpi=160);plt.close(fig)
fig,axes=plt.subplots(2,2,figsize=(13,8),layout='constrained')
for ax,(seed,pol) in zip(axes.flat,itertools.product([0,1],['fresh','inherit_A'])):
    job=f'seed{seed}/same_mod_same_pos/mix_0.1/{pol}'
    for z in [0,1]:
        for loc in ['query','operand']:
            ys=[]
            for step in [0,100,1000,10000,100000]:
                v=next(r for r in carrier if r['job']==job and r['step']==step and r['task']=='A' and r['latent']==z)
                ns=[n for n in (v['strong_nodes'] or []) if (n['node'][2]==9)==(loc=='query')]
                ys.append(min([n['node'][1]+1 for n in ns],default=np.nan))
            ax.plot(range(5),ys,'o-',label=f'p={[23,41][z]} {loc}')
    ax.set_xticks(range(5),['0','100','1k','10k','100k']);ax.set_yticks([1,2,3]);ax.set_title(f'Seed {seed}, {pol}');ax.set_ylabel('Earliest confirmed layer');ax.legend(fontsize=8)
fig.savefig(out/'carrier_locations.png',dpi=160);plt.close(fig)
lines=['# 因果路径与优化器机制分析','',
'本轮完成冻结模型的输入配对节点干预、任务条件化 Q/K/V 干预、同一模型内的末层路径因子拆解，以及 AdamW 状态单步反事实。没有新增正式训练。',
'', '## 覆盖和验证',
'16 条 10% replay 轨迹（4 种关系 × 2 种子 × 2 优化器策略），每条检查 0/100/1000/10000/100000 共 80 个 checkpoint，A/B 每个运算分别分析。节点候选在 val 选择，test 检查，每个运算 128 个配对输入。单操作数改变、双操作数同模和、双操作数不同模和四种干预分开记录；后者排除两个单操作数拼接答案。配对输入按可构造约束条件抽样，不代替普通行为评估。',
'', '48 个 checkpoint 完成细化跨层路径检查；末层 Q/K/V/残差全部 16 组合的零恢复和全部恢复预测分别与原始运行、节点恢复运行一致。192 个 AdamW 单步反事实使用原始首批数据和相同编译梯度；fresh/inherit 两端在全部 24 对照组精确复现正式实验第一步准确率。',
'', '## 可以直接支持的机制结论',
'1. seed0/same_mod_same_pos 的 A p=41：切换前第一层 query 后残差满足四种模和传递测试；fresh 和 inherit replay 终点第一层操作数位置3满足同样测试（测试准确率约99.2%/100%）。这支持功能载体位置变化，不能仅用固定探针的基底旋转解释。',
'2. 末层路径进一步区分这些载体的调用：源模型同模和 donor 只恢复 query 残差 skip 即100%；fresh 终点最小成功组合为 V+skip，99.2%；inherit 终点仅 V 即100%。这里 V 为末层全部头的 value，不能误写成单个头或整条电路已完全识别。组合充分性是在固定受损上下文中成立，仍可能存在替代路径。',
'3. seed0 的 p=23 源模型与 fresh 终点均能通过第二层 query 载体、末层 skip 恢复100%；inherit 终点没有通过独立筛选门槛的早期节点，不能解释成算法消失。不同运算、种子没有统一的位置迁移规律。',
'4. TASK 因果门控有不同实现：fresh 终点 p=23 的 L1/H2 K 替换可恢复 A 答案100%，p=41 的 L1/H1 V 替换亦100%；inherit 终点 p=41 的 L1/H1 完整 query-head 替换100%。因果 mask 下数字位置表示在 TASK 翻转前后相同；这说明第一层 TASK key/value 也可影响任务调用，不支持“任务路由必然到第二层才开始”的普遍断言。',
'5. A→A 第一步保持历史二阶矩 v、重置 m 和 step 后，8 个来源 A 准确率均约99.8%–100%；完全重置则13.4%–85.0%。因此历史 v 已足以抑制这一步的功能破坏。保留 m 却重置 v 的部分反事实会产生极大更新，不能作为建议训练方案；长期效果不能由单步推出。',
'', '## 仍未闭合的问题',
'本轮直接解释了若干 replay 电路重组案例、TASK 调用方式和优化器首步破坏原因，尚未把五个研究问题全部解决。A 对 B 的算法/路由迁移仍需要 B-only 与匹配 A→B 的同型因果载体和必要性检验；无 replay 的旧 A 计算是否可经其他上下文调用不能由普通 A-input probe 阴性排除；沿轨迹暂时失效阶段缺乏正确配对样本时不能解释内部算法已经消失；最低 replay 与 overlap 的规律仍需原有比例训练结果的匹配统计；末层路径实验尚非完整加法/模约简数学算法逆向。',
'', '## 方法边界与修正',
'满足模和传递测试的节点可能同时包含操作数、原始和及其他信息，不能据此断言该处已完成数学意义的模约简。头编号只在同一模型内解释，跨种子按功能角色比较。节点筛选使用 val，但 Q/K/V 头扫描仍属探索性证据，应在新数据/更多种子确认。',
'初版 step JSON 中 paths 字段使用的缓存遍历不触发 block 替换，因此该字段无效，不纳入结论；节点和 TASK 测试使用正常 forward，有效。跨层路径以 refined_*.json version=2 为准，已检查替换缓存与原生 forward 一致。',
'', '## 文件',
'causal_summary.json：逐种子、逐运算、逐阶段候选及确认载体与任务门控。final_bridge.json：所有末层组合结果。adam_factor_summary.json：192 条状态拆解。完整配对输入和原始节点/细化路径结果留在服务器 causal_paths_20261010 中。图只生成 PNG。','']
(out/'因果机制分析报告.md').write_text('\n'.join(lines),encoding='utf-8')
with zipfile.ZipFile(base/'causal_report.zip','w',zipfile.ZIP_DEFLATED) as z:
    for p in out.iterdir():z.write(p,'report/'+p.name)
print('report',out,'carriers',len(carrier),'bridge',len(bridge['records']),'adam',len(compact),'archive_bytes',(base/'causal_report.zip').stat().st_size)
