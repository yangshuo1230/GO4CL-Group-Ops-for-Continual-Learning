"""Presentation-sized PNG charts from the consolidated Phase 2 CSVs."""
import argparse
import csv
from collections import defaultdict
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ap=argparse.ArgumentParser()
ap.add_argument('--root',required=True)
ROOT=Path(ap.parse_args().root)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':12,'axes.titlesize':15,
                    'axes.labelsize':12,'xtick.labelsize':11,'ytick.labelsize':11,
                    'axes.spines.top':False,'axes.spines.right':False,
                    'figure.facecolor':'white','axes.facecolor':'white',
                    'legend.frameon':False,'lines.linewidth':2.4})
COLORS=['#3D6FA3','#E3A13B','#58A28D','#BC6275']
NAMES={'20261004_113430':'Protocol pilot: wd=.5','20261004_140942':'Protocol pilot: wd=.8',
       '20261004_150305':'Protocol pilot: wd=.3','20261004_202208':'Original 27-cell grid',
       '20261005_154626':'Negative-task pilot','20261006_131444':'27-cell replay grid',
       '20261008_135656':'17-cell no replay','20261008_184031':'17-cell shared-A replay',
       '01_core_fresh':'Core: fresh optimizer','02_core_preserve':'Core: carried optimizer',
       '03_replay_overlap':'8-cell replay','04_no_replay_anchors':'4-cell no replay'}
PROT={'a_only':'A only','b_only':'B only','joint':'Joint A+B','sequential_ab':'A then B',
      'sequential_ab_replay':'A then B + replay','interleaved':'Interleaved',
      'a_only_continued':'A continued','sequential_ba':'B then A'}

def load(name):
    with (ROOT/name).open(encoding='utf-8',newline='') as f:return list(csv.DictReader(f))
def num(x):
    try:return float(x)
    except (TypeError,ValueError):return np.nan
def values(rs,k):return [num(r.get(k)) for r in rs if np.isfinite(num(r.get(k)))]
def avg(rs,k):
    v=values(rs,k);return np.mean(v) if v else np.nan
def finish(fig,name):
    for ax in fig.axes:
        if getattr(ax,'_review_bar',False):
            ax.set_title(ax.get_title(),pad=56)
    fig.savefig(ROOT/(name+'.png'),dpi=220,bbox_inches='tight',facecolor='white')
    plt.close(fig)
def axes(n,title,cols=2,height=4):
    rows=(n+cols-1)//cols
    fig,axs=plt.subplots(rows,cols,figsize=(6.2*cols,height*rows),layout='constrained',squeeze=False)
    fig.suptitle(title,fontsize=18,fontweight='normal')
    for ax in axs.flat[n:]:ax.set_visible(False)
    return fig,list(axs.flat[:n])
def bar(ax,labels,series,ylabel='Accuracy',ylim=(0,1.12),annotate=True):
    width=.76/len(series);x=np.arange(len(labels))
    for j,(name,groups) in enumerate(series):
        means=[np.mean(g) if len(g) else np.nan for g in groups]
        std=[np.std(g,ddof=1) if len(g)>1 else 0 for g in groups]
        xp=x+(j-(len(series)-1)/2)*width
        ax.bar(xp,means,width*.92,label=name,color=COLORS[j%len(COLORS)],
               yerr=std if any(std) else None,capsize=3,error_kw={'elinewidth':1,'alpha':.65})
        if annotate:
            for a,b in zip(xp,means):
                if np.isfinite(b):ax.annotate(f'{b:.2f}',(a,b),xytext=(0,5),textcoords='offset points',
                                             ha='center',fontsize=9)
    ax.set_xticks(x,labels);ax.set_ylabel(ylabel)
    if ylim:ax.set_ylim(*ylim)
    ax.grid(axis='y',color='#E5E9EF',lw=.8);ax.set_axisbelow(True)
    ax.legend(loc='lower center',bbox_to_anchor=(.5,1.01),ncol=min(3,len(series)),fontsize=10)
    ax._review_bar=True
def cond(c):return c.replace('s0.5','s.5').replace('o0.5','o.5').replace('m0.5','m.5').replace('_','\n')

behavior=load('all_behavior.csv')
summary=load('batch_protocol_summary.csv')
batches=list(dict.fromkeys(r['batch'] for r in summary))
# Split batches into separate panels; no tall 40-row heatmap.
fig,axs=axes(len(batches),'Phase 2: each experiment batch shown separately',cols=3,height=3.8)
for ax,b in zip(axs,batches):
    rs=[r for r in summary if r['batch']==b]
    labs=[PROT.get(r['protocol'],r['protocol'])+'\nn='+r['n'] for r in rs]
    bar(ax,labs,[('A',[[num(r['A_mean'])] for r in rs]),('B',[[num(r['B_mean'])] for r in rs])],annotate=False)
    ax.tick_params(axis='x',labelsize=8);plt.setp(ax.get_xticklabels(),rotation=25,ha='right')
    ax.set_title(NAMES.get(Path(b).name,Path(b).name),pad=24)
finish(fig,'01_all_batches_behavior')
for b in batches:
    rs=[r for r in behavior if r['batch']==b]
    ps=list(dict.fromkeys(r['protocol'] for r in rs))
    fig,ax=plt.subplots(figsize=(max(8,len(ps)*1.65),5),layout='constrained')
    bar(ax,[PROT[p]+'\nn='+str(sum(r['protocol']==p for r in rs)) for p in ps],
        [(t,[values([r for r in rs if r['protocol']==p],t) for p in ps]) for t in 'AB'])
    ax.set_title(NAMES[Path(b).name],pad=26)
    finish(fig,'01_batch_'+Path(b).name)
for b in batches:
    if 'relation_matrix/' not in b:continue
    rs=[r for r in behavior if r['batch']==b]
    cs=sorted({r['condition'] for r in rs});ps=list(dict.fromkeys(r['protocol'] for r in rs))
    # Each group has at most nine cells and its own readable plot.
    groups=[cs[i:i+9] for i in range(0,len(cs),9)]
    fig,axs=axes(len(groups)*2,NAMES.get(Path(b).name,Path(b).name),cols=2,height=4.4)
    for i,group in enumerate(groups):
        for j,task in enumerate('AB'):
            ax=axs[2*i+j]
            bar(ax,[cond(c) for c in group],[(PROT[p],[values([r for r in rs if r['condition']==c and r['protocol']==p],task) for c in group]) for p in ps],annotate=False)
            ax.set_title(f'Task {task} / cells {i*9+1}-{i*9+len(group)}',pad=24)
            ax.tick_params(axis='x',labelsize=10)
    finish(fig,'02_overlap_'+Path(b).name)

fig,axs=axes(3,'Optimizer state comparison (A pretrained independently)',cols=3)
for ax,key,title in zip(axs,['A_switch','auc','B'],['A at switch','B learning AUC','B final accuracy']):
    bar(ax,['m=0','m=.5','m=1'],[(name,[values([r for r in behavior if tag in r['batch'] and r['protocol']=='sequential_ab' and num(r['rho_mod'])==rho],key) for rho in [0,.5,1]]) for tag,name in [('01_core_fresh','Fresh'),('02_core_preserve','Carried')]],annotate=True)
    ax.set_title(title,pad=24)
finish(fig,'03_optimizer_comparison')
rs=[r for r in behavior if 'next80_formal' in r['batch'] and '02_core_preserve' not in r['batch'] and r['protocol'] in ['sequential_ab','sequential_ab_replay']]
cs=sorted({r['condition'] for r in rs})
fig,axs=axes(2,'Final behavior: no replay versus 10% replay',cols=1,height=4.6)
for ax,task in zip(axs,'AB'):
    bar(ax,[cond(c) for c in cs],[(name,[values([r for r in rs if r['condition']==c and r['protocol']==p],task) for c in cs]) for p,name in [('sequential_ab','No replay'),('sequential_ab_replay','10% replay')]],annotate=False)
    ax.set_title('Task '+task,pad=24)
finish(fig,'04_replay_endpoints')

comp=load('component_interventions.csv');names=list(dict.fromkeys(r['intervention'] for r in comp))
fig,axs=axes(2,'Which A parameters help B learn?',cols=1,height=5)
for ax,key,title,scale in [(axs[0],'b_exposure_auc','B validation AUC',1),(axs[1],'stable_steps_to_gen','Stable time to 90%',1000)]:
    groups=[values([r for r in comp if r['intervention']==n],key) for n in names]
    bar(ax,[n.replace('_','\n') for n in names],[('Mean across available seeds',[[v/scale for v in g] for g in groups])],ylabel='AUC' if scale==1 else 'B exposure steps (thousands)',ylim=(0,1.12) if scale==1 else (0,100),annotate=True)
    ax.set_title(title,pad=24);ax.tick_params(axis='x',labelsize=9)
    if scale==1000:
        for i,g in enumerate(groups):
            if len(g)<3:ax.text(i,92,f'{3-len(g)}/3\ncensored',ha='center',fontsize=9,color='#A84555')
finish(fig,'05_component_interventions')

mech=load('mechanism_operations.csv')
families=['F1_forget_m0','F2_partial_m1','Rm_noreplay_m1','Rp_replay_m1','Rh_replay075']
family_names=['No shared modulus','Partial functional overlap','Shared modulus / no replay','Shared modulus / replay','Replay failure case']
for filename,keys,labels,title in [('06_probe_behavior',['behavior_acc','probe_L1','ladder_sum_probe_L2_post'],['Behavior','L1 probe','L2 probe'],'A behavior and sum probes'),('07_routing',['operand_mass_L0','operand_mass_L1','operand_mass_L2'],['Layer 0','Layer 1','Layer 2'],'A query attention on its operands')]:
    fig,axs=axes(5,title,cols=2,height=4)
    for ax,f,name in zip(axs,families,family_names):
        rs=[r for r in mech if r['label']==f+'_finalA']
        bar(ax,['p'+r['modulus'] for r in rs],[(label,[[num(r.get(k))] for r in rs]) for k,label in zip(keys,labels)],annotate=False)
        ax.set_title(name,pad=24)
    finish(fig,filename)
sites=['L0_mid','L0_post','L1_mid','L1_post','L2_mid','L2_post']
for filename,prefix,title in [('08_harmonic','harmonic_sum_R2_','Harmonic sum decoding across layers'),('12_probe_ladder_before_after','ladder_sum_probe_','Sum probe across layers: before and after B')]:
    # One compact overview and separate 2x2 per-operation figures.
    fig,axs=axes(5,title+' (operation means)',cols=2,height=4)
    for ax,f,name in zip(axs,families,family_names):
        for suffix,label,color in [('_thetaA','Before B: A',COLORS[0]),('_finalA','After B: A',COLORS[3]),('_finalB','After B: B',COLORS[2])]:
            rs=[r for r in mech if r['label']==f+suffix]
            ax.plot(range(6),[avg(rs,prefix+s) for s in sites],'-o',color=color,label=label,ms=5)
        ax.set_title(name);ax.set_xticks(range(6),['L0 attn','L0 MLP','L1 attn','L1 MLP','L2 attn','L2 MLP'],rotation=25,ha='right',fontsize=10)
        ax.set_ylim(-.15,1.08);ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
        ax.set_ylabel('Harmonic R2' if prefix.startswith('harmonic') else 'Probe accuracy')
    finish(fig,filename)
    if prefix=='ladder_sum_probe_':
        for f,name in zip(families,family_names):
            fig,axs=axes(4,name+': individual operations',cols=2,height=3.9)
            base=[r for r in mech if r['label']==f+'_thetaA']
            for ax,b in zip(axs,base):
                for suffix,label,color in [('_thetaA','Before B: A',COLORS[0]),('_finalA','After B: A',COLORS[3]),('_finalB','After B: B',COLORS[2])]:
                    r=next(r for r in mech if r['label']==f+suffix and r['latent_id']==b['latent_id'])
                    ax.plot(range(6),[num(r.get(prefix+s)) for s in sites],'-o',color=color,ms=5,label=f'{label}, p{r["modulus"]}, acc={num(r["behavior_acc"]):.2f}')
                ax.set_title('Operation '+b['latent_id']);ax.set_ylim(0,1.06)
                ax.set_xticks(range(6),['L0 attn','L0 MLP','L1 attn','L1 MLP','L2 attn','L2 MLP'],rotation=25,ha='right',fontsize=10)
                ax.set_ylabel('Sum probe accuracy');ax.grid(axis='y',alpha=.2);ax.legend(fontsize=9)
            finish(fig,'12_ladder_'+f)

fig,axs=axes(2,'Readable answers and model behavior',cols=2)
for ax,label,title in zip(axs,['F1_forget_m0_finalA','Rp_replay_m1_finalA'],['Forgotten A','Replay A']):
    rs=[r for r in mech if r['label']==label]
    bar(ax,['p'+r['modulus'] for r in rs],[(legend,[[num(r.get(k))] for r in rs]) for k,legend in [('behavior_acc','Behavior'),('probe_L1','L1 probe'),('ladder_sum_probe_L2_post','L2 probe')]],annotate=True)
    ax.set_title(title,pad=24)
finish(fig,'17_key_probe_behavior_cases')
fig,axs=axes(2,'Probe readability and behavior are distinct measurements',cols=2)
for ax,key,title in zip(axs,['probe_L1','ladder_sum_probe_L2_post'],['Layer 1','Layer 2 post-MLP']):
    for task,color in [('A',COLORS[0]),('B',COLORS[1])]:
        rs=[r for r in mech if r['role']=='phase_b_final' and r['task']==task]
        ax.scatter([num(r['behavior_acc']) for r in rs],[num(r.get(key)) for r in rs],s=65,alpha=.75,color=color,label='Task '+task)
    point=next(r for r in mech if r['label']=='F1_forget_m0_finalA' and r['modulus']=='53')
    ax.annotate('Forgotten A / p53',(num(point['behavior_acc']),num(point[key])),xytext=(.42,.65),
                arrowprops={'arrowstyle':'->','color':'#555'},fontsize=11)
    ax.plot([0,1],[0,1],'--',color='#AAA',lw=1)
    ax.set_xlim(-.03,1.05);ax.set_ylim(-.03,1.05);ax.set_xlabel('Behavior test accuracy')
    ax.set_ylabel('Sum probe test accuracy');ax.set_title(title);ax.legend();ax.grid(alpha=.15)
finish(fig,'09_readability_vs_behavior')

transfer=load('matched_transfer.csv');groups=list(dict.fromkeys((r['batch'],r['protocol']) for r in transfer))
fig,ax=plt.subplots(figsize=(11,5.5),layout='constrained')
gs=[values([r for r in transfer if (r['batch'],r['protocol'])==g],'delta_auc') for g in groups]
means=[np.mean(g) for g in gs];std=[np.std(g,ddof=1) if len(g)>1 else 0 for g in gs]
ax.barh(range(len(groups)),means,xerr=std,color=COLORS[0],capsize=3)
ax.set_yticks(range(len(groups)),[NAMES[Path(b).name]+' / '+PROT[p] for b,p in groups],fontsize=10)
ax.axvline(0,color='#777',lw=1);ax.set_xlabel('AUC gain over matched B-only (mean and SD)')
ax.set_title('Forward transfer within each batch');ax.grid(axis='x',alpha=.2);finish(fig,'13_all_matched_transfer')
negative=load('negative_matched.csv');ps=sorted({r['protocol'] for r in negative})
fig,axs=axes(2,'Negative-task pilot: matched completed runs only',cols=2)
for ax,task in zip(axs,'AB'):
    bar(ax,[PROT[p] for p in ps],[(legend,[values([r for r in negative if r['protocol']==p],task+'_'+k) for p in ps]) for k,legend in [('without','Without negatives'),('negative','With negatives')]],annotate=True)
    ax.set_title('Task '+task,pad=24);ax.tick_params(axis='x',labelsize=10)
finish(fig,'14_negative_matched')

partition=load('partition_endpoints.csv')
fig,axs=axes(3,'Task partition: three seeds pass AB gate; one seed stops',cols=3)
for ax,b,name in zip(axs,['ab_then_c','abc_joint','ab_continued'],['AB Joint then C','ABC Joint','AB Joint continued']):
    rs=[r for r in partition if r['branch']==b]
    bar(ax,list('ABC'),[('Final test accuracy',[values(rs,t) for t in 'ABC'])]);ax.set_title(name,pad=24)
finish(fig,'10_task_partition')
counter=load('partition_counterfactual.csv');seeds=sorted({r['seed'] for r in counter})
fig,axs=axes(len(seeds),'TASK_A / TASK_B predictions switch to C during C training',cols=3)
for ax,seed in zip(axs,seeds):
    rs=[r for r in counter if r['seed']==seed and num(r['step']) in [0,1000,5000] and (r['tag']=='ab_joint' or r['tag'].startswith('c_step'))]
    for task,color in zip('AB',COLORS):
        ax.plot([num(r['step'])/1000 for r in rs],[num(r[task+'_matches_C']) for r in rs],'-o',color=color,label='TASK_'+task,ms=6)
    ax.set_title('Model '+seed);ax.set_xlabel('C steps (thousands)');ax.set_ylabel('Fraction matching C answer');ax.set_ylim(0,1.08);ax.legend();ax.grid(alpha=.2)
finish(fig,'11_task_token_takeover')

failures=load('replay_failure_moduli.csv')
for start in range(0,len(failures),6):
    subset=failures[start:start+6];fig,axs=axes(len(subset),'Replay failures: which A operation is lost?',cols=2,height=3.8)
    for ax,r in zip(axs,subset):
        mods=sorted([k for k in r if k.startswith('p') and np.isfinite(num(r[k]))],key=lambda k:int(k[1:]))
        bar(ax,mods,[('A operation accuracy',[[num(r[k])] for k in mods])],annotate=True)
        job=r['job_id'].replace('sequential_ab_replay_','').split('_d64')[0]
        ax.set_title(NAMES[Path(r['batch']).name]+'\n'+job,fontsize=11,pad=27)
    finish(fig,'18_replay_failure_moduli' if start==0 else '18_replay_failures_page'+str(start//6+1))

# Early-learning zooms: no single panel overlays all moduli and all protocols.
curves=load('per_modulus/curves_modulus.csv')
for rho in [0,.5,1]:
    subset=[r for r in curves if num(r['rho_mod'])==rho];mods=sorted({int(r['modulus']) for r in subset})
    fig,axs=axes(len(mods),f'B learning by modulus / overlap m={rho:g}',cols=2,height=3.7)
    for ax,p in zip(axs,mods):
        for proto,label,color in [('b_only','B only',COLORS[0]),('sequential_ab','A then B',COLORS[1])]:
            points=defaultdict(list)
            for r in subset:
                if int(r['modulus'])==p and r['protocol']==proto:points[num(r['b_exposure_step'])/1000].append(num(r['acc']))
            xs=sorted(points);ys=np.array([np.mean(points[x]) for x in xs]);sd=np.array([np.std(points[x],ddof=1) if len(points[x])>1 else 0 for x in xs])
            ax.plot(xs,ys,color=color,label=label);ax.fill_between(xs,np.maximum(ys-sd,0),np.minimum(ys+sd,1),color=color,alpha=.12)
        ax.set_xlim(0,30);ax.set_ylim(0,1.05);ax.set_title('p = '+str(p));ax.set_xlabel('B exposure steps (thousands)');ax.set_ylabel('Validation accuracy');ax.legend();ax.grid(alpha=.2)
    finish(fig,'per_modulus/curves_rho'+f'{rho:g}')
matched=load('per_modulus/matched_modulus.csv')
groups=['rho_mod=0','rho_mod=0.5/nonshared','rho_mod=0.5/shared','rho_mod=1']
fig,axs=axes(2,'Shared and nonshared modulus transfer',cols=2)
for ax,key,title,scale in [(axs[0],'delta_auc','AUC gain',1),(axs[1],'delta_stable_t90','Stable t90 speedup',1000)]:
    gs=[values([r for r in matched if r['overlap_group']==g],key) for g in groups]
    bar(ax,['m=0\nnonshared','m=.5\nnonshared','m=.5\nshared','m=1\nshared'],[('Operation-run mean and SD',[[v/scale for v in g] for g in gs])],ylabel='Delta AUC' if scale==1 else 'Steps saved (thousands)',ylim=None,annotate=False)
    ax.axhline(0,color='#777',lw=1);ax.set_title(title,pad=24)
    for i,g in enumerate(gs):ax.text(i,ax.get_ylim()[0],f'n={len(g)}',ha='center',va='bottom',fontsize=9)
finish(fig,'per_modulus/delta_shared_vs_nonshared')

fig,axs=axes(2,'Shared-modulus condition: p23 versus the other moduli',cols=2)
for ax,key,title,scale in [(axs[0],'delta_auc','AUC gain',1),(axs[1],'delta_stable_t90','Stable t90 speedup',1000)]:
    rs=[r for r in matched if num(r['rho_mod'])==1]
    gs=[values([r for r in rs if (int(r['modulus'])==23)==is23],key) for is23 in [True,False]]
    bar(ax,['p23','Other moduli'],[('Operation-run mean and SD',[[v/scale for v in g] for g in gs])],
        ylabel='Delta AUC' if scale==1 else 'Steps saved (thousands)',ylim=None,annotate=False)
    ax.set_title(title,pad=24);ax.axhline(0,color='#777',lw=1)
finish(fig,'per_modulus/p23_vs_other')
fig,ax=plt.subplots(figsize=(9,4.8),layout='constrained')
for proto,label,color in [('b_only','B only',COLORS[0]),('sequential_ab','A then B',COLORS[1])]:
    rs=[r for r in curves if r['task_seed']=='1' and r['model_seed']=='2' and r['modulus']=='29'
        and num(r['rho_mod'])==0 and r['protocol']==proto]
    ax.plot([num(r['b_exposure_step'])/1000 for r in rs],[num(r['acc']) for r in rs],color=color,label=label)
ax.set_title('Negative transfer case: p29, task seed 1, model seed 2')
ax.set_xlabel('B exposure steps (thousands)');ax.set_ylabel('Validation accuracy')
ax.set_ylim(0,1.05);ax.legend();ax.grid(alpha=.2);finish(fig,'per_modulus/anomaly_ts1_ms2_p29')
ops=load('per_modulus/curves_operation.csv')
for rho in [0,.5,1]:
    fig,axs=axes(8,f'Operation-specific A forgetting and B learning / m={rho:g}',cols=2,height=3.7)
    for ax,(seed,latent) in zip(axs,[(s,l) for s in [0,1] for l in range(4)]):
        for task,color in [('A',COLORS[3]),('B',COLORS[0])]:
            rs=[r for r in ops if num(r['rho_mod'])==rho and int(r['task_seed'])==seed and int(r['latent_id'])==latent and r['task']==task]
            points=defaultdict(list)
            for r in rs:points[num(r['b_exposure_step'])/1000].append(num(r['acc']))
            xs=sorted(points);ys=[np.mean(points[x]) for x in xs]
            mod=rs[0]['modulus'] if rs else '?'
            ax.plot(xs,ys,color=color,label=f'{task}: p{mod}')
        ax.set_xlim(0,30);ax.set_ylim(0,1.05);ax.set_title(f'Task seed {seed} / operation {latent}')
        ax.set_xlabel('B exposure steps (thousands)');ax.set_ylabel('Validation accuracy');ax.legend();ax.grid(alpha=.2)
    finish(fig,'per_modulus/joint_ops_rho'+f'{rho:g}')

weights=load('legacy_weight_delta.csv');groups=sorted({r['group'] for r in weights})
fig,axs=axes(2,'Parameter changes during B training (legacy replay grid)',cols=2,height=6)
for j,(proto,label,color) in enumerate([('sequential_ab','No replay',COLORS[0]),('sequential_ab_replay','10% replay',COLORS[2])]):
    means=[avg([r for r in weights if r['protocol']==proto and r['group']==g],'rel_l2') for g in groups]
    axs[0].barh(np.arange(len(groups))+(j-.5)*.35,means,height=.32,label=label,color=color)
    rs=[r for r in weights if r['protocol']==proto and r['group']=='ALL']
    axs[1].scatter([num(r['rel_l2']) for r in rs],[num(r['A_test_acc']) for r in rs],label=label,color=color,s=60,alpha=.7)
axs[0].set_yticks(range(len(groups)),[g.replace('_',' ') for g in groups]);axs[0].set_xlabel('Relative parameter change');axs[0].legend()
axs[1].set_xlabel('Total relative parameter change');axs[1].set_ylabel('A final test accuracy');axs[1].legend();axs[1].grid(alpha=.2)
finish(fig,'15_legacy_weight_change')
spectra=load('legacy_fourier.csv');fig,axs=axes(2,'Embedding Fourier measurements (original grid)',cols=2)
for ax in axs:ax.grid(alpha=.2)
for shared,color in [('0',COLORS[1]),('1',COLORS[0])]:
    rs=[r for r in spectra if r['in_A']==shared]
    axs[0].scatter([num(r['frac_after_A']) for r in rs],[num(r['frac_after_B_final']) for r in rs],color=color,s=50,alpha=.6,label='Shared modulus' if shared=='1' else 'Nonshared modulus')
    axs[1].scatter([num(r['spec_cos_A_final']) for r in rs],[num(r['A_test']) for r in rs],color=color,s=50,alpha=.6,label='Shared modulus' if shared=='1' else 'Nonshared modulus')
axs[0].set_xlabel('Peak fraction after A');axs[0].set_ylabel('Peak fraction after B')
axs[1].set_xlabel('Spectral cosine: after A versus after B');axs[1].set_ylabel('A behavior after B')
for ax in axs:ax.legend(fontsize=10)
finish(fig,'16_legacy_fourier_preservation')

# Index includes all PNGs, including retained diagnostic scatter plots.
images=sorted(ROOT.rglob('*.png'))
lines=['# Phase 2 PNG figure gallery','','Large-font bar and line charts. Error bars/bands denote SD, not confidence intervals. Missing stable t90 is censored.','',
       'Start with 04 (replay), 10 (task partition), 17 (probe / behavior), and individual 12_ladder plots.','']
for p in images:
    rel=p.relative_to(ROOT).as_posix();lines += [f'## {p.stem}',f'![{p.stem}]({rel})','']
(ROOT/'图表索引.md').write_text('\n'.join(lines),encoding='utf-8')
print(f'Updated PNG charts; gallery contains {len(images)} images.')
