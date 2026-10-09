"""Read-only consolidation of Phase 2 experiments; write a separate review bundle."""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def read(p):
    return json.loads(p.read_text(encoding='utf-8'))


def write_csv(p, rows):
    if not rows:
        return
    keys = list(dict.fromkeys(k for row in rows for k in row))
    with p.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def mean(rows, key):
    xs = [float(r[key]) for r in rows if r.get(key) is not None]
    return float(np.mean(xs)) if xs else None


def save(fig, name):
    if not fig.get_constrained_layout():
        fig.tight_layout()
    fig.savefig(OUT / f'{name}.png', dpi=220, bbox_inches='tight')
    plt.close(fig)


def heat(ax, values, labels, columns, title, vmin=0, vmax=1):
    a = np.array(values, dtype=float)
    im = ax.imshow(a, aspect='auto', vmin=vmin, vmax=vmax, cmap='viridis')
    ax.set_xticks(range(len(columns)), columns, fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_title(title, fontsize=10)
    if len(labels) < 30:
        for i in range(len(labels)):
            for j in range(len(columns)):
                if np.isfinite(a[i, j]):
                    ax.text(j, i, f'{a[i,j]:.2f}', ha='center', va='center', fontsize=6,
                            color='white' if a[i,j] < .5 else 'black')
    return im


ap = argparse.ArgumentParser()
ap.add_argument('--out', required=True)
args = ap.parse_args()
OUT = Path(args.out)
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})

# Raw job results include interrupted batches without an aggregate report.
records, inventory = [], []
roots = sorted({p.parent.parent for p in Path('runs/phase2').rglob('runs/*/job_result.json')})
for runs_dir in roots:
    batch = str(runs_dir.parent)
    results = [read(p) for p in sorted(runs_dir.glob('*/job_result.json'))]
    ok = [r for r in results if r.get('status') == 'ok']
    config = {}
    for r in ok:
        cfg, m = r.get('wandb_config', {}), r.get('metrics', {})
        config = cfg
        before = m.get('A_test_acc_at_switch', (m.get('after_a') or {}).get('A_test_acc'))
        if before is None and m.get('forgetting_A_from_switch') is not None:
            before = m['A_test_acc'] + m['forgetting_A_from_switch']
        rec = {'batch': batch, 'job_id': r['job_id'], 'protocol': r.get('protocol',cfg.get('protocol')),
               'condition': cfg.get('condition'), 'task_seed': cfg.get('task_seed'),
               'model_seed': r.get('model_seed',cfg.get('model_seed')),
               'rho_slot': cfg.get('rho_slot'), 'rho_operand': cfg.get('rho_operand'),
               'rho_mod': cfg.get('rho_mod'), 'A': m.get('A_test_acc'), 'B': m.get('B_test_acc'),
               'A_switch': before, 'retention_from_switch': m.get('A_test_acc')/before if before else None,
               'optimizer': m.get('optimizer_transition', cfg.get('optimizer_transition','legacy/unspecified')),
               'null_task': cfg.get('null_task_tokens',False), 'data_dir':r.get('data_dir'),
               'auc': (m.get('behavior') or {}).get('b_exposure_auc')}
        records.append(rec)
    report = next(runs_dir.parent.glob('phase2_*_report.json'), None)
    batch_cfg = read(report).get('config',{}) if report else {}
    inventory.append({'batch':batch, 'completed':len(ok), 'errors':len(results)-len(ok),
                      'run_dirs':sum(p.is_dir() for p in runs_dir.iterdir()),
                      'protocols':','.join(sorted({r.get('protocol','') for r in ok})),
                      'fixed_a':batch_cfg.get('fixed_a','unknown'), 'share_a':batch_cfg.get('share_a','unknown'),
                      'optimizer':batch_cfg.get('optimizer_transition','legacy/unspecified'),
                      'wd':config.get('weight_decay'), 'train_frac':config.get('train_frac'),
                      'null_task':config.get('null_task_tokens',False), 'report_exists':bool(report)})
write_csv(OUT/'inventory.csv',inventory)
write_csv(OUT/'all_behavior.csv',records)

aggregates = []
for batch in sorted({r['batch'] for r in records}):
    rs=[r for r in records if r['batch']==batch]
    for protocol in sorted({r['protocol'] for r in rs}):
        sub=[r for r in rs if r['protocol']==protocol]
        aggregates.append({'batch':batch,'protocol':protocol,'n':len(sub),
                           'A_mean':mean(sub,'A'),'B_mean':mean(sub,'B'),
                           'A_switch_mean':mean(sub,'A_switch'),'A_retention_mean':mean(sub,'retention_from_switch'),
                           'B_auc_mean':mean(sub,'auc')})
write_csv(OUT/'batch_protocol_summary.csv',aggregates)
fig,ax=plt.subplots(figsize=(10, max(5,len(aggregates)*.24)))
labels=[f"{r['batch'].replace('runs/phase2/','')} | {r['protocol']} (n={r['n']})" for r in aggregates]
im=heat(ax,[[r['A_mean'],r['B_mean']] for r in aggregates],labels,['A final test','B final test'],
        'All Phase 2 batches: separate settings, no pooled comparison')
fig.colorbar(im,ax=ax,shrink=.5);save(fig,'01_all_batches_behavior')

# Every relation batch gets a separate overlap overview, including incomplete negative runs.
for batch in sorted({r['batch'] for r in records if 'relation_matrix/' in r['batch']}):
    rs=[r for r in records if r['batch']==batch]
    protocols=sorted({r['protocol'] for r in rs})
    conditions=sorted({r['condition'] for r in rs})
    fig,axs=plt.subplots(1,2,figsize=(max(9,len(protocols)*2.2),max(4,len(conditions)*.27)),layout='constrained')
    for ax,key in zip(axs,['A','B']):
        values=[[mean([r for r in rs if r['condition']==c and r['protocol']==p],key)
                 for p in protocols] for c in conditions]
        im=heat(ax,values,conditions,protocols,f'{key} final test | {Path(batch).name}')
    fig.colorbar(im,ax=axs.tolist(),shrink=.5)
    save(fig,'02_overlap_'+Path(batch).name)

# Fresh vs carried optimizer: plot independently trained A starts explicitly.
nextroot=Path('runs/phase2/next80_formal')
fig,axs=plt.subplots(1,3,figsize=(13,4))
for ax,key,title in zip(axs,['A_switch','auc','B'],['A accuracy at switch','B validation AUC','B final test']):
    for tag,color in [('01_core_fresh','tab:blue'),('02_core_preserve','tab:orange')]:
        rs=[r for r in records if tag in r['batch'] and r['protocol']=='sequential_ab']
        for i,rho in enumerate([0,.5,1]):
            vals=[r[key] for r in rs if r['rho_mod']==rho and r.get(key) is not None]
            x=i+(-.08 if 'fresh' in tag else .08)
            ax.scatter(np.full(len(vals),x),vals,color=color,s=16,alpha=.6)
            if vals:ax.plot(x,np.mean(vals),'_',color=color,ms=15,label=tag if i==0 else None)
    ax.set_xticks([0,1,2],['m=0','m=.5','m=1']);ax.set_title(title);ax.set_ylim(0,1.05)
axs[0].legend(fontsize=7);fig.suptitle('Fresh / preserve: independent A checkpoints across batches')
save(fig,'03_optimizer_comparison')

# Per-condition no replay / replay endpoints. Cross-batch starting A differs.
fig,axs=plt.subplots(1,2,figsize=(13,5))
seq=[r for r in records if 'next80_formal' in r['batch'] and r['protocol'] in ['sequential_ab','sequential_ab_replay'] and '02_core_preserve' not in r['batch']]
conditions=sorted({r['condition'] for r in seq})
for ax,key in zip(axs,['A','B']):
    for offset,proto,color in [(-.12,'sequential_ab','tab:red'),(.12,'sequential_ab_replay','tab:green')]:
        for i,c in enumerate(conditions):
            vals=[r[key] for r in seq if r['protocol']==proto and r['condition']==c]
            if vals:
                ax.scatter(np.full(len(vals),i+offset),vals,s=16,color=color,alpha=.6)
                ax.plot(i+offset,np.mean(vals),'_',ms=16,color=color,label=proto if i==0 else None)
    ax.set_xticks(range(len(conditions)),conditions,rotation=65,ha='right',fontsize=7)
    ax.set_ylim(0,1.05);ax.set_title(f'{key} final test');ax.legend(fontsize=7)
fig.suptitle('No replay / 10% replay: cross-batch comparison')
save(fig,'04_replay_endpoints')

# Reuse reviewed per-modulus analysis with raw sources untouched.
from go4cl.phases.phase2.modulus_analysis import run_modulus_analysis
run_modulus_analysis(nextroot/'01_core_fresh', OUT/'per_modulus')

# Component interventions retain per-seed scatter and censoring counts.
cp=nextroot/'05_component_reset/summary/component_transfer_summary.csv'
with cp.open() as f:comp=list(csv.DictReader(f))
write_csv(OUT/'component_interventions.csv',comp)
names=list(dict.fromkeys(r['intervention'] for r in comp))
fig,axs=plt.subplots(1,2,figsize=(12,5))
for i,name in enumerate(names):
    sub=[r for r in comp if r['intervention']==name]
    for ax,key,scale in [(axs[0],'b_exposure_auc',1),(axs[1],'stable_steps_to_gen',1000)]:
        vals=[float(r[key])/scale for r in sub if r[key]]
        ax.scatter(vals,np.full(len(vals),i),s=24)
        if vals:ax.plot(np.mean(vals),i,'|',ms=16,color='black')
    missing=sum(not r['stable_steps_to_gen'] for r in sub)
    if missing:axs[1].text(85,i,f'{missing}/3 censored',fontsize=7)
for ax in axs:ax.set_yticks(range(len(names)),names);ax.invert_yaxis()
axs[0].set_xlabel('B validation AUC');axs[1].set_xlabel('Stable t90 (k exposure steps)')
fig.suptitle('Parameter source interventions: 3 model seeds, task seed 0 only')
save(fig,'05_component_interventions')

# Mechanism reports: preserve distinct metrics and precise provenance.
mech=[]
for p in sorted(nextroot.glob('mech_suite/*/*report.json')):
    d=read(p);cfg=d['config']
    for op in d.get('per_op',[]):
        s=op['summary']
        row={'label':p.parent.name,'task':cfg['task'],'role':cfg['ckpt_kind'],
             'checkpoint':cfg['ckpt_path'],'modulus':op['modulus'],'latent_id':op['latent_id'],
             'behavior_acc':s['baseline_acc'],'n_test':s.get('n_test')}
        for layer,v in op.get('probes_by_layer',{}).items():
            row['probe_'+layer]=v['probe_sum_acc']
            row['random_probe_'+layer]=v.get('probe_sum_acc_random')
        for layer in op.get('attention',{}).get('layers',[]):
            row['operand_mass_L'+str(layer['layer'])]=layer['operand_mass']
        for site,v in op.get('composition',{}).get('harmonic',{}).get('by_site',{}).items():
            row['harmonic_sum_R2_'+site]=v.get('R2_sum_mean')
        for site,v in op.get('composition',{}).get('ladder',{}).get('by_site',{}).items():
            row['ladder_sum_probe_'+site]=v.get('probe_sum')
        energies=np.array(op.get('fourier_digit_emb',{}).get('energy_by_freq',[]))
        if len(energies)>1:
            e=energies[1:]/energies[1:].sum()
            row['embedding_nonDC_entropy']=float(-(e*np.log(e+1e-30)).sum()/np.log(len(e)))
        row['top1_ablation_drop']=-s.get('top1_imp_delta',0)
        mech.append(row)
write_csv(OUT/'mechanism_operations.csv',mech)
for prefix,cols,title in [('06_probe_behavior',['behavior_acc','probe_L0','probe_L1'],'Behavior and linear sum probes (different decoders)'),
                          ('07_routing',['operand_mass_L0','operand_mass_L1','operand_mass_L2'],'Query attention mass on own operands'),
                          ('08_harmonic',['harmonic_sum_R2_L0_mid','harmonic_sum_R2_L0_post','harmonic_sum_R2_L1_post','harmonic_sum_R2_L2_post'],'Harmonic sum R2: not classification accuracy')]:
    fig,ax=plt.subplots(figsize=(10,max(6,len(mech)*.19)))
    labels=[f"{r['label']} / p{r['modulus']}" for r in mech]
    im=heat(ax,[[r.get(c) for c in cols] for r in mech],labels,cols,title,
            vmin=-.2 if prefix=='08_harmonic' else 0)
    fig.colorbar(im,ax=ax,shrink=.4);save(fig,prefix)
fig,axs=plt.subplots(1,2,figsize=(10,4))
for task,color in [('A','tab:blue'),('B','tab:orange')]:
    rs=[r for r in mech if r['task']==task and r['role']=='phase_b_final']
    axs[0].scatter([r['behavior_acc'] for r in rs],[r.get('probe_L1',np.nan) for r in rs],color=color,label=task,alpha=.7)
    axs[1].scatter([r['behavior_acc'] for r in rs],[r.get('harmonic_sum_R2_L2_post',np.nan) for r in rs],color=color,label=task,alpha=.7)
for ax in axs:ax.set_xlabel('Behavior test accuracy');ax.set_xlim(0,1.05);ax.legend()
axs[0].plot([0,1],[0,1],'k--',lw=.7);axs[0].set_ylabel('L1 linear sum probe accuracy')
axs[1].set_ylabel('L2 post harmonic sum R2');save(fig,'09_readability_vs_behavior')

# Task-partition experiment is scientifically relevant although stored separately.
partition, gates, counter = [], [], []
for p in sorted(Path('runs/task_partition').glob('seeds_*/*/summary.json')):
    d=read(p);g=d.get('gate',{});seed=p.parent.name
    gates.append({'seed':seed,'passed':g.get('passed'),'A':g.get('A_test_acc'),'B':g.get('B_test_acc'),'flip_rate':g.get('task_token_flip_rate')})
    for branch,acc in d.get('final_test_accuracy',{}).items():
        partition.append({'seed':seed,'branch':branch,**acc})
    for m in d.get('mechanism_summary',[]):
        matrix=m.get('counterfactual_matrix')
        if matrix:
            counter.append({'seed':seed,'tag':m['tag'],'step':m.get('routing_step'),
                            **{f'{a}_matches_{b}':matrix[i][j] for i,a in enumerate('ABC') for j,b in enumerate('ABC')}})
write_csv(OUT/'partition_gates.csv',gates);write_csv(OUT/'partition_endpoints.csv',partition);write_csv(OUT/'partition_counterfactual.csv',counter)
if partition:
    fig,axs=plt.subplots(1,3,figsize=(11,4));branches=['ab_then_c','abc_joint','ab_continued']
    for ax,branch in zip(axs,branches):
        rs=[r for r in partition if r['branch']==branch]
        for i,task in enumerate('ABC'):
            vals=[r[task] for r in rs];ax.scatter(np.full(len(vals),i),vals);ax.plot(i,np.mean(vals),'_',ms=18,color='black')
        ax.set_xticks(range(3),list('ABC'));ax.set_ylim(0,1.05);ax.set_title(branch)
    fig.suptitle(f'AB joint -> C without replay: {sum(bool(g["passed"]) for g in gates)}/{len(gates)} seeds pass AB gate')
    save(fig,'10_task_partition')
    fig,axs=plt.subplots(1,len(set(r['seed'] for r in counter)),figsize=(12,3.5),squeeze=False)
    for ax,seed in zip(axs[0],sorted(set(r['seed'] for r in counter))):
        rs=[r for r in counter if r['seed']==seed and r['step'] in [0,1000,5000]
            and (r['tag']=='ab_joint' or r['tag'].startswith('c_step'))]
        for token in 'AB':
            ax.plot([r['step'] for r in rs],[r[f'{token}_matches_C'] for r in rs],'o-',label=f'TASK_{token}: matches C')
        ax.set_title(seed);ax.set_xlabel('C training steps');ax.set_ylim(0,1.05);ax.legend(fontsize=7)
    save(fig,'11_task_token_takeover')

# Layer-by-layer probe ladder makes delayed computation visible.
sites=['L0_mid','L0_post','L1_mid','L1_post','L2_mid','L2_post']
families=['F1_forget_m0','F2_partial_m1','Rm_noreplay_m1','Rp_replay_m1','Rh_replay075']
fig,axs=plt.subplots(len(families),4,figsize=(15,13),sharex=True,sharey=True)
for i,family in enumerate(families):
    baseline=[r for r in mech if r['label']==family+'_thetaA']
    for j,r in enumerate(baseline):
        ax=axs[i,j]
        for suffix,color in [('_thetaA','tab:blue'),('_finalA','tab:red'),('_finalB','tab:green')]:
            matches=[x for x in mech if x['label']==family+suffix and x['latent_id']==r['latent_id']]
            if not matches:continue
            item=matches[0]
            ax.plot(range(6),[item.get('ladder_sum_probe_'+s,np.nan) for s in sites],'-o',ms=3,color=color,
                    label=f"{suffix[1:]} p{item['modulus']} / behavior={item['behavior_acc']:.2f}")
        ax.set_title(f'{family} / op{r["latent_id"]}',fontsize=8)
        ax.set_ylim(0,1.05);ax.set_xticks(range(6),sites,rotation=60,fontsize=6)
        ax.legend(fontsize=5,loc='upper left')
save(fig,'12_probe_ladder_before_after')

# Matched forward transfer is kept within each batch.
transfer=[]
for batch in sorted({r['batch'] for r in records}):
    rs=[r for r in records if r['batch']==batch]
    idx={(r['condition'],r['task_seed'],r['model_seed']):r for r in rs if r['protocol']=='b_only'}
    for r in rs:
        b=idx.get((r['condition'],r['task_seed'],r['model_seed']))
        if b and r['protocol'] in ['sequential_ab','joint','interleaved'] and r['auc'] is not None and b['auc'] is not None:
            transfer.append({'batch':batch,'protocol':r['protocol'],'condition':r['condition'],
                             'task_seed':r['task_seed'],'model_seed':r['model_seed'],'rho_mod':r['rho_mod'],
                             'delta_auc':r['auc']-b['auc']})
write_csv(OUT/'matched_transfer.csv',transfer)
groups=list(dict.fromkeys((r['batch'],r['protocol']) for r in transfer))
fig,ax=plt.subplots(figsize=(11,max(4,len(groups)*.45)))
for i,(batch,proto) in enumerate(groups):
    vals=[r['delta_auc'] for r in transfer if r['batch']==batch and r['protocol']==proto]
    ax.scatter(vals,np.full(len(vals),i),alpha=.6,s=20)
    ax.plot(np.mean(vals),i,'|',color='black',ms=15)
ax.axvline(0,color='gray',ls='--');ax.set_xlabel('Validation AUC: protocol minus matched B-only')
ax.set_yticks(range(len(groups)),[f'{Path(b).name}/{p}' for b,p in groups],fontsize=8)
save(fig,'13_all_matched_transfer')

# Incomplete negative batch: only matched conditions, never pooled with full grid.
base=[r for r in records if r['batch'].endswith('20261004_202208')]
neg=[r for r in records if r['batch'].endswith('20261005_154626')]
negative=[]
for r in neg:
    matches=[x for x in base if (x['condition'],x['protocol'],x['task_seed'],x['model_seed'])==
             (r['condition'],r['protocol'],r['task_seed'],r['model_seed'])]
    if matches:
        b=matches[0];negative.append({'condition':r['condition'],'protocol':r['protocol'],
                                     'A_without':b['A'],'A_negative':r['A'],'B_without':b['B'],'B_negative':r['B']})
write_csv(OUT/'negative_matched.csv',negative)
fig,axs=plt.subplots(1,2,figsize=(9,4))
for ax,task in zip(axs,'AB'):
    for proto in sorted({r['protocol'] for r in negative}):
        rs=[r for r in negative if r['protocol']==proto]
        ax.scatter([r[task+'_without'] for r in rs],[r[task+'_negative'] for r in rs],label=proto)
    ax.plot([0,1],[0,1],'k--',lw=.7);ax.set_xlim(0,1.05);ax.set_ylim(0,1.05)
    ax.set_xlabel(f'{task} without negatives');ax.set_ylabel(f'{task} with negatives');ax.legend(fontsize=7)
fig.suptitle('Negative-task pilot: 22 completed runs, matched subset only')
save(fig,'14_negative_matched')

# Existing posthoc weight and Fourier results.
wp=Path('runs/phase2/relation_matrix/20261006_131444/weight_delta_switch/group_delta.csv')
if wp.exists():
    with wp.open() as f:weights=list(csv.DictReader(f))
    write_csv(OUT/'legacy_weight_delta.csv',weights)
    groups=sorted({r['group'] for r in weights})
    fig,axs=plt.subplots(1,2,figsize=(13,5))
    for proto,color in [('sequential_ab','tab:red'),('sequential_ab_replay','tab:green')]:
        vals=[np.mean([float(r['rel_l2']) for r in weights if r['protocol']==proto and r['group']==g]) for g in groups]
        axs[0].plot(range(len(groups)),vals,'o-',color=color,label=proto)
        rs=[r for r in weights if r['protocol']==proto and r['group']=='ALL']
        axs[1].scatter([float(r['rel_l2']) for r in rs],[float(r['A_test_acc']) for r in rs],color=color,label=proto)
    axs[0].set_xticks(range(len(groups)),groups,rotation=65,ha='right',fontsize=7)
    axs[0].set_ylabel('Mean relative parameter change');axs[1].set_xlabel('Total relative parameter change')
    axs[1].set_ylabel('A final test accuracy')
    for ax in axs:ax.legend(fontsize=7)
    save(fig,'15_legacy_weight_change')
fp=Path('runs/phase2/relation_matrix/20261004_202208/fourier_B_after_A_vs_after_B.csv')
if fp.exists():
    with fp.open() as f:spectra=list(csv.DictReader(f))
    write_csv(OUT/'legacy_fourier.csv',spectra)
    fig,axs=plt.subplots(1,2,figsize=(10,4))
    for shared,color in [('0','tab:orange'),('1','tab:blue')]:
        rs=[r for r in spectra if r['in_A']==shared]
        axs[0].scatter([float(r['frac_after_A']) for r in rs],[float(r['frac_after_B_final']) for r in rs],color=color,label=f'in A={shared}',alpha=.5)
        axs[1].scatter([float(r['spec_cos_A_final']) for r in rs],[float(r['A_test']) for r in rs],color=color,label=f'in A={shared}',alpha=.5)
    axs[0].set_xlabel('Embedding peak fraction after A');axs[0].set_ylabel('Embedding peak fraction after B')
    axs[1].set_xlabel('Embedding spectral cosine: after A / after B');axs[1].set_ylabel('A behavior after B')
    for ax in axs:ax.legend(fontsize=7)
    save(fig,'16_legacy_fourier_preservation')

fig,axs=plt.subplots(1,2,figsize=(11,4))
cases=[('F1_forget_m0_finalA','Forgotten A'),('Rp_replay_m1_finalA','Replay A')]
for ax,(label,title) in zip(axs,cases):
    rs=[r for r in mech if r['label']==label]
    for offset,key,legend in [(-.22,'behavior_acc','Behavior'),(0,'probe_L1','L1 sum probe'),(.22,'ladder_sum_probe_L2_post','L2 sum probe')]:
        ax.bar(np.arange(len(rs))+offset,[r.get(key,np.nan) for r in rs],width=.22,label=legend)
    ax.set_xticks(range(len(rs)),[f'p{r["modulus"]}' for r in rs]);ax.set_ylim(0,1.08)
    ax.set_title(title, pad=40)
    ax.legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,1.13),ncol=3)
save(fig,'17_key_probe_behavior_cases')

replay_fail=[]
for p in Path('runs/phase2').rglob('runs/*/job_result.json'):
    d=read(p)
    if d.get('status')!='ok' or d.get('protocol')!='sequential_ab_replay':continue
    m=d['metrics']
    if m['A_test_acc']>=.9:continue
    row={'batch':str(p.parent.parent.parent),'job_id':d['job_id'],'A':m['A_test_acc'],'B':m['B_test_acc']}
    for k,v in m.items():
        if k.startswith('A_test_acc/p'):row[k.split('/')[-1]]=v
    replay_fail.append(row)
write_csv(OUT/'replay_failure_moduli.csv',replay_fail)
if replay_fail:
    mods=sorted({k for r in replay_fail for k in r if k.startswith('p')},key=lambda k:int(k[1:]))
    fig,ax=plt.subplots(figsize=(12,max(4,len(replay_fail)*.35)))
    labels=[r['batch'].split('/')[-1]+' / '+r['job_id'].replace('sequential_ab_replay_','').replace('_d64_L3_wd0.3_steps100000','') for r in replay_fail]
    im=heat(ax,[[r.get(k) for k in mods] for r in replay_fail],labels,mods,'Replay runs with A < .9: per-modulus test accuracy')
    fig.colorbar(im,ax=ax,shrink=.7);save(fig,'18_replay_failure_moduli')

lines=['# Phase 2 complete results review','','Sources are read-only. Settings remain separate; counts are result records, not all independent repetitions.','',
       '## Batch inventory','','| batch | completed | errors | protocols | fixed A | shared A | optimizer | negatives |',
       '|---|---:|---:|---|---|---|---|---|']
for r in inventory:
    lines.append(f"| {r['batch']} | {r['completed']} | {r['errors']} | {r['protocols']} | {r['fixed_a']} | {r['share_a']} | {r['optimizer']} | {r['null_task']} |")
lines+=['','## Endpoints by batch and protocol','','| batch | protocol | n | A | B | A at switch | retention |','|---|---|---:|---:|---:|---:|---:|']
fmt=lambda v: '' if v is None else f'{v:.4f}'
for r in aggregates:
    lines.append(f"| {r['batch']} | {r['protocol']} | {r['n']} | {fmt(r['A_mean'])} | {fmt(r['B_mean'])} | {fmt(r['A_switch_mean'])} | {fmt(r['A_retention_mean'])} |")
lines+=['','## Interpretation boundaries','',
        '- Legacy overlap batches vary A across cells; cell-matched comparisons are stronger than cross-cell causal claims.',
        '- Optimizer fresh/preserve and replay/no-replay next80 stages retrain A separately. A start accuracy differs.',
        '- Retention uses explicit test endpoints at switch and final, never a phase-B-only history maximum.',
        '- Mechanism suite has selected representative checkpoints, not a multi-seed mechanism confirmation.',
        '- Mechanism evaluation sample sizes differ from 256-per-operation training endpoint evaluation; do not equate endpoints.',
        '- Probe accuracy, harmonic R2, spectral entropy and model accuracy are different metrics.',
        '- Per-operation curves within one model are correlated; they are not independent model seeds.',
        '- Negative batch is incomplete. Smoke/dry-run outputs are excluded from scientific results.',
        '- Task-partition: report gate failures as well as completed branches; results apply to passed seeds.',
        '- Protocol pilots use s1_o1_m1 (fully overlapping functions); success there is not a general forgetting control.',
        '', '## Figures','']
for p in sorted(OUT.glob('*.png')):lines += [f'### {p.stem}',f'![{p.stem}]({p.name})','']
lines+=['','## Per-modulus figures','']
for p in sorted((OUT/'per_modulus').glob('*.png')):lines += [f'![{p.stem}](per_modulus/{p.name})','']
(OUT/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'out':str(OUT),'batches':len(inventory),'behavior_records':len(records),'mechanism_operation_records':len(mech),'figures':len(list(OUT.rglob('*.png'))),'partition_gate_passed':sum(bool(g['passed']) for g in gates),'partition_total':len(gates)},indent=2))
import subprocess
import sys
subprocess.run([sys.executable, str(Path(__file__).with_name('plot_phase2_review.py')),
                '--root',str(OUT)],check=True)
