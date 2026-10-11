"""Task alignment, full-batch exposures, and actual A-checkpoint handoff on CPU."""
import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from go4cl.data.generate import build_shared_residue_splits
from go4cl.data.packed import replay_pack_counts
from go4cl.phases.phase2.mechanism_suite import KINDS, Stream, run_job, tasks
from go4cl.phases.phase2.one_layer_sequential import build_plan, config_for_job, task_pair


def config():
    return yaml.safe_load(Path('configs/phase2/one_layer_sequential.yaml').read_text())


def test_plan_and_alignment():
    cfg = config()
    jobs = build_plan(cfg)
    sources = [j for j in jobs if j['source_job']]
    branches = [j for j in jobs if not j['source_job']]
    assert len(jobs) == len({j['id'] for j in jobs}) == 72
    assert len(sources) == 8 and len(branches) == 64
    assert all(j['source'] is None for j in sources)
    for source in sources:
        children = [j for j in branches if j['source'] == source['id']]
        assert len(children) == 8
        assert all(j['A'] == source['A'] and j['model'] == source['model'] for j in children)
    for n in (1, 2):
        for kind in KINDS:
            a, b = task_pair(cfg, n, kind)
            assert a.n_ops == b.n_ops == n
            for x, y in zip(a.operations, b.operations):
                assert (x.modulus == y.modulus) == kind.startswith('same_mod')
                assert (x.operand_pair == y.operand_pair) == kind.endswith('same_pos')
                assert (x.slot == y.slot) == (n == 1)
            if kind.startswith('diff_mod'):
                assert set(a.moduli()).isdisjoint(b.moduli())
    assert {(j['model']['d_model'], j['model']['d_mlp'], j['model']['d_model']//j['model']['n_heads']) for j in jobs} == {(64,256,32), (128,512,64)}
    assert all(j['model']['n_layers']==1 and j['model']['n_heads']==2 for j in jobs)
    assert {j['replay_ratio'] for j in branches} == {0.,.1}


@pytest.mark.parametrize('n_ops',[1,2,4])
@pytest.mark.parametrize('ratio,source',[(0.,True),(0.,False),(.1,False)])
def test_stream_exposure_labels_and_resume(n_ops,ratio,source):
    cfg=config()
    if n_ops==4:
        aa,b,_=tasks();a=aa[KINDS[1]]
    else:
        a,b=task_pair(cfg,n_ops,KINDS[-1])
    splits=build_shared_residue_splits([a,b],data_seed=cfg['data_seed'])
    stream=Stream(a,b,splits,8192,ratio,1701,source)
    batch=stream.next()
    assert batch['tokens'].shape==(8192,10)
    for t in [a,b]:
        for op in t.operations:
            mask=(batch['task_ids']==t.task_id)&(batch['slots']==op.slot)
            selected=batch['tokens'][mask]
            assert torch.equal(batch['labels'][mask],(selected[:,op.i]+selected[:,op.j])%op.modulus)
            for row in selected[:12]:
                pair=tuple(sorted([int(row[op.i])%op.modulus,int(row[op.j])%op.modulus]))
                assert pair in splits[op.modulus].train
    npacks=8192//n_ops
    na,nb=(npacks,0) if source else (0,npacks) if ratio==0 else replay_pack_counts(npacks,ratio)
    assert stream.counts=={'A':na*n_ops,'B':nb*n_ops}
    assert int((batch['task_ids']==0).sum())==na*n_ops
    state=copy.deepcopy(stream.state_dict());expected=stream.next()
    again=Stream(a,b,splits,8192,ratio,1701,source);again.load_state_dict(state);actual=again.next()
    assert all(torch.equal(actual[k],expected[k]) for k in expected)
    assert again.counts==stream.counts


def test_four_op_stream_keeps_original_sampling():
    aa,b,_=tasks();a=aa[KINDS[1]]
    splits=build_shared_residue_splits([a,b],data_seed=20261009)
    stream=Stream(a,b,splits,80,.1,1701)
    batch=stream.next()
    parts=[stream.builders['A'].sample_train_batch(np.random.default_rng(1701),2),
           stream.builders['B'].sample_train_batch(np.random.default_rng(1701+10007),18)]
    assert all(np.array_equal(batch[k].numpy(),np.concatenate([p[k] for p in parts])) for k in batch)


@pytest.mark.parametrize('n_ops,width',[(1,64),(2,128)])
def test_cpu_checkpoint_branches_and_gradient_diagnostics(tmp_path,n_ops,width):
    torch.set_num_threads(1)
    cfg=config()
    cfg['train'].update(max_steps=2,batch_size=16,compile_model=False)
    cfg.update(eval_per_operation=4,diagnostic_batch_size=16,mastery_accuracy=0.,mastery_window=1,
               checkpoint_early_every=1,gradient_steps=[0,2])
    jobs=build_plan(cfg)
    source=next(j for j in jobs if j['source_job'] and j['operations_per_task']==n_ops and j['model']['d_model']==width and j['seed']==0)
    scfg=config_for_job(cfg,source)
    run_job(source,scfg,tmp_path,'cpu')
    donor=torch.load(tmp_path/source['id']/'checkpoints/final.pt',map_location='cpu',weights_only=False)
    assert donor['step']==2 and donor['stream_state']['counts']=={'A':32,'B':0}
    for ratio in [0.,.1]:
        branch=next(j for j in jobs if j['source']==source['id'] and j['relation']==KINDS[-1] and j['replay_ratio']==ratio)
        bcfg=config_for_job(cfg,branch);run_job(branch,bcfg,tmp_path,'cpu')
        directory=tmp_path/branch['id']
        initial=torch.load(directory/'checkpoints/step_000000.pt',map_location='cpu',weights_only=False)
        assert all(torch.equal(v,initial['model_state'][k]) for k,v in donor['model_state'].items())
        assert initial['optimizer_state']['state']=={}
        final=torch.load(directory/'checkpoints/final.pt',map_location='cpu',weights_only=False)
        assert final['step']==2 and sum(final['stream_state']['counts'].values())==32
        assert final['model_config']['d_model']==width and final['model_config']['d_mlp']==4*width
        assert len(final['model_state']['pos_emb.weight'])==10
        diagnostic=torch.load(directory/'gradients/step_000000.pt',map_location='cpu',weights_only=False)
        assert len(diagnostic['gradients'])==2+2*n_ops+1
        result=json.loads((directory/'result.json').read_text())
        assert result['status']=='complete' and result['requested_replay_ratio']==ratio
        counts=result['query_exposure'];assert counts['B']>0 and (counts['A']>0)==(ratio>0)
        before=(directory/'result.json').read_bytes();run_job(branch,bcfg,tmp_path,'cpu')
        assert before==(directory/'result.json').read_bytes()
