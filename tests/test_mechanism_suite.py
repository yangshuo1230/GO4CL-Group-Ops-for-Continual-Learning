"""Contract checks for task matching, stream continuation and diagnostic isolation."""
import copy
import json
import os
from pathlib import Path

import torch
import yaml

from go4cl.data.generate import build_shared_residue_splits
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.phase2.mechanism_suite import (
    KINDS, Stream, build_plan, eval_loaders, gradient_snapshot, run_job, schedule, selected_jobs, tasks,
)
from go4cl.train.loop import TrainConfig, build_optimizer


def config():
    return yaml.safe_load(Path('configs/phase2/mechanism_suite.yaml').read_text())


def tiny():
    cfg = config()
    cfg['model'] = dict(n_layers=1, d_model=8, n_heads=2, d_mlp=16)
    cfg['train'].update(max_steps=2, batch_size=8, compile_model=False)
    cfg.update(eval_per_operation=4, diagnostic_batch_size=16, mastery_accuracy=0., mastery_window=1,
               checkpoint_early_every=1, gradient_steps=[0, 2])
    return cfg


def test_tasks_and_deduplication():
    aa, b, overlap = tasks()
    assert [o.modulus for o in b.operations] == [23, 41, 31, 47]
    for kind, a in aa.items():
        for z in range(4):
            x, y = a.operations[z], b.operations[z]
            assert x.slot != y.slot
            assert (x.modulus == y.modulus) == (z < 2 and kind.startswith('same_mod'))
            assert (x.operand_pair == y.operand_pair) == (z < 2 and kind.endswith('same_pos'))
        assert a.operations[2:] == aa[KINDS[0]].operations[2:]
    plan = build_plan(config())
    assert len(plan) == 142
    assert len(selected_jobs(plan, ['core'])) == 26
    assert len(selected_jobs(plan, ['components'])) == 88
    assert len(selected_jobs(plan, ['overlap'])) == 44
    for name, task_b in overlap.items():
        a = aa[KINDS[1]]
        shared = sum(x.modulus == y.modulus for x, y in zip(a.operations, task_b.operations))
        assert shared / 4 == float(name[1:])
        assert not any(x.operand_pair == y.operand_pair or x.slot == y.slot for x, y in zip(a.operations, task_b.operations))


def test_stream_exact_continuation():
    aa, b, _ = tasks(); a = aa[KINDS[1]]
    splits = build_shared_residue_splits([a, b], data_seed=1)
    first = Stream(a, b, splits, 80, .1, 17)
    first.next(); state = copy.deepcopy(first.state_dict()); expected = first.next()
    second = Stream(a, b, splits, 80, .1, 17); second.load_state_dict(state)
    actual = second.next()
    assert all(torch.equal(actual[k], expected[k]) for k in expected)
    assert first.counts == second.counts


def test_gradients_do_not_change_training_state(tmp_path):
    torch.set_num_threads(1)
    cfg = tiny(); aa, b, _ = tasks(); a = aa[KINDS[1]]
    splits = build_shared_residue_splits([a, b], data_seed=cfg['data_seed'])
    model = ModularTransformer(ModelConfig(**cfg['model']))
    opt = build_optimizer(model, TrainConfig(**cfg['train'], device='cpu'))
    batch = Stream(a, b, splits, 16, 0., 7).next()
    model(batch['tokens'], batch['labels'])['loss'].backward(); opt.step(); opt.zero_grad(set_to_none=True)
    state = copy.deepcopy(model.state_dict()); moments = copy.deepcopy(opt.state_dict()); rng = torch.get_rng_state().clone()
    job = dict(replay_ratio=.1)
    gradient_snapshot(model, opt, a, b, splits, eval_loaders(a, b, splits, cfg), cfg, job, 1, tmp_path / 'g.pt', torch.device('cpu'))
    assert all(torch.equal(state[k], v) for k, v in model.state_dict().items())
    assert torch.equal(rng, torch.get_rng_state())
    for key, value in opt.state_dict()['state'].items():
        assert all(torch.equal(moments['state'][key][k], v) for k, v in value.items())
    snap = torch.load(tmp_path / 'g.pt', weights_only=False)
    assert 'B/adamw' in snap['deltas'] and 'mix_0.1/adam_no_decay' in snap['deltas']
    assert len(snap['gradients']) == 11


def test_cpu_source_and_all_branch_types(tmp_path):
    torch.set_num_threads(1)
    cfg = tiny(); plan = build_plan(cfg)
    # Two initialization seeds and full, replay, reset, keep, fresh branches.
    for seed in cfg['seeds']:
        prefix = f'seed{seed}/'
        source = next(j for j in plan if j['id'] == prefix + 'source/' + KINDS[1])
        run_job(source, cfg, tmp_path, 'cpu')
        for suffix in ['main/B_only', *[f'main/{KINDS[1]}/{i}' for i in
                       ['full_A', 'replay_0.1', 'reset_attention', 'keep_mlp']]]:
            job = next(j for j in plan if j['id'] == prefix + suffix)
            run_job(job, cfg, tmp_path, 'cpu')
            final = torch.load(tmp_path / job['id'] / 'checkpoints/final.pt', weights_only=False)
            assert final['step'] == 2 and len(final['history']) == 3
            run_job(job, cfg, tmp_path, 'cpu')  # completed jobs are immutable/idempotent


def test_exact_resume_and_failed_source_gate(tmp_path):
    torch.set_num_threads(1)
    cfg = tiny(); job = next(j for j in build_plan(cfg) if j['source_job'])
    run_job(job, cfg, tmp_path, 'cpu')
    directory = tmp_path / job['id']; cp = directory / 'checkpoints'
    before = torch.load(cp / 'final.pt', weights_only=False)
    (cp / 'final.pt').unlink(); (directory / 'result.json').unlink(); (cp / 'latest.pt').unlink()
    os.link(cp / 'step_000001.pt', cp / 'latest.pt')
    # Own test output: remove step2 so resumed execution writes it again.
    (cp / 'step_000002.pt').unlink()
    run_job(job, cfg, tmp_path, 'cpu')
    after = torch.load(cp / 'final.pt', weights_only=False)
    assert all(torch.equal(v, after['model_state'][k]) for k, v in before['model_state'].items())
    assert before['stream_state'] == after['stream_state']
    for name, state in before['optimizer_state']['state'].items():
        assert all(torch.equal(v, after['optimizer_state']['state'][name][k]) for k, v in state.items())
    failed_cfg = tiny(); failed_cfg['mastery_window'] = 10
    failed_root = tmp_path / 'failed'; plan = build_plan(failed_cfg)
    source = next(j for j in plan if j['source_job'])
    run_job(source, failed_cfg, failed_root, 'cpu')
    branch = next(j for j in plan if j['source'] == source['id'])
    run_job(branch, failed_cfg, failed_root, 'cpu')
    result = json.loads((failed_root / branch['id'] / 'result.json').read_text())
    assert result['status'] == 'blocked_source_not_mastered'
    assert not (failed_root / branch['id'] / 'checkpoints').exists()


def test_offline_probes_causals_and_gradient_summary(tmp_path):
    from go4cl.phases.phase2.mechanism_suite_analysis import causals, probe_data, probes_for_task, summarize, vector_stats
    torch.set_num_threads(1)
    cfg = tiny(); aa, b, _ = tasks(); a = aa[KINDS[1]]
    model = ModularTransformer(ModelConfig(**cfg['model'])); model.eval()
    splits = build_shared_residue_splits([a, b], data_seed=cfg['data_seed'])
    data = probe_data(a, splits, 4, 77)
    fixed, scores, _ = probes_for_task(model, a, data, 'cpu', 2, 18)
    _, second, _ = probes_for_task(model, a, data, 'cpu', 2, 18, fixed)
    assert len(scores) == 24 and all(abs(s['fixed_test'] - s['retrained_test']) < 1e-7 for s in second)
    causal = causals(model, model, a, data, 'cpu')
    assert causal['baseline'] == causal['interventions']['donor_resid_post/L0/all_tokens']
    source = next(j for j in build_plan(cfg) if j['source_job'])
    run_job(source, cfg, tmp_path, 'cpu'); summarize(tmp_path)
    assert (tmp_path / 'summary/behavior_summary.csv').exists()
    assert (tmp_path / source['id'] / 'behavior.png').exists()
    opt = build_optimizer(model, TrainConfig(**cfg['train'], device='cpu'))
    gradient_snapshot(model, opt, a, b, splits, eval_loaders(a, b, splits, cfg), cfg,
                      dict(replay_ratio=.1), 0, tmp_path / 'diagnostic.pt', torch.device('cpu'))
    snap = torch.load(tmp_path / 'diagnostic.pt', weights_only=False)
    stats = vector_stats(snap)
    assert 'layer_0' in stats and stats['global']['norms']['A'] > 0
