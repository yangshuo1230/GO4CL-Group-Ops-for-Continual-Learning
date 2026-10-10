"""CPU-only integration checks using existing donors, isolated temporary outputs."""
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import tempfile

import torch

spec = importlib.util.spec_from_file_location('optimizer_switch', 'scripts/phase2/optimizer_switch.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
torch.set_num_threads(1)
args = SimpleNamespace(source_root='runs/phase2/mechanism_suite_20261009',
    seeds=[0], kinds=['same_mod_diff_pos'], objectives=list(m.OBJECTIVES),
    policies=list(m.POLICIES), max_steps=2, a_stream='continue')
jobs = m.build_plan(args)
for j in jobs:
    c = j['config']
    c['train'].update(batch_size=64, compile_model=False)
    c.update(eval_per_operation=8, diagnostic_batch_size=32, gradient_steps=[0, 1, 2])

with tempfile.TemporaryDirectory(prefix='optimizer_switch_cpu_') as temp:
    root = Path(temp)
    for objective in m.OBJECTIVES:
        pair = [j for j in jobs if j['objective'] == objective]
        state_before, hashes = [], []
        for job in pair:
            model, opt, stream, donor, *_ = m.setup(job, torch.device('cpu'))
            assert all(torch.equal(v.cpu(), donor['model_state'][n]) for n, v in model.state_dict().items())
            if job['optimizer_policy'] == 'fresh':
                assert not opt.state
            else:
                assert opt.state
                assert all(float(v['step']) == donor['step'] for v in opt.state.values())
                for actual, original in zip(opt.state.values(), donor['optimizer_state']['state'].values()):
                    assert torch.equal(actual['exp_avg'], original['exp_avg'])
                    assert torch.equal(actual['exp_avg_sq'], original['exp_avg_sq'])
            state_before.append(copy.deepcopy(stream.state_dict()))
            hashes.append(m.batch_hash(stream.next()))
            if objective == 'A':
                assert state_before[-1]['rng'] == donor['stream_state']['rng']
        assert hashes[0] == hashes[1], objective
        for job in pair:
            m.run_job(job, root, 'cpu')
        a, b = [torch.load(root / j['id'] / 'checkpoints/final.pt', weights_only=False) for j in pair]
        assert all(torch.equal(v, b['history'][0]['metrics']['A'][n]) if isinstance(v, torch.Tensor) else v == b['history'][0]['metrics']['A'][n] for n, v in a['history'][0]['metrics']['A'].items())
        for step in [1, 2]:
            assert a['history'][step]['last_update']['batch_sha256'] == b['history'][step]['last_update']['batch_sha256']
        assert all(float(v['step']) == 2 for v in a['optimizer_state']['state'].values())
        assert all(float(v['step']) == donor['step'] + 2 for v in b['optimizer_state']['state'].values())
        assert (root / pair[0]['id'] / 'gradients/step_000001.pt').exists()
        assert (root / pair[1]['id'] / 'result.json').exists()
        # Verify skip of completed work and actual resume from the last checkpoint.
        m.run_job(pair[0], root, 'cpu')
        (root / pair[1]['id'] / 'result.json').unlink()
        m.run_job(pair[1], root, 'cpu')
        resumed = torch.load(root / pair[1]['id'] / 'checkpoints/final.pt', weights_only=False)
        assert resumed['step'] == 2
        assert len(resumed['history']) == 3
    print('PASS: six CPU trajectories, identical pair weights/data, inherited moments/counters, gradients, resume and skip')
