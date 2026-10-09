"""Offline trajectory analysis for the paired mechanism suite; PNG figures only."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from go4cl.analysis.probes import fit_linear_probe
from go4cl.data.eval_contexts import make_eval_context_loader
from go4cl.data.generate import build_shared_residue_splits
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import load_checkpoint, write_json


def probe_data(task, splits, n, seed):
    # Separate residue-pair splits; every checkpoint sees identical examples.
    return {split: next(iter(make_eval_context_loader(task, splits, target_split=split,
             n_per_operation=n, seed=seed + i))) for i, split in enumerate(('train', 'val', 'test'))}


def features(model, data, device):
    return {split: model.forward_with_cache(batch['tokens'].to(device)) for split, batch in data.items()}


def targets(data, op, target, device):
    if target == 'sum':
        return {s: b['labels'][b['latent_ids'] == op.latent_id].to(device) for s, b in data.items()}
    index = op.i if target == 'xi' else op.j
    return {s: (b['tokens'][b['latent_ids'] == op.latent_id, index] % op.modulus).to(device) for s, b in data.items()}


def probes_for_task(model, task, data, device, steps, seed, fixed=None):
    cache = features(model, data, device); probes = {}; scores = []
    for op in task.operations:
        for layer in range(model.cfg.n_layers):
            for site in ('resid_mid', 'resid_post'):
                x = {s: c[site][layer][data[s]['latent_ids'] == op.latent_id, -1] for s, c in cache.items()}
                for target in ('xi', 'xj', 'sum'):
                    y = targets(data, op, target, device)
                    key = f'lat{op.latent_id}/{site}/{layer}/{target}'
                    result = fit_linear_probe(x['train'], y['train'], {s: x[s] for s in ('val', 'test')},
                        {s: y[s] for s in ('val', 'test')}, n_classes=op.modulus, steps=steps, seed=seed)
                    probes[key] = {'weight': result['weight'], 'bias': result['bias']}
                    row = dict(operation=op.to_dict(), layer=layer, site=site, target=target,
                               retrained_train=result['train_acc'], retrained_val=result['eval_acc']['val'],
                               retrained_test=result['eval_acc']['test'])
                    if fixed is not None:
                        p = fixed[key]
                        logits = F.linear(x['test'], p['weight'].to(device), p['bias'].to(device))
                        row['fixed_test'] = float((logits.argmax(-1) == y['test']).float().mean())
                    scores.append(row)
    return probes, scores, cache


def op_metrics(logits, batch):
    losses = F.cross_entropy(logits, batch['labels'].to(logits.device), reduction='none')
    correct = logits.argmax(-1).cpu() == batch['labels']
    return {str(z): dict(accuracy=float(correct[batch['latent_ids'] == z].float().mean()),
                         loss=float(losses[(batch['latent_ids'] == z).to(logits.device)].mean())) for z in range(4)}


def causals(model, reference, task, data, device):
    batch = data['test']; tokens = batch['tokens'].to(device)
    current = model.forward_with_cache(tokens); donor = reference.forward_with_cache(tokens)
    out = {'baseline': op_metrics(current['logits'], batch), 'routing': [], 'interventions': {}}
    for layer, att in enumerate(current['attn']):
        for op in task.operations:
            mask = (batch['latent_ids'] == op.latent_id).to(device)
            for head in range(model.cfg.n_heads):
                mass = att[mask, head, -1, op.i] + att[mask, head, -1, op.j]
                out['routing'].append(dict(latent=op.latent_id, layer=layer, head=head,
                    operand_mass=float(mass.mean()), task_mass=float(att[mask, head, -1, 8].mean())))
        for scope in ('query', 'all_tokens'):
            patched = current['resid_post'][layer].clone()
            if scope == 'query': patched[:, -1] = donor['resid_post'][layer][:, -1]
            else: patched = donor['resid_post'][layer]
            logits = model.continue_from_layer(patched, layer_idx=layer)['logits']
            out['interventions'][f'donor_resid_post/L{layer}/{scope}'] = op_metrics(logits, batch)
        for head in range(model.cfg.n_heads):
            def zero_head(module, inputs, h=head):
                x = inputs[0].clone()
                start = h * model.cfg.d_head
                x[:, -1, start:start + model.cfg.d_head] = 0
                return (x,)
            handle = model.blocks[layer].attn.out.register_forward_pre_hook(zero_head)
            try:
                with torch.no_grad(): logits = model(tokens)['logits']
            finally: handle.remove()
            out['interventions'][f'zero_query_head/L{layer}/H{head}'] = op_metrics(logits, batch)
        def zero_query(module, inputs, output):
            x = output.clone(); x[:, -1] = 0; return x
        handle = model.blocks[layer].mlp.register_forward_hook(zero_query)
        try:
            with torch.no_grad(): logits = model(tokens)['logits']
        finally: handle.remove()
        out['interventions'][f'zero_query_mlp/L{layer}'] = op_metrics(logits, batch)
    flipped = tokens.clone(); flipped[:, 8] = 65 if task.task_id == 0 else 64
    with torch.no_grad(): logits = model(flipped)['logits']
    out['interventions']['flip_task_token_scored_against_original_task'] = op_metrics(logits, batch)
    return out


def vector_stats(snapshot):
    gradients = snapshot['gradients']; a = gradients['A']; b = gradients['B']
    groups = {'global': [(n, None) for n in a]}
    for sl in snapshot['parameter_groups']:
        for group in [sl['group'], *([f"layer_{sl['layer']}"] if sl['layer'] is not None else [])]:
            groups.setdefault(group, []).append((sl['name'], sl['rows']))
    result = {}
    for group, slices in groups.items():
        def vec(values):
            return torch.cat([(values[n] if rows is None else values[n][rows[0]:rows[1]]).flatten().double() for n, rows in slices])
        va, vb = vec(a), vec(b)
        cosine = lambda x, y: float(torch.dot(x, y) / (x.norm() * y.norm())) if x.norm() > 0 and y.norm() > 0 else None
        result[group] = dict(norms={name: float(vec(v).norm()) for name, v in gradients.items()},
                            A_B_cosine=cosine(va, vb),
                            update_A_gradient_cosines={name: cosine(vec(v), va) for name, v in snapshot['deltas'].items()},
                            update_norms={name: float(vec(v).norm()) for name, v in snapshot['deltas'].items()})
    return result


def summarize(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []; out = root / 'summary'; out.mkdir(exist_ok=True)
    for path in root.glob('seed*/**/result.json'):
        directory = path.parent; result = json.loads(path.read_text()); resolved = json.loads((directory / 'resolved.json').read_text())
        row = dict(job=resolved['job']['id'], seed=resolved['job']['seed'], status=result['status'],
                   requested_replay=resolved['job']['replay_ratio'], source=resolved['job']['source'],
                   source_mastered=result.get('source_mastered'))
        if result['status'] == 'complete':
            history = json.loads((directory / 'history.json').read_text()); threshold = resolved['config']['mastery_accuracy']; window = resolved['config']['mastery_window']
            def passes(record, name):
                return all(v['accuracy'] >= threshold for v in record['metrics'][name]['by_operation'].values())
            row.update(A_final=result['final_val']['A']['macro_operation_accuracy'], B_final=result['final_val']['B']['macro_operation_accuracy'],
                       effective_replay=result['effective_replay_ratio'], seconds=result['training_and_analysis_seconds'],
                       A_maintained_at_sampled_points=all(passes(h, 'A') for h in history),
                       final_window_joint_pass=len(history) >= window and all(passes(h, t) for h in history[-window:] for t in ['A', 'B']))
            fig, ax = plt.subplots(figsize=(8, 4))
            for name in ['A', 'B']:
                for z in range(4):
                    keys = list(history[0]['metrics'][name]['by_operation'])
                    key = next(k for k in keys if f'/lat{z}/' in k)
                    ax.plot([h['step'] for h in history], [h['metrics'][name]['by_operation'][key]['accuracy'] for h in history], label=f'{name}/lat{z}')
            ax.set(xlabel='Training step', ylabel='Held-out operation accuracy', ylim=(0, 1.02), title=resolved['job']['id'])
            ax.legend(ncol=4, fontsize=7); fig.tight_layout(); fig.savefig(directory / 'behavior.png', dpi=150); plt.close(fig)
            gradient_rows = []
            for gpath in sorted((directory / 'gradients').glob('step_*.pt')):
                snap = torch.load(gpath, map_location='cpu', weights_only=False)
                gradient_rows.append(dict(step=snap['step'], statistics=vector_stats(snap), update_effects=snap['effects']))
            if gradient_rows: write_json(directory / 'gradient_summary.json', gradient_rows)
        rows.append(row)
    write_json(out / 'behavior_summary.json', rows)
    if rows:
        keys = sorted(set().union(*(r.keys() for r in rows)))
        with (out / 'behavior_summary.csv').open('w', newline='') as handle:
            writer = csv.DictWriter(handle, keys); writer.writeheader(); writer.writerows(rows)


def probe_figures(report, directory):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for task in ['A', 'B']:
        records = [r for r in report if r.get('task') == task and 'step' in r]
        if not records: continue
        for z in range(4):
            fig, ax = plt.subplots(figsize=(7, 4))
            layers = sorted({p['layer'] for p in records[0]['probes']})
            for layer in layers:
                points = [(r['step'], next(p for p in r['probes'] if p['operation']['latent_id'] == z
                          and p['site'] == 'resid_post' and p['layer'] == layer and p['target'] == 'sum')) for r in records]
                line, = ax.plot([s for s, _ in points], [p['retrained_test'] for _, p in points], label=f'L{layer+1} retrained')
                ax.plot([s for s, _ in points], [p['fixed_test'] for _, p in points], '--', color=line.get_color(), label=f'L{layer+1} fixed')
            ax.set(xlabel='Training step', ylabel='Held-out sum probe accuracy', ylim=(0, 1.02), title=f'{task}/lat{z} post-MLP query')
            ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(directory / f'sum_probe_{task}_lat{z}.png', dpi=150); plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', default='runs/phase2/mechanism_suite_20261009')
    parser.add_argument('--job', help='e.g. seed0/main/same_mod_diff_pos/replay_0.1')
    parser.add_argument('--steps', default='0,1000,10000,100000', help='all or comma-separated checkpoint steps')
    parser.add_argument('--probe-steps', type=int, default=400)
    parser.add_argument('--n-per-operation', type=int, default=512)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--causal', action='store_true', help='Head/MLP zeroing and same-input donor activation patches')
    args = parser.parse_args(); torch.set_num_threads(1); root = Path(args.root)
    if not args.job:
        summarize(root); return
    directory = root / args.job; resolved = json.loads((directory / 'resolved.json').read_text()); job = resolved['job']; cfg = resolved['config']
    a = TaskSpec.from_dict(job['A']); b = TaskSpec.from_dict(job['B']); splits = build_shared_residue_splits([a, b], data_seed=cfg['data_seed'])
    reference_path = root / job['source'] / 'checkpoints/final.pt' if job['source'] else directory / 'checkpoints/final.pt'
    reference, _ = load_checkpoint(reference_path, map_location=args.device); reference.eval()
    data = {name: probe_data(t, splits, args.n_per_operation, cfg['eval_seed'] + 50000 + t.task_id * 1009) for name, t in [('A', a), ('B', b)]}
    fixed = {}; report = []; analysis_dir = directory / 'analysis'; analysis_dir.mkdir(exist_ok=True)
    for name, t in [('A', a), ('B', b)]:
        fixed[name], scores, _ = probes_for_task(reference, t, data[name], args.device, args.probe_steps, 4401)
        report.append(dict(reference=str(reference_path), task=name, probes=scores))
    torch.save(fixed, analysis_dir / 'fixed_probes.pt')
    selected = None if args.steps == 'all' else set(map(int, args.steps.split(',')))
    for path in sorted((directory / 'checkpoints').glob('step_*.pt')):
        step = int(path.stem.split('_')[1])
        if selected is not None and step not in selected: continue
        model, _ = load_checkpoint(path, map_location=args.device); model.eval(); records = []
        for name, t in [('A', a), ('B', b)]:
            probes, scores, _ = probes_for_task(model, t, data[name], args.device, args.probe_steps, 4401, fixed[name])
            torch.save(probes, analysis_dir / f'probes_{step:06d}_{name}.pt')
            row = dict(step=step, task=name, probes=scores)
            if args.causal: row['causal'] = causals(model, reference, t, data[name], args.device)
            records.append(row)
        write_json(analysis_dir / f'step_{step:06d}.json', records); report.extend(records)
    write_json(analysis_dir / 'report.json', dict(settings=vars(args), reference=str(reference_path), records=report,
               interpretation='Probe readability and zero-ablation necessity are distinct. Donor patches can fail from interface mismatch; failed recovery is not evidence of erased knowledge.'))
    probe_figures(report, analysis_dir)


if __name__ == '__main__':
    main()
