"""Generate fixed datasets from the A/B task relation design."""

from __future__ import annotations

from pathlib import Path

from go4cl.data.generate import generate_task_datasets, save_datasets
from go4cl.data.residue_pairs import assert_disjoint
from go4cl.tasks.relations import build_task_pair


def run_generate(args) -> None:
    pair = build_task_pair(
        rho_slot=args.rho_slot,
        rho_operand=args.rho_operand,
        rho_mod=args.rho_mod,
        task_seed=args.task_seed,
    )
    manifest, datasets = generate_task_datasets(
        pair,
        data_seed=args.data_seed,
        n_aliases_per_pair=args.n_aliases,
        n_nuisance_contexts=args.n_nuisance,
        experiment_id=args.experiment_id,
    )
    for split in manifest.residue_splits.values():
        assert_disjoint(split)

    out = Path(args.out)
    save_datasets(out, manifest, datasets)
    print(f"wrote dataset to {out}")
    print(f"pair_id={pair.pair_id}")
    print(
        f"overlaps: slot={pair.rho_slot} operand={pair.rho_operand} mod={pair.rho_mod}"
    )
    print(f"dataset_hash={manifest.dataset_hash}")
    for task_name, splits in datasets.items():
        for split_name, exs in splits.items():
            print(f"  {task_name}/{split_name}: {len(exs)} examples")
