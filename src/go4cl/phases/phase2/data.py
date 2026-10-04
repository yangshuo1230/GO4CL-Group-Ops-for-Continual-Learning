"""Phase 2 A/B datasets: packed online train, shared residue splits."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from go4cl.phases.common import ratios_from_train_frac

PHASE2_GENERATION_RULE = (
    "Phase 2 packed online: one residue-pair train/val/test split per modulus, "
    "shared by every A/B operation that uses that modulus. Val/test aliases are "
    "written to disk as the nuisance control; primary eval is packed_id rebuilt "
    "from the manifest. Train is not pre-expanded. Direction 'swap' exchanges "
    "the task-token assignment of the same overlap pair under the same data_seed."
)


def prepare_phase2_dataset(
    out: Path,
    *,
    rho_slot: float,
    rho_operand: float,
    rho_mod: float,
    task_seed: int,
    data_seed: int,
    n_aliases: int,
    train_frac: float = 0.8,
    direction: str = "forward",
) -> dict[str, Any]:
    """Build or reuse one packed A/B dataset under ``out/data/``."""
    from go4cl.data.generate import generate_task_datasets, save_datasets
    from go4cl.data.manifest import DataManifest
    from go4cl.data.residue_pairs import assert_disjoint
    from go4cl.tasks.relations import build_task_pair, swap_ab

    if direction not in {"forward", "swap"}:
        raise ValueError(f"direction must be forward|swap, got {direction}")

    ratios = ratios_from_train_frac(train_frac)
    tag = (
        f"p2_s{rho_slot:g}_o{rho_operand:g}_m{rho_mod:g}"
        f"_{direction}_ts{task_seed}_ds{data_seed}_a{n_aliases}"
        f"_tr{train_frac:g}_pack1"
    )
    data_dir = out / "data" / tag
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = DataManifest.load(manifest_path)
        print(
            f"[data] reuse {data_dir}  "
            f"A_val={manifest.samples_per_slot['A']['val']}  "
            f"train_mode={manifest.train_mode}"
        )
    else:
        pair = build_task_pair(
            rho_slot=rho_slot,
            rho_operand=rho_operand,
            rho_mod=rho_mod,
            task_seed=task_seed,
        )
        if direction == "swap":
            pair = swap_ab(pair)
        manifest, datasets = generate_task_datasets(
            pair,
            data_seed=data_seed,
            n_aliases_per_pair=n_aliases,
            n_nuisance_contexts=1,
            ratios=ratios,
            experiment_id=tag,
            skip_train=True,
        )
        manifest.generation_rule = PHASE2_GENERATION_RULE
        manifest.train_mode = "packed_online"
        for split in manifest.residue_splits.values():
            assert_disjoint(split)
        save_datasets(data_dir, manifest, datasets)
        print(
            f"[data] wrote {data_dir}  "
            f"A_val={manifest.samples_per_slot['A']['val']}  "
            f"B_val={manifest.samples_per_slot['B']['val']}  "
            f"hash={manifest.dataset_hash[:12]}"
        )

    pair = manifest.task_pair
    return {
        "tag": tag,
        "data_dir": str(data_dir),
        "rho_slot": pair.rho_slot,
        "rho_operand": pair.rho_operand,
        "rho_mod": pair.rho_mod,
        "direction": direction,
        "task_seed": int(task_seed),
        "data_seed": int(data_seed),
        "pair_id": pair.pair_id,
        "n_aliases": n_aliases,
        "train_frac": train_frac,
        "ratios": list(ratios),
        "n_train_a": 0,
        "n_val_a": int(manifest.samples_per_slot["A"]["val"]),
        "n_test_a": int(manifest.samples_per_slot["A"]["test"]),
        "n_val_b": int(manifest.samples_per_slot["B"]["val"]),
        "n_test_b": int(manifest.samples_per_slot["B"]["test"]),
        "train_mode": manifest.train_mode,
        "dataset_hash": manifest.dataset_hash,
        "generation_rule": PHASE2_GENERATION_RULE,
    }
