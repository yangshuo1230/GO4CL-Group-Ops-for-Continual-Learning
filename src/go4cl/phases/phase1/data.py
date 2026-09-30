"""Phase 1A dataset construction (single-op).

Only operand positions that enter the queried op are split by residue pair
and expanded with ``n_aliases``. The other six digit positions are sampled
uniformly from the digit vocab ``0..63`` once per alias.

Two split modes (exactly one required):

- ``train_frac``: fixed train/val/test **ratios** (legacy path)
- ``n_train_pairs``: fixed train residue-pair **count** (val/test = remainder)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from go4cl.data.residue_pairs import all_unordered_pairs
from go4cl.phases.common import ratios_from_train_frac

PHASE1A_GENERATION_RULE_FRAC = (
    "Phase 1A: partition unordered residue pairs (r_i, r_j) of the queried "
    "operand positions by train_frac (val/test share the remainder); for each "
    "pair emit n_aliases raw aliases x_i≡r1, x_j≡r2 (mod p) with x in 0..63; "
    "fill the six irrelevant digit positions uniformly from 0..63 once per "
    "alias (no nuisance expansion)."
)

PHASE1A_GENERATION_RULE_FIXED = (
    "Phase 1A: partition unordered residue pairs (r_i, r_j) of the queried "
    "operand positions with a fixed train count n_train_pairs (val/test share "
    "the remainder, stratified); for each pair emit n_aliases raw aliases "
    "x_i≡r1, x_j≡r2 (mod p) with x in 0..63; fill the six irrelevant digit "
    "positions uniformly from 0..63 once per alias (no nuisance expansion)."
)

DEFAULT_BATCH_SIZE = 2048
DEFAULT_N_TRAIN_PAIRS = 150


def prepare_single_op_dataset(
    out: Path,
    *,
    modulus: int,
    n_aliases: int,
    task_seed: int,
    data_seed: int,
    train_frac: float | None = None,
    n_train_pairs: int | None = None,
) -> dict[str, Any]:
    """Build or reuse a Phase 1A single-op fixed dataset under ``out/data/``.

    Pass exactly one of ``train_frac`` or ``n_train_pairs``.
    """
    from go4cl.data.generate import generate_task_datasets, save_datasets
    from go4cl.data.manifest import DataManifest
    from go4cl.data.residue_pairs import assert_disjoint
    from go4cl.tasks.single_op import build_single_op_pair

    if (train_frac is None) == (n_train_pairs is None):
        raise ValueError("pass exactly one of train_frac or n_train_pairs")

    n_total = len(all_unordered_pairs(modulus))
    if train_frac is not None:
        ratios = ratios_from_train_frac(train_frac)
        tag = (
            f"single_p{modulus}_tr{train_frac:g}"
            f"_ts{task_seed}_ds{data_seed}_a{n_aliases}_p1a"
        )
        split_mode = "train_frac"
        generation_rule = PHASE1A_GENERATION_RULE_FRAC
        gen_kwargs: dict[str, Any] = {"ratios": ratios}
    else:
        assert n_train_pairs is not None
        ratios = None
        tag = (
            f"single_p{modulus}_ntp{n_train_pairs}"
            f"_ts{task_seed}_ds{data_seed}_a{n_aliases}_p1a"
        )
        split_mode = "n_train_pairs"
        generation_rule = PHASE1A_GENERATION_RULE_FIXED
        gen_kwargs = {"n_train_pairs": int(n_train_pairs)}

    data_dir = out / "data" / tag
    manifest_path = data_dir / "manifest.json"
    if manifest_path.exists():
        manifest = DataManifest.load(manifest_path)
        print(
            f"[data] reuse {data_dir}  "
            f"A_train={manifest.samples_per_slot['A']['train']}"
        )
        split = next(iter(manifest.residue_splits.values()))
        n_train_pairs_realized = len(split.train)
    else:
        if n_train_pairs is not None and n_train_pairs >= n_total:
            print(
                f"[data][warn] p={modulus}: n_train_pairs={n_train_pairs} >= "
                f"total unordered pairs={n_total}; train will be capped after "
                f"val/test reserves"
            )
        pair = build_single_op_pair(modulus, task_seed=task_seed)
        manifest, datasets = generate_task_datasets(
            pair,
            data_seed=data_seed,
            n_aliases_per_pair=n_aliases,
            n_nuisance_contexts=1,
            experiment_id=tag,
            **gen_kwargs,
        )
        manifest.generation_rule = generation_rule
        for split in manifest.residue_splits.values():
            assert_disjoint(split)
        save_datasets(data_dir, manifest, datasets)
        split = next(iter(manifest.residue_splits.values()))
        n_train_pairs_realized = len(split.train)
        if n_train_pairs is not None and n_train_pairs_realized < n_train_pairs:
            print(
                f"[data][warn] p={modulus}: requested n_train_pairs={n_train_pairs} "
                f"but only {n_train_pairs_realized} available after val/test reserves "
                f"(total pairs={n_total})"
            )
        extra = (
            f"ratios={ratios}"
            if ratios is not None
            else f"n_train_pairs={n_train_pairs_realized}/{n_train_pairs}"
        )
        print(
            f"[data] wrote {data_dir}  "
            f"A_train={manifest.samples_per_slot['A']['train']}  "
            f"{extra}  hash={manifest.dataset_hash[:12]}"
        )

    realized_ratios = list(next(iter(manifest.residue_splits.values())).ratios)
    return {
        "tag": tag,
        "data_dir": str(data_dir),
        "modulus": modulus,
        "split_mode": split_mode,
        "train_frac": train_frac,
        "n_train_pairs": n_train_pairs,
        "n_train_pairs_realized": n_train_pairs_realized,
        "n_total_pairs": n_total,
        "ratios": realized_ratios,
        "n_aliases": n_aliases,
        "n_train_a": int(manifest.samples_per_slot["A"]["train"]),
        "n_val_a": int(manifest.samples_per_slot["A"]["val"]),
        "n_test_a": int(manifest.samples_per_slot["A"]["test"]),
        "dataset_hash": manifest.dataset_hash,
        "pair_id": manifest.task_pair.pair_id,
        "op": manifest.task_pair.task_a.operations[0].to_dict(),
        "generation_rule": generation_rule,
        "batch_size": DEFAULT_BATCH_SIZE,
    }
