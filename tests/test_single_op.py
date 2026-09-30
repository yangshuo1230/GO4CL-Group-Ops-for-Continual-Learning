"""Tests for single-op tasks and phase-1 data generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from go4cl.constants import PRIMES
from go4cl.data.generate import generate_task_datasets, save_datasets
from go4cl.data.residue_pairs import assert_disjoint
from go4cl.phases.common import ratios_from_train_frac
from go4cl.tasks.single_op import build_single_op_pair, build_single_op_task
from go4cl.tasks.spec import TaskSpec


def test_single_op_task_ok() -> None:
    task = build_single_op_task(17, slot=2, i=3, j=5)
    assert task.is_single_op
    assert task.n_ops == 1
    assert task.moduli() == (17,)
    assert task.by_slot()[2].modulus == 17


def test_full_task_still_requires_perfect_matching() -> None:
    from go4cl.tasks.relations import build_task_pair

    pair = build_task_pair(task_seed=0)
    assert pair.task_a.n_ops == 4
    assert not pair.task_a.is_single_op


@pytest.mark.parametrize("p", PRIMES)
def test_single_op_pair_all_moduli(p: int) -> None:
    pair = build_single_op_pair(p, task_seed=p)
    assert pair.task_a.is_single_op
    assert pair.task_a.operations[0].modulus == p
    assert pair.task_b.operations[0].modulus == p
    assert pair.task_a.task_token != pair.task_b.task_token


def test_single_op_dataset_generation(tmp_path: Path) -> None:
    pair = build_single_op_pair(17, task_seed=0)
    ratios = ratios_from_train_frac(0.6)
    manifest, datasets = generate_task_datasets(
        pair,
        data_seed=0,
        n_aliases_per_pair=2,
        n_nuisance_contexts=1,
        ratios=ratios,
        experiment_id="test_single",
    )
    for split in manifest.residue_splits.values():
        assert_disjoint(split)
    assert set(datasets.keys()) == {"A", "B"}
    assert len(datasets["A"]["train"]) > 0
    # Only one slot present
    slots = {e.slot for e in datasets["A"]["train"]}
    assert len(slots) == 1
    mods = {e.modulus for e in datasets["A"]["train"]}
    assert mods == {17}
    save_datasets(tmp_path, manifest, datasets)
    assert (tmp_path / "manifest.json").exists()
    assert (tmp_path / "A" / "train.npz").exists()


def test_invalid_partial_overlap_ops() -> None:
    from go4cl.tasks.spec import Operation

    with pytest.raises(ValueError, match="operand positions must not overlap"):
        TaskSpec(
            name="A",
            task_id=0,
            operations=(
                Operation(latent_id=0, i=0, j=1, modulus=7, slot=0),
                Operation(latent_id=1, i=1, j=2, modulus=11, slot=1),
            ),
        )
