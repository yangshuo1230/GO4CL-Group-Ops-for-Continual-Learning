"""Tests for residue-pair splits."""

from __future__ import annotations

import pytest

from go4cl.constants import PRIMES
from go4cl.data.residue_pairs import (
    all_unordered_pairs,
    assert_disjoint,
    pair_label,
    stratified_residue_pair_split,
    stratified_residue_pair_split_fixed_train,
)


@pytest.mark.parametrize("p", PRIMES)
def test_split_covers_all_pairs(p: int) -> None:
    split = stratified_residue_pair_split(p, data_seed=0)
    assert_disjoint(split)
    assert len(split.train) + len(split.val) + len(split.test) == len(
        all_unordered_pairs(p)
    )


@pytest.mark.parametrize("p", [23, 29, 53])
def test_val_test_cover_labels_when_possible(p: int) -> None:
    split = stratified_residue_pair_split(p, data_seed=1)
    # Every label with >=3 pairs should appear in val and test
    from collections import Counter

    def labels(pairs):
        return {pair_label(pr, p) for pr in pairs}

    by_y = Counter(pair_label(pr, p) for pr in all_unordered_pairs(p))
    rich = {y for y, n in by_y.items() if n >= 3}
    assert rich <= labels(split.val)
    assert rich <= labels(split.test)


def test_splits_deterministic() -> None:
    a = stratified_residue_pair_split(31, data_seed=42)
    b = stratified_residue_pair_split(31, data_seed=42)
    assert a.train == b.train and a.val == b.val and a.test == b.test


@pytest.mark.parametrize("p", PRIMES)
def test_fixed_train_covers_all_pairs(p: int) -> None:
    split = stratified_residue_pair_split_fixed_train(
        p, n_train_pairs=150, data_seed=0
    )
    assert_disjoint(split)
    total = len(all_unordered_pairs(p))
    assert len(split.train) + len(split.val) + len(split.test) == total
    assert len(split.train) <= 150
    # Enough leftover after ~2 reserved pairs per class → exact 150
    if total > 150 + 2 * p:
        assert len(split.train) == 150


def test_fixed_train_deterministic() -> None:
    a = stratified_residue_pair_split_fixed_train(31, n_train_pairs=150, data_seed=7)
    b = stratified_residue_pair_split_fixed_train(31, n_train_pairs=150, data_seed=7)
    assert a.train == b.train and a.val == b.val and a.test == b.test
