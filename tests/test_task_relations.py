"""Tests for A/B task relation construction."""

from __future__ import annotations

import pytest

from go4cl.tasks.relations import VALID_OVERLAPS, build_task_pair, compute_overlaps, swap_ab


@pytest.mark.parametrize("rho_slot", VALID_OVERLAPS)
@pytest.mark.parametrize("rho_operand", VALID_OVERLAPS)
@pytest.mark.parametrize("rho_mod", VALID_OVERLAPS)
def test_build_task_pair_overlaps(rho_slot, rho_operand, rho_mod) -> None:
    pair = build_task_pair(
        rho_slot=rho_slot,
        rho_operand=rho_operand,
        rho_mod=rho_mod,
        task_seed=0,
    )
    got = compute_overlaps(pair.task_a, pair.task_b)
    assert got == (rho_slot, rho_operand, rho_mod)


def test_perfect_matching_constraint() -> None:
    pair = build_task_pair(rho_slot=0.0, rho_operand=1.0, rho_mod=0.0, task_seed=3)
    for task in (pair.task_a, pair.task_b):
        used = []
        for op in task.operations:
            used.extend([op.i, op.j])
        assert sorted(used) == list(range(8))


def test_swap_ab_preserves_overlaps() -> None:
    pair = build_task_pair(rho_slot=0.5, rho_operand=0.0, rho_mod=1.0, task_seed=5)
    swapped = swap_ab(pair)
    assert compute_overlaps(swapped.task_a, swapped.task_b) == (
        pair.rho_slot,
        pair.rho_operand,
        pair.rho_mod,
    )
