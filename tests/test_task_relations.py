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


@pytest.mark.parametrize("rho_slot", VALID_OVERLAPS)
@pytest.mark.parametrize("rho_operand", VALID_OVERLAPS)
@pytest.mark.parametrize("rho_mod", VALID_OVERLAPS)
def test_fixed_a_overlaps(rho_slot, rho_operand, rho_mod) -> None:
    pair = build_task_pair(
        rho_slot=rho_slot,
        rho_operand=rho_operand,
        rho_mod=rho_mod,
        task_seed=0,
        fixed_a=True,
    )
    assert compute_overlaps(pair.task_a, pair.task_b) == (
        rho_slot,
        rho_operand,
        rho_mod,
    )
    assert pair.pair_id.endswith("_fixedA")


def test_fixed_a_identical_across_rho() -> None:
    """Same task_seed + fixed_a ⇒ identical Task A for every overlap cell."""
    seed = 7
    cells = [
        (0.0, 0.0, 0.0),
        (0.0, 0.5, 1.0),
        (0.5, 0.0, 0.5),
        (1.0, 1.0, 1.0),
        (0.5, 1.0, 0.0),
    ]
    refs = None
    for rho_slot, rho_operand, rho_mod in cells:
        pair = build_task_pair(
            rho_slot=rho_slot,
            rho_operand=rho_operand,
            rho_mod=rho_mod,
            task_seed=seed,
            fixed_a=True,
        )
        ops = tuple(
            (op.latent_id, op.i, op.j, op.modulus, op.slot)
            for op in sorted(pair.task_a.operations, key=lambda o: o.latent_id)
        )
        if refs is None:
            refs = ops
        else:
            assert ops == refs


def test_legacy_a_varies_with_rho() -> None:
    """Without fixed_a, Task A still depends on ρ (legacy joint RNG)."""
    a0 = build_task_pair(rho_slot=0.0, rho_operand=0.0, rho_mod=0.0, task_seed=0)
    a1 = build_task_pair(rho_slot=1.0, rho_operand=1.0, rho_mod=1.0, task_seed=0)

    def sig(t):
        return tuple(
            (op.latent_id, op.i, op.j, op.modulus, op.slot)
            for op in sorted(t.task_a.operations, key=lambda o: o.latent_id)
        )

    assert sig(a0) != sig(a1)
