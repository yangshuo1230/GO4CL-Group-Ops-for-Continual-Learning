"""Tests for Phase 1B multi-op task construction."""

from __future__ import annotations

from pathlib import Path

from go4cl.tasks.multi_op import (
    build_multi_op_pair,
    choose_base_moduli,
    moduli_for_variant,
)


def test_variants_share_structure() -> None:
    seed = 3
    one = build_multi_op_pair("one", task_seed=seed)
    diff = build_multi_op_pair("four_diff", task_seed=seed)
    same = build_multi_op_pair("pair_same", task_seed=seed)

    assert one.task_a.n_ops == 1
    assert diff.task_a.n_ops == 4
    assert same.task_a.n_ops == 4

    # First latent edge/slot match across variants
    o0 = one.task_a.operations[0]
    d0 = diff.task_a.by_latent()[0]
    s0 = same.task_a.by_latent()[0]
    assert o0.operand_pair == d0.operand_pair == s0.operand_pair
    assert o0.slot == d0.slot == s0.slot
    assert o0.modulus == d0.modulus == s0.modulus

    # pair_same shares modulus on latents 0 and 1; four_diff does not
    assert same.task_a.by_latent()[0].modulus == same.task_a.by_latent()[1].modulus
    assert diff.task_a.by_latent()[0].modulus != diff.task_a.by_latent()[1].modulus

    # Remaining edges match between four_diff and pair_same
    for z in range(4):
        assert (
            diff.task_a.by_latent()[z].operand_pair
            == same.task_a.by_latent()[z].operand_pair
        )
        assert diff.task_a.by_latent()[z].slot == same.task_a.by_latent()[z].slot


def test_all_same_moduli() -> None:
    pair = build_multi_op_pair("all_same", task_seed=0)
    mods = [op.modulus for op in pair.task_a.operations]
    assert len(set(mods)) == 1


def test_prepare_multi_op_dataset(tmp_path: Path) -> None:
    from go4cl.phases.phase1.data import prepare_multi_op_dataset

    meta = prepare_multi_op_dataset(
        tmp_path,
        variant="pair_same",
        n_aliases=2,
        task_seed=0,
        data_seed=0,
        train_frac=0.6,
    )
    assert meta["n_ops"] == 4
    assert meta["n_train_a"] > 0
    assert (tmp_path / "data" / meta["tag"] / "manifest.json").exists()
    # Shared modulus ops reuse one residue split → still one split entry per unique p
    base = choose_base_moduli(0)
    mods = moduli_for_variant("pair_same", base)
    assert mods[0] == mods[1]
