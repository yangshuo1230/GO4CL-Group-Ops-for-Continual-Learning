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
    all_same = build_multi_op_pair("all_same", task_seed=seed)
    diff = build_multi_op_pair("four_diff", task_seed=seed)
    same = build_multi_op_pair("pair_same", task_seed=seed)

    assert all_same.task_a.n_ops == 4
    assert diff.task_a.n_ops == 4
    assert same.task_a.n_ops == 4

    # Edges/slots identical across 4-op variants; only moduli differ
    for z in range(4):
        a = all_same.task_a.by_latent()[z]
        d = diff.task_a.by_latent()[z]
        s = same.task_a.by_latent()[z]
        assert a.operand_pair == d.operand_pair == s.operand_pair
        assert a.slot == d.slot == s.slot

    # all_same: every latent uses focal p
    assert len({op.modulus for op in all_same.task_a.operations}) == 1
    assert all_same.task_a.by_latent()[0].modulus == diff.task_a.by_latent()[0].modulus

    # pair_same shares modulus on latents 0 and 1; four_diff does not
    assert same.task_a.by_latent()[0].modulus == same.task_a.by_latent()[1].modulus
    assert diff.task_a.by_latent()[0].modulus != diff.task_a.by_latent()[1].modulus


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
    assert meta["n_train_a"] == 0
    assert meta["n_val_a"] > 0
    assert meta["train_mode"] == "packed_online"
    assert "_pack1" in meta["tag"]
    assert (tmp_path / "data" / meta["tag"] / "manifest.json").exists()
    # Shared modulus ops reuse one residue split → still one split entry per unique p
    base = choose_base_moduli(0)
    mods = moduli_for_variant("pair_same", base)
    assert mods[0] == mods[1]


def test_sample_packed_examples_four_ops() -> None:
    import numpy as np

    from go4cl.data.generate import build_shared_residue_splits
    from go4cl.data.packed import sample_packed_examples
    from go4cl.tasks.multi_op import build_multi_op_pair

    pair = build_multi_op_pair("four_diff", task_seed=0)
    task = pair.task_a
    splits = build_shared_residue_splits([task], data_seed=0)
    rng = np.random.default_rng(0)
    exs = sample_packed_examples(rng, task, splits)
    assert len(exs) == 4
    digits = exs[0].tokens[:8]
    assert all(e.tokens[:8] == digits for e in exs)
    slots = {e.slot for e in exs}
    assert slots == {0, 1, 2, 3}
    used: list[int] = []
    for op, e in zip(task.operations, exs, strict=True):
        used.extend([op.i, op.j])
        assert e.modulus == op.modulus
        assert e.label == (digits[op.i] + digits[op.j]) % op.modulus
        assert e.tokens[8] == task.task_token
    assert sorted(used) == list(range(8))
    # Same seed is reproducible; next draw can differ (with-replacement)
    rng_a = np.random.default_rng(1)
    rng_b = np.random.default_rng(1)
    a = sample_packed_examples(rng_a, task, splits)
    b = sample_packed_examples(rng_b, task, splits)
    assert [e.tokens for e in a] == [e.tokens for e in b]
    later = [sample_packed_examples(rng_a, task, splits) for _ in range(16)]
    assert any(
        [e.tokens for e in pack] != [e.tokens for e in a] for pack in later
    )


def test_packed_train_loader_batch_shape() -> None:
    from go4cl.data.generate import build_shared_residue_splits
    from go4cl.data.packed import make_packed_multi_op_train_loader
    from go4cl.tasks.multi_op import build_multi_op_pair

    pair = build_multi_op_pair("four_diff", task_seed=0)
    task = pair.task_a
    splits = build_shared_residue_splits([task], data_seed=0)
    loader = make_packed_multi_op_train_loader(task, splits, batch_size=8, seed=0)
    assert loader.n_packs == 2
    batch = next(iter(loader))
    assert tuple(batch["tokens"].shape) == (8, 10)
    assert batch["labels"].shape == (8,)
    # Two packs × 4 queries; slots should be two copies of {0,1,2,3}
    slots = set(batch["slots"].tolist())
    assert slots == {0, 1, 2, 3}


def test_sample_packed_examples_one_op() -> None:
    import numpy as np

    from go4cl.data.generate import build_shared_residue_splits
    from go4cl.data.packed import sample_packed_examples
    from go4cl.tasks.multi_op import build_multi_op_pair

    pair = build_multi_op_pair("one", task_seed=0)
    task = pair.task_a
    splits = build_shared_residue_splits([task], data_seed=0)
    exs = sample_packed_examples(np.random.default_rng(0), task, splits)
    assert len(exs) == 1
    op = task.operations[0]
    digits = exs[0].tokens[:8]
    assert exs[0].label == (digits[op.i] + digits[op.j]) % op.modulus

