"""TASK-slot reject mixing for the null-token probe."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from go4cl.constants import (
    CONTEXT_LENGTH,
    NUM_DIGITS,
    SEQ_LEN_OPERANDS,
    TOKEN_TASK_A,
    TOKEN_TASK_B,
)
from go4cl.data.null_task import (
    foreign_task_token_ids,
    mix_null_task_batch,
    wrap_null_task_loader,
)
from go4cl.phases.phase2.grid import conditions_by_name


def _batch(n: int = 8, task_token: int = TOKEN_TASK_A) -> dict[str, torch.Tensor]:
    tokens = torch.zeros(n, CONTEXT_LENGTH, dtype=torch.long)
    tokens[:, SEQ_LEN_OPERANDS] = task_token
    return {
        "tokens": tokens,
        "labels": torch.arange(n, dtype=torch.long) % 7 + 1,
        "task_ids": torch.zeros(n, dtype=torch.long),
    }


def test_phase_a_pool_includes_b_but_not_a() -> None:
    ids = foreign_task_token_ids("a")
    assert TOKEN_TASK_B in ids
    assert TOKEN_TASK_A not in ids
    assert set(range(NUM_DIGITS)).issubset(ids)


def test_phase_b_and_joint_are_neither_a_nor_b() -> None:
    for phase in ("b", "joint"):
        ids = foreign_task_token_ids(phase)
        assert TOKEN_TASK_A not in ids
        assert TOKEN_TASK_B not in ids
        assert ids == tuple(range(NUM_DIGITS))


def test_mix_appends_fixed_label_and_foreign_task_slot() -> None:
    rng = np.random.default_rng(0)
    ids = foreign_task_token_ids("a")
    out = mix_null_task_batch(
        _batch(8), rng, token_ids=ids, ratio=0.5, label=0
    )
    assert out["tokens"].shape[0] == 8 + 4
    extra_task = out["tokens"][8:, SEQ_LEN_OPERANDS].tolist()
    assert all(int(t) in ids for t in extra_task)
    assert torch.equal(out["labels"][8:], torch.zeros(4, dtype=torch.long))
    assert TOKEN_TASK_A not in extra_task


def test_phase_b_mix_never_uses_real_task_tokens() -> None:
    rng = np.random.default_rng(1)
    ids = foreign_task_token_ids("b")
    out = mix_null_task_batch(
        _batch(16, task_token=TOKEN_TASK_B),
        rng,
        token_ids=ids,
        ratio=1.0,
        label=0,
    )
    extra = set(out["tokens"][16:, SEQ_LEN_OPERANDS].tolist())
    assert TOKEN_TASK_A not in extra
    assert TOKEN_TASK_B not in extra


def test_wrapper_passthrough_when_disabled() -> None:
    class _Once:
        batch_size = 4
        n_packs = 1
        dataset = [0]

        def __iter__(self):
            yield _batch(4)

    wrapped = wrap_null_task_loader(
        _Once(), phase="a", ratio=0.25, enabled=False
    )
    assert wrapped is not None
    first = next(iter(wrapped))
    assert first["tokens"].shape[0] == 4


def test_wrapper_adds_rows() -> None:
    class _Once:
        batch_size = 8
        n_packs = 2
        dataset = [0]

        def __iter__(self):
            yield _batch(8)

    mixed = wrap_null_task_loader(_Once(), phase="joint", ratio=0.25, seed=0)
    first = next(iter(mixed))
    assert first["tokens"].shape[0] == 8 + 2
    assert mixed.n_packs == 2


def test_conditions_by_name_picks_partial_mod_overlap() -> None:
    cells = conditions_by_name(["s0.5_o0.5_m1"])
    assert len(cells) == 1
    assert cells[0].rho_slot == 0.5
    assert cells[0].rho_operand == 0.5
    assert cells[0].rho_mod == 1.0


def test_conditions_by_name_rejects_unknown() -> None:
    with pytest.raises(SystemExit):
        conditions_by_name(["not_a_cell"])
