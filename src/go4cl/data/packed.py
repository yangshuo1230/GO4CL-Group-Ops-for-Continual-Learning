"""Online packed multi-op training: one shared 8-digit context, one query each."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from go4cl.constants import NUM_DIGITS, QUERY_TOKEN_IDS, SEQ_LEN_OPERANDS
from go4cl.data.generate import Example, sample_raw_operands
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.tasks.spec import TaskSpec


def sample_packed_examples(
    rng: np.random.Generator,
    task: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    *,
    split_name: str = "train",
) -> list[Example]:
    """Draw one residue pair per op (with replacement) and pack into 8 digits.

    Four-op tasks form a perfect matching, so every digit position is written.
    Single-op tasks fill the unused six positions uniformly from 0..63.
    Returns one Example per op, sharing the same digit context, each with its
    own query token and label.
    """
    if task.n_ops == 1:
        digits = [int(rng.integers(0, NUM_DIGITS)) for _ in range(SEQ_LEN_OPERANDS)]
    else:
        digits = [0] * SEQ_LEN_OPERANDS

    pairs_used: list[tuple[int, int]] = []
    for op in task.operations:
        pool = splits[op.modulus].get(split_name)  # type: ignore[arg-type]
        if not pool:
            raise ValueError(
                f"empty {split_name} residue-pair pool for modulus {op.modulus}"
            )
        pair = pool[int(rng.integers(0, len(pool)))]
        swap = bool(rng.integers(0, 2))
        raw_i, raw_j = sample_raw_operands(rng, pair, op.modulus, swap=swap)
        digits[op.i] = raw_i
        digits[op.j] = raw_j
        pairs_used.append(pair)

    examples: list[Example] = []
    for op, pair in zip(task.operations, pairs_used, strict=True):
        tokens = tuple(digits + [task.task_token, QUERY_TOKEN_IDS[op.slot]])
        label = (int(digits[op.i]) + int(digits[op.j])) % op.modulus
        examples.append(
            Example(
                tokens=tokens,
                label=label,
                task_name=task.name,
                slot=op.slot,
                modulus=op.modulus,
                residue_pair=pair,
                split=split_name,
            )
        )
    return examples


class PackedMultiOpTrainLoader:
    """Infinite train loader: each batch is ``n_packs`` packed contexts × n_ops.

    ``batch_size`` is the number of query examples per step and must be
    divisible by ``task.n_ops``.
    """

    def __init__(
        self,
        task: TaskSpec,
        splits: dict[int, ResiduePairSplit],
        *,
        batch_size: int,
        seed: int = 0,
        split_name: str = "train",
    ) -> None:
        n_ops = task.n_ops
        if batch_size < n_ops or batch_size % n_ops != 0:
            raise ValueError(
                f"batch_size={batch_size} must be a positive multiple of "
                f"n_ops={n_ops}"
            )
        self.task = task
        self.splits = splits
        self.batch_size = int(batch_size)
        self.n_ops = n_ops
        self.n_packs = self.batch_size // n_ops
        self.seed = int(seed)
        self.split_name = split_name
        self.dataset = _PackedLen(self.n_packs)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        while True:
            yield self._next_batch(rng)

    def _next_batch(self, rng: np.random.Generator) -> dict[str, torch.Tensor]:
        tokens: list[list[int]] = []
        labels: list[int] = []
        slots: list[int] = []
        moduli: list[int] = []
        for _ in range(self.n_packs):
            for ex in sample_packed_examples(
                rng, self.task, self.splits, split_name=self.split_name
            ):
                tokens.append(list(ex.tokens))
                labels.append(ex.label)
                slots.append(ex.slot)
                moduli.append(ex.modulus)
        tid = self.task.task_id
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "slots": torch.tensor(slots, dtype=torch.long),
            "moduli": torch.tensor(moduli, dtype=torch.long),
            "task_ids": torch.full((self.batch_size,), tid, dtype=torch.long),
        }


class _PackedLen:
    def __init__(self, n: int) -> None:
        self._n = n

    def __len__(self) -> int:
        return self._n


def make_packed_multi_op_train_loader(
    task: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    *,
    batch_size: int,
    seed: int = 0,
) -> PackedMultiOpTrainLoader:
    return PackedMultiOpTrainLoader(
        task, splits, batch_size=batch_size, seed=seed
    )


def is_packed_train_loader(loader: DataLoader | PackedMultiOpTrainLoader) -> bool:
    return isinstance(loader, PackedMultiOpTrainLoader)
