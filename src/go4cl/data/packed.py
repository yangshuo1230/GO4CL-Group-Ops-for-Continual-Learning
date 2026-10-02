"""Online packed multi-op training: one shared 8-digit context, one query each."""

from __future__ import annotations

from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from go4cl.data.context import ContextBuilder
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.tasks.spec import TaskSpec


def sample_packed_examples(
    rng: np.random.Generator,
    task: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    *,
    split_name: str = "train",
):
    """Draw one residue pair per op and pack into 8 digits (via ContextBuilder)."""
    builder = ContextBuilder(task, splits)
    record = builder.sample_packed_digits(rng, default_split=split_name)  # type: ignore[arg-type]
    return builder.emit_all_queries(record, split=split_name)


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
        self.builder = ContextBuilder(task, splits)
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
        latent_ids: list[int] = []
        for _ in range(self.n_packs):
            for ex in self.builder.sample_train_pack(rng):
                tokens.append(list(ex.tokens))
                labels.append(ex.label)
                slots.append(ex.slot)
                moduli.append(ex.modulus)
                latent_ids.append(ex.latent_id)
        tid = self.task.task_id
        return {
            "tokens": torch.tensor(tokens, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "slots": torch.tensor(slots, dtype=torch.long),
            "moduli": torch.tensor(moduli, dtype=torch.long),
            "latent_ids": torch.tensor(latent_ids, dtype=torch.long),
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
