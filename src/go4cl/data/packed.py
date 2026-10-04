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
        return _torch_batch(self.builder.sample_train_batch(rng, self.n_packs))


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


def _torch_batch(arrays: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    return {
        key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in arrays.items()
    }


class BalancedPackedJointLoader:
    """One step = equal query counts from task A and task B (50/50).

    ``batch_size`` must be divisible by ``2 * n_ops`` so each task contributes
    the same number of packed contexts.
    """

    def __init__(
        self,
        task_a: TaskSpec,
        task_b: TaskSpec,
        splits: dict[int, ResiduePairSplit],
        *,
        batch_size: int,
        seed: int = 0,
    ) -> None:
        if task_a.n_ops != task_b.n_ops:
            raise ValueError(
                f"joint tasks must share n_ops, got {task_a.n_ops} and {task_b.n_ops}"
            )
        n_ops = task_a.n_ops
        if batch_size < 2 * n_ops or batch_size % (2 * n_ops) != 0:
            raise ValueError(
                f"joint batch_size={batch_size} must be a positive multiple of "
                f"2*n_ops={2 * n_ops} so each step is 50/50 A/B"
            )
        self.task_a = task_a
        self.task_b = task_b
        self.batch_size = int(batch_size)
        self.n_ops = n_ops
        self.n_packs_each = self.batch_size // (2 * n_ops)
        self.n_packs = self.n_packs_each * 2
        self.seed = int(seed)
        self._builder_a = ContextBuilder(task_a, splits)
        self._builder_b = ContextBuilder(task_b, splits)
        self.dataset = _PackedLen(self.n_packs)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng_a = np.random.default_rng(self.seed)
        rng_b = np.random.default_rng(self.seed + 10_007)
        while True:
            yield self._next_batch(rng_a, rng_b)

    def _next_batch(
        self, rng_a: np.random.Generator, rng_b: np.random.Generator
    ) -> dict[str, torch.Tensor]:
        part_a = self._builder_a.sample_train_batch(rng_a, self.n_packs_each)
        part_b = self._builder_b.sample_train_batch(rng_b, self.n_packs_each)
        merged = {
            key: np.concatenate([part_a[key], part_b[key]], axis=0) for key in part_a
        }
        return _torch_batch(merged)


class AlternatingTaskLoader:
    """Yield a full batch of A, then B, then A, ... (interleaved protocol)."""

    def __init__(self, loader_a, loader_b) -> None:
        self.loader_a = loader_a
        self.loader_b = loader_b
        self.batch_size = getattr(loader_a, "batch_size", None)
        n_a = getattr(loader_a, "n_packs", None)
        n_b = getattr(loader_b, "n_packs", None)
        self.n_packs = n_a if n_a is not None else n_b
        n = n_a if isinstance(n_a, int) else 1
        self.dataset = getattr(loader_a, "dataset", _PackedLen(n))

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        it_a = iter(self.loader_a)
        it_b = iter(self.loader_b)
        while True:
            yield next(it_a)
            yield next(it_b)
