"""Eval context loaders: packed_id (primary) and nuisance_random (control)."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from go4cl.data.context import ContextBuilder, ContextRequest
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.tasks.spec import OperationKey, TaskSpec

ContextMode = Literal["packed_id", "nuisance_random"]
SplitName = Literal["train", "val", "test"]


class EvalContextDataset(Dataset):
    """Fixed-size synthetic eval set built by ContextBuilder (not disk .npz)."""

    def __init__(self, examples: list) -> None:
        self._examples = examples

    def __len__(self) -> int:
        return len(self._examples)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self._examples[idx]
        return {
            "tokens": torch.tensor(ex.tokens, dtype=torch.long),
            "labels": torch.tensor(ex.label, dtype=torch.long),
            "slots": torch.tensor(ex.slot, dtype=torch.long),
            "moduli": torch.tensor(ex.modulus, dtype=torch.long),
            "latent_ids": torch.tensor(ex.latent_id, dtype=torch.long),
            "task_ids": torch.tensor(ex.task_id, dtype=torch.long),
        }


def build_eval_examples(
    task: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    *,
    target_split: SplitName,
    context_mode: ContextMode = "packed_id",
    distractor_split: SplitName = "train",
    n_per_operation: int = 256,
    seed: int = 0,
) -> list:
    """Sample ``n_per_operation`` examples per op under the requested context mode."""
    builder = ContextBuilder(task, splits)
    rng = np.random.default_rng(int(seed))
    out = []
    for op in task.operations:
        key = OperationKey.from_operation(op, task_id=task.task_id)
        req = ContextRequest(
            target_operation=key,
            target_split=target_split,
            distractor_split=distractor_split,
            context_mode=context_mode,
        )
        for _ in range(int(n_per_operation)):
            out.append(builder.sample_eval_for_target(rng, req))
    return out


def make_eval_context_loader(
    task: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    *,
    target_split: SplitName,
    context_mode: ContextMode = "packed_id",
    distractor_split: SplitName = "train",
    n_per_operation: int = 256,
    seed: int = 0,
    batch_size: int | None = None,
) -> DataLoader:
    examples = build_eval_examples(
        task,
        splits,
        target_split=target_split,
        context_mode=context_mode,
        distractor_split=distractor_split,
        n_per_operation=n_per_operation,
        seed=seed,
    )
    ds = EvalContextDataset(examples)
    bs = batch_size if batch_size and batch_size > 0 else max(len(ds), 1)
    return DataLoader(ds, batch_size=bs, shuffle=False)
