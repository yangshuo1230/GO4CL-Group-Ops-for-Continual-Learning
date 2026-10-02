"""PyTorch datasets and loaders for fixed modular-addition data."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, RandomSampler, WeightedRandomSampler

from go4cl.data.generate import Example, load_split_arrays

SplitName = Literal["train", "val", "test"]


class ModularAdditionDataset(Dataset):
    def __init__(
        self,
        tokens: np.ndarray,
        labels: np.ndarray,
        slots: np.ndarray | None = None,
        moduli: np.ndarray | None = None,
        task_ids: np.ndarray | None = None,
        latent_ids: np.ndarray | None = None,
    ) -> None:
        self.tokens = torch.as_tensor(tokens, dtype=torch.long)
        self.labels = torch.as_tensor(labels, dtype=torch.long)
        self.slots = (
            torch.as_tensor(slots, dtype=torch.long)
            if slots is not None
            else torch.full((len(labels),), -1, dtype=torch.long)
        )
        self.moduli = (
            torch.as_tensor(moduli, dtype=torch.long)
            if moduli is not None
            else torch.full((len(labels),), -1, dtype=torch.long)
        )
        self.task_ids = (
            torch.as_tensor(task_ids, dtype=torch.long)
            if task_ids is not None
            else torch.zeros(len(labels), dtype=torch.long)
        )
        self.latent_ids = (
            torch.as_tensor(latent_ids, dtype=torch.long)
            if latent_ids is not None
            else torch.full((len(labels),), -1, dtype=torch.long)
        )

    def __len__(self) -> int:
        return int(self.tokens.shape[0])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "tokens": self.tokens[idx],
            "labels": self.labels[idx],
            "slots": self.slots[idx],
            "moduli": self.moduli[idx],
            "task_ids": self.task_ids[idx],
            "latent_ids": self.latent_ids[idx],
        }

    @classmethod
    def from_examples(cls, examples: list[Example], task_id: int = 0) -> ModularAdditionDataset:
        tokens = np.asarray([e.tokens for e in examples], dtype=np.int64)
        labels = np.asarray([e.label for e in examples], dtype=np.int64)
        slots = np.asarray([e.slot for e in examples], dtype=np.int64)
        moduli = np.asarray([e.modulus for e in examples], dtype=np.int64)
        task_ids = np.asarray(
            [e.task_id if e.task_id >= 0 else task_id for e in examples],
            dtype=np.int64,
        )
        latent_ids = np.asarray([e.latent_id for e in examples], dtype=np.int64)
        return cls(tokens, labels, slots, moduli, task_ids, latent_ids)

    @classmethod
    def from_disk(
        cls, root: Path | str, task_name: str, split: SplitName, task_id: int
    ) -> ModularAdditionDataset:
        arrays = load_split_arrays(root, task_name, split)
        n = len(arrays["labels"])
        task_ids = np.full(n, task_id, dtype=np.int64)
        latent_ids = arrays.get("latent_ids")
        if latent_ids is None:
            latent_ids = np.full(n, -1, dtype=np.int64)
        return cls(
            arrays["tokens"],
            arrays["labels"],
            arrays["slots"],
            arrays["moduli"],
            task_ids,
            latent_ids,
        )


def concat_datasets(*datasets: ModularAdditionDataset) -> ModularAdditionDataset:
    return ModularAdditionDataset(
        tokens=torch.cat([d.tokens for d in datasets], dim=0).numpy(),
        labels=torch.cat([d.labels for d in datasets], dim=0).numpy(),
        slots=torch.cat([d.slots for d in datasets], dim=0).numpy(),
        moduli=torch.cat([d.moduli for d in datasets], dim=0).numpy(),
        task_ids=torch.cat([d.task_ids for d in datasets], dim=0).numpy(),
        latent_ids=torch.cat([d.latent_ids for d in datasets], dim=0).numpy(),
    )


def make_loader(
    dataset: Dataset,
    *,
    batch_size: int | None = None,
    shuffle: bool = True,
    num_workers: int = 0,
    drop_last: bool = False,
    replacement: bool = False,
) -> DataLoader:
    """
    Build a DataLoader.

    If ``replacement`` is True and ``batch_size`` is set, each batch is exactly
    ``batch_size`` examples drawn with replacement (even when ``batch_size`` >
    dataset size). Eval should leave ``replacement=False``.
    """
    n = len(dataset)  # type: ignore[arg-type]
    if batch_size is None or batch_size <= 0:
        bs = max(n, 1)
        replacement = False
    else:
        bs = max(int(batch_size), 1)

    if replacement:
        # One fixed-size batch per epoch; ``infinite_loader`` re-epochs each step.
        sampler = RandomSampler(dataset, replacement=True, num_samples=bs)
        return DataLoader(
            dataset,
            batch_size=bs,
            sampler=sampler,
            num_workers=num_workers,
            drop_last=False,
            pin_memory=torch.cuda.is_available(),
        )

    bs = min(bs, max(n, 1))
    return DataLoader(
        dataset,
        batch_size=bs,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        pin_memory=torch.cuda.is_available(),
    )


def make_balanced_joint_loader(
    ds_a: ModularAdditionDataset,
    ds_b: ModularAdditionDataset,
    *,
    batch_size: int | None = None,
    num_workers: int = 0,
) -> DataLoader:
    """Joint A+B loader. Full-batch uses the concatenated set once per step."""
    joint = concat_datasets(ds_a, ds_b)
    n = len(joint)
    # Full batch (default): one pass over all A∪B examples.
    if batch_size is None or batch_size <= 0 or batch_size >= n:
        return make_loader(joint, batch_size=n, shuffle=True, num_workers=num_workers)

    n_a, n_b = len(ds_a), len(ds_b)
    weights = torch.cat(
        [
            torch.full((n_a,), 0.5 / max(n_a, 1)),
            torch.full((n_b,), 0.5 / max(n_b, 1)),
        ]
    )
    sampler = WeightedRandomSampler(weights, num_samples=n, replacement=True)
    return DataLoader(
        joint,
        batch_size=int(batch_size),
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
