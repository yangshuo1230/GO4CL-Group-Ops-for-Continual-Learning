"""Shared residue splits and equal-task packed loaders for A/B/C."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import DataLoader

from go4cl.data.context import ContextBuilder
from go4cl.data.eval_contexts import make_eval_context_loader
from go4cl.data.generate import build_shared_residue_splits
from go4cl.data.manifest import hash_payload
from go4cl.data.packed import _torch_batch, make_packed_multi_op_train_loader
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.phases.common import ratios_from_train_frac
from go4cl.phases.task_partition.config import PartitionConfig
from go4cl.tasks.partition import PARTITION_ORDER, build_partition_tasks, task_descriptions
from go4cl.tasks.spec import TaskSpec
from go4cl.utils.checkpoint import write_json


class EqualPackedJointLoader:
    """One step = the same number of packed queries from each task.

    Generalizes ``BalancedPackedJointLoader`` from two tasks to N tasks.
    Sampling goes through ``ContextBuilder.sample_train_batch``.
    """

    def __init__(
        self,
        tasks: list[TaskSpec],
        splits: dict[int, ResiduePairSplit],
        *,
        batch_size: int,
        seed: int,
    ) -> None:
        if len(tasks) < 2:
            raise ValueError("joint loader needs at least two tasks")
        n_ops = tasks[0].n_ops
        if any(task.n_ops != n_ops for task in tasks):
            raise ValueError("joint tasks must share n_ops")
        unit = len(tasks) * n_ops
        if batch_size < unit or batch_size % unit != 0:
            raise ValueError(
                f"batch_size={batch_size} must be a positive multiple of "
                f"{len(tasks)}*n_ops={unit}"
            )
        self.tasks = list(tasks)
        self.batch_size = int(batch_size)
        self.n_ops = n_ops
        self.n_packs_each = self.batch_size // unit
        self.seed = int(seed)
        self._builders = [ContextBuilder(task, splits) for task in self.tasks]
        self._seeds = [self.seed + 10_007 * int(task.task_id) for task in self.tasks]

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rngs = [np.random.default_rng(seed) for seed in self._seeds]
        while True:
            parts = [
                builder.sample_train_batch(rng, self.n_packs_each)
                for builder, rng in zip(self._builders, rngs, strict=True)
            ]
            merged = {
                key: np.concatenate([part[key] for part in parts], axis=0)
                for key in parts[0]
            }
            yield _torch_batch(merged)


def prepare_dataset(out_dir: Path, cfg: PartitionConfig) -> dict[str, object]:
    """One residue split per modulus, shared by A, B, and C."""
    tasks = build_partition_tasks()
    ratios = ratios_from_train_frac(float(cfg.train_frac))
    splits = build_shared_residue_splits(
        [tasks[name] for name in PARTITION_ORDER],
        data_seed=int(cfg.data_seed),
        ratios=ratios,
    )
    split_sizes = {
        str(modulus): {
            "train": len(split.train),
            "val": len(split.val),
            "test": len(split.test),
        }
        for modulus, split in sorted(splits.items())
    }
    body = {
        "experiment": "task_partition",
        "data_seed": int(cfg.data_seed),
        "train_frac": float(cfg.train_frac),
        "ratios": list(ratios),
        "train_mode": "packed_online",
        "eval_context_mode": "packed_id",
        "tasks": {name: tasks[name].to_dict() for name in PARTITION_ORDER},
        "operations": task_descriptions(tasks),
        "residue_splits": {
            str(modulus): split.to_dict() for modulus, split in sorted(splits.items())
        },
        "split_sizes": split_sizes,
    }
    body["dataset_hash"] = hash_payload(
        {key: value for key, value in body.items() if key != "dataset_hash"}
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "manifest.json"
    write_json(path, body)
    return {
        "tasks": tasks,
        "splits": splits,
        "manifest_path": path,
        "dataset_hash": body["dataset_hash"],
        "split_sizes": split_sizes,
        "ratios": list(ratios),
        "operations": body["operations"],
    }


def build_eval_loaders(
    tasks: dict[str, TaskSpec],
    splits: dict[int, ResiduePairSplit],
    cfg: PartitionConfig,
) -> tuple[dict[str, dict[str, DataLoader]], dict[str, DataLoader]]:
    """Packed-id val/test loaders. Flat dict is what ``train_steps`` consumes."""
    nested: dict[str, dict[str, DataLoader]] = {}
    flat: dict[str, DataLoader] = {}
    for name in PARTITION_ORDER:
        task = tasks[name]
        nested[name] = {}
        for split_name, bit in (("val", 0), ("test", 1)):
            loader = make_eval_context_loader(
                task,
                splits,
                target_split=split_name,  # type: ignore[arg-type]
                context_mode="packed_id",
                distractor_split="train",
                n_per_operation=int(cfg.eval_n_per_operation),
                seed=int(cfg.eval_seed) + 1009 * int(task.task_id) + bit,
            )
            nested[name][split_name] = loader
            flat[f"{name}_{split_name}"] = loader
    return nested, flat


def make_c_only_loader(
    task_c: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    cfg: PartitionConfig,
):
    """C queries only. A and B digits are not drawn in this stream."""
    return make_packed_multi_op_train_loader(
        task_c,
        splits,
        batch_size=int(cfg.batch_size),
        seed=cfg.phase_sampler_seed("c_only"),
    )


def peek_task_ids(loader) -> list[int]:
    """One batch, then the training iterator starts again from the same seed."""
    batch = next(iter(loader))
    return [int(task_id) for task_id in batch["task_ids"].tolist()]
