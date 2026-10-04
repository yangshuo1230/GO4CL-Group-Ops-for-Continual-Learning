"""Build per-task train/eval DataLoaders for a dataset root.

Packed train is a replayable online stream (query examples per step).
Primary val/test for packed manifests is packed_id; disk splits are
nuisance_random controls. Checkpoint selection uses val only.
"""

from __future__ import annotations

from pathlib import Path

from torch.utils.data import DataLoader

from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.data.manifest import DataManifest
from go4cl.data.packed import make_packed_multi_op_train_loader


def task_loaders(
    data_root: Path,
    *,
    batch_size: int | None,
    train_replacement: bool = False,
    train_seed: int = 0,
    sampler_seed: int | None = None,
    eval_n_per_operation: int = 256,
    eval_seed: int = 0,
) -> dict[str, dict[str, DataLoader]]:
    """Build train/val/test loaders.

    For ``packed_online`` manifests:
      - train: replayable packed stream
      - val/test: **packed_id** (primary; distractors from train pool)
      - val_nuisance / test_nuisance: on-disk nuisance_random control
      - train_eval: fixed packed_id analysis set from train pools (t_mem)
      - iid: packed_id with all ops from train pools (t_iid)
    """
    from go4cl.data.context import build_analysis_dataset
    from go4cl.data.eval_contexts import make_eval_context_loader

    manifest_path = data_root / "manifest.json"
    manifest = DataManifest.load(manifest_path) if manifest_path.is_file() else None
    packed = bool(manifest and manifest.train_mode == "packed_online")
    stream_seed = int(sampler_seed if sampler_seed is not None else train_seed)

    out: dict[str, dict[str, DataLoader]] = {}
    for task_name, task_id in (("A", 0), ("B", 1)):
        out[task_name] = {}
        for split in ("train", "val", "test"):
            if packed and split == "train":
                continue
            ds = ModularAdditionDataset.from_disk(data_root, task_name, split, task_id)  # type: ignore[arg-type]
            is_train = split == "train"
            out[task_name][split] = make_loader(
                ds,
                batch_size=batch_size if is_train else None,
                shuffle=is_train and not (train_replacement and batch_size),
                replacement=bool(is_train and train_replacement and batch_size),
            )
        if packed:
            assert manifest is not None
            if batch_size is None or batch_size <= 0:
                raise ValueError("packed_online train requires a positive batch_size")
            task = (
                manifest.task_pair.task_a
                if task_name == "A"
                else manifest.task_pair.task_b
            )
            out[task_name]["train"] = make_packed_multi_op_train_loader(
                task,
                manifest.residue_splits,
                batch_size=int(batch_size),
                seed=int(stream_seed) + 10_007 * int(task_id),
            )
            # Keep disk splits as nuisance controls
            out[task_name]["val_nuisance"] = out[task_name]["val"]
            out[task_name]["test_nuisance"] = out[task_name]["test"]
            # Primary eval = packed_id
            for split_name, key in (("val", "val"), ("test", "test")):
                out[task_name][key] = make_eval_context_loader(
                    task,
                    manifest.residue_splits,
                    target_split=split_name,  # type: ignore[arg-type]
                    context_mode="packed_id",
                    distractor_split="train",
                    n_per_operation=int(eval_n_per_operation),
                    seed=int(eval_seed) + 1009 * task_id + (0 if split_name == "val" else 1),
                )
            # Fixed train-eval for t_mem (not minibatch)
            train_eval_ex = build_analysis_dataset(
                task,
                manifest.residue_splits,
                split="train",
                context_mode="packed_id",
                analysis_seed=int(eval_seed) + 17 + task_id,
                aliases_per_pair=2,
                contexts_per_pair=1,
            )
            out[task_name]["train_eval"] = make_loader(
                ModularAdditionDataset.from_examples(train_eval_ex, task_id=task_id),
                batch_size=None,
                shuffle=False,
            )
            # t_iid: all ops from train pool, packed_id
            out[task_name]["iid"] = make_eval_context_loader(
                task,
                manifest.residue_splits,
                target_split="train",
                context_mode="packed_id",
                distractor_split="train",
                n_per_operation=int(eval_n_per_operation),
                seed=int(eval_seed) + 409 + task_id,
            )
    return out



_task_loaders = task_loaders
