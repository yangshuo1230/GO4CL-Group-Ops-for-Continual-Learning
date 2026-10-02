"""Fixed dataset generation from residue-pair splits."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np

from go4cl.constants import (
    CONTEXT_LENGTH,
    NUM_DIGITS,
    QUERY_TOKEN_IDS,
    SEQ_LEN_OPERANDS,
)
from go4cl.data.manifest import DataManifest, hash_payload, moduli_used_by_tasks
from go4cl.data.residue_pairs import (
    ResiduePairSplit,
    stratified_residue_pair_split,
    stratified_residue_pair_split_fixed_train,
)
from go4cl.tasks.relations import TaskPairSpec
from go4cl.tasks.spec import Operation, TaskSpec

SplitName = Literal["train", "val", "test"]
GENERATION_RULE = (
    "sample unordered residue pair from split; expand to raw aliases "
    "x_i ≡ r1, x_j ≡ r2 (mod p) with x in 0..63; fill remaining six "
    "positions uniformly in 0..63; emit one example per query slot."
)


@dataclass(frozen=True)
class Example:
    tokens: tuple[int, ...]  # length 10
    label: int
    task_name: str
    slot: int
    modulus: int
    residue_pair: tuple[int, int]
    split: str
    latent_id: int = -1
    task_id: int = -1

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens": list(self.tokens),
            "label": self.label,
            "task_name": self.task_name,
            "slot": self.slot,
            "modulus": self.modulus,
            "residue_pair": list(self.residue_pair),
            "split": self.split,
            "latent_id": self.latent_id,
            "task_id": self.task_id,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(
            tokens=tuple(int(t) for t in d["tokens"]),
            label=int(d["label"]),
            task_name=str(d["task_name"]),
            slot=int(d["slot"]),
            modulus=int(d["modulus"]),
            residue_pair=(int(d["residue_pair"][0]), int(d["residue_pair"][1])),
            split=str(d["split"]),
            latent_id=int(d.get("latent_id", -1)),
            task_id=int(d.get("task_id", -1)),
        )


def aliases_for_residue(residue: int, modulus: int) -> list[int]:
    return [x for x in range(NUM_DIGITS) if x % modulus == residue]


def sample_raw_operands(
    rng: np.random.Generator,
    pair: tuple[int, int],
    modulus: int,
    *,
    swap: bool,
) -> tuple[int, int]:
    r1, r2 = pair
    if swap:
        r1, r2 = r2, r1
    a_opts = aliases_for_residue(r1, modulus)
    b_opts = aliases_for_residue(r2, modulus)
    return int(rng.choice(a_opts)), int(rng.choice(b_opts))


def build_sequence(
    rng: np.random.Generator,
    op: Operation,
    task: TaskSpec,
    raw_i: int,
    raw_j: int,
) -> tuple[int, ...]:
    x = [int(rng.integers(0, NUM_DIGITS)) for _ in range(SEQ_LEN_OPERANDS)]
    x[op.i] = raw_i
    x[op.j] = raw_j
    tokens = tuple(x + [task.task_token, QUERY_TOKEN_IDS[op.slot]])
    return tokens


def generate_examples_for_op(
    rng: np.random.Generator,
    task: TaskSpec,
    op: Operation,
    split: ResiduePairSplit,
    split_name: SplitName,
    *,
    n_aliases_per_pair: int,
    n_nuisance_contexts: int,
) -> list[Example]:
    examples: list[Example] = []
    pairs = split.get(split_name)
    for pair in pairs:
        for _ in range(n_aliases_per_pair):
            swap = bool(rng.integers(0, 2))
            raw_i, raw_j = sample_raw_operands(rng, pair, op.modulus, swap=swap)
            for _ctx in range(n_nuisance_contexts):
                tokens = build_sequence(rng, op, task, raw_i, raw_j)
                label = (raw_i + raw_j) % op.modulus
                assert label == op.evaluate(tokens[:SEQ_LEN_OPERANDS])
                examples.append(
                    Example(
                        tokens=tokens,
                        label=label,
                        task_name=task.name,
                        slot=op.slot,
                        modulus=op.modulus,
                        residue_pair=pair,
                        split=split_name,
                        latent_id=op.latent_id,
                        task_id=task.task_id,
                    )
                )
    return examples


def build_shared_residue_splits(
    tasks: list[TaskSpec],
    *,
    data_seed: int,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    n_train_pairs: int | None = None,
) -> dict[int, ResiduePairSplit]:
    """One split per modulus; shared across all ops that use that modulus."""
    mods = moduli_used_by_tasks(*tasks)
    if n_train_pairs is not None:
        return {
            p: stratified_residue_pair_split_fixed_train(
                p, n_train_pairs=n_train_pairs, data_seed=data_seed + p
            )
            for p in mods
        }
    return {
        p: stratified_residue_pair_split(p, ratios=ratios, data_seed=data_seed + p)
        for p in mods
    }


def generate_task_datasets(
    task_pair: TaskPairSpec,
    *,
    data_seed: int,
    n_aliases_per_pair: int = 16,
    n_nuisance_contexts: int = 4,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    n_train_pairs: int | None = None,
    experiment_id: str = "default",
    skip_train: bool = False,
) -> tuple[DataManifest, dict[str, dict[str, list[Example]]]]:
    """
    Returns (manifest, datasets) where datasets[task_name][split] = examples.

    If ``n_train_pairs`` is set, use a fixed train residue-pair count (Phase 1A);
    otherwise split by ``ratios``.

    ``skip_train=True`` writes empty train lists (Phase 1B packed online train);
    residue splits are still stored on the manifest.
    """
    tasks = [task_pair.task_a, task_pair.task_b]
    splits = build_shared_residue_splits(
        tasks,
        data_seed=data_seed,
        ratios=ratios,
        n_train_pairs=n_train_pairs,
    )
    rng = np.random.default_rng(data_seed)

    datasets: dict[str, dict[str, list[Example]]] = {
        t.name: {"train": [], "val": [], "test": []} for t in tasks
    }
    counts: dict[str, dict[str, int]] = {
        t.name: {"train": 0, "val": 0, "test": 0} for t in tasks
    }

    for task in tasks:
        for op in task.operations:
            split = splits[op.modulus]
            for split_name in ("train", "val", "test"):
                if skip_train and split_name == "train":
                    continue
                exs = generate_examples_for_op(
                    rng,
                    task,
                    op,
                    split,
                    split_name,  # type: ignore[arg-type]
                    n_aliases_per_pair=n_aliases_per_pair,
                    n_nuisance_contexts=n_nuisance_contexts,
                )
                datasets[task.name][split_name].extend(exs)
                counts[task.name][split_name] += len(exs)

    # Hash over all example payloads for auditability
    hash_body = {
        "task_pair": task_pair.to_dict(),
        "splits": {str(k): v.to_dict() for k, v in splits.items()},
        "examples": {
            t: {s: [e.to_dict() for e in datasets[t][s]] for s in ("train", "val", "test")}
            for t in datasets
        },
        "n_aliases_per_pair": n_aliases_per_pair,
        "n_nuisance_contexts": n_nuisance_contexts,
        "n_train_pairs": n_train_pairs,
        "data_seed": data_seed,
        "skip_train": skip_train,
    }
    train_mode = "packed_online" if skip_train else "fixed"
    if train_mode == "packed_online":
        schema_version = 2
        protocol_version = 2
        train_context_mode = "packed"
        eval_context_modes = ("packed_id", "nuisance_random")
    else:
        schema_version = 1
        protocol_version = 1
        train_context_mode = "nuisance_random"
        eval_context_modes = ("nuisance_random",)
    manifest = DataManifest(
        experiment_id=experiment_id,
        data_seed=data_seed,
        task_pair=task_pair,
        residue_splits=splits,
        n_aliases_per_pair=n_aliases_per_pair,
        n_nuisance_contexts=n_nuisance_contexts,
        samples_per_slot=counts,
        generation_rule=GENERATION_RULE,
        dataset_hash=hash_payload(hash_body),
        train_mode=train_mode,
        schema_version=schema_version,
        protocol_version=protocol_version,
        train_context_mode=train_context_mode,
        eval_context_modes=eval_context_modes,
    )
    return manifest, datasets


def save_datasets(
    root: Path | str,
    manifest: DataManifest,
    datasets: dict[str, dict[str, list[Example]]],
) -> Path:
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    manifest.save(root / "manifest.json")
    for task_name, splits in datasets.items():
        for split_name, examples in splits.items():
            path = root / task_name / f"{split_name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            payload = [e.to_dict() for e in examples]
            if payload:
                tokens = np.asarray([e["tokens"] for e in payload], dtype=np.int64)
                labels = np.asarray([e["label"] for e in payload], dtype=np.int64)
                slots = np.asarray([e["slot"] for e in payload], dtype=np.int64)
                moduli = np.asarray([e["modulus"] for e in payload], dtype=np.int64)
                latent_ids = np.asarray(
                    [e.get("latent_id", -1) for e in payload], dtype=np.int64
                )
            else:
                tokens = np.zeros((0, CONTEXT_LENGTH), dtype=np.int64)
                labels = np.zeros((0,), dtype=np.int64)
                slots = np.zeros((0,), dtype=np.int64)
                moduli = np.zeros((0,), dtype=np.int64)
                latent_ids = np.zeros((0,), dtype=np.int64)
            np.savez_compressed(
                path,
                tokens=tokens,
                labels=labels,
                slots=slots,
                moduli=moduli,
                latent_ids=latent_ids,
                meta=np.asarray(payload, dtype=object),
            )
    return root


def load_split_arrays(root: Path | str, task_name: str, split: SplitName) -> dict[str, np.ndarray]:
    path = Path(root) / task_name / f"{split}.npz"
    data = np.load(path, allow_pickle=True)
    out: dict[str, np.ndarray] = {
        "tokens": data["tokens"],
        "labels": data["labels"],
        "slots": data["slots"],
        "moduli": data["moduli"],
    }
    if "latent_ids" in data.files:
        out["latent_ids"] = data["latent_ids"]
    elif "meta" in data.files and len(data["meta"]):
        meta = data["meta"]
        out["latent_ids"] = np.asarray(
            [int(m.get("latent_id", -1)) if isinstance(m, dict) else -1 for m in meta],
            dtype=np.int64,
        )
    else:
        out["latent_ids"] = np.full(len(out["labels"]), -1, dtype=np.int64)
    return out
