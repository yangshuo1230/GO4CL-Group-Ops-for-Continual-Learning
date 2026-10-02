"""Dataset manifest persistence and hashing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.tasks.relations import TaskPairSpec
from go4cl.tasks.spec import TaskSpec


def stable_json_dumps(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"))


def hash_payload(obj: Any) -> str:
    blob = stable_json_dumps(obj).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


@dataclass
class DataManifest:
    """Fully describes a fixed dataset so it can be audited or rebuilt."""

    experiment_id: str
    data_seed: int
    task_pair: TaskPairSpec
    residue_splits: dict[int, ResiduePairSplit]  # modulus -> split
    n_aliases_per_pair: int
    n_nuisance_contexts: int
    samples_per_slot: dict[str, dict[str, int]]  # task -> split -> count
    generation_rule: str
    dataset_hash: str
    train_mode: str = "fixed"  # fixed | packed_online
    schema_version: int = 1
    protocol_version: int = 1
    train_context_mode: str = "nuisance_random"
    eval_context_modes: tuple[str, ...] = ("nuisance_random",)

    def to_dict(self) -> dict[str, Any]:
        return {
            "experiment_id": self.experiment_id,
            "data_seed": self.data_seed,
            "task_pair": self.task_pair.to_dict(),
            "residue_splits": {
                str(p): split.to_dict() for p, split in sorted(self.residue_splits.items())
            },
            "n_aliases_per_pair": self.n_aliases_per_pair,
            "n_nuisance_contexts": self.n_nuisance_contexts,
            "samples_per_slot": self.samples_per_slot,
            "generation_rule": self.generation_rule,
            "dataset_hash": self.dataset_hash,
            "train_mode": self.train_mode,
            "schema_version": self.schema_version,
            "protocol_version": self.protocol_version,
            "train_context_mode": self.train_context_mode,
            "eval_context_modes": list(self.eval_context_modes),
            "operation_keys": [
                k.to_dict()
                for t in (self.task_pair.task_a, self.task_pair.task_b)
                for k in t.operation_keys()
            ],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> DataManifest:
        splits = {
            int(p): ResiduePairSplit.from_dict(s) for p, s in d["residue_splits"].items()
        }
        return cls(
            experiment_id=str(d["experiment_id"]),
            data_seed=int(d["data_seed"]),
            task_pair=TaskPairSpec.from_dict(d["task_pair"]),
            residue_splits=splits,
            n_aliases_per_pair=int(d["n_aliases_per_pair"]),
            n_nuisance_contexts=int(d["n_nuisance_contexts"]),
            samples_per_slot=d["samples_per_slot"],
            generation_rule=str(d["generation_rule"]),
            dataset_hash=str(d["dataset_hash"]),
            train_mode=str(d.get("train_mode", "fixed")),
            schema_version=int(d.get("schema_version", 1)),
            protocol_version=int(d.get("protocol_version", 1)),
            train_context_mode=str(d.get("train_context_mode", "nuisance_random")),
            eval_context_modes=tuple(
                d.get("eval_context_modes", ("nuisance_random",))
            ),
        )

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(stable_json_dumps(self.to_dict()) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> DataManifest:
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def moduli_used_by_tasks(*tasks: TaskSpec) -> list[int]:
    mods: set[int] = set()
    for task in tasks:
        for op in task.operations:
            mods.add(op.modulus)
    return sorted(mods)


def assert_manifest_compatible(
    existing: DataManifest,
    *,
    expected_schema_version: int | None = None,
    expected_data_seed: int | None = None,
    expected_train_mode: str | None = None,
    expected_task_pair_hash: str | None = None,
) -> None:
    """Refuse silent reuse of incompatible data directories."""
    errors: list[str] = []
    if expected_schema_version is not None and existing.schema_version != expected_schema_version:
        errors.append(
            f"schema_version {existing.schema_version} != {expected_schema_version}"
        )
    if expected_data_seed is not None and existing.data_seed != expected_data_seed:
        errors.append(f"data_seed {existing.data_seed} != {expected_data_seed}")
    if expected_train_mode is not None and existing.train_mode != expected_train_mode:
        errors.append(f"train_mode {existing.train_mode} != {expected_train_mode}")
    if expected_task_pair_hash is not None:
        got = hash_payload(existing.task_pair.to_dict())
        if got != expected_task_pair_hash:
            errors.append("task_pair hash mismatch")
    if errors:
        raise ValueError("manifest reuse rejected: " + "; ".join(errors))
