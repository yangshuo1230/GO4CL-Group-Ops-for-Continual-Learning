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
