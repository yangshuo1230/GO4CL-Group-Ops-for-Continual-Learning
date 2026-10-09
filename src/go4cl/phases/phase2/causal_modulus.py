"""Fixed-B modulus causal grid.

B is held fixed while A's target operation either keeps that modulus or
swaps it for a prime that does not appear in B. The other three A operations
stay identical, so modulus sharing is not confounded with B's modulus difficulty.

Formal size: 2 targets × 3 protocols × 2 constructions × 3 model seeds = 36.
B-only jobs are keyed by the B task, so same_p and different_p do not each
train B from scratch. A checkpoints are keyed by the A task and model seed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from go4cl.constants import LATENT_OPS, PRIMES
from go4cl.data.generate import build_shared_residue_splits
from go4cl.data.manifest import hash_payload
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.tasks.relations import TaskPairSpec, compute_overlaps
from go4cl.tasks.spec import Operation, TaskSpec, perfect_matchings

TARGET_MODULI: tuple[int, ...] = (23, 41)
CONSTRUCTIONS: tuple[int, ...] = (0, 1)
PROTOCOLS: tuple[str, ...] = ("b_only", "same_p_to_b", "different_p_to_b")
MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)
MASTERY_THRESHOLD = 0.9

# Positive delta_auc means sequential is above B-only. Defined here so the
# causal metrics module and the existing-result analysis share one sign.
N_MAIN_RUNS = (
    len(TARGET_MODULI) * len(PROTOCOLS) * len(CONSTRUCTIONS) * len(MODEL_SEEDS)
)


def _sorted_edges(matching: tuple) -> tuple:
    return tuple(sorted(matching, key=lambda edge: (min(edge), max(edge))))


def _slots(construction: int) -> tuple[list[int], list[int]]:
    """B slots, then A slots. Latent 0's A slot is never B's slot."""
    if int(construction) % 2 == 0:
        b_slots = [0, 1, 2, 3]
    else:
        b_slots = [2, 3, 0, 1]
    a_slots = [(slot + 1) % 4 for slot in b_slots]
    if len(set(a_slots)) != 4 or a_slots[0] == b_slots[0]:
        raise RuntimeError("A/B slot assignment collided on the target operation")
    return b_slots, a_slots


def _moduli(target: int, construction: int) -> dict[str, list[int]]:
    if int(target) not in PRIMES:
        raise ValueError(f"target modulus {target} is not in PRIMES")
    others = [prime for prime in PRIMES if prime != int(target)]
    shift = (int(construction) * 3) % len(others)
    rotated = others[shift:] + others[:shift]
    b_rest = rotated[:3]
    a_rest = rotated[3:6]
    replacement = rotated[6]
    b_mods = [int(target), *b_rest]
    if replacement in b_mods or replacement in a_rest or replacement == int(target):
        raise RuntimeError("replacement modulus collides with B or the other A ops")
    if set(a_rest) & set(b_mods):
        raise RuntimeError("A's non-target moduli overlap B")
    return {
        "b": b_mods,
        "a_same": [int(target), *a_rest],
        "a_diff": [int(replacement), *a_rest],
        "replacement": [int(replacement)],
    }


def _ops(edges: tuple, mods: list[int], slots: list[int]) -> tuple[Operation, ...]:
    ops: list[Operation] = []
    for latent in LATENT_OPS:
        i, j = sorted(edges[latent])
        ops.append(
            Operation(
                latent_id=int(latent),
                i=int(i),
                j=int(j),
                modulus=int(mods[latent]),
                slot=int(slots[latent]),
            )
        )
    return tuple(ops)


def _pair(task_a: TaskSpec, task_b: TaskSpec, *, target: int, construction: int, variant: str) -> TaskPairSpec:
    rho_slot, rho_operand, rho_mod = compute_overlaps(task_a, task_b)
    return TaskPairSpec(
        task_a=task_a,
        task_b=task_b,
        rho_slot=float(rho_slot),
        rho_operand=float(rho_operand),
        rho_mod=float(rho_mod),
        task_seed=int(construction),
        pair_id=f"causal_p{target}_c{construction}_{variant}",
    )


@dataclass(frozen=True)
class CausalTasks:
    target: int
    construction: int
    same: TaskPairSpec
    different: TaskPairSpec
    replacement_modulus: int

    @property
    def task_b(self) -> TaskSpec:
        return self.same.task_b


def build_causal_tasks(target: int, construction: int) -> CausalTasks:
    """One B, plus same-modulus and different-modulus versions of A."""
    mods = _moduli(target, construction)
    matchings = perfect_matchings()
    b_edges = _sorted_edges(matchings[int(construction) % len(matchings)])
    a_edges = None
    for offset in range(1, len(matchings) + 1):
        cand = _sorted_edges(matchings[(int(construction) + offset) % len(matchings)])
        if cand[0] != b_edges[0]:
            a_edges = cand
            break
    if a_edges is None:
        raise RuntimeError("no A matching with a different target-operation edge")
    b_slots, a_slots = _slots(construction)
    task_b = TaskSpec(name="B", task_id=1, operations=_ops(b_edges, mods["b"], b_slots))
    task_same = TaskSpec(
        name="A", task_id=0, operations=_ops(a_edges, mods["a_same"], a_slots)
    )
    task_diff = TaskSpec(
        name="A", task_id=0, operations=_ops(a_edges, mods["a_diff"], a_slots)
    )
    built = CausalTasks(
        target=int(target),
        construction=int(construction),
        same=_pair(task_same, task_b, target=target, construction=construction, variant="same"),
        different=_pair(
            task_diff, task_b, target=target, construction=construction, variant="different"
        ),
        replacement_modulus=int(mods["replacement"][0]),
    )
    validate_causal_tasks(built)
    return built


def validate_causal_tasks(built: CausalTasks) -> None:
    """Raise if the same/different contrast is not isolated to one modulus."""
    same_ops = built.same.task_a.by_latent()
    diff_ops = built.different.task_a.by_latent()
    b_ops = built.task_b.by_latent()
    if built.same.task_b.to_dict() != built.different.task_b.to_dict():
        raise RuntimeError("same_p and different_p do not share B")
    target_op = b_ops[0]
    if int(target_op.modulus) != int(built.target):
        raise RuntimeError("B latent 0 is not the target modulus")
    if int(same_ops[0].modulus) != int(built.target):
        raise RuntimeError("same_p A does not contain the target modulus")
    if same_ops[0].operand_pair == target_op.operand_pair:
        raise RuntimeError("same_p target operation reuses B's operands")
    if int(same_ops[0].slot) == int(target_op.slot):
        raise RuntimeError("same_p target operation reuses B's output slot")
    if int(diff_ops[0].modulus) == int(built.target):
        raise RuntimeError("different_p still uses the target modulus")
    if int(diff_ops[0].modulus) in {int(op.modulus) for op in built.task_b.operations}:
        raise RuntimeError("different_p modulus collides with B")
    for latent in (1, 2, 3):
        if same_ops[latent].to_dict() != diff_ops[latent].to_dict():
            raise RuntimeError(f"A operation {latent} differs between same and different")
    if {int(op.modulus) for op in same_ops.values()} & {
        int(op.modulus) for op in b_ops.values()
    } != {int(built.target)}:
        raise RuntimeError("same_p shares a modulus with B other than the target")


def task_hash(task: TaskSpec) -> str:
    return hash_payload(task.to_dict())


def split_hash(splits: dict[int, ResiduePairSplit]) -> str:
    payload = {str(mod): split.to_dict() for mod, split in sorted(splits.items())}
    return hash_payload(payload)


def b_dataset_hash(task_b: TaskSpec, splits: dict[int, ResiduePairSplit], data_seed: int) -> str:
    """Identity of packed B: the task, the shared residue splits, and data_seed.

    Train rows are sampled online, so this hash is the distribution identity
    rather than a hash of an expanded example list.
    """
    return hash_payload(
        {
            "task_b": task_b.to_dict(),
            "splits": {str(mod): split.to_dict() for mod, split in sorted(splits.items())},
            "data_seed": int(data_seed),
        }
    )


def residue_splits_for(built: CausalTasks, *, data_seed: int) -> dict[int, ResiduePairSplit]:
    """Splits from the union of same-A and B, reused for the different-A pair.

    different-A adds one modulus. Those extra pairs are split with the same
    seed rule; B's moduli are unchanged, so B's split hash stays the same when
    recomputed on the B-only modulus set.
    """
    return build_shared_residue_splits(
        [built.same.task_a, built.different.task_a, built.task_b],
        data_seed=int(data_seed),
        ratios=(0.8, 0.1, 0.1),
    )


def b_split_hash(built: CausalTasks, splits: dict[int, ResiduePairSplit]) -> str:
    b_mods = {int(op.modulus) for op in built.task_b.operations}
    return split_hash({mod: splits[mod] for mod in sorted(b_mods)})


@dataclass(frozen=True)
class CausalJob:
    job_id: str
    protocol: str
    target_modulus: int
    construction: int
    model_seed: int
    b_task_hash: str
    a_task_hash: str
    b_only_key: str
    a_source_key: str | None
    pair_id: str
    replacement_modulus: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "protocol": self.protocol,
            "target_modulus": self.target_modulus,
            "construction": self.construction,
            "model_seed": self.model_seed,
            "b_task_hash": self.b_task_hash,
            "a_task_hash": self.a_task_hash,
            "b_only_key": self.b_only_key,
            "a_source_key": self.a_source_key,
            "pair_id": self.pair_id,
            "replacement_modulus": self.replacement_modulus,
            "optimizer_transition": "fresh",
            "replay_ratio": 0.0,
            "mastery_threshold": MASTERY_THRESHOLD,
            "metric": "target operation accuracy, not overall B accuracy",
        }


def plan_causal_jobs(
    *,
    targets: tuple[int, ...] = TARGET_MODULI,
    constructions: tuple[int, ...] = CONSTRUCTIONS,
    protocols: tuple[str, ...] = PROTOCOLS,
    model_seeds: tuple[int, ...] = MODEL_SEEDS,
) -> list[CausalJob]:
    jobs: list[CausalJob] = []
    seen_b: set[str] = set()
    for target in targets:
        for construction in constructions:
            built = build_causal_tasks(int(target), int(construction))
            b_hash = task_hash(built.task_b)
            for protocol in protocols:
                pair = built.same if protocol != "different_p_to_b" else built.different
                a_hash = task_hash(pair.task_a)
                for model_seed in model_seeds:
                    b_key = f"{b_hash}_ms{int(model_seed)}"
                    a_key = None if protocol == "b_only" else f"{a_hash}_ms{int(model_seed)}"
                    if protocol == "b_only":
                        if b_key in seen_b:
                            continue
                        seen_b.add(b_key)
                    jobs.append(
                        CausalJob(
                            job_id=(
                                f"{protocol}_p{int(target)}_c{int(construction)}"
                                f"_ms{int(model_seed)}_optfresh"
                            ),
                            protocol=protocol,
                            target_modulus=int(target),
                            construction=int(construction),
                            model_seed=int(model_seed),
                            b_task_hash=b_hash,
                            a_task_hash=a_hash,
                            b_only_key=b_key,
                            a_source_key=a_key,
                            pair_id=pair.pair_id,
                            replacement_modulus=built.replacement_modulus,
                        )
                    )
    return jobs


def job_count_report(jobs: list[CausalJob]) -> dict[str, Any]:
    b_only = [job for job in jobs if job.protocol == "b_only"]
    a_keys = {job.a_source_key for job in jobs if job.a_source_key}
    b_keys = {job.b_only_key for job in b_only}
    by_protocol: dict[str, int] = {}
    for job in jobs:
        by_protocol[job.protocol] = by_protocol.get(job.protocol, 0) + 1
    return {
        "n_main_runs": len(jobs),
        "expected_main_runs": N_MAIN_RUNS,
        "by_protocol": by_protocol,
        "n_b_only": len(b_only),
        "n_unique_b_only": len(b_keys),
        "n_a_sources": len(a_keys),
        "b_only_deduped": len(b_only) == len(b_keys),
        "a_sources_deduped": len(a_keys) == len({job.a_source_key for job in jobs if job.a_source_key}),
    }


def a_mastery_spec(task_a: TaskSpec, threshold: float = MASTERY_THRESHOLD) -> dict[str, Any]:
    """Per-operation val keys that must clear ``threshold`` before B starts."""
    return {
        "threshold": float(threshold),
        "keys": [f"A_val_acc/{op.report_key()}" for op in task_a.operation_keys()],
        "rule": "every A operation, not the macro average",
    }


def write_dry_run(out_dir: Path, jobs: list[CausalJob], *, data_seed: int = 0) -> dict[str, Any]:
    """Write the job manifest and check B hashes. Does not train."""
    out_dir.mkdir(parents=True, exist_ok=True)
    identities: list[dict[str, Any]] = []
    for target in TARGET_MODULI:
        for construction in CONSTRUCTIONS:
            built = build_causal_tasks(target, construction)
            splits = residue_splits_for(built, data_seed=data_seed)
            b_splits = b_split_hash(built, splits)
            dataset = b_dataset_hash(built.task_b, {m: splits[m] for m in splits if m in {op.modulus for op in built.task_b.operations}}, data_seed)
            same_b = b_dataset_hash(
                built.same.task_b,
                {m: splits[m] for m in {op.modulus for op in built.task_b.operations}},
                data_seed,
            )
            diff_b = b_dataset_hash(
                built.different.task_b,
                {m: splits[m] for m in {op.modulus for op in built.task_b.operations}},
                data_seed,
            )
            if same_b != diff_b or same_b != dataset:
                raise RuntimeError("B dataset hash diverged between same_p and different_p")
            identities.append(
                {
                    "target_modulus": target,
                    "construction": construction,
                    "task_b_hash": task_hash(built.task_b),
                    "task_a_same_hash": task_hash(built.same.task_a),
                    "task_a_different_hash": task_hash(built.different.task_a),
                    "b_residue_split_hash": b_splits,
                    "b_dataset_hash": dataset,
                    "replacement_modulus": built.replacement_modulus,
                    "a_mastery": a_mastery_spec(built.same.task_a),
                }
            )
    report = job_count_report(jobs)
    if report["n_main_runs"] != N_MAIN_RUNS:
        raise RuntimeError(
            f"causal grid has {report['n_main_runs']} main runs, expected {N_MAIN_RUNS}"
        )
    if not report["b_only_deduped"]:
        raise RuntimeError("B-only jobs were not deduplicated")
    payload = {
        "experiment": "causal_modulus",
        "dry_run": True,
        "formal_gpu_started": False,
        "optimizer_transition": "fresh",
        "replay_ratio": 0.0,
        "mastery_threshold": MASTERY_THRESHOLD,
        "counts": report,
        "identities": identities,
        "jobs": [job.to_dict() for job in jobs],
        "note": (
            "Invalid runs are those whose A operations miss the mastery "
            "threshold. They must be excluded from formal statistics."
        ),
    }
    (out_dir / "jobs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def dataset_tag(target: int, construction: int, variant: str) -> str:
    return f"causal_p{int(target)}_c{int(construction)}_{variant}"


def prepare_causal_datasets(
    out_dir: Path,
    *,
    data_seed: int = 0,
    n_aliases: int = 16,
    train_frac: float = 0.8,
) -> dict[tuple[int, int, str], Path]:
    """Write packed datasets for same_p and different_p. B-only reuses same_p."""
    from go4cl.data.generate import generate_task_datasets, save_datasets
    from go4cl.data.manifest import DataManifest
    from go4cl.data.residue_pairs import assert_disjoint
    from go4cl.phases.common import ratios_from_train_frac

    ratios = ratios_from_train_frac(train_frac)
    paths: dict[tuple[int, int, str], Path] = {}
    for target in TARGET_MODULI:
        for construction in CONSTRUCTIONS:
            built = build_causal_tasks(target, construction)
            expected_b = task_hash(built.task_b)
            for variant, pair in (("same", built.same), ("different", built.different)):
                tag = dataset_tag(target, construction, variant)
                data_dir = out_dir / "data" / tag
                manifest_path = data_dir / "manifest.json"
                if manifest_path.is_file():
                    manifest = DataManifest.load(manifest_path)
                    if task_hash(manifest.task_pair.task_b) != expected_b:
                        raise RuntimeError(f"B task hash changed in {data_dir}")
                    if task_hash(manifest.task_pair.task_a) != task_hash(pair.task_a):
                        raise RuntimeError(f"A task hash changed in {data_dir}")
                else:
                    manifest, datasets = generate_task_datasets(
                        pair,
                        data_seed=int(data_seed),
                        n_aliases_per_pair=int(n_aliases),
                        n_nuisance_contexts=1,
                        ratios=ratios,
                        experiment_id=tag,
                        skip_train=True,
                    )
                    manifest.train_mode = "packed_online"
                    manifest.fixed_a = False
                    manifest.task_a_hash = task_hash(pair.task_a)
                    for split in manifest.residue_splits.values():
                        assert_disjoint(split)
                    save_datasets(data_dir, manifest, datasets)
                paths[(int(target), int(construction), variant)] = data_dir
    return paths
