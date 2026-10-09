"""Fixed replay-ratio coverage grid.

``replay_ratio`` stays 0.1. Coverage only changes how many distinct A train
residue pairs are in the replay buffer. Every coverage of one Task A and
model seed branches from the same theta_A. Low coverage repeats those pairs
with replacement so the number of A replay examples per step stays fixed.

Formal size: 2 conditions × 6 coverages × 3 model seeds = 36.
Train and test residue pairs are disjoint, so covered vs uncovered is scored
on the train split. The full A test set is reported separately.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from go4cl.data.generate import build_shared_residue_splits
from go4cl.data.manifest import hash_payload
from go4cl.data.packed import replay_pack_counts
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.metrics.continual import _auc, b_exposure, stable_time_to_threshold
from go4cl.tasks.relations import TaskPairSpec, build_task_pair
from go4cl.tasks.spec import TaskSpec

COVERAGES: tuple[float, ...] = (0.01, 0.05, 0.10, 0.25, 0.50, 1.00)
CONDITIONS: tuple[tuple[str, float, float, float], ...] = (
    ("s0_o0_m0", 0.0, 0.0, 0.0),
    ("s0.5_o0.5_m1", 0.5, 0.5, 1.0),
)
MODEL_SEEDS: tuple[int, ...] = (0, 1, 2)
REPLAY_RATIO = 0.1
BUFFER_SEED = 0
TASK_SEED = 0
N_MAIN_RUNS = len(CONDITIONS) * len(COVERAGES) * len(MODEL_SEEDS)

EVAL_SPLITS = (
    "A_test_all",
    "A_train_covered",
    "A_train_uncovered",
    "A_per_operation",
    "B_per_operation",
    "retention_A_from_switch",
    "forgetting_A_from_switch",
    "B_auc",
    "B_stable_t90",
)


def _k_pairs(n_train: int, coverage: float) -> int:
    if n_train < 1:
        raise ValueError("train pool is empty")
    return max(1, min(int(n_train), int(round(float(coverage) * int(n_train)))))


def select_coverage_pairs(
    task_a: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    coverage: float,
    *,
    buffer_seed: int = BUFFER_SEED,
) -> dict[str, Any]:
    """Stratify the subset by latent operation. The seed is not the model seed."""
    rng = np.random.default_rng(int(buffer_seed))
    per_op = []
    pairs_by_latent: dict[int, list[list[int]]] = {}
    for op in task_a.operations:
        pool = list(splits[int(op.modulus)].train)
        k = _k_pairs(len(pool), coverage)
        chosen_idx = sorted(int(i) for i in rng.choice(len(pool), size=k, replace=False))
        chosen = [list(pool[i]) for i in chosen_idx]
        pairs_by_latent[int(op.latent_id)] = chosen
        per_op.append(
            {
                "latent_id": int(op.latent_id),
                "modulus": int(op.modulus),
                "n_train": len(pool),
                "n_selected": k,
                "actual_coverage": k / len(pool),
                "repeat_factor": len(pool) / k,
            }
        )
    coverages = [row["actual_coverage"] for row in per_op]
    return {
        "buffer_seed": int(buffer_seed),
        "requested_coverage": float(coverage),
        "train_pairs_by_latent": pairs_by_latent,
        "per_operation": per_op,
        "actual_coverage_min": min(coverages),
        "actual_coverage_max": max(coverages),
        "unique_pairs": sum(row["n_selected"] for row in per_op),
    }


def replay_example_count(batch_size: int, n_ops: int, frac_a: float = REPLAY_RATIO) -> dict[str, int]:
    n_packs = int(batch_size) // int(n_ops)
    n_a, n_b = replay_pack_counts(n_packs, frac_a)
    return {
        "n_packs": n_packs,
        "n_packs_a": n_a,
        "n_packs_b": n_b,
        "a_replay_examples_per_step": n_a * int(n_ops),
    }


@dataclass(frozen=True)
class CoverageJob:
    job_id: str
    condition: str
    coverage: float
    model_seed: int
    a_source_key: str
    task_a_hash: str
    buffer: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "job_id": self.job_id,
            "condition": self.condition,
            "coverage": self.coverage,
            "model_seed": self.model_seed,
            "a_source_key": self.a_source_key,
            "task_a_hash": self.task_a_hash,
            "replay_ratio": REPLAY_RATIO,
            "optimizer_transition": "fresh",
            "protocol": "sequential_ab_replay",
            "theta_a_shared": True,
            "buffer_seed": self.buffer["buffer_seed"],
            "actual_coverage_min": self.buffer["actual_coverage_min"],
            "actual_coverage_max": self.buffer["actual_coverage_max"],
            "unique_pairs": self.buffer["unique_pairs"],
            "per_operation": self.buffer["per_operation"],
            "train_pairs_by_latent": self.buffer["train_pairs_by_latent"],
            "eval": list(EVAL_SPLITS),
            "covered_uncovered_split": "train residue pairs; test pairs are disjoint from the replay buffer",
            "metrics_use": "retention_A_from_switch",
        }


def _pair(condition: tuple[str, float, float, float]) -> TaskPairSpec:
    name, rho_slot, rho_operand, rho_mod = condition
    return build_task_pair(
        rho_slot=rho_slot,
        rho_operand=rho_operand,
        rho_mod=rho_mod,
        task_seed=TASK_SEED,
        pair_id=name,
        fixed_a=True,
    )


def plan_coverage_jobs(
    *,
    data_seed: int = 0,
    batch_size: int = 8192,
    buffer_seed: int = BUFFER_SEED,
) -> list[CoverageJob]:
    jobs: list[CoverageJob] = []
    for condition in CONDITIONS:
        pair = _pair(condition)
        splits = build_shared_residue_splits(
            [pair.task_a, pair.task_b],
            data_seed=int(data_seed),
            ratios=(0.8, 0.1, 0.1),
        )
        a_hash = hash_payload(pair.task_a.to_dict())
        counts = replay_example_count(batch_size, pair.task_a.n_ops, REPLAY_RATIO)
        for coverage in COVERAGES:
            buffer = select_coverage_pairs(
                pair.task_a, splits, coverage, buffer_seed=buffer_seed
            )
            buffer["replay_counts"] = counts
            for model_seed in MODEL_SEEDS:
                jobs.append(
                    CoverageJob(
                        job_id=(
                            f"coverage_{condition[0]}_c{coverage:.2f}"
                            f"_ts{TASK_SEED}_ms{int(model_seed)}_optfresh"
                        ),
                        condition=condition[0],
                        coverage=float(coverage),
                        model_seed=int(model_seed),
                        a_source_key=f"{a_hash}_ms{int(model_seed)}",
                        task_a_hash=a_hash,
                        buffer=buffer,
                    )
                )
    return jobs


def job_count_report(jobs: list[CoverageJob]) -> dict[str, Any]:
    by_source: dict[str, set[float]] = {}
    for job in jobs:
        by_source.setdefault(job.a_source_key, set()).add(job.coverage)
    replay_counts = {job.buffer["replay_counts"]["a_replay_examples_per_step"] for job in jobs}
    return {
        "n_main_runs": len(jobs),
        "expected_main_runs": N_MAIN_RUNS,
        "n_a_sources": len(by_source),
        "coverages_per_source": {key: len(values) for key, values in by_source.items()},
        "a_replay_examples_per_step": sorted(replay_counts),
        "shared_theta_a": all(len(values) == len(COVERAGES) for values in by_source.values()),
    }


def write_dry_run(out_dir: Path, jobs: list[CoverageJob]) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    report = job_count_report(jobs)
    if report["n_main_runs"] != N_MAIN_RUNS:
        raise RuntimeError(
            f"coverage grid has {report['n_main_runs']} runs, expected {N_MAIN_RUNS}"
        )
    if not report["shared_theta_a"]:
        raise RuntimeError("coverages do not all branch from the same theta_A")
    if report["a_replay_examples_per_step"] != [jobs[0].buffer["replay_counts"]["a_replay_examples_per_step"]]:
        raise RuntimeError("A replay example count is not identical across coverages")
    sources = []
    seen_sources: set[str] = set()
    for job in jobs:
        if job.a_source_key in seen_sources:
            continue
        seen_sources.add(job.a_source_key)
        sources.append(
            {
                "a_source_key": job.a_source_key,
                "task_a_hash": job.task_a_hash,
                "model_seed": job.model_seed,
                "protocol": "a_only",
                "shared_by_coverages": list(COVERAGES),
            }
        )
    payload = {
        "experiment": "replay_coverage",
        "dry_run": True,
        "formal_gpu_started": False,
        "replay_ratio": REPLAY_RATIO,
        "coverages": list(COVERAGES),
        "buffer_seed": BUFFER_SEED,
        "task_seed": TASK_SEED,
        "fixed_a": True,
        "optimizer_transition": "fresh",
        "counts": report,
        "a_pretrain": sources,
        "eval": list(EVAL_SPLITS),
        "note": (
            "Covered and uncovered residue pairs are partitions of the A train "
            "split. The A test split does not intersect the replay buffer."
        ),
        "jobs": [job.to_dict() for job in jobs],
    }
    (out_dir / "jobs.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return payload


def b_curve_metrics(history: list[dict[str, Any]], switch_step: float | None) -> dict[str, Any]:
    """B AUC and stable t90 on the project B-exposure axis, including replay scaling."""
    series: list[tuple[float, float]] = []
    for row in history:
        if "B_val_acc" not in row or "step" not in row:
            continue
        exposure = b_exposure("sequential_ab_replay", float(row["step"]), switch_step)
        if exposure is None:
            continue
        series.append((float(exposure), float(row["B_val_acc"])))
    series.sort(key=lambda item: item[0])
    return {
        "B_auc": _auc(series),
        "B_stable_t90": stable_time_to_threshold(series),
        "B_stable_t90_censored": stable_time_to_threshold(series) is None,
    }


def attach_coverage_metrics(session, coverage: dict[str, Any]) -> None:
    """Score covered and uncovered A train pairs and copy switch retention.

    Uses ``retention_A_from_switch`` already stored on the session. Does not
    recompute the history-max retention.
    """
    from go4cl.data.eval_contexts import make_eval_context_loader
    from go4cl.data.manifest import DataManifest
    from go4cl.data.packed import restrict_train_pairs
    from go4cl.metrics.behavioral import evaluate

    manifest = DataManifest.load(session.data_root / "manifest.json")
    task_a = manifest.task_pair.task_a
    selected = {
        int(latent): pairs
        for latent, pairs in coverage["train_pairs_by_latent"].items()
    }
    covered_splits = restrict_train_pairs(task_a, manifest.residue_splits, selected)
    from go4cl.data.residue_pairs import ResiduePairSplit

    uncovered: dict[int, ResiduePairSplit] = {}
    uncovered_empty = False
    for op in task_a.operations:
        base = manifest.residue_splits[int(op.modulus)]
        chosen = {tuple(int(a) for a in pair) for pair in selected[int(op.latent_id)]}
        left = tuple(pair for pair in base.train if pair not in chosen)
        if not left:
            uncovered_empty = True
            continue
        uncovered[int(op.modulus)] = ResiduePairSplit(
            modulus=base.modulus,
            train=left,
            val=base.val,
            test=base.test,
            ratios=base.ratios,
            data_seed=base.data_seed,
        )
    device = session.device
    covered_loader = make_eval_context_loader(
        task_a, covered_splits, target_split="train", n_per_operation=64, seed=0
    )
    covered_eval = evaluate(session.model, covered_loader, device)
    session.metrics["A_train_covered_acc"] = covered_eval.accuracy
    session.metrics["A_train_covered_by_operation"] = covered_eval.by_operation
    if uncovered_empty or not uncovered:
        session.metrics["A_train_uncovered_acc"] = None
        session.metrics["A_train_uncovered_empty"] = True
        session.metrics["A_train_uncovered_by_operation"] = {}
    else:
        uncovered_loader = make_eval_context_loader(
            task_a, uncovered, target_split="train", n_per_operation=64, seed=1
        )
        uncovered_eval = evaluate(session.model, uncovered_loader, device)
        session.metrics["A_train_uncovered_acc"] = uncovered_eval.accuracy
        session.metrics["A_train_uncovered_by_operation"] = uncovered_eval.by_operation
    session.metrics["A_test_all"] = session.metrics.get("after_b", {}).get("A_test_acc")
    session.metrics["coverage_eval_note"] = (
        "A_test_all is the full A test split. Covered and uncovered are train pairs."
    )
    session.metrics.update(
        b_curve_metrics(session.history, session.metrics.get("switch_step"))
    )
