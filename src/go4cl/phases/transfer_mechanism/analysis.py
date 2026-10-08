"""Per-operation transfer tables and checkpoint-mixing scores.

``b_exposure_steps_to_gen`` / ``first_reach_steps_to_gen`` are the first
B-exposure evaluation at or above the threshold (the phase-2 definition).
``stable_steps_to_gen`` is the B-exposure of the evaluation that completes
``stable_window`` consecutive evaluations at or above the threshold.
Necessity and sufficiency are not clipped to [0, 1].
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from go4cl.analysis.reporting import write_csv_rows
from go4cl.constants import LATENT_OPS
from go4cl.data.manifest import hash_payload
from go4cl.tasks.relations import TaskPairSpec, build_task_pair, compute_overlaps

GEN_THRESHOLD = 0.9
STABLE_WINDOW = 5

PER_OP_FIELDS = (
    "condition",
    "task_seed",
    "model_seed",
    "operation",
    "latent_id",
    "slot",
    "operand_pair",
    "modulus",
    "modulus_seen_in_A",
    "same_latent_modulus_as_A",
    "protocol",
    "b_exposure_auc",
    "first_reach_steps_to_gen",
    "stable_steps_to_gen",
    "b_exposure_steps_to_gen",
    "baseline_b_exposure_auc",
    "baseline_first_reach_steps_to_gen",
    "baseline_stable_steps_to_gen",
    "baseline_b_exposure_steps_to_gen",
    "delta_b_exposure_auc",
    "delta_first_reach_steps_to_gen",
    "stable_delta_steps_to_gen",
    "delta_b_exposure_steps_to_gen",
)

SUMMARY_FIELDS = (
    "condition",
    "modulus_group",
    "n_rows",
    "mean_delta_b_exposure_auc",
    "mean_stable_delta_steps_to_gen",
    "mean_delta_b_exposure_steps_to_gen",
    "mean_b_exposure_auc",
    "mean_baseline_b_exposure_auc",
)

COMPONENT_FIELDS = (
    "condition",
    "task_seed",
    "model_seed",
    "intervention",
    "groups_from_A",
    "groups_from_init",
    "pre_B_B_acc",
    "final_B_acc",
    "b_exposure_auc",
    "stable_steps_to_gen",
    "first_reach_steps_to_gen",
    "necessity_score",
    "sufficiency_score",
)

SHARED_GROUP = "shared-modulus operations"
NOVEL_GROUP = "novel-modulus operations"


def task_a_hash(pair: TaskPairSpec) -> str:
    return hash_payload(pair.task_a.to_dict())


def assert_shared_task_a(
    task_seed: int,
    conditions: Sequence[Mapping[str, Any]],
) -> str:
    """Task A is identical across overlap conditions when ``fixed_a`` is on."""
    hashes: list[str] = []
    specs: list[dict[str, Any]] = []
    b_hashes: list[str] = []
    for cond in conditions:
        pair = build_task_pair(
            rho_slot=float(cond["rho_slot"]),
            rho_operand=float(cond["rho_operand"]),
            rho_mod=float(cond["rho_mod"]),
            task_seed=int(task_seed),
            fixed_a=True,
        )
        got = compute_overlaps(pair.task_a, pair.task_b)
        want = (
            float(cond["rho_slot"]),
            float(cond["rho_operand"]),
            float(cond["rho_mod"]),
        )
        if any(abs(a - b) > 1e-9 for a, b in zip(got, want)):
            raise RuntimeError(f"{cond['name']} overlaps {got} != {want}")
        if not pair.pair_id.endswith("_fixedA"):
            raise RuntimeError(f"{pair.pair_id} is not marked fixed_a")
        hashes.append(task_a_hash(pair))
        specs.append(pair.task_a.to_dict())
        b_hashes.append(hash_payload(pair.task_b.to_dict()))
    if len(set(hashes)) != 1 or any(spec != specs[0] for spec in specs[1:]):
        raise RuntimeError(
            f"fixed_a Task A diverged across conditions for task_seed={task_seed}"
        )
    if len(conditions) > 1 and len(set(b_hashes)) != len(conditions):
        raise RuntimeError(
            f"Task B did not change across modulus conditions for task_seed={task_seed}"
        )
    return hashes[0]


def b_operation_table(pair: TaskPairSpec) -> list[dict[str, Any]]:
    """One row per latent B operation.

    ``op{k}`` is latent id ``k``, not the query slot. Slot is its own column.
    ``modulus_seen_in_A`` is whether that modulus appears on any A operation.
    ``same_latent_modulus_as_A`` is whether this latent keeps A's modulus.
    """
    task_a = pair.task_a.by_latent()
    task_b = pair.task_b.by_latent()
    moduli_a = {op.modulus for op in pair.task_a.operations}
    rows: list[dict[str, Any]] = []
    for latent in LATENT_OPS:
        op = task_b[latent]
        rows.append(
            {
                "operation": f"op{int(latent)}",
                "latent_id": int(latent),
                "slot": int(op.slot),
                "operand_pair": "+".join(str(x) for x in sorted(op.operand_pair)),
                "modulus": int(op.modulus),
                "modulus_seen_in_A": bool(op.modulus in moduli_a),
                "same_latent_modulus_as_A": bool(op.modulus == task_a[latent].modulus),
            }
        )
    return rows


def exposure_auc(series: Sequence[tuple[float, float]]) -> float | None:
    """Trapezoid AUC divided by the x-span. Matches ``metrics.continual._auc``."""
    if len(series) < 2:
        return None
    area = 0.0
    for (x0, y0), (x1, y1) in zip(series, series[1:]):
        dt = x1 - x0
        if dt <= 0:
            continue
        area += dt * (y0 + y1) / 2.0
    span = series[-1][0] - series[0][0]
    if span <= 0:
        return None
    return area / span


def first_reach_steps_to_gen(
    series: Sequence[tuple[float, float]], threshold: float
) -> float | None:
    for step, value in series:
        if value >= threshold:
            return float(step)
    return None


def stable_steps_to_gen(
    series: Sequence[tuple[float, float]],
    threshold: float,
    window: int,
) -> float | None:
    """B-exposure of the eval that completes ``window`` consecutive hits."""
    width = int(window)
    if width < 1:
        raise ValueError(f"stable window must be >= 1, got {window}")
    if len(series) < width:
        return None
    for index in range(width - 1, len(series)):
        chunk = series[index - width + 1 : index + 1]
        if all(value >= threshold for _, value in chunk):
            return float(chunk[-1][0])
    return None


def summarize_series(
    series: Sequence[tuple[float, float]],
    *,
    threshold: float = GEN_THRESHOLD,
    window: int = STABLE_WINDOW,
) -> dict[str, float | None]:
    first = first_reach_steps_to_gen(series, threshold)
    stable = stable_steps_to_gen(series, threshold, window)
    return {
        "b_exposure_auc": exposure_auc(series),
        "first_reach_steps_to_gen": first,
        "stable_steps_to_gen": stable,
        # Same definition as phase-2 ``b_exposure_steps_to_gen`` (first hit).
        "b_exposure_steps_to_gen": first,
    }


def curve_from_history(
    history: Sequence[Mapping[str, Any]], key: str
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in history:
        if row.get("curve") not in (None, "B"):
            continue
        if "b_exposure" not in row and row.get("curve") != "B":
            continue
        value = row.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            continue
        exposure = row.get("b_exposure", row.get("step"))
        if not isinstance(exposure, (int, float)) or isinstance(exposure, bool):
            continue
        points.append((float(exposure), float(value)))
    points.sort(key=lambda item: item[0])
    return points


def necessity_score(
    auc_full_a: float | None,
    auc_reset: float | None,
    auc_full_fresh: float | None,
) -> float | None:
    denom = _denom(auc_full_a, auc_full_fresh)
    if denom is None or auc_reset is None or auc_full_a is None:
        return None
    return (float(auc_full_a) - float(auc_reset)) / denom


def sufficiency_score(
    auc_keep: float | None,
    auc_full_a: float | None,
    auc_full_fresh: float | None,
) -> float | None:
    denom = _denom(auc_full_a, auc_full_fresh)
    if denom is None or auc_keep is None or auc_full_fresh is None:
        return None
    return (float(auc_keep) - float(auc_full_fresh)) / denom


def _denom(auc_full_a: float | None, auc_full_fresh: float | None) -> float | None:
    if auc_full_a is None or auc_full_fresh is None:
        return None
    denom = float(auc_full_a) - float(auc_full_fresh)
    if denom == 0.0:
        return None
    return denom


def _delta(run: Any, base: Any, *, higher_is_better: bool) -> float | None:
    if not isinstance(run, (int, float)) or not isinstance(base, (int, float)):
        return None
    if isinstance(run, bool) or isinstance(base, bool):
        return None
    if higher_is_better:
        return float(run) - float(base)
    return float(base) - float(run)


def per_op_transfer_rows(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Join each protocol's per-op curve to the matched ``b_only`` baseline."""
    baselines: dict[tuple, dict[str, Any]] = {}
    prepared: list[tuple[Mapping[str, Any], dict[str, Any]]] = []
    for job in jobs:
        if job.get("status") not in {None, "ok"}:
            continue
        ops = list(job.get("operations") or [])
        stats = _per_op_stats(job, ops)
        prepared.append((job, stats))
        if job.get("protocol") == "b_only":
            for op in ops:
                baselines[
                    (
                        job.get("condition"),
                        int(job["task_seed"]),
                        int(job["model_seed"]),
                        op["operation"],
                    )
                ] = stats[op["operation"]]
    rows: list[dict[str, Any]] = []
    for job, stats in prepared:
        for op in job.get("operations") or []:
            name = op["operation"]
            run_stats = stats[name]
            key = (
                job.get("condition"),
                int(job["task_seed"]),
                int(job["model_seed"]),
                name,
            )
            base = baselines.get(key)
            if job.get("protocol") == "b_only":
                base = run_stats
            row = {
                "condition": job.get("condition"),
                "task_seed": int(job["task_seed"]),
                "model_seed": int(job["model_seed"]),
                "operation": name,
                "latent_id": op.get("latent_id"),
                "slot": op.get("slot"),
                "operand_pair": op.get("operand_pair"),
                "modulus": op.get("modulus"),
                "modulus_seen_in_A": op.get("modulus_seen_in_A"),
                "same_latent_modulus_as_A": op.get("same_latent_modulus_as_A"),
                "protocol": job.get("protocol"),
                "b_exposure_auc": run_stats.get("b_exposure_auc"),
                "first_reach_steps_to_gen": run_stats.get("first_reach_steps_to_gen"),
                "stable_steps_to_gen": run_stats.get("stable_steps_to_gen"),
                "b_exposure_steps_to_gen": run_stats.get("b_exposure_steps_to_gen"),
            }
            if base is None:
                row["baseline_b_exposure_auc"] = None
                row["baseline_first_reach_steps_to_gen"] = None
                row["baseline_stable_steps_to_gen"] = None
                row["baseline_b_exposure_steps_to_gen"] = None
                row["delta_b_exposure_auc"] = None
                row["delta_first_reach_steps_to_gen"] = None
                row["stable_delta_steps_to_gen"] = None
                row["delta_b_exposure_steps_to_gen"] = None
            else:
                row["baseline_b_exposure_auc"] = base.get("b_exposure_auc")
                row["baseline_first_reach_steps_to_gen"] = base.get(
                    "first_reach_steps_to_gen"
                )
                row["baseline_stable_steps_to_gen"] = base.get("stable_steps_to_gen")
                row["baseline_b_exposure_steps_to_gen"] = base.get(
                    "b_exposure_steps_to_gen"
                )
                row["delta_b_exposure_auc"] = _delta(
                    run_stats.get("b_exposure_auc"),
                    base.get("b_exposure_auc"),
                    higher_is_better=True,
                )
                row["delta_first_reach_steps_to_gen"] = _delta(
                    run_stats.get("first_reach_steps_to_gen"),
                    base.get("first_reach_steps_to_gen"),
                    higher_is_better=False,
                )
                row["stable_delta_steps_to_gen"] = _delta(
                    run_stats.get("stable_steps_to_gen"),
                    base.get("stable_steps_to_gen"),
                    higher_is_better=False,
                )
                row["delta_b_exposure_steps_to_gen"] = row["delta_first_reach_steps_to_gen"]
            rows.append(row)
    return rows


def modulus_summary_rows(per_op_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Mean transfer on sequential_ab, split by ``modulus_seen_in_A``."""
    sequential = [row for row in per_op_rows if row.get("protocol") == "sequential_ab"]
    conditions = list(dict.fromkeys(str(row.get("condition")) for row in sequential))
    groups = (
        (True, SHARED_GROUP),
        (False, NOVEL_GROUP),
    )
    out: list[dict[str, Any]] = []
    scopes: list[tuple[str, list[Mapping[str, Any]]]] = [
        (condition, [row for row in sequential if str(row.get("condition")) == condition])
        for condition in conditions
    ]
    if sequential:
        scopes.append(("all", list(sequential)))
    for condition, pool in scopes:
        for flag, label in groups:
            chosen = [
                row
                for row in pool
                if _as_bool(row.get("modulus_seen_in_A")) is flag
            ]
            out.append(
                {
                    "condition": condition,
                    "modulus_group": label,
                    "n_rows": len(chosen),
                    "mean_delta_b_exposure_auc": _mean(
                        row.get("delta_b_exposure_auc") for row in chosen
                    ),
                    "mean_stable_delta_steps_to_gen": _mean(
                        row.get("stable_delta_steps_to_gen") for row in chosen
                    ),
                    "mean_delta_b_exposure_steps_to_gen": _mean(
                        row.get("delta_b_exposure_steps_to_gen") for row in chosen
                    ),
                    "mean_b_exposure_auc": _mean(
                        row.get("b_exposure_auc") for row in chosen
                    ),
                    "mean_baseline_b_exposure_auc": _mean(
                        row.get("baseline_b_exposure_auc") for row in chosen
                    ),
                }
            )
    return out


def component_transfer_rows(jobs: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple, list[Mapping[str, Any]]] = {}
    for job in jobs:
        if job.get("status") not in {None, "ok"}:
            continue
        if "intervention" not in job:
            continue
        key = (job.get("condition"), int(job["task_seed"]), int(job["model_seed"]))
        grouped.setdefault(key, []).append(job)
    rows: list[dict[str, Any]] = []
    for members in grouped.values():
        summaries = {str(job["intervention"]): _aggregate_summary(job) for job in members}
        full_a = summaries.get("full_A", {}).get("b_exposure_auc")
        full_fresh = summaries.get("full_fresh", {}).get("b_exposure_auc")
        anchors = _denom(full_a, full_fresh) is not None
        for job in members:
            name = str(job["intervention"])
            summary = summaries[name]
            kind = str(job.get("kind") or _kind_from_name(name))
            necessity = None
            sufficiency = None
            if kind == "reset":
                necessity = necessity_score(
                    full_a, summary.get("b_exposure_auc"), full_fresh
                )
            elif kind == "keep":
                sufficiency = sufficiency_score(
                    summary.get("b_exposure_auc"), full_a, full_fresh
                )
            elif kind == "full_A" and anchors:
                sufficiency = 1.0
            elif kind == "full_fresh" and anchors:
                necessity = 1.0
                sufficiency = 0.0
            rows.append(
                {
                    "condition": job.get("condition"),
                    "task_seed": int(job["task_seed"]),
                    "model_seed": int(job["model_seed"]),
                    "intervention": name,
                    "groups_from_A": _join(job.get("groups_from_A")),
                    "groups_from_init": _join(job.get("groups_from_init")),
                    "pre_B_B_acc": _pre_b_acc(job),
                    "final_B_acc": _final_b_acc(job, summary),
                    "b_exposure_auc": summary.get("b_exposure_auc"),
                    "stable_steps_to_gen": summary.get("stable_steps_to_gen"),
                    "first_reach_steps_to_gen": summary.get("first_reach_steps_to_gen"),
                    "necessity_score": necessity,
                    "sufficiency_score": sufficiency,
                }
            )
    return rows


def analyze_modulus_dir(root: Path | str) -> dict[str, Path]:
    root = Path(root)
    jobs = _load_jobs(root, "*/*/ts*_ms*/metrics.json")
    rows = per_op_transfer_rows(jobs)
    summary = modulus_summary_rows(rows)
    out = root / "summary"
    per_path = write_csv_rows(out / "per_op_transfer.csv", _cells(rows), PER_OP_FIELDS)
    sum_path = write_csv_rows(
        out / "modulus_specificity_summary.csv", _cells(summary), SUMMARY_FIELDS
    )
    return {"per_op_transfer": per_path, "modulus_specificity_summary": sum_path}


def analyze_component_dir(root: Path | str) -> dict[str, Path]:
    root = Path(root)
    jobs = _load_jobs(root, "*/ts*_ms*/metrics.json")
    rows = component_transfer_rows(jobs)
    path = write_csv_rows(
        root / "summary" / "component_transfer_summary.csv",
        _cells(rows),
        COMPONENT_FIELDS,
    )
    return {"component_transfer_summary": path}


def _per_op_stats(
    job: Mapping[str, Any], operations: Sequence[Mapping[str, Any]]
) -> dict[str, dict[str, Any]]:
    threshold = float(job.get("gen_threshold", GEN_THRESHOLD))
    window = int(job.get("stable_window", STABLE_WINDOW))
    history = list(job.get("history") or [])
    stored = job.get("per_op") or {}
    out: dict[str, dict[str, Any]] = {}
    for op in operations:
        name = str(op["operation"])
        if history:
            series = curve_from_history(history, f"B_test_acc/{name}")
            out[name] = summarize_series(series, threshold=threshold, window=window)
        elif isinstance(stored, dict) and name in stored:
            out[name] = dict(stored[name])
        else:
            out[name] = summarize_series((), threshold=threshold, window=window)
    return out


def _aggregate_summary(job: Mapping[str, Any]) -> dict[str, Any]:
    threshold = float(job.get("gen_threshold", GEN_THRESHOLD))
    window = int(job.get("stable_window", STABLE_WINDOW))
    history = list(job.get("history") or [])
    if history:
        return summarize_series(
            curve_from_history(history, "B_test_acc"),
            threshold=threshold,
            window=window,
        )
    stored = job.get("b_curve")
    if isinstance(stored, dict):
        return dict(stored)
    return {
        "b_exposure_auc": job.get("b_exposure_auc"),
        "stable_steps_to_gen": job.get("stable_steps_to_gen"),
        "first_reach_steps_to_gen": job.get("first_reach_steps_to_gen"),
        "b_exposure_steps_to_gen": job.get("b_exposure_steps_to_gen"),
    }


def _pre_b_acc(job: Mapping[str, Any]) -> Any:
    if "pre_B/B_test_acc" in job:
        return job["pre_B/B_test_acc"]
    for row in job.get("history") or []:
        if row.get("pre_b") and "B_test_acc" in row:
            return row["B_test_acc"]
    return None


def _final_b_acc(job: Mapping[str, Any], summary: Mapping[str, Any]) -> Any:
    if job.get("final_B_acc") is not None and not job.get("history"):
        return job.get("final_B_acc")
    last = None
    for row in job.get("history") or []:
        if "B_test_acc" in row and (
            row.get("curve") == "B" or "b_exposure" in row
        ):
            last = row["B_test_acc"]
    if last is not None:
        return last
    return job.get("final_B_acc", summary.get("final_B_acc"))


def _kind_from_name(name: str) -> str:
    if name == "full_A":
        return "full_A"
    if name == "full_fresh":
        return "full_fresh"
    if name.startswith("reset_"):
        return "reset"
    if name.startswith("keep_"):
        return "keep"
    return "other"


def _join(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return ",".join(str(item) for item in value)


def _as_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        if value.lower() == "true":
            return True
        if value.lower() == "false":
            return False
    return None


def _mean(values: Iterable[Any]) -> float | None:
    nums = [
        float(value)
        for value in values
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    ]
    if not nums:
        return None
    return sum(nums) / len(nums)


def _cells(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for row in rows:
        out: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                out[key] = ""
            elif isinstance(value, bool):
                out[key] = "true" if value else "false"
            else:
                out[key] = value
        rendered.append(out)
    return rendered


def _load_jobs(root: Path, pattern: str) -> list[dict[str, Any]]:
    jobs_root = root / "jobs"
    if not jobs_root.is_dir():
        return []
    loaded: list[dict[str, Any]] = []
    for path in sorted(jobs_root.glob(pattern)):
        metrics = json.loads(path.read_text(encoding="utf-8"))
        if metrics.get("status") not in {None, "ok"}:
            continue
        history_path = path.with_name("eval_history.jsonl")
        history: list[dict[str, Any]] = []
        if history_path.is_file():
            for line in history_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    history.append(json.loads(line))
        metrics["history"] = history
        loaded.append(metrics)
    return loaded
