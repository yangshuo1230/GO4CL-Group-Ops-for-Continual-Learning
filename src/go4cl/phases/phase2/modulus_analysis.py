"""Matched per-modulus and per-operation curves for an existing Phase 2 stamp.

Reads ``eval_history.jsonl`` only. Sequential steps are converted with
``b_exposure_step = global_step - switch_step``. Missing stable or first t90
stays in the table as censored. This module does not rewrite histories,
checkpoints, or the stamp's existing summary files.
"""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from go4cl.metrics.continual import _auc, _first_reach, stable_time_to_threshold

STABLE_WINDOW = 5
STABLE_THRESHOLD = 0.9
MATCHED_FIELDS = [
    "condition",
    "task_seed",
    "model_seed",
    "modulus",
    "shared_modulus",
    "b_only_auc",
    "sequential_auc",
    "delta_auc",
    "b_only_first_t90",
    "sequential_first_t90",
    "b_only_stable_t90",
    "sequential_stable_t90",
    "delta_stable_t90",
    "b_only_final_acc",
    "sequential_final_acc",
    "censored_first_t90",
    "censored_stable_t90",
    "rho_mod",
    "rho_slot",
    "rho_operand",
    "overlap_group",
]


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_history(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _moduli_from_keys(row: dict[str, Any], prefix: str) -> set[int]:
    found: set[int] = set()
    token = f"{prefix}/p"
    for key in row:
        if key.startswith(token) and key[len(token) :].isdigit():
            found.add(int(key[len(token) :]))
    return found


def _series(history: list[dict[str, Any]], key: str, switch_step: float | None) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for row in history:
        if key not in row or "step" not in row:
            continue
        step = float(row["step"])
        exposure = step if switch_step is None else step - float(switch_step)
        if exposure < 0:
            continue
        points.append((exposure, float(row[key])))
    points.sort(key=lambda item: item[0])
    return points


def _curve_numbers(series: list[tuple[float, float]]) -> dict[str, Any]:
    first = _first_reach(series, STABLE_THRESHOLD)
    stable = stable_time_to_threshold(
        series, threshold=STABLE_THRESHOLD, window=STABLE_WINDOW
    )
    return {
        "auc": _auc(series),
        "first_t90": first,
        "stable_t90": stable,
        "final_acc": series[-1][1] if series else None,
        "censored_first_t90": first is None,
        "censored_stable_t90": stable is None,
    }


def _blank(value: Any) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def overlap_group(rho_mod: float, shared: bool) -> str:
    if abs(float(rho_mod) - 0.0) < 1e-9:
        return "rho_mod=0"
    if abs(float(rho_mod) - 1.0) < 1e-9:
        return "rho_mod=1"
    if abs(float(rho_mod) - 0.5) < 1e-9:
        return "rho_mod=0.5/shared" if shared else "rho_mod=0.5/nonshared"
    return f"rho_mod={rho_mod:g}"


def _index_runs(runs_dir: Path) -> dict[tuple, dict[str, Any]]:
    indexed: dict[tuple, dict[str, Any]] = {}
    for run_dir in sorted(path for path in runs_dir.iterdir() if path.is_dir()):
        cfg_path = run_dir / "config_resolved.json"
        hist_path = run_dir / "eval_history.jsonl"
        if not cfg_path.is_file() or not hist_path.is_file():
            continue
        cfg = _load_json(cfg_path)
        protocol = str(cfg.get("protocol") or "")
        if protocol not in {"b_only", "sequential_ab"}:
            continue
        metrics_path = run_dir / "metrics.json"
        metrics = _load_json(metrics_path) if metrics_path.is_file() else {}
        key = (str(cfg["condition"]), int(cfg["task_seed"]), int(cfg["model_seed"]))
        slot = indexed.setdefault(key, {})
        if protocol in slot:
            raise RuntimeError(f"duplicate {protocol} for {key}")
        switch = metrics.get("switch_step")
        slot[protocol] = {
            "run_dir": run_dir,
            "cfg": cfg,
            "metrics": metrics,
            "history": _load_history(hist_path),
            "switch_step": None if protocol == "b_only" else float(switch),
        }
    return indexed


def _task_ops(data_root: str | None) -> dict[str, list[dict[str, Any]]]:
    if not data_root:
        return {}
    manifest = Path(data_root) / "manifest.json"
    if not manifest.is_file():
        return {}
    payload = _load_json(manifest)
    pair = payload.get("task_pair") or {}
    out: dict[str, list[dict[str, Any]]] = {}
    for name in ("task_a", "task_b"):
        task = pair.get(name) or {}
        out[name] = list(task.get("operations") or [])
    return out


def _matched_rows(indexed: dict[tuple, dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    matched: list[dict[str, Any]] = []
    modulus_curves: list[dict[str, Any]] = []
    operation_curves: list[dict[str, Any]] = []
    for (condition, task_seed, model_seed), sides in sorted(indexed.items()):
        if "b_only" not in sides or "sequential_ab" not in sides:
            continue
        b_only = sides["b_only"]
        sequential = sides["sequential_ab"]
        cfg = sequential["cfg"]
        rho_mod = float(cfg["rho_mod"])
        seq_hist = sequential["history"]
        if not seq_hist or not b_only["history"]:
            continue
        a_mods = _moduli_from_keys(seq_hist[-1], "A_val_acc")
        b_mods = _moduli_from_keys(seq_hist[-1], "B_val_acc")
        if not b_mods:
            b_mods = _moduli_from_keys(b_only["history"][-1], "B_val_acc")
        ops = _task_ops(cfg.get("data_root"))
        for modulus in sorted(b_mods):
            shared = modulus in a_mods
            b_series = _series(b_only["history"], f"B_val_acc/p{modulus}", None)
            s_series = _series(
                seq_hist, f"B_val_acc/p{modulus}", sequential["switch_step"]
            )
            b_stats = _curve_numbers(b_series)
            s_stats = _curve_numbers(s_series)
            both_first = (
                b_stats["first_t90"] is not None and s_stats["first_t90"] is not None
            )
            both_stable = (
                b_stats["stable_t90"] is not None and s_stats["stable_t90"] is not None
            )
            both_auc = b_stats["auc"] is not None and s_stats["auc"] is not None
            row = {
                "condition": condition,
                "task_seed": task_seed,
                "model_seed": model_seed,
                "modulus": modulus,
                "shared_modulus": shared,
                "b_only_auc": b_stats["auc"],
                "sequential_auc": s_stats["auc"],
                "delta_auc": (
                    s_stats["auc"] - b_stats["auc"] if both_auc else None
                ),
                "b_only_first_t90": b_stats["first_t90"],
                "sequential_first_t90": s_stats["first_t90"],
                "b_only_stable_t90": b_stats["stable_t90"],
                "sequential_stable_t90": s_stats["stable_t90"],
                "delta_stable_t90": (
                    b_stats["stable_t90"] - s_stats["stable_t90"] if both_stable else None
                ),
                "b_only_final_acc": b_stats["final_acc"],
                "sequential_final_acc": s_stats["final_acc"],
                "censored_first_t90": not both_first,
                "censored_stable_t90": not both_stable,
                "rho_mod": rho_mod,
                "rho_slot": float(cfg["rho_slot"]),
                "rho_operand": float(cfg["rho_operand"]),
                "overlap_group": overlap_group(rho_mod, shared),
            }
            matched.append(row)
            for protocol, series in (("b_only", b_series), ("sequential_ab", s_series)):
                for exposure, acc in series:
                    modulus_curves.append(
                        {
                            "condition": condition,
                            "task_seed": task_seed,
                            "model_seed": model_seed,
                            "rho_mod": rho_mod,
                            "modulus": modulus,
                            "shared_modulus": shared,
                            "protocol": protocol,
                            "b_exposure_step": exposure,
                            "acc": acc,
                        }
                    )
        for task_name, prefix, task_id in (
            ("task_a", "A_val_acc", 0),
            ("task_b", "B_val_acc", 1),
        ):
            for op in ops.get(task_name, []):
                key = (
                    f"{prefix}/task{task_id}/lat{int(op['latent_id'])}/slot{int(op['slot'])}"
                )
                series = _series(seq_hist, key, sequential["switch_step"])
                for exposure, acc in series:
                    operation_curves.append(
                        {
                            "condition": condition,
                            "task_seed": task_seed,
                            "model_seed": model_seed,
                            "rho_mod": rho_mod,
                            "task": "A" if task_name == "task_a" else "B",
                            "latent_id": int(op["latent_id"]),
                            "slot": int(op["slot"]),
                            "modulus": int(op["modulus"]),
                            "b_exposure_step": exposure,
                            "acc": acc,
                        }
                    )
    return matched, modulus_curves, operation_curves


def _stat_block(values: list[float]) -> dict[str, Any]:
    finite = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    if not finite:
        return {"mean": None, "median": None, "std": None, "n_finite": 0}
    return {
        "mean": statistics.fmean(finite),
        "median": statistics.median(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "n_finite": len(finite),
    }


def summarize_groups(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean, median, std, positive-transfer rate, and censored counts.

    Censored t90 rows stay in ``n``. They are omitted only from the t90 mean.
    """
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        buckets[str(row["overlap_group"])].append(row)
        buckets[f"p{int(row['modulus'])}"].append(row)
    summary = []
    for name in sorted(buckets):
        group = buckets[name]
        auc = _stat_block([row["delta_auc"] for row in group])
        stable = _stat_block(
            [
                row["delta_stable_t90"]
                for row in group
                if not row["censored_stable_t90"]
            ]
        )
        positive = [
            row["delta_auc"]
            for row in group
            if row["delta_auc"] is not None and math.isfinite(float(row["delta_auc"]))
        ]
        n_pos = sum(1 for value in positive if value > 0)
        summary.append(
            {
                "group": name,
                "n": len(group),
                "n_censored_stable_t90": sum(1 for row in group if row["censored_stable_t90"]),
                "n_censored_first_t90": sum(1 for row in group if row["censored_first_t90"]),
                "delta_auc_mean": auc["mean"],
                "delta_auc_median": auc["median"],
                "delta_auc_std": auc["std"],
                "delta_auc_n_finite": auc["n_finite"],
                "positive_transfer_fraction": (n_pos / len(positive)) if positive else None,
                "delta_stable_t90_mean": stable["mean"],
                "delta_stable_t90_median": stable["median"],
                "delta_stable_t90_std": stable["std"],
                "delta_stable_t90_n_finite": stable["n_finite"],
            }
        )
    return summary


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and not fields:
        path.write_text("", encoding="utf-8")
        return
    names = fields or list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _blank(row.get(key)) for key in names})


def _mean_std(groups: dict[Any, list[float]]) -> list[tuple[Any, float, float, int]]:
    out = []
    for key in sorted(groups):
        values = groups[key]
        mean = statistics.fmean(values)
        std = statistics.stdev(values) if len(values) > 1 else 0.0
        out.append((key, mean, std, len(values)))
    return out


def _plot(out_dir: Path, rows: list[dict[str, Any]], modulus_curves: list[dict[str, Any]], operation_curves: list[dict[str, Any]]) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    written: list[str] = []
    by_rho: dict[float, list[dict[str, Any]]] = defaultdict(list)
    for point in modulus_curves:
        by_rho[float(point["rho_mod"])].append(point)
    for rho, points in sorted(by_rho.items()):
        moduli = sorted({int(point["modulus"]) for point in points})
        fig, axes = plt.subplots(1, len(moduli), figsize=(3.2 * len(moduli), 3.4), sharey=True)
        if len(moduli) == 1:
            axes = [axes]
        for ax, modulus in zip(axes, moduli):
            for protocol, color in (("b_only", "C0"), ("sequential_ab", "C1")):
                bucket: dict[float, list[float]] = defaultdict(list)
                for point in points:
                    if int(point["modulus"]) == modulus and point["protocol"] == protocol:
                        bucket[float(point["b_exposure_step"])].append(float(point["acc"]))
                xs, means, stds = [], [], []
                for step, mean, std, _n in _mean_std(bucket):
                    xs.append(step)
                    means.append(mean)
                    stds.append(std)
                if not xs:
                    continue
                ax.plot(xs, means, color=color, label=protocol)
                ax.fill_between(
                    xs,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    color=color,
                    alpha=0.2,
                )
            ax.set_title(f"p{modulus}")
            ax.set_xlabel("B exposure steps")
            ax.set_ylim(0, 1)
        axes[0].set_ylabel("B val acc")
        axes[0].legend(fontsize=8)
        fig.suptitle(f"rho_mod={rho:g}  mean ± std")
        fig.tight_layout()
        path = out_dir / f"curves_rho{rho:g}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path.name)

    summary = {row["group"]: row for row in summarize_groups(rows)}
    order = ["rho_mod=0", "rho_mod=0.5/nonshared", "rho_mod=0.5/shared", "rho_mod=1"]
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6))
    labels = [name for name in order if name in summary]
    axes[0].bar(range(len(labels)), [summary[name]["delta_auc_mean"] or 0 for name in labels])
    axes[0].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[0].set_ylabel("mean ΔAUC (sequential − B-only)")
    axes[0].axhline(0, color="black", linewidth=0.6)
    axes[1].bar(
        range(len(labels)),
        [summary[name]["delta_stable_t90_mean"] or 0 for name in labels],
    )
    axes[1].set_xticks(range(len(labels)), labels, rotation=20, ha="right")
    axes[1].set_ylabel("mean Δstable-t90 (B-only − sequential)")
    axes[1].axhline(0, color="black", linewidth=0.6)
    fig.tight_layout()
    path = out_dir / "delta_shared_vs_nonshared.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    p23 = [row["delta_auc"] for row in rows if int(row["modulus"]) == 23 and row["delta_auc"] is not None]
    other = [row["delta_auc"] for row in rows if int(row["modulus"]) != 23 and row["delta_auc"] is not None]
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    ax.bar(
        ["p23", "other moduli"],
        [
            statistics.fmean(p23) if p23 else 0,
            statistics.fmean(other) if other else 0,
        ],
    )
    ax.axhline(0, color="black", linewidth=0.6)
    ax.set_ylabel("mean ΔAUC")
    ax.set_title("p=23 vs other B moduli")
    fig.tight_layout()
    path = out_dir / "p23_vs_other.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    anomaly = [
        point
        for point in modulus_curves
        if int(point["task_seed"]) == 1
        and int(point["model_seed"]) == 2
        and int(point["modulus"]) == 29
    ]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    conditions = sorted({point["condition"] for point in anomaly})
    for condition in conditions:
        for protocol in ("b_only", "sequential_ab"):
            pts = [
                point
                for point in anomaly
                if point["condition"] == condition and point["protocol"] == protocol
            ]
            pts.sort(key=lambda point: float(point["b_exposure_step"]))
            if pts:
                ax.plot(
                    [float(point["b_exposure_step"]) for point in pts],
                    [float(point["acc"]) for point in pts],
                    label=f"{condition} {protocol}",
                )
    ax.set_xlabel("B exposure steps")
    ax.set_ylabel("B val acc p29")
    ax.set_title("task_seed=1 model_seed=2 p=29")
    ax.set_ylim(0, 1)
    if conditions:
        ax.legend(fontsize=7)
    fig.tight_layout()
    path = out_dir / "anomaly_ts1_ms2_p29.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    written.append(path.name)

    for rho in sorted({float(point["rho_mod"]) for point in operation_curves}):
        fig, axes = plt.subplots(2, 1, figsize=(7.5, 6), sharex=True)
        for ax, task_seed in zip(axes, (0, 1)):
            subset = [
                point
                for point in operation_curves
                if float(point["rho_mod"]) == rho and int(point["task_seed"]) == task_seed
            ]
            series_keys = sorted(
                {
                    (point["task"], int(point["latent_id"]), int(point["modulus"]))
                    for point in subset
                }
            )
            for task, latent, modulus in series_keys:
                bucket: dict[float, list[float]] = defaultdict(list)
                for point in subset:
                    if (
                        point["task"] == task
                        and int(point["latent_id"]) == latent
                        and int(point["modulus"]) == modulus
                    ):
                        bucket[float(point["b_exposure_step"])].append(float(point["acc"]))
                xs, means, _stds, _n = zip(*_mean_std(bucket)) if bucket else ([], [], [], [])
                if xs:
                    style = "-" if task == "B" else "--"
                    ax.plot(xs, means, linestyle=style, label=f"{task} lat{latent} p{modulus}")
            ax.set_ylim(0, 1)
            ax.set_title(f"rho_mod={rho:g} task_seed={task_seed}")
            ax.legend(fontsize=6, ncol=2)
        axes[-1].set_xlabel("B exposure steps")
        axes[0].set_ylabel("val acc")
        fig.tight_layout()
        path = out_dir / f"joint_ops_rho{rho:g}.png"
        fig.savefig(path, dpi=120)
        plt.close(fig)
        written.append(path.name)
    return written


def run_modulus_analysis(stamp_root: Path | str, out_dir: Path | str | None = None) -> dict[str, Any]:
    stamp_root = Path(stamp_root)
    out_dir = Path(out_dir) if out_dir else stamp_root / "modulus_operation_analysis"
    indexed = _index_runs(stamp_root / "runs")
    rows, modulus_curves, operation_curves = _matched_rows(indexed)
    if not rows:
        raise RuntimeError(f"no matched B-only / sequential_ab histories under {stamp_root}")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize_groups(rows)
    _write_csv(out_dir / "matched_modulus.csv", rows, MATCHED_FIELDS)
    _write_csv(out_dir / "group_summary.csv", summary)
    _write_csv(out_dir / "curves_modulus.csv", modulus_curves)
    _write_csv(out_dir / "curves_operation.csv", operation_curves)
    plots = _plot(out_dir, rows, modulus_curves, operation_curves)
    notes = {
        "b_exposure_step": "global_step - switch_step for sequential_ab; global_step for b_only",
        "delta_auc": "sequential_auc - b_only_auc",
        "delta_stable_t90": "b_only_stable_t90 - sequential_stable_t90; blank when either side is censored",
        "stable_t90": f"{STABLE_WINDOW} consecutive evals at accuracy >= {STABLE_THRESHOLD}",
        "censored": "missing t90 is kept and flagged; it is not dropped",
        "per_modulus_loss": "not present in 01_core_fresh histories; overall B_val_loss is not used as a per-modulus loss",
        "n_matched_rows": len(rows),
        "plots": plots,
        "formal_histories_modified": False,
    }
    (out_dir / "notes.json").write_text(
        json.dumps(notes, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {"out_dir": str(out_dir), "n_rows": len(rows), "summary": summary, "plots": plots}
