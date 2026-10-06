#!/usr/bin/env python3
"""Phase 2 continual-learning report figures from existing stamps (no training).

Reads:
  108-run grid  runs/phase2/relation_matrix/20261004_202208
  protocols     runs/phase2/protocols/20261004_150305
  null-negative runs/phase2/relation_matrix/20261005_154626

Writes PNG under runs/phase2/report_figures/{core,appendix}
plus derived CSVs and README. Does not overwrite original experiment files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAMP_GRID = ROOT / "runs/phase2/relation_matrix/20261004_202208"
STAMP_PROTO = ROOT / "runs/phase2/protocols/20261004_150305"
STAMP_NULL = ROOT / "runs/phase2/relation_matrix/20261005_154626"
OUT = ROOT / "runs/phase2/report_figures"

PROTOCOLS = ("a_only", "b_only", "joint", "sequential_ab")
PROTO_LABEL = {
    "a_only": "A-only",
    "b_only": "B-only",
    "joint": "Joint",
    "sequential_ab": "Sequential A→B",
}
COLOR_A = "#4C78A8"
COLOR_B = "#F58518"
COLOR_JOINT = "#54A24B"
COLOR_SEQ = "#E45756"
COLOR_MOD = {0.0: "#4C78A8", 0.5: "#54A24B", 1.0: "#E45756"}
MARKER_SLOT = {0.0: "o", 0.5: "s", 1.0: "^"}
RHO_LEVELS = (0.0, 0.5, 1.0)
HIGHLIGHT = ("s0_o0_m0", "s0.5_o0.5_m1", "s1_o1_m1")
REP_COND = "s0.5_o0.5_m1"
SWITCH_STEP = 100_000
NULL_PROTOS = ("a_only", "b_only", "joint", "sequential_ab")
NOTES: list[str] = []
SKIPPED: list[str] = []


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
            "grid.linewidth": 0.6,
        }
    )


def _save(fig: plt.Figure, stem: Path) -> list[Path]:
    stem.parent.mkdir(parents=True, exist_ok=True)
    png = stem.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    if png.stat().st_size == 0:
        raise RuntimeError(f"empty figure {png}")
    return [png]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _to_float(value: Any) -> float:
    if value is None or value is True or value is False:
        return float("nan")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if isinstance(value, float) and math.isnan(value):
            return float("nan")
        return float(value)
    text = str(value).strip()
    if text == "":
        return float("nan")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _finite(values: list[float]) -> list[float]:
    return [v for v in values if math.isfinite(v)]


def _mean(values: list[float]) -> float:
    xs = _finite(values)
    return float(np.mean(xs)) if xs else float("nan")


def _median(values: list[float]) -> float:
    xs = _finite(values)
    return float(np.median(xs)) if xs else float("nan")


def _fmt(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "NA"
    return f"{value:.{digits}f}"


def _rankdata(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(1, n + 1, dtype=np.float64)
    xs = x[order]
    i = 0
    while i < n:
        j = i
        while j + 1 < n and xs[j + 1] == xs[i]:
            j += 1
        if j > i:
            avg = float(ranks[order[i : j + 1]].mean())
            ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(xs: list[float], ys: list[float]) -> tuple[float, int]:
    pairs = [(a, b) for a, b in zip(xs, ys, strict=True) if math.isfinite(a) and math.isfinite(b)]
    n = len(pairs)
    if n < 3:
        return float("nan"), n
    a = np.asarray([p[0] for p in pairs], dtype=np.float64)
    b = np.asarray([p[1] for p in pairs], dtype=np.float64)
    ra, rb = _rankdata(a), _rankdata(b)
    ra = ra - ra.mean()
    rb = rb - rb.mean()
    denom = float(np.sqrt((ra**2).sum() * (rb**2).sum()))
    if denom <= 0:
        return float("nan"), n
    return float((ra * rb).sum() / denom), n


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["step"]))
    return rows


def _numeric_row(row: dict[str, str], extra: dict[str, Any] | None = None) -> dict[str, Any]:
    out: dict[str, Any] = dict(row)
    for key, val in row.items():
        if key in {
            "protocol",
            "condition",
            "direction",
            "status",
            "job_id",
            "wandb_url",
            "grok_order",
        }:
            continue
        num = _to_float(val)
        if not (math.isnan(num) and str(val).strip() not in {"", "nan", "NaN"}):
            if str(val).strip() != "" and math.isfinite(num):
                out[key] = num
            elif str(val).strip() == "":
                out[key] = float("nan")
            else:
                out[key] = num
        else:
            out[key] = num
    if extra:
        out.update(extra)
    out["source"] = str(STAMP_GRID / "phase2_relation-matrix_summary.csv")
    return out


def load_grid() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    summary_raw = _read_csv(STAMP_GRID / "phase2_relation-matrix_summary.csv")
    transfer_raw = _read_csv(STAMP_GRID / "phase2_relation-matrix_transfer.csv")
    summary = [_numeric_row(r) for r in summary_raw]
    transfer = [_numeric_row(r) for r in transfer_raw]
    for row in summary:
        row["rho_slot"] = float(row["rho_slot"])
        row["rho_operand"] = float(row["rho_operand"])
        row["rho_mod"] = float(row["rho_mod"])
    for row in transfer:
        row["rho_slot"] = float(row["rho_slot"])
        row["rho_operand"] = float(row["rho_operand"])
        row["rho_mod"] = float(row["rho_mod"])
        row["condition"] = (
            f"s{int(row['rho_slot']) if row['rho_slot'] in (0.0, 1.0) else row['rho_slot']}"
            f"_o{int(row['rho_operand']) if row['rho_operand'] in (0.0, 1.0) else row['rho_operand']}"
            f"_m{int(row['rho_mod']) if row['rho_mod'] in (0.0, 1.0) else row['rho_mod']}"
        )
        # normalize 0.5
        row["condition"] = cond_name(row["rho_slot"], row["rho_operand"], row["rho_mod"])
        row["source"] = str(STAMP_GRID / "phase2_relation-matrix_transfer.csv")
    return summary, transfer


def cond_name(slot: float, operand: float, mod: float) -> str:
    def _tok(v: float, prefix: str) -> str:
        if abs(v - 0.0) < 1e-9:
            return f"{prefix}0"
        if abs(v - 1.0) < 1e-9:
            return f"{prefix}1"
        if abs(v - 0.5) < 1e-9:
            return f"{prefix}0.5"
        return f"{prefix}{v:g}"

    return _tok(slot, "s") + "_" + _tok(operand, "o") + "_" + _tok(mod, "m")


def validate_grid(summary: list[dict[str, Any]], transfer: list[dict[str, Any]]) -> None:
    if len(summary) != 108:
        raise AssertionError(f"expected 108 summary rows, got {len(summary)}")
    prots = {r["protocol"] for r in summary}
    if prots != set(PROTOCOLS):
        raise AssertionError(f"protocol set {prots}")
    counts = Counter(r["protocol"] for r in summary)
    for p in PROTOCOLS:
        if counts[p] != 27:
            raise AssertionError(f"{p} has {counts[p]} rows")
    seq_t = [r for r in transfer if r["protocol"] == "sequential_ab"]
    if len(seq_t) != 27:
        raise AssertionError(f"sequential transfer rows {len(seq_t)}")
    seeds = {(int(r["task_seed"]), int(r["model_seed"])) for r in summary}
    if seeds != {(0, 0)}:
        NOTES.append(f"summary seeds are {sorted(seeds)}; figures still treat points as conditions")
    statuses = {r["status"] for r in summary}
    if statuses != {"ok"}:
        raise AssertionError(f"non-ok status in 108-run grid: {statuses}")


def seq_rows(summary: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [r for r in summary if r["protocol"] == "sequential_ab"]


def by_protocol(summary: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {p: [] for p in PROTOCOLS}
    for row in summary:
        out[row["protocol"]].append(row)
    return out


def grid_matrix(rows: list[dict[str, Any]], value_key: str, slot: float) -> np.ndarray:
    mat = np.full((3, 3), np.nan)
    idx = {v: i for i, v in enumerate(RHO_LEVELS)}
    n = 0
    for row in rows:
        if abs(float(row["rho_slot"]) - slot) > 1e-9:
            continue
        i = idx[float(row["rho_mod"])]
        j = idx[float(row["rho_operand"])]
        val = float(row[value_key])
        if math.isnan(val):
            SKIPPED.append(f"NaN {value_key} slot={slot} {row.get('condition')}")
            continue
        mat[i, j] = val
        n += 1
    if n != 9:
        raise RuntimeError(f"{value_key} slot={slot}: expected 9 cells, got {n}")
    return mat


def _heatmap(
    ax,
    mat: np.ndarray,
    *,
    cmap: str,
    vmin: float,
    vmax: float,
    title: str,
    slot: float,
    highlight: bool = True,
    fmt: str = ".2f",
    center0: bool = False,
) -> Any:
    if center0:
        im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    else:
        im = ax.imshow(mat, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
    ax.set_xticks(range(3), ["0", "0.5", "1"])
    ax.set_yticks(range(3), ["0", "0.5", "1"])
    ax.set_xlabel("operand overlap")
    ax.set_ylabel("modulus overlap")
    ax.set_title(title)
    ax.grid(False)
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            if math.isnan(v):
                ax.text(j, i, "NA", ha="center", va="center", color="k", fontsize=8)
                continue
            ax.text(
                j,
                i,
                format(v, fmt),
                ha="center",
                va="center",
                fontsize=8,
                color="white" if abs(v - vmin) / max(vmax - vmin, 1e-9) > 0.55 else "k",
            )
    if highlight:
        idx = {v: i for i, v in enumerate(RHO_LEVELS)}
        for name in HIGHLIGHT:
            # parse
            m = re.fullmatch(r"s([0-9.]+)_o([0-9.]+)_m([0-9.]+)", name)
            if not m:
                continue
            s, o, md = (float(m.group(1)), float(m.group(2)), float(m.group(3)))
            if abs(s - slot) > 1e-9:
                continue
            ax.add_patch(
                Rectangle(
                    (idx[o] - 0.5, idx[md] - 0.5),
                    1,
                    1,
                    fill=False,
                    ec="k",
                    lw=1.8,
                )
            )
    return im


def job_dir(job_id: str) -> Path:
    return STAMP_GRID / "runs" / job_id


def curve_xy(history: list[dict[str, Any]], key: str) -> tuple[list[float], list[float]]:
    xs, ys = [], []
    for row in history:
        if key not in row:
            continue
        val = _to_float(row[key])
        if not math.isfinite(val):
            continue
        xs.append(float(row["step"]))
        ys.append(val)
    return xs, ys


def _strip(ax, x: float, values: list[float], color: str, *, rng: np.random.Generator) -> None:
    xs = _finite(values)
    jitter = rng.uniform(-0.12, 0.12, size=len(xs))
    ax.scatter(np.full(len(xs), x) + jitter, xs, s=22, c=color, alpha=0.55, edgecolors="none", zorder=3)


def _mean_bar(ax, x: float, values: list[float], color: str) -> float:
    m = _mean(values)
    if math.isfinite(m):
        ax.plot([x - 0.18, x + 0.18], [m, m], color=color, lw=2.4, zorder=5)
    return m


# ---------------------------------------------------------------------------
# Negative-task completeness
# ---------------------------------------------------------------------------


def _parse_null_job(name: str) -> tuple[str, str] | None:
    m = re.match(
        r"^(a_only|b_only|joint|sequential_ab)_(s[0-9.]+_o[0-9.]+_m[0-9.]+)_",
        name,
    )
    if not m:
        return None
    return m.group(1), m.group(2)


def inspect_negative() -> dict[str, Any]:
    jobs_path = STAMP_NULL / "jobs.json"
    planned: list[dict[str, Any]] = []
    if jobs_path.is_file():
        blob = json.loads(jobs_path.read_text(encoding="utf-8"))
        if isinstance(blob, dict):
            candidates = blob.get("jobs") or blob.get("records") or []
            if not candidates:
                for val in blob.values():
                    if (
                        isinstance(val, list)
                        and val
                        and isinstance(val[0], dict)
                        and "job_id" in val[0]
                    ):
                        candidates = val
                        break
        elif isinstance(blob, list):
            candidates = blob
        else:
            candidates = []
        for rec in candidates:
            jid = rec.get("job_id", "")
            parsed = _parse_null_job(jid)
            if parsed is None:
                continue
            prot, cond = parsed
            planned.append(
                {
                    "job_id": jid,
                    "protocol": rec.get("protocol", prot),
                    "condition": cond,
                    "out_dir": rec.get("out_dir", ""),
                }
            )
    run_root = STAMP_NULL / "runs"
    present = {p.name: p for p in run_root.iterdir()} if run_root.is_dir() else {}
    if not planned:
        for name, path in present.items():
            parsed = _parse_null_job(name)
            if parsed is None:
                continue
            prot, cond = parsed
            planned.append({"job_id": name, "protocol": prot, "condition": cond, "out_dir": str(path)})

    expected_conds = [cond_name(s, o, m) for s in RHO_LEVELS for o in RHO_LEVELS for m in RHO_LEVELS]
    expected_jobs = {(p, c) for p in NULL_PROTOS for c in expected_conds}
    status_rows = []
    have: set[tuple[str, str]] = set()
    complete_ok: set[tuple[str, str]] = set()
    for rec in planned:
        key = (rec["protocol"], rec["condition"])
        have.add(key)
        d = run_root / rec["job_id"]
        jr = d / "job_result.json"
        st = "missing_dir"
        a_acc = b_acc = float("nan")
        if jr.is_file():
            payload = json.loads(jr.read_text(encoding="utf-8"))
            st = str(payload.get("status") or "")
            metrics = payload.get("metrics") or {}
            a_acc = _to_float(metrics.get("A_test_acc", payload.get("A_test_acc")))
            b_acc = _to_float(metrics.get("B_test_acc", payload.get("B_test_acc")))
            if st == "ok":
                complete_ok.add(key)
        elif d.is_dir():
            st = "incomplete_no_job_result"
        status_rows.append(
            {
                "protocol": rec["protocol"],
                "condition": rec["condition"],
                "job_id": rec["job_id"],
                "status": st,
                "A_test_acc": a_acc if math.isfinite(a_acc) else "",
                "B_test_acc": b_acc if math.isfinite(b_acc) else "",
                "source": str(d),
            }
        )
    missing = sorted(expected_jobs - {(r["protocol"], r["condition"]) for r in status_rows})
    for prot, cond in missing:
        status_rows.append(
            {
                "protocol": prot,
                "condition": cond,
                "job_id": "",
                "status": "not_started",
                "A_test_acc": "",
                "B_test_acc": "",
                "source": str(STAMP_NULL),
            }
        )

    def has_ok(prot: str, cond: str) -> bool:
        return (prot, cond) in complete_ok

    target_ok = all(has_ok(p, REP_COND) for p in ("a_only", "sequential_ab", "joint"))
    # matched sequential conditions with both baseline (grid) and null
    seq_null_ok = {c for p, c in complete_ok if p == "sequential_ab"}
    matched_seq = sorted(seq_null_ok)
    n_ok = sum(1 for r in status_rows if r["status"] == "ok")
    n_expected = len(expected_jobs)
    complete_grid = n_ok == n_expected and all(
        has_ok(p, c) for p in NULL_PROTOS for c in expected_conds
    )
    info = {
        "n_expected": n_expected,
        "n_ok": n_ok,
        "n_status_rows": len(status_rows),
        "by_protocol_ok": {
            p: sum(1 for q, _c in complete_ok if q == p) for p in NULL_PROTOS
        },
        "target_s0.5_o0.5_m1": {
            p: has_ok(p, REP_COND) for p in NULL_PROTOS
        },
        "target_complete_for_fig": bool(target_ok),
        "matched_sequential_conditions": matched_seq,
        "complete_full_grid": complete_grid,
        "rows": status_rows,
    }
    return info


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------


def fig_2_1() -> list[Path]:
    fig, ax = plt.subplots(figsize=(11.2, 6.4))
    ax.set_xlim(0, 12.2)
    ax.set_ylim(0, 10.2)
    ax.axis("off")
    ax.set_title("Four Phase-2 protocols share initialization; only sequential switches the task")

    def box(x, y, w, h, text, fc, ec="#333"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h, boxstyle="round,pad=0.04", facecolor=fc,
                edgecolor=ec, lw=0.9, alpha=0.95,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    rows = [
        (8.2, "A-only", "Train A only  (0–100k)", COLOR_A + "33", "θA  ·  measure A retention; B is untrained eval"),
        (6.2, "B-only", "Train B only  (0–100k)", COLOR_B + "33", "θB  ·  from-scratch B baseline"),
        (4.2, "Joint", "Train A and B together  (0–100k)", COLOR_JOINT + "33", "θjoint  ·  capacity check"),
        (2.2, "Sequential A→B", "Train A  (0–100k) then B only  (100k–200k)", COLOR_SEQ + "33", "θA→B  ·  A retention & B transfer"),
    ]
    box(0.3, 9.45, 1.6, 0.55, r"$\theta_0$", "#EEE")
    ax.text(2.15, 9.72, "shared initialization", ha="left", va="center", fontsize=8, color="#555")
    for y, name, bar, fc, note in rows:
        ax.text(0.35, y + 0.55, name, fontsize=10, fontweight="bold", va="center")
        box(2.4, y, 6.2, 0.85, bar, fc)
        ax.text(8.8, y + 0.42, note, fontsize=8, va="center")
    ax.plot([5.5, 5.5], [2.1, 3.15], color=COLOR_SEQ, lw=1.6)
    ax.text(5.5, 1.85, "switch step = 100k", ha="center", color=COLOR_SEQ, fontsize=9)
    ax.annotate(
        "",
        xy=(8.55, 2.55),
        xytext=(8.55, 3.05),
        arrowprops=dict(arrowstyle="-|>", color=COLOR_SEQ, lw=1.0),
    )
    ax.text(9.7, 1.15, "A retention: Acc_A after B\nB transfer: B curve vs B-only (exposure AUC)", fontsize=8)
    ax.text(0.35, 0.35, "Schematic only — no experimental numbers.", fontsize=8, color="#666")
    return _save(fig, OUT / "core/fig_2_1_protocol_design")


def fig_2_2(summary: list[dict[str, Any]]) -> list[Path]:
    grouped = by_protocol(summary)
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.15, 1.05]})
    ax = axes[0]
    ax.set_title("A. Final held-out accuracy across 27 overlap conditions")
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("test accuracy")
    rng = np.random.default_rng(0)
    xticks, xlabels = [], []
    for i, prot in enumerate(PROTOCOLS):
        base = i * 2.2
        rows = grouped[prot]
        a_vals = [float(r["A_test_acc"]) for r in rows]
        b_vals = [float(r["B_test_acc"]) for r in rows]
        _strip(ax, base, a_vals, COLOR_A, rng=rng)
        _strip(ax, base + 0.7, b_vals, COLOR_B, rng=rng)
        _mean_bar(ax, base, a_vals, COLOR_A)
        _mean_bar(ax, base + 0.7, b_vals, COLOR_B)
        xticks.append(base + 0.35)
        xlabels.append(PROTO_LABEL[prot])
    ax.set_xticks(xticks, xlabels, rotation=12)
    ax.plot([], [], color=COLOR_A, lw=2.4, label="Task A mean")
    ax.plot([], [], color=COLOR_B, lw=2.4, label="Task B mean")
    ax.scatter([], [], s=22, c="#888", label="one condition (1 seed)")
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.02,
        0.02,
        "Points: 27 overlap conditions · 1 task seed × 1 model seed\n"
        "Spread is variation across conditions, not seed uncertainty",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )

    ax = axes[1]
    ax.set_title("B. Learning curves (representative condition)")
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("step")
    ax.set_ylabel("test accuracy")
    ax.axvline(SWITCH_STEP, color="#666", ls="--", lw=1.0, label="switch = 100k")
    wanted = {
        "sequential_ab": COLOR_SEQ,
        "joint": COLOR_JOINT,
    }
    for prot, color in wanted.items():
        recs = [r for r in grouped[prot] if r["condition"] == REP_COND]
        if not recs:
            SKIPPED.append(f"missing {prot} {REP_COND} history")
            continue
        hist = _load_jsonl(job_dir(recs[0]["job_id"]) / "eval_history.jsonl")
        xa, ya = curve_xy(hist, "A_test_acc")
        xb, yb = curve_xy(hist, "B_test_acc")
        ax.plot(xa, ya, color=color, lw=1.6, label=f"{PROTO_LABEL[prot]} A")
        ax.plot(xb, yb, color=color, lw=1.6, ls="--", label=f"{PROTO_LABEL[prot]} B")
    ax.legend(frameon=False, fontsize=7, ncol=2)
    ax.text(
        0.02,
        0.02,
        f"Representative condition: {REP_COND}\nNot the grid-wide mean",
        transform=ax.transAxes,
        fontsize=7,
        va="bottom",
    )
    fig.suptitle("Capacity is sufficient, but sequential training overwrites Task A", y=1.03)
    return _save(fig, OUT / "core/fig_2_2_protocol_outcomes")


def fig_2_3(summary: list[dict[str, Any]]) -> list[Path]:
    rows = seq_rows(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), layout="constrained")
    last = None
    for ax, slot in zip(axes, RHO_LEVELS, strict=True):
        mat = grid_matrix(rows, "A_test_acc", slot)
        last = _heatmap(
            ax, mat, cmap="YlGnBu", vmin=0, vmax=1,
            title=f"slot overlap = {slot:g}", slot=slot,
        )
    fig.colorbar(last, ax=axes, fraction=0.02, pad=0.02, label="final A test acc")
    fig.suptitle("Task overlap determines which parts of Task A survive", y=1.06)
    fig.text(
        0.5,
        -0.04,
        "Boxes: s0_o0_m0, s0.5_o0.5_m1, s1_o1_m1. High retention can mean B retrained shared A/B "
        "functions, not an isolated Task-A memory. s1_o1_m1 is near-identical A/B maps — not isolation.",
        ha="center",
        fontsize=8,
    )
    return _save(fig, OUT / "core/fig_2_3_overlap_retention")


def _b_training_curve(rec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """B test accuracy vs steps of B training (not global step)."""
    hist = _load_jsonl(job_dir(rec["job_id"]) / "eval_history.jsonl")
    protocol = rec["protocol"]
    switch = _to_float(rec.get("switch_step"))
    if protocol == "sequential_ab" and not math.isfinite(switch):
        switch = float(SWITCH_STEP)
    xs, ys = [], []
    for row in hist:
        acc = _to_float(row.get("B_test_acc"))
        if not math.isfinite(acc):
            continue
        step = float(row["step"])
        if protocol == "sequential_ab":
            if step <= switch:
                continue
            xs.append(step - switch)
        elif protocol == "b_only":
            xs.append(step)
        else:
            continue
        ys.append(acc)
    if not xs:
        return np.array([]), np.array([])
    order = np.argsort(xs)
    return np.asarray(xs, dtype=np.float64)[order], np.asarray(ys, dtype=np.float64)[order]


def _interp_on_grid(xs: np.ndarray, ys: np.ndarray, grid: np.ndarray) -> np.ndarray:
    out = np.full(grid.shape, np.nan, dtype=np.float64)
    if xs.size < 2:
        return out
    mask = (grid >= xs[0]) & (grid <= xs[-1])
    out[mask] = np.interp(grid[mask], xs, ys)
    return out


def fig_2_4(summary: list[dict[str, Any]], transfer: list[dict[str, Any]]) -> list[Path]:
    """Direct comparison: B accuracy during B training, with vs without A first."""
    grouped = by_protocol(summary)
    seq_t = [r for r in transfer if r["protocol"] == "sequential_ab"]
    grid = np.arange(1000.0, 80_000.0 + 1, 1000.0)
    fig = plt.figure(figsize=(12.6, 7.4), layout="constrained")
    gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 0.95])
    curve_axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    bar_ax = fig.add_subplot(gs[1, :])

    for ax, mod in zip(curve_axes, RHO_LEVELS, strict=True):
        seq_rows_m = [r for r in grouped["sequential_ab"] if abs(float(r["rho_mod"]) - mod) < 1e-9]
        b_rows_m = [r for r in grouped["b_only"] if abs(float(r["rho_mod"]) - mod) < 1e-9]
        seq_stack, b_stack = [], []
        for rec in b_rows_m:
            xs, ys = _b_training_curve(rec)
            if xs.size:
                ax.plot(xs, ys, color=COLOR_B, lw=0.8, alpha=0.28)
                b_stack.append(_interp_on_grid(xs, ys, grid))
        for rec in seq_rows_m:
            xs, ys = _b_training_curve(rec)
            if xs.size:
                ax.plot(xs, ys, color=COLOR_SEQ, lw=0.8, alpha=0.28)
                seq_stack.append(_interp_on_grid(xs, ys, grid))
        if b_stack:
            with np.errstate(all="ignore"):
                b_mean = np.nanmean(np.vstack(b_stack), axis=0)
            ax.plot(grid, b_mean, color=COLOR_B, lw=2.4, label="B-only mean")
        if seq_stack:
            with np.errstate(all="ignore"):
                s_mean = np.nanmean(np.vstack(seq_stack), axis=0)
            ax.plot(grid, s_mean, color=COLOR_SEQ, lw=2.4, label="after training A")
        ax.axhline(0.9, color="#888", ls=":", lw=0.8)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(0, 80_000)
        ax.set_title(f"modulus overlap = {mod:g}")
        ax.set_xlabel("B training steps")
        if mod == 0.0:
            ax.set_ylabel("B test accuracy")
            ax.legend(frameon=False, fontsize=7, loc="lower right")
        ax.text(0.02, 0.02, "9 conditions (faint) + mean", transform=ax.transAxes, fontsize=7, va="bottom")

    rng = np.random.default_rng(4)
    width = 0.34
    label_y = []
    for i, mod in enumerate(RHO_LEVELS):
        base = [
            float(r["baseline_b_exposure_steps_to_gen"])
            for r in seq_t
            if abs(float(r["rho_mod"]) - mod) < 1e-9
        ]
        seq_s = [
            float(r["b_exposure_steps_to_gen"])
            for r in seq_t
            if abs(float(r["rho_mod"]) - mod) < 1e-9
        ]
        mb, ms = _mean(base), _mean(seq_s)
        bar_ax.bar(i - width / 2, mb if math.isfinite(mb) else 0, width=width, color=COLOR_B, zorder=2)
        bar_ax.bar(i + width / 2, ms if math.isfinite(ms) else 0, width=width, color=COLOR_SEQ, zorder=2)
        _strip(bar_ax, i - width / 2, base, COLOR_B, rng=rng)
        _strip(bar_ax, i + width / 2, seq_s, COLOR_SEQ, rng=rng)
        if math.isfinite(mb) and math.isfinite(ms):
            faster = mb - ms
            label = (
                f"{_fmt(faster, 0)} steps faster"
                if faster > 0
                else f"{_fmt(-faster, 0)} steps slower"
            )
            label_y.append((i, max(mb, ms), label))
    bar_ax.set_xticks(range(3), ["0", "0.5", "1"])
    bar_ax.set_xlabel("modulus overlap")
    bar_ax.set_ylabel("B steps to 90% test acc")
    bar_ax.set_title("How many B-training steps to reach 90%  ·  shorter bar = A helped B")
    bar_ax.legend(
        handles=[
            Patch(facecolor=COLOR_B, label="B-only (no A)"),
            Patch(facecolor=COLOR_SEQ, label="after training A"),
        ],
        frameon=False,
        loc="upper right",
    )
    _, ymax = bar_ax.get_ylim()
    bar_ax.set_ylim(0, ymax * 1.18)
    for i, y0, label in label_y:
        bar_ax.text(i, y0 + 0.03 * ymax, label, ha="center", va="bottom", fontsize=9)

    n_pos = sum(1 for r in seq_t if float(r["delta_b_exposure_auc"]) > 0)
    fig.suptitle("Training A first speeds up B — mainly when they share moduli", y=1.02)
    fig.text(
        0.5,
        -0.02,
        "Top: x-axis is B training time (sequential counted after the 100k switch). "
        "Bottom: first time B test acc ≥ 0.9; missing runs omitted from the mean (not treated as 0). "
        f"{n_pos}/27 conditions have a higher B-exposure AUC after A. "
        "Faint lines = overlap conditions, not seeds.",
        ha="center",
        fontsize=8,
    )
    return _save(fig, OUT / "core/fig_2_4_forward_transfer")


def fig_appendix_transfer_auc(transfer: list[dict[str, Any]]) -> list[Path]:
    seq = [r for r in transfer if r["protocol"] == "sequential_ab"]
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), layout="constrained")
    vmax = max(
        (
            abs(float(r["delta_b_exposure_auc"]))
            for r in seq
            if math.isfinite(float(r["delta_b_exposure_auc"]))
        ),
        default=0.15,
    )
    vmax = max(vmax, 0.15)
    last = None
    for ax, slot in zip(axes, RHO_LEVELS, strict=True):
        mat = grid_matrix(seq, "delta_b_exposure_auc", slot)
        last = _heatmap(
            ax, mat, cmap="RdBu", vmin=-vmax, vmax=vmax,
            title=f"slot overlap = {slot:g}", slot=slot, fmt=".2f", center0=True,
        )
    fig.colorbar(last, ax=axes, fraction=0.02, pad=0.02, label=r"$\Delta$ B-exposure AUC")
    fig.suptitle("Appendix: B-exposure AUC after A minus B-only", y=1.06)
    return _save(fig, OUT / "appendix/appendix_forward_transfer_auc")


def fig_2_5(summary: list[dict[str, Any]], transfer: list[dict[str, Any]]) -> list[Path]:
    seq = {r["condition"]: r for r in seq_rows(summary)}
    tr = [r for r in transfer if r["protocol"] == "sequential_ab"]
    xs, ys, colors, markers, names = [], [], [], [], []
    for r in tr:
        s = seq[r["condition"]]
        xs.append(float(s["forgetting_A_from_switch"]))
        ys.append(float(r["delta_b_exposure_auc"]))
        colors.append(COLOR_MOD[float(r["rho_mod"])])
        markers.append(MARKER_SLOT[float(r["rho_slot"])])
        names.append(r["condition"])
    rho, n = spearman(xs, ys)
    fig, ax = plt.subplots(figsize=(7.4, 5.4), layout="constrained")
    ax.axhline(0.0, color="#666", ls="--", lw=0.9)
    for x, y, c, m, name in zip(xs, ys, colors, markers, names, strict=True):
        ax.scatter([x], [y], c=c, marker=m, s=48, edgecolors="k", linewidths=0.4, zorder=3)
        if name in HIGHLIGHT:
            offset = {
                "s0_o0_m0": (-72, 10),
                "s0.5_o0.5_m1": (8, -14),
                "s1_o1_m1": (8, 8),
            }[name]
            ax.annotate(name, (x, y), textcoords="offset points", xytext=offset, fontsize=8)
    coef = np.polyfit(xs, ys, 1)
    xr = np.linspace(min(xs), max(xs), 50)
    ax.plot(xr, np.polyval(coef, xr), color="#888", lw=1.0, ls=":")
    ax.set_xlabel("forgetting_A_from_switch")
    ax.set_ylabel(r"$\Delta$ B-exposure AUC")
    ax.set_title("Task A can be forgotten while still accelerating Task B")
    ax.text(0.98, 0.97, "Reuse without\nbehavioral retention", transform=ax.transAxes,
            ha="right", va="top", fontsize=9, color="#333")
    ax.text(0.02, 0.02, f"Spearman ρ={_fmt(rho, 2)}  n={n} conditions\n"
            "Fit is descriptive; not a causal claim",
            transform=ax.transAxes, fontsize=8, va="bottom")
    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MOD[0.0], markeredgecolor="k", label="mod overlap 0"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MOD[0.5], markeredgecolor="k", label="mod overlap 0.5"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=COLOR_MOD[1.0], markeredgecolor="k", label="mod overlap 1"),
        Line2D([0], [0], marker="o", color="k", linestyle="None", label="slot overlap 0"),
        Line2D([0], [0], marker="s", color="k", linestyle="None", label="slot overlap 0.5"),
        Line2D([0], [0], marker="^", color="k", linestyle="None", label="slot overlap 1"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=7, loc="lower right")
    return _save(fig, OUT / "core/fig_2_5_forgetting_transfer_dissociation")


def fig_2_6() -> list[Path]:
    fig, ax = plt.subplots(figsize=(9.2, 7.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title(
        "Shared computation enables transfer; overwritten routing produces forgetting"
    )

    def box(x, y, w, h, text, fc):
        ax.add_patch(
            FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04",
                           facecolor=fc, edgecolor="#333", lw=0.9, alpha=0.95)
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)

    box(2.0, 10.4, 6.0, 1.0, "Task A training", COLOR_A + "55")
    box(2.0, 8.7, 6.0, 1.0, "Shared digit / Fourier features", "#F2CF5B")
    box(2.0, 7.0, 6.0, 1.0, "Operation-specific routing and result binding", "#B279A2")
    box(2.0, 5.3, 6.0, 1.0, "Task B training", COLOR_B + "55")
    box(0.4, 3.2, 4.4, 1.3, "reuses shared features\n→ positive forward transfer", COLOR_JOINT + "66")
    box(5.2, 3.2, 4.4, 1.3, "rewrites routing / binding\n→ Task-A forgetting", COLOR_SEQ + "66")
    box(2.0, 1.2, 6.0, 1.2, "Task token is weakly causal\nA/B fail to form isolated task spaces", "#EEE")
    for y0, y1 in ((10.4, 9.7), (8.7, 8.0), (7.0, 6.3)):
        ax.annotate("", xy=(5, y1), xytext=(5, y0),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0))
    ax.annotate("", xy=(2.6, 4.5), xytext=(4.2, 5.3),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0))
    ax.annotate("", xy=(7.4, 4.5), xytext=(5.8, 5.3),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0))
    ax.annotate("", xy=(5, 2.4), xytext=(5, 3.2),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0))
    ax.text(
        5,
        0.35,
        "Hypothesized mechanism consistent with current evidence — not a fully proven circuit.",
        ha="center",
        fontsize=8,
        color="#555",
    )
    return _save(fig, OUT / "core/fig_2_6_cl_mechanism_summary")


def fig_appendix_marginal(summary: list[dict[str, Any]]) -> list[Path]:
    rows = seq_rows(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharey=True, layout="constrained")
    factors = (
        ("rho_slot", "slot overlap"),
        ("rho_operand", "operand overlap"),
        ("rho_mod", "modulus overlap"),
    )
    rng = np.random.default_rng(2)
    for ax, (key, lab) in zip(axes, factors, strict=True):
        ax.set_title(lab)
        ax.set_ylim(-0.02, 1.02)
        ax.set_xlabel(lab)
        for i, level in enumerate(RHO_LEVELS):
            vals = [float(r["A_test_acc"]) for r in rows if abs(float(r[key]) - level) < 1e-9]
            _strip(ax, i, vals, COLOR_SEQ, rng=rng)
            _mean_bar(ax, i, vals, "k")
        ax.set_xticks(range(3), ["0", "0.5", "1"])
        ax.text(0.02, 0.02, "9 conditions / level\nno seed CI", transform=ax.transAxes, fontsize=7)
    axes[0].set_ylabel("final A test acc (sequential)")
    fig.suptitle("Marginal Task-A retention vs each overlap factor", y=1.04)
    return _save(fig, OUT / "appendix/appendix_overlap_marginal_effects")


def fig_appendix_B(summary: list[dict[str, Any]]) -> list[Path]:
    rows = seq_rows(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), layout="constrained")
    last = None
    for ax, slot in zip(axes, RHO_LEVELS, strict=True):
        mat = grid_matrix(rows, "B_test_acc", slot)
        last = _heatmap(ax, mat, cmap="YlOrBr", vmin=0, vmax=1,
                        title=f"slot overlap = {slot:g}", slot=slot)
    fig.colorbar(last, ax=axes, fraction=0.02, pad=0.02, label="final B test acc")
    fig.suptitle("Task B is learned in almost every sequential condition", y=1.06)
    return _save(fig, OUT / "appendix/appendix_B_final_accuracy")


def fig_appendix_forget(summary: list[dict[str, Any]]) -> list[Path]:
    rows = seq_rows(summary)
    fig, axes = plt.subplots(1, 3, figsize=(11.8, 3.9), layout="constrained")
    last = None
    for ax, slot in zip(axes, RHO_LEVELS, strict=True):
        mat = grid_matrix(rows, "forgetting_A_from_switch", slot)
        last = _heatmap(ax, mat, cmap="YlOrRd", vmin=0, vmax=1,
                        title=f"slot overlap = {slot:g}", slot=slot)
    fig.colorbar(last, ax=axes, fraction=0.02, pad=0.02, label="forgetting_A_from_switch")
    fig.suptitle("Task-A drop from switch to the end of B", y=1.06)
    return _save(fig, OUT / "appendix/appendix_forgetting_from_switch")


def fig_appendix_curves(summary: list[dict[str, Any]]) -> list[Path]:
    rows = {r["condition"]: r for r in seq_rows(summary)}
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharey=True, layout="constrained")
    for ax, name in zip(axes, HIGHLIGHT, strict=True):
        ax.set_title(name)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlabel("step")
        ax.axvline(SWITCH_STEP, color="#666", ls="--", lw=1.0)
        rec = rows[name]
        hist = _load_jsonl(job_dir(rec["job_id"]) / "eval_history.jsonl")
        xa, ya = curve_xy(hist, "A_test_acc")
        xb, yb = curve_xy(hist, "B_test_acc")
        ax.plot(xa, ya, color=COLOR_A, lw=1.6, label="A")
        ax.plot(xb, yb, color=COLOR_B, lw=1.6, label="B")
        ax.legend(frameon=False, fontsize=8)
    axes[0].set_ylabel("test accuracy")
    fig.suptitle("Sequential A→B curves at three overlap extremes", y=1.04)
    return _save(fig, OUT / "appendix/appendix_sequential_curves_selected_conditions")


def fig_appendix_fourier_forget() -> list[Path]:
    rows = [_numeric_row(r) for r in _read_csv(STAMP_GRID / "fourier_A_after_forget.csv")]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2), sharey=False, layout="constrained")
    for ax, ykey, ylab in (
        (axes[0], "emb_spec_cos", "digit-emb spectrum cosine"),
        (axes[1], "unembed_spec_cos", "unembed spectrum cosine"),
    ):
        ax.set_title(ylab)
        ax.set_xlabel("final A test acc (run-level)")
        ax.set_ylim(-0.05, 1.05)
        for r in rows:
            in_b = int(float(r["in_B"]))
            ax.scatter(
                [float(r["A_test"])],
                [float(r[ykey])],
                c=COLOR_B if in_b else COLOR_A,
                s=22,
                alpha=0.7,
                edgecolors="none",
            )
        ax.set_ylabel(ylab)
    axes[0].scatter([], [], c=COLOR_B, label="modulus in B")
    axes[0].scatter([], [], c=COLOR_A, label="modulus not in B")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Fourier structure after Task-A behavioral forgetting", y=1.04)
    fig.text(
        0.5,
        -0.05,
        "Checkpoint-level descriptive evidence. Some spectra can remain similar after Acc_A collapses. "
        "Condition/modulus variation is large. Cosine ≠ intact algorithmic circuit.",
        ha="center",
        fontsize=8,
    )
    return _save(fig, OUT / "appendix/appendix_fourier_after_forgetting")


def fig_appendix_fourier_B() -> list[Path]:
    rows = [_numeric_row(r) for r in _read_csv(STAMP_GRID / "fourier_B_after_A_vs_after_B.csv")]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), layout="constrained")
    # frequency match rates
    ax = axes[0]
    ax.set_title("A. Top-frequency match rate")
    ax.set_ylim(0, 1.05)
    match_ab = _mean([float(r["k_match_final"]) for r in rows])
    match_bb = _mean([float(r["k_match_b_only"]) for r in rows])
    ax.bar([0, 1], [match_ab, match_bb], color=[COLOR_A, COLOR_B], width=0.55)
    ax.set_xticks([0, 1], ["after A vs\nseq-final", "seq-final vs\nB-only"])
    ax.set_ylabel("fraction of (condition, modulus)")

    ax = axes[1]
    ax.set_title("B. Spectrum cosine")
    ax.set_ylim(-0.05, 1.05)
    rng = np.random.default_rng(3)
    _strip(ax, 0, [float(r["spec_cos_A_final"]) for r in rows], COLOR_A, rng=rng)
    _mean_bar(ax, 0, [float(r["spec_cos_A_final"]) for r in rows], "k")
    _strip(ax, 1, [float(r["spec_cos_final_bonly"]) for r in rows], COLOR_B, rng=rng)
    _mean_bar(ax, 1, [float(r["spec_cos_final_bonly"]) for r in rows], "k")
    ax.set_xticks([0, 1], ["after A vs\nseq-final", "seq-final vs\nB-only"])
    ax.set_ylabel("energy-spectrum cosine (skip DC)")

    ax = axes[2]
    ax.set_title("C. Top-frequency energy fraction")
    ax.set_ylim(0, 1.05)
    for i, (key, color, lab) in enumerate(
        (
            ("frac_after_A", COLOR_A, "after A"),
            ("frac_after_B_final", COLOR_SEQ, "after seq B"),
            ("frac_b_only", COLOR_B, "B-only"),
        )
    ):
        vals = [float(r[key]) for r in rows]
        _strip(ax, i, vals, color, rng=rng)
        _mean_bar(ax, i, vals, "k")
    ax.set_xticks([0, 1, 2], ["after A", "after seq B", "B-only"], rotation=15)
    ax.set_ylabel("top-bin energy fraction")
    fig.suptitle("B-modulus Fourier: after A, after sequential B, and B-only", y=1.05)
    return _save(fig, OUT / "appendix/appendix_B_fourier_transfer")


# ---------------------------------------------------------------------------
# Stats + README
# ---------------------------------------------------------------------------


def collect_stats(
    summary: list[dict[str, Any]],
    transfer: list[dict[str, Any]],
    neg: dict[str, Any],
) -> dict[str, Any]:
    grouped = by_protocol(summary)
    seq = seq_rows(summary)
    seq_t = [r for r in transfer if r["protocol"] == "sequential_ab"]
    proto_means = {}
    for p in PROTOCOLS:
        proto_means[p] = {
            "A": _mean([float(r["A_test_acc"]) for r in grouped[p]]),
            "B": _mean([float(r["B_test_acc"]) for r in grouped[p]]),
        }
    forget = [float(r["forgetting_A_from_switch"]) for r in seq]
    auc = [float(r["delta_b_exposure_auc"]) for r in seq_t]
    steps = [float(r["delta_b_exposure_steps_to_gen"]) for r in seq_t]
    steps_ok = _finite(steps)
    marginal = {}
    for key in ("rho_slot", "rho_operand", "rho_mod"):
        marginal[key] = {
            lv: _mean([float(r["A_test_acc"]) for r in seq if abs(float(r[key]) - lv) < 1e-9])
            for lv in RHO_LEVELS
        }
    auc_by_mod = {
        lv: _mean([float(r["delta_b_exposure_auc"]) for r in seq_t if abs(float(r["rho_mod"]) - lv) < 1e-9])
        for lv in RHO_LEVELS
    }
    steps_by_mod = {}
    n_steps_by_mod = {}
    for lv in RHO_LEVELS:
        xs = [
            float(r["delta_b_exposure_steps_to_gen"])
            for r in seq_t
            if abs(float(r["rho_mod"]) - lv) < 1e-9
        ]
        steps_by_mod[lv] = _mean(xs)
        n_steps_by_mod[lv] = len(_finite(xs))
    proto_stamp = STAMP_PROTO / "phase2_protocols_summary.csv"
    proto_n = len(_read_csv(proto_stamp)) if proto_stamp.is_file() else 0
    return {
        "n_summary": len(summary),
        "n_per_protocol": dict(Counter(r["protocol"] for r in summary)),
        "proto_means": proto_means,
        "mean_forget_switch": _mean(forget),
        "n_forget": len(_finite(forget)),
        "n_pos_auc": sum(1 for v in auc if v > 0),
        "mean_auc": _mean(auc),
        "median_auc": _median(auc),
        "n_auc": len(_finite(auc)),
        "n_steps_saved": len(steps_ok),
        "mean_steps_saved": _mean(steps),
        "marginal": marginal,
        "auc_by_mod": auc_by_mod,
        "steps_by_mod": steps_by_mod,
        "n_steps_by_mod": n_steps_by_mod,
        "proto_sanity_n": proto_n,
        "neg": neg,
        "spearman": spearman(
            [
                float({x["condition"]: x for x in seq}[r["condition"]]["forgetting_A_from_switch"])
                for r in seq_t
            ],
            auc,
        ),
    }


def write_readme(
    stats: dict[str, Any],
    written: list[Path],
    neg: dict[str, Any],
) -> None:
    pm = stats["proto_means"]
    mar = stats["marginal"]
    lines = [
        "# Phase 2 continual-learning report figures",
        "",
        "Generated by `scripts/phase2/plot_phase2_report.py`. 只读取已有结果，未改训练代码 / checkpoint。",
        "",
        "## 数据来源",
        "",
        f"- 主网格（108 run）：`{STAMP_GRID.relative_to(ROOT)}`",
        f"- protocol sanity（单 condition 多协议）：`{STAMP_PROTO.relative_to(ROOT)}`（{stats['proto_sanity_n']} 行）",
        f"- 负样本 / task-boundary：`{STAMP_NULL.relative_to(ROOT)}`",
        "",
        "主网格：27 overlap × 4 protocol（a_only / b_only / joint / sequential_ab），"
        "task_seed=0、model_seed=0，wd=0.3，每段 100k steps。",
        "",
        "## 重新计算的关键数字",
        "",
        f"- summary 行数：**{stats['n_summary']}**（期望 108）",
        f"- 每协议行数：{stats['n_per_protocol']}",
        f"- Sequential mean final A / B：{_fmt(pm['sequential_ab']['A'])} / {_fmt(pm['sequential_ab']['B'])}",
        f"- Joint mean final A / B：{_fmt(pm['joint']['A'])} / {_fmt(pm['joint']['B'])}",
        f"- A-only mean final A / B：{_fmt(pm['a_only']['A'])} / {_fmt(pm['a_only']['B'])}",
        f"- B-only mean final A / B：{_fmt(pm['b_only']['A'])} / {_fmt(pm['b_only']['B'])}",
        f"- Sequential mean forgetting_A_from_switch：{_fmt(stats['mean_forget_switch'])}（n={stats['n_forget']}）",
        "- 边际 final A（sequential，每 level 9 个 condition）：",
        f"  - slot overlap 0 / 0.5 / 1：{_fmt(mar['rho_slot'][0.0])} / {_fmt(mar['rho_slot'][0.5])} / {_fmt(mar['rho_slot'][1.0])}",
        f"  - operand overlap 0 / 0.5 / 1：{_fmt(mar['rho_operand'][0.0])} / {_fmt(mar['rho_operand'][0.5])} / {_fmt(mar['rho_operand'][1.0])}",
        f"  - modulus overlap 0 / 0.5 / 1：{_fmt(mar['rho_mod'][0.0])} / {_fmt(mar['rho_mod'][0.5])} / {_fmt(mar['rho_mod'][1.0])}",
        f"- 正 ΔB-AUC condition 数：**{stats['n_pos_auc']}/27**",
        f"- mean / median ΔB-AUC：{_fmt(stats['mean_auc'])} / {_fmt(stats['median_auc'])}（n={stats['n_auc']}）",
        f"- mean ΔB-AUC by modulus overlap 0 / 0.5 / 1：{_fmt(stats['auc_by_mod'][0.0])} / {_fmt(stats['auc_by_mod'][0.5])} / {_fmt(stats['auc_by_mod'][1.0])}",
        f"- mean steps saved (Δ steps-to-gen vs B-only; 缺失不入均值) by mod 0 / 0.5 / 1："
        f"{_fmt(stats['steps_by_mod'][0.0], 0)} (n={stats['n_steps_by_mod'][0.0]}) / "
        f"{_fmt(stats['steps_by_mod'][0.5], 0)} (n={stats['n_steps_by_mod'][0.5]}) / "
        f"{_fmt(stats['steps_by_mod'][1.0], 0)} (n={stats['n_steps_by_mod'][1.0]})",
        f"- Spearman(forgetting_A_from_switch, ΔAUC) ρ={_fmt(stats['spearman'][0], 2)}, n={stats['spearman'][1]}",
        "",
        "## 指标定义",
        "",
        "- **final A/B test acc**：summary 的 `A_test_acc` / `B_test_acc`（held-out test）。",
        "- **forgetting_A_from_switch**：`max Acc_A − Acc_A(after B)`（`src/go4cl/metrics/behavioral.py`）。",
        "- **b_exposure_auc**：把 B 准确率曲线映射到 B-example 暴露轴后的归一化梯形面积"
        "（sequential 只计 switch 之后；joint 的暴露为 step/2）。",
        "- **delta_b_exposure_auc**：sequential（或 joint）相对 **匹配的 B-only** 的 AUC 差；正值表示 A 预训练后 B 曲线更高。",
        "- **delta_b_exposure_steps_to_gen**：B-only 达到 0.9 的暴露步 − 该 run 达到 0.9 的暴露步（正值=更快）。"
        "缺失（从未达到 0.9）**不参与均值**。",
        "- Fourier：`emb_spec_cos` / `unembed_spec_cos` / `spec_cos_*` 为 `energy_cosine(..., skip_dc=True)`，"
        "即 skip-DC 能量谱余弦，不是完整电路相似度。`k_match_*` 为 top frequency 是否相同；"
        "`frac_*` 为 top-bin energy fraction。",
        "",
        "## 负样本实验（task-boundary supervision，不是 replay）",
        "",
        f"- 期望 job 数（4 protocol × 27 condition）：**{neg['n_expected']}**",
        f"- status=ok：**{neg['n_ok']}**",
        f"- 按协议 ok 数：{neg['by_protocol_ok']}",
        f"- 目标 condition `{REP_COND}`：{neg['target_s0.5_o0.5_m1']}",
        f"- 是否生成主图 fig_2_7：**否（网格不完整）**" if not neg["complete_full_grid"] else "- 已生成 fig_2_7。",
        f"- 当前 **不能** 得出 task-boundary 是否降低遗忘的结论。",
        "- 详见 `derived/negative_experiment_status.csv`。",
        "",
        "## Figures",
        "",
    ]
    captions = {
        "fig_2_1_protocol_design": {
            "src": "schematic（无数值）",
            "filter": "四协议",
            "metric": "无",
            "yes": "协议时间线与测量位置。",
            "no": "任何准确率或迁移幅度。",
        },
        "fig_2_2_protocol_outcomes": {
            "src": "108-run summary + eval_history",
            "filter": "Panel A 全部 27 condition；Panel B 仅 s0.5_o0.5_m1",
            "metric": "final A/B test acc；曲线为 A_test_acc / B_test_acc",
            "yes": "容量足够（joint 双高）；sequential 平均严重遗忘 A 且 B 仍能学会。",
            "no": "多种子稳健性；把单条曲线当成 27-condition 平均。",
        },
        "fig_2_3_overlap_retention": {
            "src": "summary sequential_ab",
            "filter": "27 sequential condition",
            "metric": "final A_test_acc",
            "yes": "A 保留随 overlap 变化，全重叠最高。",
            "no": "高保留 = 独立 A 记忆；s1_o1_m1 上的任务隔离。",
        },
        "fig_2_4_forward_transfer": {
            "src": "eval_history.jsonl（sequential vs 匹配 B-only）+ transfer.csv 的 steps-to-gen",
            "filter": "按 modulus overlap 分成 3×9 个 condition",
            "metric": "B_test_acc vs B 训练步；B 达到 0.9 的暴露步（缺失不入均值）",
            "yes": "共享模数时，先训 A 会让 B 的准确率爬得更快。",
            "no": "ΔAUC 热图的逐格细节（见附录）；多种子推断。",
        },
        "fig_2_5_forgetting_transfer_dissociation": {
            "src": "summary + transfer sequential",
            "filter": "n=27",
            "metric": "forgetting_A_from_switch vs delta_b_exposure_auc",
            "yes": "遗忘与正迁移可以同时出现。",
            "no": "相关=因果；复用的具体电路。",
        },
        "fig_2_6_cl_mechanism_summary": {
            "src": "Phase 1 机制 + Phase 2 行为的假设总结",
            "filter": "无新数据",
            "metric": "无",
            "yes": "与现有证据相一致的工作假说。",
            "no": "已证明的完整机制。",
        },
        "appendix_forward_transfer_auc": {
            "src": "phase2_relation-matrix_transfer.csv sequential_ab",
            "filter": "27 sequential vs matched B-only",
            "metric": "delta_b_exposure_auc",
            "yes": "每个 overlap 格子上 AUC 差的正负。",
            "no": "比曲线图更抽象，不宜单独作为主结论。",
        },
        "appendix_overlap_marginal_effects": {
            "src": "sequential summary",
            "filter": "每 level 9 condition",
            "metric": "final A_test_acc",
            "yes": "三个 overlap 因素各自的边际保留。",
            "no": "因素独立性（它们在网格中正交设计，但是单 seed）。",
        },
        "appendix_B_final_accuracy": {
            "src": "sequential summary",
            "filter": "27",
            "metric": "final B_test_acc",
            "yes": "A 保留差异通常不是因为 B 没学会。",
            "no": "B 内部 op 是否均匀学会。",
        },
        "appendix_forgetting_from_switch": {
            "src": "sequential summary",
            "filter": "27",
            "metric": "forgetting_A_from_switch",
            "yes": "switch 后 A 掉多少。",
            "no": "掉点发生在哪一层/哪个频率。",
        },
        "appendix_sequential_curves_selected_conditions": {
            "src": "eval_history.jsonl",
            "filter": "s0_o0_m0, s0.5_o0.5_m1, s1_o1_m1",
            "metric": "A/B test acc vs step",
            "yes": "极端 overlap 下遗忘/迁移的时间进程。",
            "no": "网格平均曲线。",
        },
        "appendix_fourier_after_forgetting": {
            "src": "fourier_A_after_forget.csv",
            "filter": "每 condition 的 A 模数（约 4×27 行）",
            "metric": "emb_spec_cos, unembed_spec_cos vs A_test",
            "yes": "行为遗忘后频谱余弦可仍较高（描述性）。",
            "no": "完整 A 电路仍在。",
        },
        "appendix_B_fourier_transfer": {
            "src": "fourier_B_after_A_vs_after_B.csv",
            "filter": "B 模数 × 27 condition",
            "metric": "k_match_*, spec_cos_*, frac_*",
            "yes": "B 主频/能谱在 A 后、sequential 后、B-only 之间的描述对照。",
            "no": "B 电路被 A 完整预形成。",
        },
    }
    cores = sorted({p.stem for p in written if p.parent.name == "core"})
    apps = sorted({p.stem for p in written if p.parent.name == "appendix"})
    lines.append("### Core")
    for name in cores:
        cap = captions.get(name, {})
        lines.append(f"- `{name}`")
        if cap:
            lines.append(f"  - 数据来源：{cap['src']}")
            lines.append(f"  - 筛选：{cap['filter']}")
            lines.append(f"  - 指标：{cap['metric']}")
            lines.append(f"  - 可支持：{cap['yes']}")
            lines.append(f"  - 不能支持：{cap['no']}")
    lines += ["", "### Appendix", ""]
    for name in apps:
        cap = captions.get(name, {})
        lines.append(f"- `{name}`")
        if cap:
            lines.append(f"  - 数据来源：{cap['src']}")
            lines.append(f"  - 筛选：{cap['filter']}")
            lines.append(f"  - 指标：{cap['metric']}")
            lines.append(f"  - 可支持：{cap['yes']}")
            lines.append(f"  - 不能支持：{cap['no']}")
    lines += ["", "## 主汇报建议", "", "优先使用 fig_2_2 … fig_2_6。负样本图未进入主报告。", ""]
    if NOTES:
        lines += ["## Notes", ""] + [f"- {n}" for n in NOTES]
    if SKIPPED:
        lines += ["", "## Skipped / incomplete", ""] + [f"- {s}" for s in SKIPPED]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _wipe_figures() -> None:
    for sub in ("core", "appendix"):
        d = OUT / sub
        d.mkdir(parents=True, exist_ok=True)
        for path in list(d.glob("*.png")) + list(d.glob("*.pdf")):
            path.unlink()
    (OUT / "derived").mkdir(parents=True, exist_ok=True)


def main() -> None:
    argparse.ArgumentParser(description=__doc__).parse_args()
    _style()
    _wipe_figures()
    summary, transfer = load_grid()
    validate_grid(summary, transfer)
    neg = inspect_negative()
    _write_csv(
        OUT / "derived/negative_experiment_status.csv",
        neg["rows"],
        ["protocol", "condition", "job_id", "status", "A_test_acc", "B_test_acc", "source"],
    )

    written: list[Path] = []
    written += fig_2_1()
    written += fig_2_2(summary)
    written += fig_2_3(summary)
    written += fig_2_4(summary, transfer)
    written += fig_2_5(summary, transfer)
    written += fig_2_6()
    if neg["complete_full_grid"]:
        NOTES.append("Negative grid unexpectedly complete; fig_2_7 was not implemented in this run.")
    else:
        NOTES.append(
            f"Negative/task-boundary experiment incomplete: {neg['n_ok']}/{neg['n_expected']} ok; "
            "fig_2_7 not generated."
        )
    written += fig_appendix_marginal(summary)
    written += fig_appendix_transfer_auc(transfer)
    written += fig_appendix_B(summary)
    written += fig_appendix_forget(summary)
    written += fig_appendix_curves(summary)
    written += fig_appendix_fourier_forget()
    written += fig_appendix_fourier_B()

    stats = collect_stats(summary, transfer, neg)
    proto_rows = []
    grouped = by_protocol(summary)
    for p in PROTOCOLS:
        proto_rows.append(
            {
                "protocol": p,
                "n": len(grouped[p]),
                "mean_A_test_acc": _mean([float(r["A_test_acc"]) for r in grouped[p]]),
                "mean_B_test_acc": _mean([float(r["B_test_acc"]) for r in grouped[p]]),
                "source": str(STAMP_GRID / "phase2_relation-matrix_summary.csv"),
            }
        )
    _write_csv(
        OUT / "derived/protocol_means.csv",
        proto_rows,
        ["protocol", "n", "mean_A_test_acc", "mean_B_test_acc", "source"],
    )
    write_readme(stats, written, neg)

    pngs = [p for p in written if p.suffix == ".png"]
    print(f"wrote {len(pngs)} PNG under {OUT}")
    for p in sorted(pngs):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
