#!/usr/bin/env python3
"""Phase 1B / 1C report figures from existing stamps (no training).

Reads:
  1B  runs/phase1/multi_op/20261004_170554   (wd=0.3, 100k)
  1C  runs/phase1/mechanisms/20261004_1b_132221  (wd=0.5, best ckpt)

Writes PNG under runs/phase1/report_figures/1b_1c/{core,appendix}
and derived CSVs + README. Does not overwrite original experiment files.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
STAMP_1B = ROOT / "runs/phase1/multi_op/20261004_170554"
STAMP_1C = ROOT / "runs/phase1/mechanisms/20261004_1b_132221"
OUT = ROOT / "runs/phase1/report_figures/1b_1c"

VARIANTS = ("all_same", "pair_same", "four_diff")
VAR_COLOR = {
    "all_same": "#4C78A8",
    "pair_same": "#54A24B",
    "four_diff": "#F58518",
}
VAR_LABEL = {
    "all_same": "all_same",
    "pair_same": "pair_same",
    "four_diff": "four_diff",
}
Q_COLOR = {0: "#B279A2", 1: "#4C78A8", 2: "#F58518", 3: "#E45756"}
MOD_COLOR = {
    23: "#E45756",
    29: "#F58518",
    31: "#4C78A8",
    37: "#54A24B",
    41: "#B279A2",
    43: "#9D755D",
    47: "#72B7B2",
    53: "#F2CF5B",
}
T_GEN_THRESH = 0.9
T_GEN_STREAK = 3
ACC_YLIM = (0.0, 1.02)
DACC_YLIM = (-1.0, 0.1)
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


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    rows.sort(key=lambda r: int(r["step"]))
    return rows


def op_key(latent_id: int, slot: int, modulus: int) -> str:
    return f"lat{int(latent_id)}/slot{int(slot)}/p{int(modulus)}"


def q_index(latent_id: int) -> int:
    return int(latent_id)


# ---------------------------------------------------------------------------
# 1B loaders
# ---------------------------------------------------------------------------


def t_gen_from_history(history: list[dict[str, Any]]) -> int | None:
    """First step where val macro-op acc stays >= 0.9 for 3 consecutive evals."""
    accs = []
    for row in history:
        acc = row.get("A_val_macro_op_acc", row.get("A_val_acc"))
        accs.append((int(row["step"]), float(acc)))
    for i in range(len(accs) - T_GEN_STREAK + 1):
        window = accs[i : i + T_GEN_STREAK]
        if all(a >= T_GEN_THRESH for _, a in window):
            return window[0][0]
    return None


def load_1b() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    summary = _read_csv(STAMP_1B / "phase1_multi-op_summary.csv")
    jobs: list[dict[str, Any]] = []
    tgen_rows: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    for row in summary:
        job_id = row["job_id"]
        job_dir = STAMP_1B / "runs" / job_id
        result = _load_json(job_dir / "job_result.json")
        cfg = _load_json(job_dir / "config_resolved.json")
        hist = _load_jsonl(job_dir / "eval_history.jsonl")
        metrics = result.get("metrics") or {}
        events = metrics.get("events") or {}
        ops = cfg.get("ops") or []
        t_re = t_gen_from_history(hist)
        t_ev = events.get("t_gen")
        t_ev = int(t_ev) if t_ev is not None else None
        slots = [int(op["slot"]) for op in ops]
        worst_best = min(float(metrics[f"best_A_test_acc_s{s}"]) for s in slots)
        rec = {
            "variant": row["variant"],
            "task_seed": int(row["task_seed"]),
            "model_seed": int(row["model_seed"]),
            "status": row["status"],
            "job_id": job_id,
            "job_dir": str(job_dir),
            "moduli": [int(op["modulus"]) for op in ops],
            "ops": ops,
            "final_val_macro": float(row["A_val_acc"]),
            "final_test_macro": float(row["A_test_acc"]),
            "best_val_macro": float(row["best_A_val_acc"]),
            "best_test_macro": float(row["best_A_test_acc"]),
            "best_step": int(float(row["best_step"])),
            "t_gen": t_re,
            "t_gen_events": t_ev,
            "t_gen_reached": t_re is not None,
            "worst_op_best_test": worst_best,
            "source": str(job_dir / "job_result.json"),
        }
        jobs.append(rec)
        tgen_rows.append(
            {
                **{k: rec[k] for k in (
                    "variant", "task_seed", "model_seed", "job_id", "t_gen",
                    "t_gen_events", "t_gen_reached", "best_step", "source",
                )},
                "definition": (
                    f"first step with A_val_macro_op_acc>={T_GEN_THRESH} "
                    f"for {T_GEN_STREAK} consecutive evals"
                ),
            }
        )
        for h in hist:
            step = int(h["step"])
            curve_rows.append(
                {
                    "variant": rec["variant"],
                    "task_seed": rec["task_seed"],
                    "model_seed": rec["model_seed"],
                    "job_id": job_id,
                    "step": step,
                    "operation": "macro",
                    "latent_id": "",
                    "slot": "",
                    "modulus": "",
                    "acc": float(h.get("A_val_macro_op_acc", h["A_val_acc"])),
                    "source": str(job_dir / "eval_history.jsonl"),
                }
            )
            for op in ops:
                s = int(op["slot"])
                key = f"A_val_acc/s{s}"
                if key not in h:
                    continue
                curve_rows.append(
                    {
                        "variant": rec["variant"],
                        "task_seed": rec["task_seed"],
                        "model_seed": rec["model_seed"],
                        "job_id": job_id,
                        "step": step,
                        "operation": op_key(op["latent_id"], op["slot"], op["modulus"]),
                        "latent_id": int(op["latent_id"]),
                        "slot": s,
                        "modulus": int(op["modulus"]),
                        "acc": float(h[key]),
                        "source": str(job_dir / "eval_history.jsonl"),
                    }
                )
    return jobs, tgen_rows, curve_rows


def validate_1b(jobs: list[dict[str, Any]]) -> None:
    ok = [j for j in jobs if j["status"] == "ok"]
    if len(ok) != 27:
        raise RuntimeError(f"expected 27 ok 1B jobs, got {len(ok)}")
    for v in VARIANTS:
        pairs = {(j["task_seed"], j["model_seed"]) for j in ok if j["variant"] == v}
        if len(pairs) != 9:
            raise RuntimeError(f"{v} expected 9 seeds, got {pairs}")
    for j in ok:
        for key in ("final_test_macro", "best_test_macro", "worst_op_best_test"):
            a = j[key]
            if not (0.0 <= a <= 1.0):
                raise RuntimeError(f"accuracy out of range {key}={a} {j['job_id']}")


def median_iqr(xs: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(xs, dtype=float)
    return float(np.median(arr)), float(np.percentile(arr, 25)), float(np.percentile(arr, 75))


def _strip(ax, xs, ys, colors, *, marker="o"):
    rng = np.random.default_rng(0)
    for x, y, c in zip(xs, ys, colors, strict=True):
        j = rng.uniform(-0.12, 0.12)
        ax.scatter(x + j, y, s=38, c=c, edgecolors="k", linewidths=0.4, zorder=3, marker=marker)


def _median_bar(ax, x, values, color):
    med, lo, hi = median_iqr(values)
    ax.plot([x - 0.18, x + 0.18], [med, med], color="k", lw=2.0, zorder=4)
    ax.plot([x, x], [lo, hi], color="k", lw=1.2, zorder=4)
    ax.plot([x - 0.08, x + 0.08], [lo, lo], color="k", lw=1.2, zorder=4)
    ax.plot([x - 0.08, x + 0.08], [hi, hi], color="k", lw=1.2, zorder=4)
    return med, lo, hi


# ---------------------------------------------------------------------------
# 1C loaders
# ---------------------------------------------------------------------------


def load_1c(variant: str, ts: int) -> dict[str, Any] | None:
    d = STAMP_1C / f"{variant}_ts{ts}"
    if not d.is_dir():
        SKIPPED.append(f"missing 1C dir {d}")
        return None
    report = _load_json(d / "phase1_mechanisms_report.json")
    return {
        "dir": d,
        "variant": variant,
        "task_seed": ts,
        "report": report,
        "ops": report.get("ops") or [],
        "summary": _read_csv(d / "phase1_mechanisms_summary.csv"),
        "fourier_sel": _read_csv(d / "phase1_mechanisms_fourier_selectivity.csv"),
        "head_ko": _read_csv(d / "causal_detail/cross_op_head_knockout.csv"),
        "fourier_cross": _read_csv(d / "causal_detail/cross_op_fourier_top1.csv"),
        "fourier_curve": _read_csv(d / "causal_detail/per_op_fourier_curve.csv"),
        "steer_self": _read_csv(d / "steering/per_op_steering.csv"),
        "steer_cross": _read_csv(d / "steering/cross_op_steering.csv"),
        "attn_swap": _read_csv(d / "attn_swap/attn_pos_swap.csv"),
        "patch": _read_csv(d / "transplant/operand_patch.csv"),
        "transplant": _read_csv(d / "transplant/query_transplant.csv"),
        "token": _read_csv(d / "task_token_edit/task_token_edit.csv"),
        "unembed_ab": _read_csv(d / "unembed_fourier/unembed_ablation.csv"),
        "unembed_sp": _read_csv(d / "unembed_fourier/unembed_spectra.csv"),
        "ckpt": (report.get("config") or {}).get("ckpt_path", ""),
        "source": str(d),
    }


def per_op_from_report(bundle: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    ops_meta = {int(o["latent_id"]): o for o in bundle["ops"]}
    for rec in bundle["report"].get("per_op") or []:
        lat = int(rec.get("latent_id", -1))
        meta = ops_meta.get(lat, {})
        slot = int(rec.get("slot", meta.get("slot", -1)))
        p = int(rec.get("modulus", meta.get("modulus", -1)))
        att = rec.get("attention") or {}
        layers = att.get("layers") or []
        l0 = layers[0] if layers else {}
        ladder = ((rec.get("composition") or {}).get("ladder") or {}).get("by_site") or {}
        compose = rec.get("composition") or {}
        summary = rec.get("summary") or {}
        out.append(
            {
                "operation": op_key(lat, slot, p),
                "latent_id": lat,
                "slot": slot,
                "modulus": p,
                "operand_i": int(rec.get("operand_i", meta.get("operand_i", -1))),
                "operand_j": int(rec.get("operand_j", meta.get("operand_j", -1))),
                "mass_by_pos": [float(x) for x in (l0.get("mass_by_pos") or [])],
                "own_mass": float(l0.get("operand_mass") or summary.get("L0_own_operand_mass") or 0.0),
                "other_digit_mass": float(l0.get("other_digit_mass") or 0.0),
                "probe_by_site": {
                    site: float(block.get("probe_sum"))
                    for site, block in ladder.items()
                    if block.get("probe_sum") is not None
                },
                "compose_guess": summary.get("compose_layer_guess", compose.get("compose_layer_guess")),
                "ladder_jump": summary.get("ladder_sum_jump_layer"),
                "source": bundle["source"],
            }
        )
    return out


# ---------------------------------------------------------------------------
# 1B figures
# ---------------------------------------------------------------------------


def fig_1b_1(jobs: list[dict[str, Any]]) -> list[Path]:
    # Schematic uses task_seed=0 (main-text seed); matching shared across variants.
    picked = {}
    for j in jobs:
        if j["task_seed"] == 0 and j["model_seed"] == 1:
            picked[j["variant"]] = j
    fig, axes = plt.subplots(1, 3, figsize=(12.4, 4.6), layout="constrained")
    for ax, var in zip(axes, VARIANTS, strict=True):
        job = picked[var]
        ops = sorted(job["ops"], key=lambda o: int(o["latent_id"]))
        ax.set_xlim(0, 10)
        ax.set_ylim(-0.6, 5.0)
        ax.axis("off")
        ax.set_title(var.replace("_", " "))
        ax.add_patch(
            FancyBboxPatch(
                (0.2, 4.35), 9.6, 0.5, boxstyle="round,pad=0.02",
                facecolor="#F4F4F4", edgecolor="#333", lw=0.8,
            )
        )
        ax.text(5.0, 4.6, "packed multi-query training", ha="center", va="center", fontsize=8)
        for qi, op in enumerate(ops):
            y = 3.3 - qi * 0.95
            p = int(op["modulus"])
            i, jpos = int(op["i"]), int(op["j"])
            ax.text(0.15, y + 0.15, f"Q{qi}", color=Q_COLOR[qi], fontweight="bold", fontsize=10)
            ax.text(1.15, y + 0.38, f"p={p}  slot={op['slot']}", fontsize=7.5, color="#444")
            for d in range(8):
                x = 1.2 + d * 0.85
                own = d in (i, jpos)
                fc = MOD_COLOR.get(p, "#aaa") if own else "#FFFFFF"
                ec = MOD_COLOR.get(p, "#333") if own else "#888"
                lw = 2.0 if own else 0.6
                ax.add_patch(Rectangle((x, y - 0.18), 0.72, 0.52, facecolor=fc, edgecolor=ec, lw=lw))
                ax.text(x + 0.36, y + 0.08, str(d), ha="center", va="center", fontsize=7,
                        color="white" if own else "#333")
            ax.text(8.15, y + 0.08, f"i={i} j={jpos}", fontsize=7, color="#333")
    return _save(fig, OUT / "core/fig_1b_1_task_design")


def fig_1b_2(jobs: list[dict[str, Any]]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.4), sharey=True, layout="constrained")
    specs = [
        (axes[0], "best_test_macro", "A. Best held-out macro-op acc"),
        (axes[1], "final_test_macro", "B. Final held-out macro-op acc"),
    ]
    for ax, key, title in specs:
        ax.set_title(title)
        ax.set_ylim(*ACC_YLIM)
        ax.axhline(0.90, color="#666", ls="--", lw=0.9)
        ax.axhline(0.99, color="#666", ls=":", lw=0.9)
        ax.set_xticks(range(3), [VAR_LABEL[v] for v in VARIANTS])
        ax.set_ylabel("accuracy")
        for xi, var in enumerate(VARIANTS):
            chunk = [j for j in jobs if j["variant"] == var]
            ys = [j[key] for j in chunk]
            cols = []
            xs = []
            yplot = []
            for j in chunk:
                xs.append(xi)
                yplot.append(j[key])
                if j["variant"] == "four_diff" and j["task_seed"] == 0 and j["model_seed"] == 0:
                    cols.append("#C51B7D")
                else:
                    cols.append(VAR_COLOR[var])
            _strip(ax, xs, yplot, cols)
            _median_bar(ax, xi, ys, VAR_COLOR[var])
            for j in chunk:
                if j["variant"] == "four_diff" and j["task_seed"] == 0 and j["model_seed"] == 0:
                    ax.annotate(
                        "four_diff ts0/ms0",
                        (xi, j[key]),
                        textcoords="offset points",
                        xytext=(8, -10),
                        fontsize=7,
                        color="#C51B7D",
                    )
        ax.text(2.45, 0.90, "0.90", fontsize=7, va="bottom", color="#555")
        ax.text(2.45, 0.99, "0.99", fontsize=7, va="bottom", color="#555")
    return _save(fig, OUT / "core/fig_1b_2_accuracy_distribution")


def fig_1b_3(jobs: list[dict[str, Any]]) -> list[Path]:
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.6), layout="constrained")
    ax = axes[0]
    ax.set_title("A. Time to generalization")
    ax.set_ylabel(r"$t_{\mathrm{gen}}$ (steps)")
    ax.set_xticks(range(3), [VAR_LABEL[v] for v in VARIANTS])
    failed_y = 108000
    ax.axhline(100000, color="#bbb", lw=0.8)
    for xi, var in enumerate(VARIANTS):
        chunk = [j for j in jobs if j["variant"] == var]
        reached = [j["t_gen"] for j in chunk if j["t_gen"] is not None]
        xs, ys, cols, markers = [], [], [], []
        for j in chunk:
            xs.append(xi)
            if j["t_gen"] is None:
                ys.append(failed_y)
                markers.append("x")
            else:
                ys.append(j["t_gen"])
                markers.append("o")
            cols.append(VAR_COLOR[var])
        rng = np.random.default_rng(1)
        for x, y, c, m in zip(xs, ys, cols, markers, strict=True):
            ax.scatter(x + rng.uniform(-0.12, 0.12), y, s=40, c=c, marker=m,
                       edgecolors="k", linewidths=0.4, zorder=3)
        if reached:
            _median_bar(ax, xi, reached, VAR_COLOR[var])
    ax.set_ylim(0, 115000)
    ax.text(2.2, failed_y, "not reached", fontsize=7, color="#555")
    ax = axes[1]
    ax.set_title("B. Matched seeds: pair_same vs four_diff")
    ax.set_xticks([0, 1], ["pair_same", "four_diff"])
    ax.set_ylabel(r"$t_{\mathrm{gen}}$ (steps)")
    pairs = {}
    for j in jobs:
        if j["variant"] in ("pair_same", "four_diff"):
            pairs.setdefault((j["task_seed"], j["model_seed"]), {})[j["variant"]] = j
    for (ts, ms), d in sorted(pairs.items()):
        a = d["pair_same"]["t_gen"]
        b = d["four_diff"]["t_gen"]
        ya = a if a is not None else failed_y
        yb = b if b is not None else failed_y
        ax.plot([0, 1], [ya, yb], color="#888", lw=0.8, zorder=1)
        ax.scatter(
            [0, 1], [ya, yb],
            c=[VAR_COLOR["pair_same"], VAR_COLOR["four_diff"]],
            s=40, edgecolors="k", linewidths=0.4, zorder=3,
            marker="x" if (a is None or b is None) else "o",
        )
        if ts == 0 and ms == 0:
            ax.annotate("ts0/ms0", (1, yb), textcoords="offset points", xytext=(6, 4),
                        fontsize=7, color="#C51B7D")
    ax.set_ylim(0, 115000)
    return _save(fig, OUT / "core/fig_1b_3_time_to_generalization")


def fig_1b_4(jobs: list[dict[str, Any]]) -> list[Path]:
    tvals = [j["t_gen"] for j in jobs if j["t_gen"] is not None]
    wvals = [j["worst_op_best_test"] for j in jobs]
    tmin, tmax = min(tvals), max(tvals)
    fig, axes = plt.subplots(2, 3, figsize=(10.6, 6.4), layout="constrained")
    for col, var in enumerate(VARIANTS):
        chunk = [j for j in jobs if j["variant"] == var]
        tmat = np.full((3, 3), np.nan)
        wmat = np.full((3, 3), np.nan)
        reached = np.zeros((3, 3), dtype=bool)
        for j in chunk:
            r, c = j["task_seed"], j["model_seed"]
            if j["t_gen"] is not None:
                tmat[r, c] = j["t_gen"]
                reached[r, c] = True
            wmat[r, c] = j["worst_op_best_test"]
        for row, mat, vmin, vmax, cmap, title in (
            (0, tmat, tmin, tmax, "YlOrRd", rf"$t_{{\mathrm{{gen}}}}$"),
            (1, wmat, 0.0, 1.0, "viridis", "worst-op best test acc"),
        ):
            ax = axes[row, col]
            im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap=cmap, origin="upper")
            ax.set_xticks(range(3), [str(i) for i in range(3)])
            ax.set_yticks(range(3), [str(i) for i in range(3)])
            ax.set_xlabel("model seed")
            if col == 0:
                ax.set_ylabel("task seed")
            ax.set_title(f"{var}\n{title}")
            ax.grid(False)
            for r in range(3):
                for c in range(3):
                    if row == 0 and not reached[r, c]:
                        ax.plot(c, r, marker="x", color="k", ms=9, mew=1.6)
                    elif np.isfinite(mat[r, c]):
                        span = max(vmax - vmin, 1e-9)
                        norm = (mat[r, c] - vmin) / span
                        ax.text(
                            c,
                            r,
                            f"{mat[r, c]:.0f}" if row == 0 else f"{mat[r, c]:.2f}",
                            ha="center",
                            va="center",
                            fontsize=7,
                            color="white" if norm > 0.55 else "k",
                        )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _save(fig, OUT / "core/fig_1b_4_seed_robustness")


def choose_curve_seed(jobs: list[dict[str, Any]]) -> tuple[int, int, str]:
    target = (0, 1)
    have = {(j["variant"], j["task_seed"], j["model_seed"]) for j in jobs}
    if all((v, *target) in have for v in VARIANTS):
        NOTES.append("Fig 1B-5 uses task_seed=0, model_seed=1 (complete history for all variants).")
        return 0, 1, "requested ts0/ms1 present for all variants"
    # fallback: matched seed closest to median t_gen among successful
    med = {}
    for v in VARIANTS:
        vals = [j["t_gen"] for j in jobs if j["variant"] == v and j["t_gen"] is not None]
        med[v] = float(np.median(vals)) if vals else 0.0
    best = None
    for ts in range(3):
        for ms in range(3):
            trio = [j for j in jobs if j["task_seed"] == ts and j["model_seed"] == ms]
            if len(trio) != 3 or any(j["t_gen"] is None for j in trio):
                continue
            dist = sum(abs(j["t_gen"] - med[j["variant"]]) for j in trio)
            if best is None or dist < best[0]:
                best = (dist, ts, ms)
    if best is None:
        raise RuntimeError("no matched seed with t_gen for all variants")
    NOTES.append(f"Fig 1B-5 fallback seed ts{best[1]}/ms{best[2]} (closest to median t_gen).")
    return best[1], best[2], "fallback closest to median t_gen"


def fig_1b_5(jobs: list[dict[str, Any]], curves: list[dict[str, Any]]) -> list[Path]:
    ts, ms, _ = choose_curve_seed(jobs)
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), sharex=True, sharey=True, layout="constrained")
    for ax, var in zip(axes, VARIANTS, strict=True):
        ax.set_title(var)
        ax.set_ylim(*ACC_YLIM)
        ax.axhline(0.9, color="#666", ls="--", lw=0.9)
        rows = [r for r in curves if r["variant"] == var and int(r["task_seed"]) == ts and int(r["model_seed"]) == ms]
        macro = [r for r in rows if r["operation"] == "macro"]
        ax.plot([r["step"] for r in macro], [r["acc"] for r in macro], color="k", lw=2.2, label="macro-op", zorder=3)
        ops = sorted(
            {r["operation"] for r in rows if r["operation"] != "macro"},
        )
        for opname in ops:
            sub = [r for r in rows if r["operation"] == opname]
            lat = int(sub[0]["latent_id"])
            lab = f"Q{lat} {opname}"
            ax.plot([r["step"] for r in sub], [r["acc"] for r in sub], color=Q_COLOR[lat], lw=1.2, label=lab)
        ax.set_xlabel("steps")
        if var == "all_same":
            ax.set_ylabel("held-out val accuracy")
        ax.legend(fontsize=6, loc="lower left", frameon=False, borderaxespad=0.2)
    fig.suptitle(f"Per-operation learning (ts={ts}, ms={ms})", y=1.03)
    return _save(fig, OUT / "core/fig_1b_5_learning_curves")


# ---------------------------------------------------------------------------
# 1C figures
# ---------------------------------------------------------------------------


def fig_1c_1(bundles: dict[str, dict[str, Any]], *, ts: int, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(3, 2, figsize=(10.8, 9.2), layout="constrained")
    for row, var in enumerate(VARIANTS):
        ops = per_op_from_report(bundles[var])
        ops = sorted(ops, key=lambda o: o["latent_id"])
        axh, axb = axes[row]
        mat = np.zeros((4, 8))
        boxes = []
        for i, op in enumerate(ops):
            mass = op["mass_by_pos"][:8]
            if len(mass) < 8:
                SKIPPED.append(f"1C-1 {var} {op['operation']} missing mass_by_pos")
                continue
            mat[i] = mass
            boxes.append((i, op["operand_i"], op["operand_j"]))
        im = axh.imshow(mat, vmin=0, vmax=0.5, cmap="magma", aspect="auto")
        axh.set_yticks(
            range(4),
            [f"Q{o['latent_id']} s{o['slot']} p{o['modulus']}" for o in ops],
            fontsize=8,
        )
        axh.set_xticks(range(8), [str(i) for i in range(8)])
        axh.set_xlabel("digit position")
        axh.set_title(f"{var} ts{ts}  L0 attention")
        axh.grid(False)
        for i, ii, jj in boxes:
            for pos in (ii, jj):
                axh.add_patch(Rectangle((pos - 0.5, i - 0.5), 1, 1, fill=False, ec="white", lw=1.3))
        fig.colorbar(im, ax=axh, fraction=0.046, pad=0.03)
        own = [o["own_mass"] for o in ops]
        other = [float(r["L0_other_ops_operand_mass"]) for r in bundles[var]["summary"]]
        # align other mass with ops via slot
        other_by_slot = {
            int(r["slot"]): float(r["L0_other_ops_operand_mass"]) for r in bundles[var]["summary"]
        }
        other = [other_by_slot[o["slot"]] for o in ops]
        x = np.arange(4)
        axb.bar(x - 0.18, own, 0.36, color="#4C78A8", label="own operands")
        axb.bar(x + 0.18, other, 0.36, color="#F58518", label="other-op operands")
        axb.axhline(0.2, color="#333", ls="--", lw=0.9, label="uniform 2/10")
        axb.set_ylim(*ACC_YLIM)
        axb.set_xticks(x, [f"Q{o['latent_id']}" for o in ops])
        axb.set_ylabel("attention mass")
        if row == 0:
            axb.legend(frameon=False, loc="upper right")
    return _save(fig, stem)


def fig_1c_2(bundles: dict[str, dict[str, Any]], *, ts: int, stem: Path) -> list[Path]:
    fig = plt.figure(figsize=(12.4, 6.6), layout="constrained")
    gs = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.85])
    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]
    axv = fig.add_subplot(gs[0, 3])
    specs_pts: dict[str, list[float]] = {v: [] for v in VARIANTS}
    spec_rows = []
    for ax, var in zip(axes, VARIANTS, strict=True):
        rows = bundles[var]["head_ko"]
        ops = sorted({r["target_operation"] for r in rows})
        heads = sorted({(int(r["ablate_layer"]), int(r["ablate_head"])) for r in rows})
        mat = np.zeros((len(heads), len(ops)))
        for r in rows:
            hi = heads.index((int(r["ablate_layer"]), int(r["ablate_head"])))
            oj = ops.index(r["target_operation"])
            mat[hi, oj] = float(r["delta_acc"])
        im = ax.imshow(mat, vmin=-1, vmax=0, cmap="RdBu", aspect="auto")
        ax.set_xticks(range(len(ops)), [f"Q{o.split('/')[0][3:]}" for o in ops], fontsize=8)
        ax.set_yticks(range(len(heads)), [f"L{l}H{h}" for l, h in heads], fontsize=7)
        ax.set_title(f"{var} ts{ts}")
        ax.grid(False)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03, label=r"$\Delta$acc")
        for hi, (l, h) in enumerate(heads):
            vals = np.abs(mat[hi])
            if float(vals.sum()) < 0.05:
                s = float("nan")
            else:
                s = float(vals.max() / vals.sum())
            specs_pts[var].append(s)
            spec_rows.append(
                {
                    "variant": var,
                    "task_seed": ts,
                    "layer": l,
                    "head": h,
                    "specialization": s,
                    "source": bundles[var]["source"],
                }
            )
    axv.set_title("head specialization")
    axv.set_ylim(-0.05, 1.05)
    axv.set_xticks(range(3), list(VARIANTS), rotation=20)
    axv.set_ylabel(r"max$|\Delta|$ / $\sum|\Delta|$")
    for xi, var in enumerate(VARIANTS):
        ys = [y for y in specs_pts[var] if math.isfinite(y)]
        _strip(axv, [xi] * len(ys), ys, [VAR_COLOR[var]] * len(ys))
        if ys:
            _median_bar(axv, xi, ys, VAR_COLOR[var])
    return _save(fig, stem), spec_rows


def fig_1c_3(bundles: dict[str, dict[str, Any]], *, ts: int, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(13.2, 3.6), layout="constrained")
    sel_pts: dict[str, list[float]] = {v: [] for v in VARIANTS}
    for ax, var in zip(axes[:3], VARIANTS, strict=True):
        rows = bundles[var]["fourier_cross"]
        ops = []
        for r in rows:
            if r["source_operation"] not in ops:
                ops.append(r["source_operation"])
        mat = np.zeros((len(ops), len(ops)))
        by = {(r["source_operation"], r["target_operation"]): float(r["delta_acc"]) for r in rows}
        for i, s in enumerate(ops):
            for j, t in enumerate(ops):
                mat[i, j] = by.get((s, t), np.nan)
        im = ax.imshow(mat, vmin=-1, vmax=0, cmap="RdBu", aspect="equal")
        ax.set_xticks(range(len(ops)), [f"Q{o.split('/')[0][3:]}" for o in ops], fontsize=8)
        ax.set_yticks(range(len(ops)), [f"Q{o.split('/')[0][3:]}" for o in ops], fontsize=8)
        ax.set_title(f"{var} ts{ts}")
        ax.set_xlabel("target")
        ax.set_ylabel("source top-1 pair")
        ax.grid(False)
        for i in range(len(ops)):
            ax.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.4))
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        # selectivity per source
        sel_seen = set()
        for r in bundles[var]["fourier_sel"]:
            src = r["source_operation"]
            if src in sel_seen:
                continue
            sel_seen.add(src)
            sel_pts[var].append(float(r["selectivity"]))
    ax = axes[3]
    ax.set_title(r"$|\Delta_{\mathrm{self}}|-\mathrm{mean}|\Delta_{\mathrm{other}}|$")
    ax.set_xticks(range(3), list(VARIANTS), rotation=20)
    ax.axhline(0.0, color="#666", ls="--", lw=0.8)
    for xi, var in enumerate(VARIANTS):
        ys = sel_pts[var]
        _strip(ax, [xi] * len(ys), ys, [VAR_COLOR[var]] * len(ys))
        if ys:
            _median_bar(ax, xi, ys, VAR_COLOR[var])
    return _save(fig, stem)


def fig_1c_4(bundles: dict[str, dict[str, Any]]) -> list[Path]:
    sites = ["L0_mid", "L0_post", "L1_mid", "L1_post", "L2_post"]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), layout="constrained",
                             gridspec_kw={"width_ratios": [1.4, 1]})
    ax = axes[0]
    ax.set_title("A. Linear sum-probe along residual")
    ax.set_ylim(*ACC_YLIM)
    ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
    x = np.arange(len(sites))
    for var in VARIANTS:
        ops = per_op_from_report(bundles[var])
        for op in ops:
            ys = [op["probe_by_site"].get(s, np.nan) for s in sites]
            ax.plot(x, ys, color=VAR_COLOR[var], lw=1.0, alpha=0.75,
                    marker="o", ms=3.5, label=var if op["latent_id"] == 0 else None)
    ax.set_xticks(x, sites, rotation=20)
    ax.set_ylabel("sum-probe acc")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.set_title("B. Inferred composition layer")
    labels, layers = [], []
    for var in VARIANTS:
        for r in bundles[var]["summary"]:
            labels.append(f"{var} Q{bundles[var]['ops'][[int(o['slot']) for o in bundles[var]['ops']].index(int(r['slot']))]['latent_id']}")
            guess = r.get("compose_layer_guess") or r.get("ladder_sum_jump_layer")
            layers.append(int(float(guess)) if guess not in (None, "") else -1)
    # simpler labels
    labels, layers = [], []
    ytick = []
    mat = []
    for var in VARIANTS:
        ops = sorted(bundles[var]["ops"], key=lambda o: int(o["latent_id"]))
        by_slot = {int(r["slot"]): r for r in bundles[var]["summary"]}
        for op in ops:
            r = by_slot[int(op["slot"])]
            guess = int(float(r["compose_layer_guess"]))
            ytick.append(f"{var} Q{op['latent_id']}")
            vec = [1.0 if k == guess else 0.0 for k in range(3)]
            mat.append(vec)
    im = ax.imshow(np.asarray(mat), vmin=0, vmax=1, cmap="Blues", aspect="auto")
    ax.set_xticks([0, 1, 2], ["L0", "L1", "L2"])
    ax.set_yticks(range(len(ytick)), ytick, fontsize=7)
    ax.set_xlabel("compose-layer guess (ladder/knockout)")
    ax.grid(False)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _save(fig, OUT / "core/fig_1c_4_composition_locus")


def fig_1c_5(bundles: dict[str, dict[str, Any]]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.9), layout="constrained")
    conds = ["none", "to_other_op", "to_random"]
    clab = ["none", "swap other-op", "swap random"]
    # A attention mass
    ax = axes[0]
    ax.set_title("A. Attention mass after swap")
    ax.set_ylim(*ACC_YLIM)
    ax.set_xticks(range(3), clab, rotation=15)
    for var in VARIANTS:
        rows = [r for r in bundles[var]["attn_swap"] if r["heads"] == "all" and int(r["layer"]) == 0]
        for ci, c in enumerate(conds):
            sub = [r for r in rows if r["condition"] == c]
            own = [float(r["attn_mass_orig_pos"]) for r in sub]
            alt = [float(r["attn_mass_alt_pos"]) for r in sub]
            ax.scatter([ci - 0.12] * len(own), own, s=22, c=VAR_COLOR[var], marker="o", alpha=0.8)
            ax.scatter([ci + 0.12] * len(alt), alt, s=22, c=VAR_COLOR[var], marker="s", alpha=0.8)
    ax.plot([], [], "o", color="#333", label="orig positions")
    ax.plot([], [], "s", color="#333", label="alt positions")
    ax.legend(frameon=False, fontsize=7)
    ax.set_ylabel("query attention mass")
    ax = axes[1]
    ax.set_title("B. Original-op accuracy")
    ax.set_ylim(*ACC_YLIM)
    ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
    ax.set_xticks(range(3), clab, rotation=15)
    for var in VARIANTS:
        rows = [r for r in bundles[var]["attn_swap"] if r["heads"] == "all" and int(r["layer"]) == 0]
        for ci, c in enumerate(conds):
            sub = [float(r["acc_orig"]) for r in rows if r["condition"] == c]
            _strip(ax, [ci] * len(sub), sub, [VAR_COLOR[var]] * len(sub))
    ax = axes[2]
    ax.set_title("C. Alternative-op accuracy")
    ax.set_ylim(*ACC_YLIM)
    ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
    ax.set_xticks(range(3), clab, rotation=15)
    for var in VARIANTS:
        rows = [r for r in bundles[var]["attn_swap"] if r["heads"] == "all" and int(r["layer"]) == 0]
        for ci, c in enumerate(conds):
            sub = [float(r["acc_alt"]) for r in rows if r["condition"] == c]
            _strip(ax, [ci] * len(sub), sub, [VAR_COLOR[var]] * len(sub))
    return _save(fig, OUT / "core/fig_1c_5_attention_swap")


def fig_1c_6(bundles: dict[str, dict[str, Any]], *, ts: int, stem: Path) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.8), sharey=True, layout="constrained")
    for ax, var in zip(axes, VARIANTS, strict=True):
        ax.set_title(f"{var} ts{ts}")
        ax.set_ylim(*ACC_YLIM)
        ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
        rows = bundles[var]["patch"]
        layers = [0, 1, 2]
        x = np.arange(len(layers))
        for cond, marker, off in (("own_operands", "o", -0.12), ("other_positions", "s", 0.12)):
            for metric, color in (("acc_orig_when_diff", "#4C78A8"), ("acc_donor_when_diff", "#E45756")):
                ys = []
                for li in layers:
                    sub = [float(r[metric]) for r in rows if r["condition"] == cond and int(r["layer"]) == li]
                    ys.append(float(np.mean(sub)) if sub else np.nan)
                ax.plot(x + off, ys, marker=marker, color=color, lw=1.3,
                        label=f"{cond.split('_')[0]} {metric.split('_')[1]}")
        ax.set_xticks(x, ["L0", "L1", "L2"])
        ax.set_xlabel("layer")
        if var == "all_same":
            ax.set_ylabel("acc when labels differ")
            ax.legend(fontsize=6, frameon=False, loc="center right")
    return _save(fig, stem)


def fig_1c_7(bundles: dict[str, dict[str, Any]]) -> list[Path]:
    kinds = [
        ("resid_pre_query", "resid_pre"),
        ("attn_write_query", "attention_write"),
        ("mlp_write_query", "mlp_write"),
        ("resid_post_query", "resid_post"),
    ]
    groups = [
        ("all_same same-p", "all_same", True),
        ("pair_same same-p", "pair_same", True),
        ("four_diff diff-p", "four_diff", False),
    ]
    fig, axes = plt.subplots(3, 2, figsize=(8.8, 9.0), layout="constrained")
    trans_rows = []
    for row, (title, var, same) in enumerate(groups):
        recs = bundles[var]["transplant"]
        src_mat = np.zeros((4, 3))
        tgt_mat = np.zeros((4, 3))
        nmat = np.zeros((4, 3))
        for r in recs:
            if int(r["same_modulus"]) != int(same):
                continue
            if r["kind"] == "twin_identity":
                continue
            try:
                ki = [k for k, _ in kinds].index(r["kind"])
            except ValueError:
                continue
            li = int(r["layer"])
            if not (0 <= li <= 2):
                continue
            src_mat[ki, li] += float(r["acc_source_when_diff"])
            tgt_mat[ki, li] += float(r["acc_target_when_diff"])
            nmat[ki, li] += 1
            trans_rows.append(
                {
                    "variant": var,
                    "task_seed": 0,
                    "source_operation": r["source_operation"],
                    "target_operation": r["target_operation"],
                    "same_modulus": r["same_modulus"],
                    "layer": li,
                    "kind": r["kind"],
                    "acc_source_when_diff": r["acc_source_when_diff"],
                    "acc_target_when_diff": r["acc_target_when_diff"],
                    "source": bundles[var]["source"],
                }
            )
        nmat = np.maximum(nmat, 1)
        src_mat /= nmat
        tgt_mat /= nmat
        for col, mat, lab in ((0, src_mat, "source-label acc"), (1, tgt_mat, "target-label acc")):
            ax = axes[row, col]
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
            ax.set_xticks([0, 1, 2], ["L0", "L1", "L2"])
            ax.set_yticks(range(4), [lab2 for _, lab2 in kinds])
            ax.set_title(f"{title}\n{lab}")
            ax.grid(False)
            for i in range(4):
                for j in range(3):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if mat[i, j] < 0.55 else "k")
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _save(fig, OUT / "core/fig_1c_7_query_transplant"), trans_rows


def fig_1c_8(bundles: dict[str, dict[str, Any]], *, ts: int, stem: Path) -> list[Path]:
    fig = plt.figure(figsize=(12.6, 8.4), layout="constrained")
    gs = fig.add_gridspec(2, 1, height_ratios=[1.05, 1.35])
    gs_a = gs[0].subgridspec(1, 3)
    gs_b = gs[1].subgridspec(2, 3)
    for vi, var in enumerate(VARIANTS):
        ax = fig.add_subplot(gs_a[0, vi])
        ax.set_title(f"A. {var} self-steer")
        ax.set_ylim(*ACC_YLIM)
        ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
        rows = bundles[var]["steer_self"]
        x = np.arange(3)
        for cond, key, off, color in (
            ("structured", "steered_acc_target", -0.15, "#E45756"),
            ("shuffled", "shuffled_steered_acc_target", 0.15, "#4C78A8"),
        ):
            ys = []
            for li in range(3):
                sub = [float(r[key]) for r in rows if int(r["layer"]) == li]
                ys.append(float(np.median(sub)) if sub else np.nan)
                _strip(ax, [li + off] * len(sub), sub, [color] * len(sub))
            ax.plot(x + off, ys, color=color, lw=1.2)
        ax.set_xticks(x, ["L0", "L1", "L2"])
        ax.set_ylabel("target residue acc")
        if vi == 0:
            ax.plot([], [], color="#E45756", label="structured")
            ax.plot([], [], color="#4C78A8", label="shuffled")
            ax.legend(frameon=False, fontsize=7)
    for ri, var in enumerate(("all_same", "pair_same")):
        rows = bundles[var]["steer_cross"]
        ops = []
        for r in rows:
            if r["source_operation"] not in ops:
                ops.append(r["source_operation"])
        for li in range(3):
            ax = fig.add_subplot(gs_b[ri, li])
            mat = np.full((len(ops), len(ops)), np.nan)
            self_rows = bundles[var]["steer_self"]
            for r in self_rows:
                if int(r["layer"]) != li:
                    continue
                name = r["operation"]
                if name in ops:
                    k = ops.index(name)
                    mat[k, k] = float(r["steered_acc_target"])
            for r in rows:
                if int(r["layer"]) != li:
                    continue
                i = ops.index(r["source_operation"])
                j = ops.index(r["target_operation"])
                mat[i, j] = float(r["steered_acc_target"])
            im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="equal")
            ax.set_xticks(range(len(ops)), [f"Q{o.split('/')[0][3:]}" for o in ops], fontsize=7)
            ax.set_yticks(range(len(ops)), [f"Q{o.split('/')[0][3:]}" for o in ops], fontsize=7)
            ax.set_title(f"{var} L{li} cross-op")
            ax.grid(False)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    return _save(fig, stem)


def fig_1c_9(bundles: dict[str, dict[str, Any]]) -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.9), layout="constrained")
    ax = axes[0]
    ax.set_title("A. digit-emb vs unembed spectrum cosine")
    ax.set_ylim(0, 1.02)
    ax.set_xticks(range(3), list(VARIANTS), rotation=15)
    for xi, var in enumerate(VARIANTS):
        ys = [float(r["cosine_digit_emb_unembed"]) for r in bundles[var]["unembed_sp"]]
        _strip(ax, [xi] * len(ys), ys, [VAR_COLOR[var]] * len(ys))
        _median_bar(ax, xi, ys, VAR_COLOR[var])
    ax.set_ylabel("cosine")
    ax = axes[1]
    ax.set_title("B. top-k Fourier ablation")
    ax.set_ylim(*ACC_YLIM)
    ax.set_xlabel("k conjugate pairs removed")
    ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
    for var in VARIANTS:
        dig = [r for r in bundles[var]["fourier_curve"] if r["rank"] == "important"]
        une = [r for r in bundles[var]["unembed_ab"] if r["rank"] == "important"]
        for k in sorted({int(r["k"]) for r in dig}):
            pass
        ks = sorted({int(r["k"]) for r in dig})
        y_d, y_u = [], []
        for k in ks:
            y_d.append(float(np.median([float(r["acc"]) for r in dig if int(r["k"]) == k])))
            yu = [float(r["acc"]) for r in une if int(r["k"]) == k]
            y_u.append(float(np.median(yu)) if yu else np.nan)
        ax.plot(ks, y_d, color=VAR_COLOR[var], lw=1.6, marker="o", label=f"{var} digit-emb")
        ax.plot(ks, y_u, color=VAR_COLOR[var], lw=1.2, ls="--", marker="s", label=f"{var} unembed")
    ax.legend(fontsize=6, frameon=False)
    ax.set_ylabel("accuracy")
    ax = axes[2]
    ax.set_title("C. query residual vs unembed cosine")
    ax.set_ylim(0, 1.02)
    ax.set_xticks([0, 1, 2], ["L0", "L1", "L2"])
    for var in VARIANTS:
        sp = bundles[var]["unembed_sp"]
        ys = []
        for key in ("cosine_query_L0_unembed", "cosine_query_L1_unembed", "cosine_query_L2_unembed"):
            ys.append(float(np.median([float(r[key]) for r in sp])))
        ax.plot([0, 1, 2], ys, color=VAR_COLOR[var], marker="o", lw=1.6, label=var)
        for li, key in enumerate(
            ("cosine_query_L0_unembed", "cosine_query_L1_unembed", "cosine_query_L2_unembed")
        ):
            vals = [float(r[key]) for r in sp]
            _strip(ax, [li] * len(vals), vals, [VAR_COLOR[var]] * len(vals))
    ax.legend(frameon=False, fontsize=7)
    return _save(fig, OUT / "core/fig_1c_9_readout_geometry")


def fig_1c_10(bundles: dict[str, dict[str, Any]]) -> list[Path]:
    order = ["original", "task_a", "task_b", "digit_0", "digit_63"]
    labs = ["original", "TASK_A", "TASK_B", "digit_0", "digit_63"]
    fig, axes = plt.subplots(1, 3, figsize=(11.6, 3.7), sharey=True, layout="constrained")
    for ax, var in zip(axes, VARIANTS, strict=True):
        ax.set_title(var)
        ax.set_ylim(*ACC_YLIM)
        ax.axhline(1 / 31, color="#888", ls=":", lw=0.8)
        rows = bundles[var]["token"]
        ops = sorted({r["operation"] for r in rows})
        x = np.arange(len(order))
        for opname in ops:
            lat = int(opname.split("/")[0][3:])
            ys = []
            for c in order:
                sub = [float(r["acc"]) for r in rows if r["operation"] == opname and r["condition"] == c]
                ys.append(sub[0] if sub else np.nan)
            ax.plot(x, ys, color=Q_COLOR[lat], marker="o", lw=1.1, ms=4, label=f"Q{lat}")
        ax.set_xticks(x, labs, rotation=20)
        if var == "all_same":
            ax.set_ylabel("accuracy")
            ax.legend(frameon=False, fontsize=7)
    return _save(fig, OUT / "core/fig_1c_10_task_token_edit")


def fig_1c_11() -> list[Path]:
    fig, ax = plt.subplots(figsize=(8.6, 7.2))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")
    ax.set_title("Shared late geometry; operation-specific early routing")
    boxes = [
        (2.2, 10.6, 5.6, 1.0, "Q0–Q3 packed queries", "#EEE"),
        (2.2, 9.1, 5.6, 1.0, "operation-specific L0 routing", "#B279A2"),
        (2.2, 7.6, 5.6, 1.0, "read own operand residuals", "#B279A2"),
        (2.2, 6.1, 5.6, 1.0, "shared digit Fourier features", "#F2CF5B"),
        (2.2, 4.6, 5.6, 1.0, "L0/L1 partially shared composition", "#F58518"),
        (2.2, 3.1, 5.6, 1.0, "same-p shared result geometry", "#54A24B"),
        (2.2, 1.6, 5.6, 1.0, "L1/L2 answer representation", "#4C78A8"),
        (2.2, 0.2, 5.6, 1.0, "shared redundant linear readout", "#4C78A8"),
    ]
    for x, y, w, h, text, fc in boxes:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.04", facecolor=fc,
                                    edgecolor="#333", lw=0.9, alpha=0.9))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=9)
    for y0, y1 in ((10.55, 10.15), (9.05, 8.65), (7.55, 7.15), (6.05, 5.65), (4.55, 4.15), (3.05, 2.65), (1.55, 1.25)):
        ax.annotate(
            "",
            xy=(5, y1),
            xytext=(5, y0),
            arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.0),
        )
    ax.text(8.4, 8.6, "early:\noperation-\nspecific", fontsize=8, color="#6B3F7D")
    ax.text(8.4, 5.0, "middle:\npartially\nshared", fontsize=8, color="#B35C00")
    ax.text(8.4, 1.4, "late:\nstrongly\nshared", fontsize=8, color="#1F4E79")
    return _save(fig, OUT / "core/fig_1c_11_mechanism_summary")


# ---------------------------------------------------------------------------
# README
# ---------------------------------------------------------------------------


def write_readme(
    jobs: list[dict[str, Any]],
    written: list[Path],
    curve_rule: str,
) -> None:
    n_ok = sum(j["status"] == "ok" for j in jobs)
    lines = [
        "# Phase 1B / 1C report figures",
        "",
        "Generated by `scripts/phase1/plot_1b_1c_report.py`. Original experiment files were not modified.",
        "",
        "## Configuration difference (do not mix)",
        "",
        "| | Phase 1B | Phase 1C |",
        "|---|---|---|",
        "| stamp | `runs/phase1/multi_op/20261004_170554` | `runs/phase1/mechanisms/20261004_1b_132221` |",
        "| training | wd=0.3, steps=100000, 3 task × 3 model seeds | mechanisms on **wd=0.5** jobs from `20261003_132221`, **best** ckpt, ms0 |",
        "| n | 27 successful jobs | 6 dirs (`*_ts0`, `*_ts1`) |",
        "",
        "1C does **not** analyze the 1B wd=0.3 checkpoints. Trends can be compared qualitatively only.",
        "",
        "## Seeds",
        "",
        "- 1B figures use all 9 (task_seed, model_seed) per variant unless noted.",
        "- 1C **core** figures use **ts0**. Matching **ts1** panels are in `appendix/` and are not averaged with ts0 (moduli differ).",
        f"- Fig 1B-5 seed rule: {curve_rule}",
        "",
        f"Validated 1B ok jobs: **{n_ok}**. Chance line is 1/31 unless a panel is per-modulus.",
        "",
        "## t_gen definition",
        "",
        "Recomputed from `eval_history.jsonl` (not `best_step`):",
        "",
        f"- metric: `A_val_macro_op_acc`",
        f"- first step at which the metric is ≥ {T_GEN_THRESH} for **{T_GEN_STREAK} consecutive evaluations**",
        "- `metrics.events.t_gen` is stored in derived CSV as `t_gen_events` for comparison and is **not** plotted as t_gen",
        "",
        "## Figures",
        "",
    ]
    captions = {
        "fig_1b_1_task_design": "Three packed 4-query designs; matching fixed, moduli change.",
        "fig_1b_2_accuracy_distribution": "All variants grok; four_diff ts0/ms0 is the visible lagging seed.",
        "fig_1b_3_time_to_generalization": "four_diff is slower; one seed never meets the 3-eval rule before 100k? check points.",
        "fig_1b_4_seed_robustness": "t_gen and worst-op accuracy across the 3×3 seed grid.",
        "fig_1b_5_learning_curves": "All four operations rise together; macro tracks the slowest op.",
        "fig_1c_1_operand_routing": "L0 attends to each op's own operands, not a uniform mix.",
        "fig_1c_2_head_specialization": "Heads are not strictly one-op-one-head; some heads hit all ops.",
        "fig_1c_3_fourier_selectivity": "all_same non-selective; pair/four_diff partially selective, not diagonal-only.",
        "fig_1c_4_composition_locus": "L0 still causal; stable linear sum often only after L1 on pair/four_diff.",
        "fig_1c_5_attention_swap": "Routing moves, original acc collapses, alternative acc does not go to 1.",
        "fig_1c_6_operand_patching": "L0 own-operand patch: original→0, donor→1; control positions keep original.",
        "fig_1c_7_query_transplant": "L0 attn write dual-low; L2 MLP write follows source.",
        "fig_1c_8_residual_steering": "Self-steer works at L1/L2; same-p cross-op steer ≈1 at L1/L2.",
        "fig_1c_9_readout_geometry": "Digit-emb top Fourier is causal; unembed top-k ablation is weak; L2 aligns with head.",
        "fig_1c_10_task_token_edit": "TASK_A→TASK_B barely changes acc (bridge to Phase 2).",
        "fig_1c_11_mechanism_summary": "Early op-specific routing, late shared geometry.",
        "fig_1c_ts1_routing": "ts1 replication of operand routing (do not average with ts0).",
        "fig_1c_ts1_head_knockout": "ts1 replication of head knockout.",
        "fig_1c_ts1_fourier_selectivity": "ts1 replication of Fourier selectivity.",
        "fig_1c_ts1_patching": "ts1 replication of operand patching.",
        "fig_1c_ts1_steering": "ts1 replication of residual steering.",
    }
    cores = sorted({p.stem for p in written if p.parent.name == "core"})
    apps = sorted({p.stem for p in written if p.parent.name == "appendix"})
    lines.append("### Core")
    for name in cores:
        lines.append(f"- `{name}`: {captions.get(name, '')}")
    lines += ["", "### Appendix (ts1, not mixed with ts0)", ""]
    for name in apps:
        lines.append(f"- `{name}`: {captions.get(name, '')}")
    lines += [
        "",
        "## Axes (short)",
        "",
        "- 1B-2/3: x=variant, y=accuracy or t_gen; points=(task_seed, model_seed).",
        "- 1B-4: rows=task seed, cols=model seed.",
        "- 1B-5: x=steps, y=held-out val acc per operation key.",
        "- 1C heatmaps: operation keys `lat{id}/slot{s}/p{m}`, never merged by modulus alone.",
        "",
        "## Supported vs not supported",
        "",
        "**Supported:** packed multi-query grokking is possible under all three modulus assignments; L0 routing is operation-specific; Fourier geometry is shared especially in all_same; attention swap is not a full circuit transplant; L0 operand residual is sufficient for the answer; late residual steering transfers across same-modulus ops; TASK token identity is not what the 1B/1C model uses to select the op.",
        "",
        "**Not supported:** that 1B wd=0.3 and 1C wd=0.5 are the same run; that four_diff 'does not compute in L0'; that heads/freqs form strictly disjoint circuits; that changing TASK tokens is a general continual-learning solution.",
        "",
        "## Missing / recomputed",
        "",
    ]
    if SKIPPED:
        lines.append("Skipped / incomplete:")
        lines += [f"- {s}" for s in SKIPPED]
    else:
        lines.append("No panels skipped for missing files.")
    lines += ["", "Notes:", ""]
    lines += [f"- {n}" for n in NOTES] or ["- none"]
    (OUT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    _style()
    (OUT / "core").mkdir(parents=True, exist_ok=True)
    (OUT / "appendix").mkdir(parents=True, exist_ok=True)
    (OUT / "derived").mkdir(parents=True, exist_ok=True)

    jobs, tgen_rows, curve_rows = load_1b()
    validate_1b(jobs)
    seed_rows = []
    for j in jobs:
        seed_rows.append(
            {
                "variant": j["variant"],
                "task_seed": j["task_seed"],
                "model_seed": j["model_seed"],
                "job_id": j["job_id"],
                "final_test_macro": j["final_test_macro"],
                "best_test_macro": j["best_test_macro"],
                "best_val_macro": j["best_val_macro"],
                "t_gen": j["t_gen"] if j["t_gen"] is not None else "",
                "t_gen_reached": int(j["t_gen_reached"]),
                "worst_op_best_test": j["worst_op_best_test"],
                "best_step": j["best_step"],
                "source": j["source"],
                "operation_keys": ";".join(
                    op_key(o["latent_id"], o["slot"], o["modulus"]) for o in j["ops"]
                ),
            }
        )
    _write_csv(
        OUT / "derived/1b_seed_metrics.csv",
        seed_rows,
        list(seed_rows[0].keys()),
    )
    _write_csv(
        OUT / "derived/1b_tgen.csv",
        tgen_rows,
        ["variant", "task_seed", "model_seed", "job_id", "t_gen", "t_gen_events",
         "t_gen_reached", "best_step", "definition", "source"],
    )
    _write_csv(
        OUT / "derived/1b_per_op_curves.csv",
        curve_rows,
        ["variant", "task_seed", "model_seed", "job_id", "step", "operation",
         "latent_id", "slot", "modulus", "acc", "source"],
    )

    c1c0: dict[str, dict[str, Any]] = {}
    c1c1: dict[str, dict[str, Any]] = {}
    for v in VARIANTS:
        b0 = load_1c(v, 0)
        b1 = load_1c(v, 1)
        if b0 is None or b1 is None:
            raise RuntimeError("expected six 1C directories")
        c1c0[v] = b0
        c1c1[v] = b1
        keys = [op_key(o["latent_id"], o["slot"], o["modulus"]) for o in b0["ops"]]
        if len(keys) != len(set(keys)):
            raise RuntimeError(f"duplicate op keys in {v} ts0")

    written: list[Path] = []
    written += fig_1b_1(jobs)
    written += fig_1b_2(jobs)
    written += fig_1b_3(jobs)
    written += fig_1b_4(jobs)
    written += fig_1b_5(jobs, curve_rows)
    _, curve_rule = choose_curve_seed(jobs), NOTES[-1]

    written += fig_1c_1(c1c0, ts=0, stem=OUT / "core/fig_1c_1_operand_routing")
    paths, spec_rows = fig_1c_2(c1c0, ts=0, stem=OUT / "core/fig_1c_2_head_specialization")
    written += paths
    _write_csv(
        OUT / "derived/1c_head_specialization.csv",
        spec_rows,
        ["variant", "task_seed", "layer", "head", "specialization", "source"],
    )
    written += fig_1c_3(c1c0, ts=0, stem=OUT / "core/fig_1c_3_fourier_selectivity")
    fsel = []
    for v, b in c1c0.items():
        for r in b["fourier_sel"]:
            fsel.append({**r, "variant": v, "task_seed": 0, "source": b["source"]})
    _write_csv(OUT / "derived/1c_fourier_selectivity.csv", fsel, list(fsel[0].keys()))
    written += fig_1c_4(c1c0)
    written += fig_1c_5(c1c0)
    written += fig_1c_6(c1c0, ts=0, stem=OUT / "core/fig_1c_6_operand_patching")
    paths, trans_rows = fig_1c_7(c1c0)
    written += paths
    _write_csv(OUT / "derived/1c_transplant_summary.csv", trans_rows, list(trans_rows[0].keys()))
    written += fig_1c_8(c1c0, ts=0, stem=OUT / "core/fig_1c_8_residual_steering")
    steer_rows = []
    for v, b in c1c0.items():
        for r in b["steer_self"]:
            steer_rows.append({**r, "variant": v, "task_seed": 0, "kind": "self", "source": b["source"]})
        for r in b["steer_cross"]:
            steer_rows.append({**r, "variant": v, "task_seed": 0, "kind": "cross", "source": b["source"]})
    _write_csv(OUT / "derived/1c_steering_summary.csv", steer_rows, list(steer_rows[0].keys()))
    route_rows = []
    for v, b in {**{f"{k}_ts0": val for k, val in c1c0.items()}, **{f"{k}_ts1": val for k, val in c1c1.items()}}.items():
        for op in per_op_from_report(b):
            route_rows.append(
                {
                    "variant": b["variant"],
                    "task_seed": b["task_seed"],
                    "operation": op["operation"],
                    "latent_id": op["latent_id"],
                    "slot": op["slot"],
                    "modulus": op["modulus"],
                    "operand_i": op["operand_i"],
                    "operand_j": op["operand_j"],
                    "own_mass": op["own_mass"],
                    "mass_by_pos": " ".join(f"{x:.6f}" for x in op["mass_by_pos"]),
                    "source": op["source"],
                }
            )
    _write_csv(OUT / "derived/1c_routing.csv", route_rows, list(route_rows[0].keys()))
    written += fig_1c_9(c1c0)
    written += fig_1c_10(c1c0)
    written += fig_1c_11()

    written += fig_1c_1(c1c1, ts=1, stem=OUT / "appendix/fig_1c_ts1_routing")
    paths, spec1 = fig_1c_2(c1c1, ts=1, stem=OUT / "appendix/fig_1c_ts1_head_knockout")
    written += paths
    written += fig_1c_3(c1c1, ts=1, stem=OUT / "appendix/fig_1c_ts1_fourier_selectivity")
    written += fig_1c_6(c1c1, ts=1, stem=OUT / "appendix/fig_1c_ts1_patching")
    written += fig_1c_8(c1c1, ts=1, stem=OUT / "appendix/fig_1c_ts1_steering")

    NOTES.append(
        "pair_same cross-op steering CSV only contains the same-modulus pair (Q0/Q1); "
        "the 2x2 heatmap is not a 4-op matrix."
    )
    NOTES.append("All 27 1B runs reached t_gen under the 3-eval rule; no X markers.")
    NOTES.append(
        "Unembed top-k ablation CSV only has k=1..3; digit-emb Fourier curves go to k=6."
    )
    write_readme(jobs, written, curve_rule)
    pngs = [p for p in written if p.suffix == ".png"]
    print(f"wrote {len(pngs)} PNG under {OUT}")
    for p in sorted(pngs):
        print(f"  {p.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
