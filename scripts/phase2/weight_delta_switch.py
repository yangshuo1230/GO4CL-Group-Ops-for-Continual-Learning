#!/usr/bin/env python3
"""Compare model weights at switch (theta_A) vs after phase B (final).

Post-hoc only: reads existing sequential_ab / sequential_ab_replay checkpoints.
Does not train or overwrite original run artifacts.

Writes under ``<stamp>/weight_delta_switch/``:
  - param_delta.csv          per (run, param tensor)
  - group_delta.csv          per (run, component group)
  - summary_by_protocol.csv  mean rel-L2 by group × protocol
  - figures/*.png
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from go4cl.constants import NUM_DIGITS, NUM_QUERIES, NUM_TASKS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STAMP = ROOT / "runs/phase2/relation_matrix/20261006_131444"
PROTOCOLS = ("sequential_ab", "sequential_ab_replay")
SKIP_SUFFIXES = (".attn.mask",)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.28,
        }
    )


def _load_state(path: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    return {k: v.detach().float().cpu() for k, v in payload["model_state"].items()}


def _group_name(key: str) -> str | None:
    if key.endswith(SKIP_SUFFIXES) or key.endswith(".attn.mask"):
        return None
    if key == "tok_emb.weight":
        return "tok_emb"  # split later
    if key == "pos_emb.weight":
        return "pos_emb"
    if key.startswith("ln_f."):
        return "ln_f"
    if key.startswith("head."):
        return "head"
    m = re.match(r"blocks\.(\d+)\.(attn|mlp|ln1|ln2)\.", key)
    if m:
        layer, kind = m.group(1), m.group(2)
        if kind in {"ln1", "ln2"}:
            return f"L{layer}.ln"
        return f"L{layer}.{kind}"
    return f"other:{key}"


def _metrics(a: torch.Tensor, b: torch.Tensor) -> dict[str, float]:
    a = a.reshape(-1)
    b = b.reshape(-1)
    delta = b - a
    l2_a = float(torch.linalg.vector_norm(a).item())
    l2_b = float(torch.linalg.vector_norm(b).item())
    l2_d = float(torch.linalg.vector_norm(delta).item())
    cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
    return {
        "l2_before": l2_a,
        "l2_after": l2_b,
        "l2_delta": l2_d,
        "rel_l2": l2_d / (l2_a + 1e-12),
        "cosine": cos,
        "numel": int(a.numel()),
    }


def _accumulate(chunks: list[tuple[torch.Tensor, torch.Tensor]]) -> dict[str, float]:
    if not chunks:
        return {
            "l2_before": 0.0,
            "l2_after": 0.0,
            "l2_delta": 0.0,
            "rel_l2": 0.0,
            "cosine": float("nan"),
            "numel": 0,
        }
    a = torch.cat([x.reshape(-1) for x, _ in chunks])
    b = torch.cat([y.reshape(-1) for _, y in chunks])
    return _metrics(a, b)


def analyze_pair(
    state_a: dict[str, torch.Tensor],
    state_b: dict[str, torch.Tensor],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    param_rows: list[dict[str, Any]] = []
    buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = defaultdict(list)

    for key in state_a:
        if key not in state_b:
            continue
        if key.endswith(".attn.mask"):
            continue
        wa, wb = state_a[key], state_b[key]
        if wa.shape != wb.shape:
            continue
        m = _metrics(wa, wb)
        group = _group_name(key)
        param_rows.append({"param": key, "group": group, **m})

        if key == "tok_emb.weight":
            buckets["tok_emb.digit"].append((wa[:NUM_DIGITS], wb[:NUM_DIGITS]))
            t0, t1 = NUM_DIGITS, NUM_DIGITS + NUM_TASKS
            buckets["tok_emb.task"].append((wa[t0:t1], wb[t0:t1]))
            q0, q1 = t1, t1 + NUM_QUERIES
            buckets["tok_emb.query"].append((wa[q0:q1], wb[q0:q1]))
            buckets["tok_emb"].append((wa, wb))
        elif group is not None:
            buckets[group].append((wa, wb))

    # whole model
    buckets["ALL"] = [
        (state_a[k], state_b[k])
        for k in state_a
        if k in state_b and not k.endswith(".attn.mask") and state_a[k].shape == state_b[k].shape
    ]

    group_rows: list[dict[str, Any]] = []
    for group, chunks in sorted(buckets.items()):
        group_rows.append({"group": group, **_accumulate(chunks)})

    # share of squared delta among primary groups (exclude ALL / tok_emb aggregate)
    primary = [
        r
        for r in group_rows
        if r["group"]
        not in {"ALL", "tok_emb"}  # use digit/task/query split instead of full tok_emb
    ]
    denom = sum(float(r["l2_delta"]) ** 2 for r in primary) + 1e-12
    for r in group_rows:
        if r["group"] == "ALL":
            r["frac_sq_delta"] = 1.0
        elif r["group"] == "tok_emb":
            r["frac_sq_delta"] = float("nan")
        else:
            r["frac_sq_delta"] = float(r["l2_delta"]) ** 2 / denom
    return param_rows, group_rows


def _parse_job(job_id: str) -> dict[str, Any] | None:
    m = re.match(
        r"^(sequential_ab(?:_replay)?)_"
        r"(s(?P<slot>[0-9.]+)_o(?P<operand>[0-9.]+)_m(?P<mod>[0-9.]+))_"
        r"forward_ts(?P<ts>\d+)_ms(?P<ms>\d+)_",
        job_id,
    )
    if not m:
        return None
    return {
        "protocol": m.group(1),
        "condition": m.group(2),
        "rho_slot": float(m.group("slot")),
        "rho_operand": float(m.group("operand")),
        "rho_mod": float(m.group("mod")),
        "task_seed": int(m.group("ts")),
        "model_seed": int(m.group("ms")),
        "job_id": job_id,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def _load_final_acc(run_dir: Path) -> dict[str, float]:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.is_file():
        return {}
    metrics = json.loads(metrics_path.read_text())
    out: dict[str, float] = {}
    for key in ("A_test_acc", "B_test_acc", "forgetting_A", "forgetting_A_from_switch"):
        val = metrics.get(key)
        if isinstance(val, (int, float)):
            out[key] = float(val)
    return out


def run(stamp: Path, out_dir: Path | None = None) -> Path:
    out = out_dir or (stamp / "weight_delta_switch")
    fig_dir = out / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    runs_root = stamp / "runs"

    group_all: list[dict[str, Any]] = []
    param_all: list[dict[str, Any]] = []

    job_dirs = sorted(
        d
        for d in runs_root.iterdir()
        if d.is_dir() and d.name.startswith("sequential_ab")
    )
    for run_dir in job_dirs:
        meta = _parse_job(run_dir.name)
        if meta is None or meta["protocol"] not in PROTOCOLS:
            continue
        theta = run_dir / "ckpts" / "theta_A.pt"
        final = run_dir / "ckpts" / "final.pt"
        if not theta.is_file() or not final.is_file():
            print(f"[skip] missing ckpt: {run_dir.name}")
            continue
        state_a = _load_state(theta)
        state_b = _load_state(final)
        param_rows, group_rows = analyze_pair(state_a, state_b)
        acc = _load_final_acc(run_dir)
        recovered = float(acc.get("A_test_acc", 0.0)) >= 0.9
        for row in param_rows:
            param_all.append({**meta, **acc, "recovered_A": recovered, **row})
        for row in group_rows:
            group_all.append({**meta, **acc, "recovered_A": recovered, **row})
        print(
            f"[ok] {meta['protocol']:22s} {meta['condition']:16s} "
            f"ALL rel_l2={next(r['rel_l2'] for r in group_rows if r['group']=='ALL'):.4f} "
            f"A={acc.get('A_test_acc', float('nan')):.3f}"
        )

    group_fields = [
        "protocol",
        "condition",
        "rho_slot",
        "rho_operand",
        "rho_mod",
        "task_seed",
        "model_seed",
        "job_id",
        "A_test_acc",
        "B_test_acc",
        "forgetting_A",
        "recovered_A",
        "group",
        "numel",
        "l2_before",
        "l2_after",
        "l2_delta",
        "rel_l2",
        "cosine",
        "frac_sq_delta",
    ]
    param_fields = [
        "protocol",
        "condition",
        "job_id",
        "A_test_acc",
        "recovered_A",
        "param",
        "group",
        "numel",
        "l2_before",
        "l2_after",
        "l2_delta",
        "rel_l2",
        "cosine",
    ]
    _write_csv(out / "group_delta.csv", group_all, group_fields)
    _write_csv(out / "param_delta.csv", param_all, param_fields)

    # protocol × group means
    buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    frac_buckets: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in group_all:
        key = (row["protocol"], row["group"])
        buckets[key].append(float(row["rel_l2"]))
        if row["group"] not in {"ALL", "tok_emb"} and np.isfinite(row.get("frac_sq_delta", np.nan)):
            frac_buckets[key].append(float(row["frac_sq_delta"]))
    summary_rows: list[dict[str, Any]] = []
    for (protocol, group), vals in sorted(buckets.items()):
        fr = frac_buckets.get((protocol, group), [])
        summary_rows.append(
            {
                "protocol": protocol,
                "group": group,
                "n": len(vals),
                "mean_rel_l2": float(np.mean(vals)),
                "median_rel_l2": float(np.median(vals)),
                "mean_frac_sq_delta": float(np.mean(fr)) if fr else "",
            }
        )
    _write_csv(
        out / "summary_by_protocol.csv",
        summary_rows,
        ["protocol", "group", "n", "mean_rel_l2", "median_rel_l2", "mean_frac_sq_delta"],
    )

    _style()
    _plot_protocol_bars(summary_rows, fig_dir / "fig_rel_l2_by_group")
    _plot_frac_bars(summary_rows, fig_dir / "fig_frac_sq_delta_by_group")
    _plot_heatmap(group_all, fig_dir / "fig_rel_l2_heatmap")
    _plot_fail_vs_ok(group_all, fig_dir / "fig_replay_fail_vs_ok")
    _plot_layer_attn_mlp(group_all, fig_dir / "fig_layer_attn_vs_mlp")

    readme = out / "README.md"
    readme.write_text(
        "\n".join(
            [
                "# Weight delta at task switch (θ_A → final)",
                "",
                f"Stamp: `{stamp}`",
                "",
                "For each sequential run, compare `ckpts/theta_A.pt` (end of A) vs "
                "`ckpts/final.pt` (end of B).",
                "",
                "- `rel_l2` = ‖ΔW‖₂ / ‖W_A‖₂",
                "- `frac_sq_delta` = ‖ΔW‖₂² / Σ_groups ‖ΔW‖₂² (tok_emb split into digit/task/query)",
                "- `cosine` = cos(W_A, W_B)",
                "",
                "Protocols: `sequential_ab` (no replay) vs `sequential_ab_replay` (10% A in B).",
                "",
            ]
        ),
        encoding="utf-8",
    )
    print(f"[done] wrote {out}")
    return out


ORDER_GROUPS = [
    "tok_emb.digit",
    "tok_emb.task",
    "tok_emb.query",
    "pos_emb",
    "L0.attn",
    "L0.mlp",
    "L0.ln",
    "L1.attn",
    "L1.mlp",
    "L1.ln",
    "L2.attn",
    "L2.mlp",
    "L2.ln",
    "ln_f",
    "head",
    "ALL",
]


def _plot_protocol_bars(summary: list[dict[str, Any]], stem: Path) -> None:
    groups = [g for g in ORDER_GROUPS if g != "ALL"]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, proto in enumerate(PROTOCOLS):
        lookup = {
            r["group"]: float(r["mean_rel_l2"])
            for r in summary
            if r["protocol"] == proto
        }
        vals = [lookup.get(g, np.nan) for g in groups]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label="no replay" if proto == "sequential_ab" else "10% replay",
            color="#E45756" if proto == "sequential_ab" else "#4C78A8",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(r"mean relative L2  $\|\Delta W\|_2 / \|W_A\|_2$")
    ax.set_title("Weight change after switch: by component")
    ax.legend()
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def _plot_frac_bars(summary: list[dict[str, Any]], stem: Path) -> None:
    groups = [g for g in ORDER_GROUPS if g not in {"ALL", "tok_emb"}]
    x = np.arange(len(groups))
    width = 0.38
    fig, ax = plt.subplots(figsize=(11, 4.2))
    for i, proto in enumerate(PROTOCOLS):
        lookup = {
            r["group"]: r["mean_frac_sq_delta"]
            for r in summary
            if r["protocol"] == proto and r["mean_frac_sq_delta"] != ""
        }
        vals = [float(lookup[g]) if g in lookup else np.nan for g in groups]
        ax.bar(
            x + (i - 0.5) * width,
            vals,
            width,
            label="no replay" if proto == "sequential_ab" else "10% replay",
            color="#E45756" if proto == "sequential_ab" else "#4C78A8",
        )
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel(r"mean share of $\|\Delta W\|_2^2$")
    ax.set_title("Where squared weight change concentrates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def _plot_heatmap(group_all: list[dict[str, Any]], stem: Path) -> None:
    groups = [g for g in ORDER_GROUPS if g != "ALL"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 7), sharey=True)
    for ax, proto, title in zip(
        axes,
        PROTOCOLS,
        ("no replay", "10% replay"),
    ):
        rows = [r for r in group_all if r["protocol"] == proto and r["group"] in groups]
        conds = sorted({r["condition"] for r in rows})
        mat = np.full((len(conds), len(groups)), np.nan)
        lookup = {(r["condition"], r["group"]): float(r["rel_l2"]) for r in rows}
        for i, c in enumerate(conds):
            for j, g in enumerate(groups):
                mat[i, j] = lookup.get((c, g), np.nan)
        im = ax.imshow(mat, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(len(groups)))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(len(conds)))
        ax.set_yticklabels(conds, fontsize=7)
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="rel L2")
    fig.suptitle(r"Per-condition relative weight change $\|\Delta W\|_2/\|W_A\|_2$")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def _plot_fail_vs_ok(group_all: list[dict[str, Any]], stem: Path) -> None:
    groups = [g for g in ORDER_GROUPS if g not in {"ALL"}]
    replay = [r for r in group_all if r["protocol"] == "sequential_ab_replay"]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(groups))
    width = 0.38
    for i, (flag, label, color) in enumerate(
        (
            (True, "replay A≥0.9", "#54A24B"),
            (False, "replay A<0.9", "#E45756"),
        )
    ):
        vals = []
        for g in groups:
            xs = [
                float(r["rel_l2"])
                for r in replay
                if r["group"] == g and bool(r["recovered_A"]) == flag
            ]
            vals.append(float(np.mean(xs)) if xs else np.nan)
        ax.bar(x + (i - 0.5) * width, vals, width, label=label, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=45, ha="right")
    ax.set_ylabel("mean rel L2")
    ax.set_title("Replay runs: weight change when A recovers vs fails")
    ax.legend()
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def _plot_layer_attn_mlp(group_all: list[dict[str, Any]], stem: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    layers = [0, 1, 2]
    for ax, proto, title in zip(axes, PROTOCOLS, ("no replay", "10% replay")):
        for kind, color in (("attn", "#4C78A8"), ("mlp", "#F58518")):
            means, stds = [], []
            for layer in layers:
                xs = [
                    float(r["rel_l2"])
                    for r in group_all
                    if r["protocol"] == proto and r["group"] == f"L{layer}.{kind}"
                ]
                means.append(float(np.mean(xs)) if xs else np.nan)
                stds.append(float(np.std(xs)) if xs else 0.0)
            ax.errorbar(
                layers,
                means,
                yerr=stds,
                marker="o",
                label=kind,
                color=color,
                capsize=3,
            )
        ax.set_xticks(layers)
        ax.set_xticklabels([f"L{i}" for i in layers])
        ax.set_xlabel("layer")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("mean rel L2")
    fig.suptitle("Layer-wise attention vs MLP change after switch")
    fig.tight_layout()
    fig.savefig(stem.with_suffix(".png"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stamp",
        type=str,
        default=str(DEFAULT_STAMP),
        help="relation_matrix stamp with sequential runs",
    )
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()
    stamp = Path(args.stamp)
    if not stamp.is_absolute():
        stamp = ROOT / stamp
    out = Path(args.out) if args.out else None
    if out is not None and not out.is_absolute():
        out = ROOT / out
    run(stamp, out)


if __name__ == "__main__":
    main()
