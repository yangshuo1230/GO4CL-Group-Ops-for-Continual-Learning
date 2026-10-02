#!/usr/bin/env python3
"""Plot canonical 1A-mech figures from mech_single stamp CSVs / reports.

Defaults point at the curated p=31 runs from 2026-10-02.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

DEFAULT_COMPOSITION = Path("runs/phase1/mech_single/20261002_105034")
DEFAULT_HEAD = Path("runs/phase1/mech_single/20261002_110852")
DEFAULT_OUT = Path("runs/phase1/mech_single/p31_summary/figures")


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 200,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def _load_comp_rows(stamp: Path) -> list[dict[str, str]]:
    path = stamp / "phase1_mech-single_composition.csv"
    return list(csv.DictReader(path.open()))


def _load_abl_rows(stamp: Path) -> list[dict[str, str]]:
    path = stamp / "phase1_mech-single_ablation_curve.csv"
    return list(csv.DictReader(path.open()))


def plot_composition_locus(stamp: Path, out: Path, *, ckpt: str = "final") -> Path:
    rows = _load_comp_rows(stamp)
    sites = ["L0_mid", "L0_post", "L1_mid", "L1_post", "L2_mid", "L2_post"]
    site_labels = ["L0\nmid", "L0\npost", "L1\nmid", "L1\npost", "L2\nmid", "L2\npost"]
    ladder: dict[str, dict[str, float]] = defaultdict(dict)
    knockout: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["ckpt"] != ckpt:
            continue
        if r["kind"] == "ladder":
            ladder[r["site_or_layer"]][r["metric"]] = float(r["value"])
        elif r["kind"] == "knockout":
            knockout[r["site_or_layer"]][r["metric"]] = float(r["value"])

    fig, axes = plt.subplots(
        1, 2, figsize=(9.2, 3.6), gridspec_kw={"width_ratios": [2.2, 1]}
    )
    ax = axes[0]
    x = range(len(sites))
    ax.plot(
        x,
        [ladder[s]["probe_xi"] for s in sites],
        "o-",
        label=r"$x_i$",
        color="#4C78A8",
        lw=1.8,
        ms=5,
    )
    ax.plot(
        x,
        [ladder[s]["probe_xj"] for s in sites],
        "s-",
        label=r"$x_j$",
        color="#72B7B2",
        lw=1.8,
        ms=5,
    )
    ax.plot(
        x,
        [ladder[s]["probe_sum"] for s in sites],
        "D-",
        label=r"$(x_i+x_j)$ mod $p$",
        color="#E45756",
        lw=2.2,
        ms=5.5,
    )
    ax.axhline(1 / 31, color="0.5", ls=":", lw=1, label="chance (1/31)")
    ax.axvspan(-0.35, 1.35, color="#E45756", alpha=0.08, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(site_labels)
    ax.set_ylim(-0.02, 1.05)
    ax.set_ylabel("linear probe accuracy (test)")
    ax.set_xlabel("query residual site")
    ax.set_title("A. Where sum becomes readable")
    ax.legend(loc="center right", frameon=False)

    ax = axes[1]
    layers = ["L0", "L1", "L2"]
    xpos = range(len(layers))
    w = 0.35
    mlp = [knockout[L]["zero_mlp_delta"] for L in layers]
    attn = [knockout[L]["zero_attn_delta"] for L in layers]
    ax.bar([i - w / 2 for i in xpos], mlp, width=w, label="zero MLP", color="#E45756")
    ax.bar([i + w / 2 for i in xpos], attn, width=w, label="zero attn", color="#4C78A8")
    ax.axhline(0, color="0.3", lw=0.8)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels(layers)
    ax.set_ylabel(r"$\Delta$ test accuracy")
    ax.set_title("B. Causal component knockout")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-1.05, 0.15)

    fig.suptitle(
        rf"Composition locus · $p=31$ {ckpt}  (shade: L0 mid$\rightarrow$post)",
        y=1.02,
        fontsize=12,
    )
    fig.tight_layout()
    path = out / "fig1_composition_locus.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_harmonic(stamp: Path, out: Path, *, ckpt: str = "final") -> Path:
    rows = _load_comp_rows(stamp)
    sites = ["L0_mid", "L0_post", "L1_mid", "L1_post", "L2_mid", "L2_post"]
    site_labels = ["L0\nmid", "L0\npost", "L1\nmid", "L1\npost", "L2\nmid", "L2\npost"]
    harmonic: dict[str, dict[str, float]] = defaultdict(dict)
    for r in rows:
        if r["ckpt"] != ckpt or r["kind"] != "harmonic":
            continue
        harmonic[r["site_or_layer"]][r["metric"]] = float(r["value"])

    x = range(len(sites))
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(
        x,
        [harmonic[s]["R2_operand_mean"] for s in sites],
        "o-",
        label=r"$R^2$ operand harmonics",
        color="#4C78A8",
        lw=1.8,
        ms=5,
    )
    ax.plot(
        x,
        [harmonic[s]["R2_sum_mean"] for s in sites],
        "D-",
        label=r"$R^2$ sum harmonics",
        color="#E45756",
        lw=2.2,
        ms=5.5,
    )
    ax.axvspan(-0.35, 1.35, color="#E45756", alpha=0.08, zorder=0)
    ax.set_xticks(list(x))
    ax.set_xticklabels(site_labels)
    ax.set_ylabel(r"holdout $R^2$ (ridge)")
    ax.set_xlabel("query residual site")
    ax.set_title(rf"Frequency harmonic signature · $p=31$ {ckpt}")
    ax.legend(frameon=False, loc="lower right")
    ax.set_ylim(-0.15, 1.05)
    fig.tight_layout()
    path = out / "fig2_harmonic_signature.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_ablation(stamp: Path, out: Path, *, ckpt: str = "final") -> Path:
    rows = _load_abl_rows(stamp)
    imp: dict[int, float] = {}
    unimp: dict[int, float] = {}
    base = 1.0
    for r in rows:
        if r["ckpt"] != ckpt:
            continue
        k = int(r["k"])
        base = float(r["baseline_acc"])
        if r["kind"] == "important":
            imp[k] = float(r["acc"])
        else:
            unimp[k] = float(r["acc"])
    ks = sorted(imp)
    fig, ax = plt.subplots(figsize=(6.2, 3.5))
    ax.plot(
        ks,
        [imp[k] for k in ks],
        "D-",
        color="#E45756",
        lw=2.0,
        ms=4.5,
        label="ablate top-$k$ (important)",
    )
    ax.plot(
        ks,
        [unimp[k] for k in ks],
        "o-",
        color="#4C78A8",
        lw=1.8,
        ms=4.5,
        label="ablate bottom-$k$ (unimportant)",
    )
    ax.axhline(base, color="0.4", ls="--", lw=1, label="baseline")
    ax.axhline(1 / 31, color="0.5", ls=":", lw=1, label="chance")
    ax.set_xlabel(r"# conjugate freq-pairs removed ($k$)")
    ax.set_ylabel("test accuracy")
    ax.set_title(rf"Digit-emb Fourier ablation · $p=31$ {ckpt}")
    ax.legend(frameon=False, loc="center right")
    ax.set_ylim(-0.02, 1.08)
    ax.set_xlim(0.5, max(ks) + 0.5)
    fig.tight_layout()
    path = out / "fig3_fourier_ablation.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_head_knockout(stamp: Path, out: Path, *, ckpt: str = "final") -> Path:
    rows = _load_comp_rows(stamp)
    by: dict[str, float] = {}
    for r in rows:
        if r["ckpt"] != ckpt or r["kind"] != "head_knockout":
            continue
        if r["metric"] != "delta_acc":
            continue
        by[r["site_or_layer"]] = float(r["value"])

    n_layers, n_heads = 3, 4
    mat = [
        [by.get(f"L{li}H{h}", 0.0) for h in range(n_heads)] for li in range(n_layers)
    ]
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    im = ax.imshow(mat, cmap="RdBu", vmin=-1, vmax=0, aspect="auto")
    ax.set_xticks(range(n_heads))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_yticks(range(n_layers))
    ax.set_yticklabels([f"L{li}" for li in range(n_layers)])
    for i in range(n_layers):
        for j in range(n_heads):
            ax.text(
                j,
                i,
                f"{mat[i][j]:+.2f}",
                ha="center",
                va="center",
                color="k",
                fontsize=9,
            )
    ax.set_title(rf"Per-head zero-ablation $\Delta$acc · $p=31$ {ckpt}")
    ax.set_xlabel("attention head")
    ax.set_ylabel("layer")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$\Delta$ test acc")
    fig.tight_layout()
    path = out / "fig4_head_knockout.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_attention_from_report(stamp: Path, out: Path, *, ckpt: str = "final") -> Path | None:
    """Bar chart of L0 operand routing from report attention summary + knockout Δ."""
    report_path = stamp / f"p31_{ckpt}_report.json"
    if not report_path.is_file():
        return None
    rep = json.loads(report_path.read_text())
    layers = (rep.get("attention") or {}).get("layers") or []
    if not layers:
        return None
    L0 = layers[0]
    masses = L0.get("operand_mass_by_head") or []
    if not masses:
        return None

    # Prefer head knockout deltas from composition CSV if present
    deltas: dict[int, float] = {}
    comp_csv = stamp / "phase1_mech-single_composition.csv"
    if comp_csv.is_file():
        for r in csv.DictReader(comp_csv.open()):
            if (
                r["ckpt"] == ckpt
                and r["kind"] == "head_knockout"
                and r["metric"] == "delta_acc"
                and r["site_or_layer"].startswith("L0H")
            ):
                h = int(r["site_or_layer"].removeprefix("L0H"))
                deltas[h] = float(r["value"])

    n_heads = len(masses)
    # Report only gives combined operand mass; split is not in summary.
    # Plot total operand mass per head + delta annotation.
    fig, ax = plt.subplots(figsize=(6.0, 3.2))
    x = np.arange(n_heads)
    ax.bar(x, masses, color="#4C78A8", label="operand mass (xi+xj)")
    ax.set_xticks(list(x))
    ax.set_xticklabels([f"H{h}" for h in range(n_heads)])
    ax.set_ylabel("mean query attention on operands")
    ax.set_title(rf"L0 head routing · $p=31$ {ckpt}  ($i={rep['operand_i']}, j={rep['operand_j']}$)")
    ax.set_ylim(0, 1.15)
    ax.legend(frameon=False, loc="lower right")
    for h, m in enumerate(masses):
        d = deltas.get(h)
        label = f"{m:.2f}" if d is None else f"{m:.2f}\nΔ={d:+.2f}"
        ax.text(h, max(m, 0.02) + 0.03, label, ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    path = out / "fig5b_L0_head_routing.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--composition-stamp",
        type=Path,
        default=DEFAULT_COMPOSITION,
        help="Stamp with full ablation + composition ladder (default: 20261002_105034)",
    )
    parser.add_argument(
        "--head-stamp",
        type=Path,
        default=DEFAULT_HEAD,
        help="Stamp with head knockout (default: 20261002_110852)",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--ckpt", type=str, default="final")
    args = parser.parse_args()

    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    written.append(plot_composition_locus(args.composition_stamp, args.out, ckpt=args.ckpt))
    written.append(plot_harmonic(args.composition_stamp, args.out, ckpt=args.ckpt))
    written.append(plot_ablation(args.composition_stamp, args.out, ckpt=args.ckpt))
    written.append(plot_head_knockout(args.head_stamp, args.out, ckpt=args.ckpt))
    p = plot_attention_from_report(args.head_stamp, args.out, ckpt=args.ckpt)
    if p is not None:
        written.append(p)

    # Copy full attention heatmap from head stamp if present
    src = args.head_stamp / "figures" / "fig5_attention_patterns.png"
    if src.is_file():
        dst = args.out / "fig5_attention_patterns.png"
        dst.write_bytes(src.read_bytes())
        written.append(dst)

    print("wrote:")
    for p in written:
        print(f"  {p}")


if __name__ == "__main__":
    main()
