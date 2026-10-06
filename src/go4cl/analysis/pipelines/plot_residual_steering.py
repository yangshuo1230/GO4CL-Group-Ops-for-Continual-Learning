"""Paper figures for Phase 1A residual steering (CPU, non-interactive)."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np

DEFAULT_DATA = Path("runs/phase1/mech_single/p31_summary/residual_steering")
DEFAULT_FIG = Path("runs/phase1/mech_single/p31_summary/figures")

COND_ORDER = ("no_steering", "structured", "shuffled")
COND_COLORS = {
    "no_steering": "#8A8A8A",
    "structured": "#E45756",
    "shuffled": "#4C78A8",
    "global_structured": "#F2A93B",
}
COND_LABELS = {
    "no_steering": "no steering",
    "structured": "structured",
    "shuffled": "shuffled",
    "global_structured": "global structured",
}
SITE_ORDER = ("L0_post", "L1_post", "L2_post")
MATRIX_CMAP = "magma"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 11,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 9,
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
        }
    )


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _summary_lookup(rows: list[dict[str, str]]) -> dict[tuple[str, int, str], dict]:
    out: dict[tuple[str, int, str], dict] = {}
    for r in rows:
        key = (r["site"], int(r["delta"]), r["condition"])
        out[key] = {
            "original_acc": float(r["original_acc"]),
            "target_acc": float(r["target_acc"]),
            "n": int(r["n"]),
            "modulus": int(float(r["modulus"])),
            "alpha": float(r["alpha"]),
        }
    return out


def matrix_from_prediction_rows(
    rows: list[dict[str, str]],
    *,
    site: str,
    delta: int,
    condition: str,
    modulus: int,
) -> np.ndarray:
    m = np.zeros((modulus, modulus), dtype=np.float64)
    counts = np.zeros((modulus, modulus), dtype=np.float64)
    for r in rows:
        if r["site"] != site or int(r["delta"]) != delta or r["condition"] != condition:
            continue
        s = int(r["source_residue"])
        k = int(r["predicted_residue"])
        counts[s, k] += float(r["count"])
    denom = counts.sum(axis=1, keepdims=True)
    np.divide(counts, denom, out=m, where=denom > 0)
    return m


def _save(fig: plt.Figure, path: Path) -> list[Path]:
    path.parent.mkdir(parents=True, exist_ok=True)
    png = path.with_suffix(".png")
    fig.savefig(png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return [png]


def _panel_a(ax: plt.Axes) -> None:
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")
    ax.set_title("A. Query residual steering")
    boxes = [
        (1.1, 7.2, r"$h_s$"),
        (1.1, 4.6, r"$+\ \alpha(\mu_{s+\delta}-\mu_s)$"),
        (1.1, 2.2, r"$h_{\mathrm{steered}}$"),
        (5.6, 2.2, "continue forward"),
        (5.6, 5.4, r"pred $(s+\delta)$ mod $p$"),
    ]
    for x, y, text in boxes:
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                3.4,
                1.6,
                boxstyle="round,pad=0.08,rounding_size=0.15",
                facecolor="#F4F4F4",
                edgecolor="#333333",
                linewidth=1.0,
            )
        )
        ax.text(x + 1.7, y + 0.8, text, ha="center", va="center", fontsize=10)
    arrows = [
        ((2.8, 7.2), (2.8, 6.2)),
        ((2.8, 4.6), (2.8, 3.8)),
        ((4.5, 3.0), (5.6, 3.0)),
        ((7.3, 3.8), (7.3, 5.4)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.add_patch(
            FancyArrowPatch(
                (x0, y0),
                (x1, y1),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=1.2,
                color="#333333",
            )
        )
    ax.text(
        5.0,
        0.45,
        "query token only  ·  μ from reference split",
        ha="center",
        va="center",
        fontsize=9,
        color="#555555",
    )


def _grouped_bars(
    ax: plt.Axes,
    lookup: dict[tuple[str, int, str], dict],
    *,
    metric: str,
    delta: int,
    chance: float,
    title: str,
    ylabel: str,
    show_legend: bool,
) -> None:
    x = np.arange(len(SITE_ORDER))
    width = 0.24
    offsets = (-width, 0.0, width)
    for off, cond in zip(offsets, COND_ORDER, strict=True):
        vals = [
            lookup.get((site, delta, cond), {}).get(metric, np.nan)
            for site in SITE_ORDER
        ]
        ax.bar(
            x + off,
            vals,
            width=width,
            color=COND_COLORS[cond],
            label=COND_LABELS[cond],
        )
    ax.axhline(chance, color="#333333", ls="--", lw=1.0, zorder=0)
    ax.text(
        len(SITE_ORDER) - 0.55,
        chance + 0.03,
        f"chance 1/{int(round(1 / chance))}" if chance > 0 else "chance",
        ha="right",
        va="bottom",
        fontsize=8,
        color="#333333",
    )
    ax.set_xticks(list(x))
    ax.set_xticklabels([s.replace("_", "\n") for s in SITE_ORDER])
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if show_legend:
        ax.legend(frameon=False, loc="lower left")


def _transition_panel(
    ax: plt.Axes,
    matrix: np.ndarray,
    *,
    delta: int,
    title: str,
    overlay: str,
    show_cbar: bool,
    fig: plt.Figure,
) -> None:
    p = matrix.shape[0]
    im = ax.imshow(
        matrix.T,
        origin="lower",
        vmin=0.0,
        vmax=1.0,
        cmap=MATRIX_CMAP,
        aspect="equal",
        interpolation="nearest",
    )
    s = np.arange(p)
    if overlay == "identity":
        y = s
    elif overlay == "shift":
        y = (s + int(delta)) % p
    else:
        y = None
    if y is not None:
        ax.scatter(
            s,
            y,
            s=9,
            facecolors="none",
            edgecolors="white",
            linewidths=0.7,
            zorder=3,
        )
    ax.set_xlabel(r"source residue $s$")
    ax.set_ylabel("predicted residue")
    ax.set_title(title)
    ax.set_xticks([0, p // 2, p - 1])
    ax.set_yticks([0, p // 2, p - 1])
    ax.grid(False)
    if show_cbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label=r"$P(\hat y \mid s)$")


def plot_fig6(
    *,
    summary_rows: list[dict[str, str]],
    prediction_rows: list[dict[str, str]],
    out_stem: Path,
    delta: int = 4,
    matrix_site: str = "L0_post",
) -> list[Path]:
    _style()
    lookup = _summary_lookup(summary_rows)
    any_row = next(iter(lookup.values()))
    modulus = int(any_row["modulus"])
    chance = 1.0 / modulus
    fig, axes = plt.subplots(2, 3, figsize=(12.6, 8.2), layout="constrained")
    _panel_a(axes[0, 0])
    _grouped_bars(
        axes[0, 1],
        lookup,
        metric="target_acc",
        delta=delta,
        chance=chance,
        title=rf"B. Target accuracy  ($\delta={delta}$)",
        ylabel="target accuracy",
        show_legend=True,
    )
    _grouped_bars(
        axes[0, 2],
        lookup,
        metric="original_acc",
        delta=delta,
        chance=chance,
        title=rf"C. Original accuracy  ($\delta={delta}$)",
        ylabel="original accuracy",
        show_legend=False,
    )
    overlays = {
        "no_steering": "identity",
        "structured": "shift",
        "shuffled": "none",
    }
    titles = {
        "no_steering": rf"D. No steering  ({matrix_site})",
        "structured": rf"E. Structured  ($\delta={delta}$)",
        "shuffled": "F. Shuffled means",
    }
    for ax, cond in zip(axes[1], COND_ORDER, strict=True):
        mat = matrix_from_prediction_rows(
            prediction_rows,
            site=matrix_site,
            delta=delta,
            condition=cond,
            modulus=modulus,
        )
        _transition_panel(
            ax,
            mat,
            delta=delta,
            title=titles[cond],
            overlay=overlays[cond],
            show_cbar=(cond == "shuffled"),
            fig=fig,
        )
    return _save(fig, out_stem)


def plot_fig6b(
    *,
    summary_rows: list[dict[str, str]],
    out_stem: Path,
    deltas: list[int] | None = None,
) -> list[Path]:
    _style()
    lookup = _summary_lookup(summary_rows)
    conditions = ["structured", "shuffled"]
    if any(k[2] == "global_structured" for k in lookup):
        conditions.append("global_structured")
    deltas = deltas or sorted({k[1] for k in lookup})
    sites = [s for s in SITE_ORDER if any(k[0] == s for k in lookup)]
    n = len(conditions)
    fig, axes = plt.subplots(1, n, figsize=(4.2 * n, 3.6), layout="constrained")
    if n == 1:
        axes = [axes]
    mats = []
    for cond in conditions:
        m = np.full((len(sites), len(deltas)), np.nan)
        for i, site in enumerate(sites):
            for j, d in enumerate(deltas):
                rec = lookup.get((site, int(d), cond))
                if rec is not None:
                    m[i, j] = rec["target_acc"]
        mats.append(m)
    for ax, cond, mat in zip(axes, conditions, mats, strict=True):
        im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(len(deltas)))
        ax.set_xticklabels([str(d) for d in deltas])
        ax.set_yticks(range(len(sites)))
        ax.set_yticklabels(sites)
        ax.set_xlabel(r"$\delta$")
        ax.set_title(f"{COND_LABELS[cond]}\ntarget accuracy")
        ax.grid(False)
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                val = mat[i, j]
                if not np.isfinite(val):
                    continue
                ax.text(
                    j,
                    i,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=9,
                    color="white" if val >= 0.55 else "black",
                )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    return _save(fig, out_stem)


def plot_all(
    *,
    data_dir: Path = DEFAULT_DATA,
    figures_dir: Path = DEFAULT_FIG,
    delta: int = 4,
    matrix_site: str = "L0_post",
) -> list[Path]:
    summary = _load_csv(Path(data_dir) / "steering_summary.csv")
    preds = _load_csv(Path(data_dir) / "steering_predictions.csv")
    written: list[Path] = []
    written.extend(
        plot_fig6(
            summary_rows=summary,
            prediction_rows=preds,
            out_stem=Path(figures_dir) / "fig6_residual_steering",
            delta=delta,
            matrix_site=matrix_site,
        )
    )
    written.extend(
        plot_fig6b(
            summary_rows=summary,
            out_stem=Path(figures_dir) / "fig6b_steering_across_deltas",
        )
    )
    return written


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--figures", type=Path, default=DEFAULT_FIG)
    parser.add_argument("--delta", type=int, default=4)
    parser.add_argument("--matrix-site", type=str, default="L0_post")
    args = parser.parse_args(argv)
    written = plot_all(
        data_dir=args.data_dir,
        figures_dir=args.figures,
        delta=args.delta,
        matrix_site=args.matrix_site,
    )
    print("wrote:")
    for path in written:
        print(f"  {path}")


if __name__ == "__main__":
    main()
