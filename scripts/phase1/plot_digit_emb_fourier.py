#!/usr/bin/env python3
"""1A digit-embedding Fourier energy as polar rings, one panel per modulus.

Each frequency k sits on the circle (k=0 at the top, clockwise).
Bar height on the ring is energy / that modulus's peak bin.

Sources:
  p=31  — mech_single/p31_summary/composition_stamp/p31_final_report.json
  others — mech_single/20261002_cross/p{m}/p{m}_final_report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

CROSS = Path("runs/phase1/mech_single/20261002_cross")
P31 = Path("runs/phase1/mech_single/p31_summary/composition_stamp/p31_final_report.json")
MODULI = (19, 23, 29, 31, 37, 41, 43, 47)

RING = 1.0
COLOR_DC = "#C4C7C5"
COLOR_TOP = "#4C78A8"
COLOR_OTHER = "#9AA0A6"
COLOR_RING = "#E8EAED"


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 11,
            "figure.dpi": 150,
            "savefig.dpi": 200,
        }
    )


def _load(p: int) -> dict:
    path = P31 if p == 31 else CROSS / f"p{p}" / f"p{p}_final_report.json"
    report = json.loads(path.read_text())
    emb = report["fourier_digit_emb"]
    energy = np.asarray(emb["energy_by_freq"], dtype=np.float64)
    total = float(emb.get("total_energy") or energy.sum())
    return {
        "p": p,
        "frac": energy / total if total > 0 else energy,
        "top_freq": int(emb["top_freq"]),
        "top_energy_frac": float(emb["top_energy_frac"]),
    }


def _draw_ring(ax, row: dict, *, tick_size: int = 7) -> None:
    p = row["p"]
    frac = row["frac"]
    k_star = row["top_freq"]
    conj = (p - k_star) % p
    peak = float(max(frac.max(), 1e-9))
    scale = 0.85 / peak

    theta = 2 * np.pi * np.arange(p) / p
    width = 2 * np.pi / p * 0.82
    heights = frac * scale
    colors = [
        COLOR_DC if k == 0 else COLOR_TOP if k in {k_star, conj} else COLOR_OTHER
        for k in range(p)
    ]

    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.set_facecolor("white")
    ax.spines["polar"].set_visible(False)
    ax.grid(False)
    ax.set_yticklabels([])
    ax.set_ylim(0, RING + 0.95)

    ring_theta = np.linspace(0, 2 * np.pi, 512)
    ax.fill_between(ring_theta, 0, RING, color=COLOR_RING, zorder=0)
    ax.plot(ring_theta, np.full_like(ring_theta, RING), color="#BDC1C6", lw=1.0, zorder=1)
    ax.bar(
        theta,
        heights,
        width=width,
        bottom=RING,
        color=colors,
        edgecolor="white",
        linewidth=0.3,
        align="center",
        zorder=2,
    )

    tick_ks = sorted({0, k_star, conj})
    ax.set_xticks([2 * np.pi * k / p for k in tick_ks])
    ax.set_xticklabels([str(k) for k in tick_ks], fontsize=tick_size)
    ax.tick_params(axis="x", pad=1)
    ax.set_title(rf"$p={p}$  $k^\star={k_star},{conj}$", pad=8)


def plot_grid(rows: list[dict], out: Path) -> Path:
    fig, axes = plt.subplots(
        1,
        8,
        figsize=(22.0, 4.2),
        subplot_kw={"projection": "polar"},
    )
    for ax, row in zip(axes.ravel(), rows, strict=True):
        _draw_ring(ax, row)
    fig.suptitle(
        "1A digit embedding Fourier  ·  bar height = energy / peak bin",
        y=1.04,
    )
    fig.legend(
        handles=[
            Patch(facecolor=COLOR_TOP, label="top conjugate pair"),
            Patch(facecolor=COLOR_OTHER, label="other $k$"),
            Patch(facecolor=COLOR_DC, label=r"DC $k=0$"),
        ],
        loc="lower center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, -0.04),
        fontsize=9,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.92))
    path = out / "digit_emb_fourier_rings.png"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=CROSS / "figures")
    args = parser.parse_args()
    _style()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [_load(p) for p in MODULI]
    path = plot_grid(rows, args.out)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
