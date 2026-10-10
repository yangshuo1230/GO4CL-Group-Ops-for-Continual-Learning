#!/usr/bin/env python3
"""Digit-embedding Fourier spectra at the A→B switch, with and without replay.

Before: the shared A source at 100k (identical to step 0 of every full_A branch).
After: the 100k endpoint of A→B with no replay, and of A→B with 10% replay.
B-only is drawn as a reference spectrum of B trained from scratch.

Each panel is one modulus. The x-axis is frequency k = 1 .. (p-1)/2; k and p-k
are conjugates and carry the same energy, so only one representative is shown.
Height is that bin's share of non-DC energy.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from go4cl.analysis.fourier import analyze_digit_embedding_fourier, energy_cosine

ROOT = Path(__file__).resolve().parents[2]
SUITE = ROOT / "runs/phase2/mechanism_suite_20261009"
OUT = SUITE / "report_figures/embedding_spectra_20261010"

KINDS = (
    "same_mod_same_pos",
    "same_mod_diff_pos",
    "diff_mod_same_pos",
    "diff_mod_diff_pos",
)
KIND_TITLE = {
    "same_mod_same_pos": "Same modulus, same positions",
    "same_mod_diff_pos": "Same modulus, different positions",
    "diff_mod_same_pos": "Different modulus, same positions",
    "diff_mod_diff_pos": "Different modulus, different positions",
}
KIND_ROW = {
    "same_mod_same_pos": "Same mod\nsame pos",
    "same_mod_diff_pos": "Same mod\ndiff pos",
    "diff_mod_same_pos": "Diff mod\nsame pos",
    "diff_mod_diff_pos": "Diff mod\ndiff pos",
}
A_MODULI = {
    "same_mod_same_pos": (23, 41, 37, 53),
    "same_mod_diff_pos": (23, 41, 37, 53),
    "diff_mod_same_pos": (29, 43, 37, 53),
    "diff_mod_diff_pos": (29, 43, 37, 53),
}
B_MODULI = (23, 41, 31, 47)

SERIES = (
    ("before", "Before switch", "#222222", "-"),
    ("noreplay", "After B, no replay", "#D55E00", "-"),
    ("replay", "After B, 10% replay", "#0072B2", "-"),
    ("bonly", "B-only", "#009E73", "--"),
)


def moduli_for(kind: str) -> list[tuple[int, str]]:
    a, b = set(A_MODULI[kind]), set(B_MODULI)
    shared = [p for p in A_MODULI[kind] if p in b]
    a_only = [p for p in A_MODULI[kind] if p not in b]
    b_only = [p for p in B_MODULI if p not in a]
    return (
        [(p, "shared") for p in shared]
        + [(p, "A only") for p in a_only]
        + [(p, "B only") for p in b_only]
    )


def load_embedding(path: Path) -> nn.Embedding:
    payload = torch.load(path, map_location="cpu", weights_only=False)
    weight = payload["model_state"]["tok_emb.weight"].detach().float().cpu()
    emb = nn.Embedding(weight.shape[0], weight.shape[1])
    with torch.no_grad():
        emb.weight.copy_(weight)
    return emb


def positive_share(energy: np.ndarray) -> np.ndarray:
    """One representative of each conjugate pair, as a share of non-DC energy."""
    e = np.asarray(energy, dtype=np.float64).copy()
    e[0] = 0.0
    total = float(e.sum())
    if total <= 0:
        return e
    return e / total


def analyze(emb: nn.Embedding, modulus: int) -> dict:
    raw = analyze_digit_embedding_fourier(emb, modulus=modulus)
    energy = np.asarray(raw["energy_by_freq"], dtype=np.float64)
    share = positive_share(energy)
    half = (modulus - 1) // 2
    top = int(np.argmax(share[1 : half + 1]) + 1)
    return {
        "energy": energy,
        "share": share,
        "top_k": top,
        "top_share": float(share[top]),
        "pair_share": float(share[top] + share[modulus - top]),
    }


def collect(seed: int) -> dict:
    bonly = load_embedding(SUITE / f"seed{seed}/main/B_only/checkpoints/final.pt")
    out = {}
    for kind in KINDS:
        before = load_embedding(SUITE / f"seed{seed}/source/{kind}/checkpoints/final.pt")
        noreplay = load_embedding(SUITE / f"seed{seed}/main/{kind}/full_A/checkpoints/final.pt")
        replay = load_embedding(SUITE / f"seed{seed}/main/{kind}/replay_0.1/checkpoints/final.pt")
        embs = {"before": before, "noreplay": noreplay, "replay": replay, "bonly": bonly}
        out[kind] = {
            p: {name: analyze(emb, p) for name, emb in embs.items()}
            for p, _role in moduli_for(kind)
        }
    return out


def draw_kind_group(seed: int, spectra: dict, kinds: tuple[str, ...], path: Path) -> None:
    n_col = max(len(moduli_for(k)) for k in kinds)
    n_row = len(kinds)
    fig, axes = plt.subplots(n_row, n_col, figsize=(3.05 * n_col, 2.55 * n_row), sharey=True)
    if n_row == 1:
        axes = np.array([axes])
    for row, kind in enumerate(kinds):
        panels = moduli_for(kind)
        for col in range(n_col):
            ax = axes[row, col]
            if col >= len(panels):
                ax.axis("off")
                continue
            modulus, role = panels[col]
            block = spectra[kind][modulus]
            half = (modulus - 1) // 2
            ks = np.arange(1, half + 1)
            for name, label, color, ls in SERIES:
                share = block[name]["share"]
                ax.plot(ks, share[ks], color=color, linestyle=ls, linewidth=1.35 if name != "bonly" else 1.05, label=label)
            ax.axvline(block["before"]["top_k"], color="#222222", linestyle=":", linewidth=0.7, alpha=0.7)
            cos_nr = energy_cosine(block["before"]["energy"], block["noreplay"]["energy"])
            cos_rp = energy_cosine(block["before"]["energy"], block["replay"]["energy"])
            ax.text(
                0.98,
                0.95,
                f"cos {cos_nr:.2f} / {cos_rp:.2f}",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7.5,
                color="#444444",
            )
            ax.set_xlim(1, half)
            ax.set_ylim(0, 0.30)
            ax.grid(alpha=0.18)
            ax.tick_params(labelsize=8)
            if row == 0:
                ax.set_title(f"p={modulus}  ·  {role}", fontsize=10)
            if col == 0:
                ax.set_ylabel(KIND_ROW[kind], fontsize=8)
            if row == n_row - 1:
                ax.set_xlabel("Frequency k")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, frameon=False, fontsize=9)
    fig.suptitle(
        f"Digit-embedding Fourier  ·  seed {seed}\n"
        "Y-axis: one bin's share of non-DC energy (k and p−k are equal; only k ≤ (p−1)/2 is drawn). "
        "Dotted line: pre-switch peak. Corner: cosine(before, no replay) / cosine(before, 10% replay).",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.90))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def draw_summary(records: list[dict], path: Path) -> None:
    """Cosine with the pre-switch spectrum, and the pre-switch peak's remaining share."""
    moduli = (23, 29, 31, 37, 41, 43, 47, 53)
    seeds = sorted({r["seed"] for r in records})
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 7.2), sharex=True, sharey="row")
    metrics = (
        ("cos_noreplay", "No replay vs before"),
        ("cos_replay", "10% replay vs before"),
    )
    for col, (key, title) in enumerate(metrics):
        for row, seed in enumerate(seeds):
            ax = axes[row, col]
            subset = [r for r in records if r["seed"] == seed]
            kinds = list(dict.fromkeys(r["kind"] for r in subset))
            x = np.arange(len(moduli))
            width = 0.18
            for i, kind in enumerate(kinds):
                vals = []
                for p in moduli:
                    hit = [r[key] for r in subset if r["kind"] == kind and r["modulus"] == p]
                    vals.append(hit[0] if hit else np.nan)
                ax.bar(x + (i - 1.5) * width, vals, width, label=KIND_TITLE[kind], color=plt.cm.tab10(i))
            ax.set_ylim(0, 1.05)
            ax.axhline(1.0, color="#bbbbbb", linewidth=0.6)
            ax.set_title(f"Seed {seed}: {title}")
            ax.grid(axis="y", alpha=0.18)
            if row == 1:
                ax.set_xticks(x, [str(p) for p in moduli])
                ax.set_xlabel("Modulus")
            if col == 0:
                ax.set_ylabel("Spectrum cosine, DC excluded")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("How close the post-switch digit embedding stays to the pre-switch spectrum")
    fig.tight_layout(rect=(0, 0.08, 1, 0.94))
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "figure.dpi": 120,
            "savefig.dpi": 160,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in (0, 1):
        spectra = collect(seed)
        draw_kind_group(
            seed,
            spectra,
            ("same_mod_same_pos", "same_mod_diff_pos"),
            OUT / f"seed{seed}_same_mod.png",
        )
        draw_kind_group(
            seed,
            spectra,
            ("diff_mod_same_pos", "diff_mod_diff_pos"),
            OUT / f"seed{seed}_diff_mod.png",
        )
        for kind, panels in spectra.items():
            for modulus, block in panels.items():
                role = next(role for p, role in moduli_for(kind) if p == modulus)
                rows.append(
                    {
                        "seed": seed,
                        "kind": kind,
                        "modulus": modulus,
                        "role": role,
                        "cos_noreplay": energy_cosine(block["before"]["energy"], block["noreplay"]["energy"]),
                        "cos_replay": energy_cosine(block["before"]["energy"], block["replay"]["energy"]),
                        "cos_noreplay_bonly": energy_cosine(block["noreplay"]["energy"], block["bonly"]["energy"]),
                        "cos_replay_bonly": energy_cosine(block["replay"]["energy"], block["bonly"]["energy"]),
                        **{
                            f"{name}_top_k": block[name]["top_k"]
                            for name, *_rest in SERIES
                        },
                        **{
                            f"{name}_top_share": block[name]["top_share"]
                            for name, *_rest in SERIES
                        },
                    }
                )
    draw_summary(rows, OUT / "summary_cosine.png")
    fieldnames = list(rows[0].keys())
    with (OUT / "spectra_summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    (OUT / "spectra_summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
