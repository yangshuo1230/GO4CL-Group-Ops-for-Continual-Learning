"""Mod-p Fourier analysis of embeddings / residue-averaged residuals."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn

from go4cl.constants import NUM_DIGITS


def residue_mean_vectors(
    vectors: torch.Tensor,
    residues: torch.Tensor,
    *,
    modulus: int,
) -> torch.Tensor:
    """Average ``vectors`` [N, D] within each residue class → [p, D]."""
    d = vectors.shape[-1]
    out = torch.zeros(modulus, d, dtype=vectors.dtype)
    counts = torch.zeros(modulus, dtype=torch.long)
    for r in range(modulus):
        mask = residues == r
        n = int(mask.sum().item())
        counts[r] = n
        if n > 0:
            out[r] = vectors[mask].mean(dim=0)
    return out


def fourier_energy(residue_means: torch.Tensor) -> dict[str, Any]:
    """DFT energy spectrum over the residue axis.

    ``residue_means``: [p, D]. Returns per-frequency energy (averaged over D)
    and total energy for normalization checks.
    """
    x = residue_means.detach().cpu().numpy().astype(np.float64)  # [p, D]
    p, d = x.shape
    # FFT along residue axis
    spec = np.fft.fft(x, axis=0)  # [p, D] complex
    # Energy per frequency (Parseval: mean |x|^2 ~= mean |spec|^2 / p)
    energy = (np.abs(spec) ** 2).mean(axis=1) / max(p, 1)  # [p]
    # Frequencies 0..p-1; for real signals k and p-k are conjugates — keep all
    total = float(energy.sum())
    ranked = sorted(
        [{"freq": int(k), "energy": float(energy[k])} for k in range(p)],
        key=lambda e: e["energy"],
        reverse=True,
    )
    return {
        "modulus": p,
        "d_model": d,
        "energy_by_freq": [float(e) for e in energy],
        "total_energy": total,
        "top_freqs": ranked[: min(8, p)],
        "top_freq": ranked[0]["freq"] if ranked else 0,
        "top_energy_frac": (
            float(ranked[0]["energy"] / total) if ranked and total > 0 else 0.0
        ),
    }


def analyze_digit_embedding_fourier(
    tok_emb: nn.Embedding,
    *,
    modulus: int,
) -> dict[str, Any]:
    """Fourier on residue-averaged digit token embeddings (0..NUM_DIGITS-1)."""
    weight = tok_emb.weight.detach().cpu()[:NUM_DIGITS]  # [64, D]
    digits = torch.arange(NUM_DIGITS)
    residues = digits % modulus
    means = residue_mean_vectors(weight, residues, modulus=modulus)
    result = fourier_energy(means)
    result["source"] = "digit_tok_emb"
    return result


def analyze_query_resid_fourier(
    query_resid: torch.Tensor,
    residues: torch.Tensor,
    *,
    modulus: int,
    source: str = "query_resid_by_sum",
) -> dict[str, Any]:
    means = residue_mean_vectors(query_resid, residues, modulus=modulus)
    result = fourier_energy(means)
    result["source"] = source
    return result
