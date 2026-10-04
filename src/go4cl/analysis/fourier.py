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


def analyze_unembedding_fourier(
    head: nn.Linear,
    *,
    modulus: int,
) -> dict[str, Any]:
    """Fourier on output-head rows indexed by class residue ``0..p-1``.

    The shared head has ``C ≥ p`` classes. Valid labels for modulus ``p`` are
    exactly those rows, so this is the unembedding analogue of residue-averaged
    digit embeddings (no alias averaging).
    """
    if modulus < 2:
        raise ValueError(f"modulus must be >= 2, got {modulus}")
    weight = head.weight.detach().cpu()
    if weight.shape[0] < modulus:
        raise ValueError(
            f"unembed has {weight.shape[0]} classes < modulus {modulus}"
        )
    # [p, D] — class id is already the residue
    result = fourier_energy(weight[:modulus])
    result["source"] = "unembed_head_rows"
    result["n_classes"] = int(weight.shape[0])
    if head.bias is not None:
        bias = head.bias.detach().cpu()[:modulus]
        result["bias_fourier"] = fourier_energy(bias.unsqueeze(-1))
        result["bias_fourier"]["source"] = "unembed_bias"
    return result


def energy_cosine(
    energy_a: list[float] | torch.Tensor,
    energy_b: list[float] | torch.Tensor,
    *,
    skip_dc: bool = True,
) -> float:
    """Cosine similarity of two length-p energy spectra."""
    a = np.asarray(energy_a, dtype=np.float64).reshape(-1)
    b = np.asarray(energy_b, dtype=np.float64).reshape(-1)
    n = min(a.size, b.size)
    a = a[:n].copy()
    b = b[:n].copy()
    if skip_dc and n:
        a[0] = 0.0
        b[0] = 0.0
    na = float(np.linalg.norm(a))
    nb = float(np.linalg.norm(b))
    if na < 1e-12 or nb < 1e-12:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
