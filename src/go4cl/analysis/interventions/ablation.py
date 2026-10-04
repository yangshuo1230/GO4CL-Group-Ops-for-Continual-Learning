"""Fourier / embedding ablations and accuracy evaluation.

Does not import phases. Mutates ``model.tok_emb`` / ``model.head`` only
inside a restore-to-original try pattern.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from go4cl.analysis.fourier import (
    analyze_digit_embedding_fourier,
    analyze_unembedding_fourier,
)
from go4cl.constants import NUM_DIGITS
from go4cl.model.transformer import ModularTransformer

def project_out_freqs_from_digit_emb(
    weight: torch.Tensor,
    *,
    modulus: int,
    freqs: list[int],
) -> torch.Tensor:
    """Remove selected Fourier modes from residue-averaged digit structure.

    For each digit x, decompose the residue-mean embedding table, zero given
    frequencies, and replace each digit embedding by
    ``emb(x) - mean_r + reconstructed_mean[x%p]``.
    """
    w = weight.detach().cpu().numpy().astype(np.float64)  # [V, D] or [64, D]
    digits = w[:NUM_DIGITS]
    d = digits.shape[1]
    means = np.zeros((modulus, d), dtype=np.float64)
    counts = np.zeros(modulus, dtype=np.int64)
    for x in range(NUM_DIGITS):
        r = x % modulus
        means[r] += digits[x]
        counts[r] += 1
    counts = np.maximum(counts, 1)
    means /= counts[:, None]

    spec = np.fft.fft(means, axis=0)  # [p, D]
    for k in freqs:
        kk = int(k) % modulus
        spec[kk] = 0
        # also zero conjugate for real reconstruction when k != 0 and k != p-k
        if kk != 0 and (modulus - kk) % modulus != kk:
            spec[(modulus - kk) % modulus] = 0
    recon = np.fft.ifft(spec, axis=0).real  # [p, D]

    out = weight.detach().cpu().clone()
    for x in range(NUM_DIGITS):
        r = x % modulus
        # preserve digit-specific deviation from residue mean; replace mean
        out[x] = torch.as_tensor(
            digits[x] - means[r] + recon[r], dtype=out.dtype
        )
    return out


def project_out_freqs_from_unembed(
    weight: torch.Tensor,
    *,
    modulus: int,
    freqs: list[int],
) -> torch.Tensor:
    """Zero selected Fourier modes on unembedding rows ``0..p-1``.

    Rows ``p..C-1`` (unused for this modulus) are left unchanged.
    """
    w = weight.detach().cpu().clone()
    if w.shape[0] < modulus:
        raise ValueError(f"unembed {w.shape[0]} classes < modulus {modulus}")
    rows = w[:modulus].numpy().astype(np.float64)
    spec = np.fft.fft(rows, axis=0)
    for k in freqs:
        kk = int(k) % modulus
        spec[kk] = 0
        if kk != 0 and (modulus - kk) % modulus != kk:
            spec[(modulus - kk) % modulus] = 0
    recon = np.fft.ifft(spec, axis=0).real
    w[:modulus] = torch.as_tensor(recon, dtype=w.dtype)
    return w


@torch.no_grad()
def eval_accuracy(
    model: ModularTransformer,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        logits = model(tokens)["logits"]
        correct += int((logits.argmax(-1) == labels).sum().item())
        total += int(labels.numel())
    return correct / max(total, 1)


def _freq_pairs_by_energy(
    energy_by_freq: list[float],
    *,
    modulus: int,
    skip_dc: bool = True,
) -> list[dict[str, Any]]:
    """Group conjugate freqs {f, p-f}, rank by combined energy (high → low)."""
    pairs: list[dict[str, Any]] = []
    seen: set[int] = set()
    for f in range(modulus):
        if skip_dc and f == 0:
            continue
        conj = (modulus - f) % modulus
        rep = min(f, conj) if f != conj else f
        if rep in seen:
            continue
        seen.add(rep)
        freqs = [rep] if conj == rep else sorted({rep, conj})
        energy = float(sum(energy_by_freq[k] for k in freqs))
        pairs.append({"rep": rep, "freqs": freqs, "energy": energy})
    pairs.sort(key=lambda e: e["energy"], reverse=True)
    return pairs


def fourier_ablation_on_embeddings(
    model: ModularTransformer,
    loader: DataLoader,
    *,
    modulus: int,
    device: torch.device,
    top_k: int = 2,
    max_batches: int | None = None,
    sweep_ks: list[int] | None = None,
) -> dict[str, Any]:
    """Ablate digit-embedding Fourier modes; sweep important vs unimportant counts.

    For each k in ``sweep_ks`` (default: 1..n_pairs):
      - important: remove the k highest-energy conjugate pairs
      - unimportant: remove the k lowest-energy conjugate pairs
    and record test accuracy. Also keeps a single ``top_k`` snapshot for
    backward-compatible ``delta_acc`` summary.
    """
    model = model.to(device)
    baseline = eval_accuracy(model, loader, device=device, max_batches=max_batches)

    fourier = analyze_digit_embedding_fourier(model.tok_emb, modulus=modulus)
    pairs = _freq_pairs_by_energy(
        fourier["energy_by_freq"], modulus=modulus, skip_dc=True
    )
    n_pairs = len(pairs)
    if sweep_ks is None:
        ks = list(range(1, n_pairs + 1))
    else:
        ks = sorted({int(k) for k in sweep_ks if 1 <= int(k) <= n_pairs})
        if not ks:
            ks = list(range(1, min(n_pairs, max(top_k, 1)) + 1))

    original = model.tok_emb.weight.data.clone()

    def _eval_ablate(freq_list: list[int]) -> float:
        ablated = project_out_freqs_from_digit_emb(
            original, modulus=modulus, freqs=freq_list
        ).to(device=original.device, dtype=original.dtype)
        model.tok_emb.weight.data.copy_(ablated)
        acc = eval_accuracy(model, loader, device=device, max_batches=max_batches)
        model.tok_emb.weight.data.copy_(original)
        return acc

    important_curve: list[dict[str, Any]] = []
    unimportant_curve: list[dict[str, Any]] = []
    for k in ks:
        hi = pairs[:k]
        lo = pairs[-k:] if k <= n_pairs else pairs
        hi_freqs = [f for p in hi for f in p["freqs"]]
        lo_freqs = [f for p in lo for f in p["freqs"]]
        hi_acc = _eval_ablate(hi_freqs)
        lo_acc = _eval_ablate(lo_freqs)
        important_curve.append(
            {
                "k": k,
                "freqs": hi_freqs,
                "reps": [p["rep"] for p in hi],
                "energy_sum": float(sum(p["energy"] for p in hi)),
                "acc": hi_acc,
                "delta_acc": float(hi_acc - baseline),
            }
        )
        unimportant_curve.append(
            {
                "k": k,
                "freqs": lo_freqs,
                "reps": [p["rep"] for p in lo],
                "energy_sum": float(sum(p["energy"] for p in lo)),
                "acc": lo_acc,
                "delta_acc": float(lo_acc - baseline),
            }
        )

    # Legacy single-point summary at top_k pairs (compute if missing from sweep)
    k0 = min(max(int(top_k), 1), n_pairs) if n_pairs else 0
    if k0 and important_curve:
        point = next((c for c in important_curve if c["k"] == k0), None)
        if point is None:
            hi = pairs[:k0]
            hi_freqs = [f for p in hi for f in p["freqs"]]
            hi_acc = _eval_ablate(hi_freqs)
            point = {
                "k": k0,
                "freqs": hi_freqs,
                "reps": [p["rep"] for p in hi],
                "energy_sum": float(sum(p["energy"] for p in hi)),
                "acc": hi_acc,
                "delta_acc": float(hi_acc - baseline),
            }
        ablated_acc = float(point["acc"])
        ablated_freqs = list(point["freqs"])
        delta_acc = float(point["delta_acc"])
    else:
        ablated_acc = baseline
        ablated_freqs = []
        delta_acc = 0.0

    # Extra controls at k0 (random freqs, Frobenius-matched random, magnitude-only)
    controls: dict[str, Any] = {}
    if k0 and n_pairs:
        rng = np.random.default_rng(0)
        # Random conjugate pairs (same count as top-k)
        rand_idx = rng.choice(n_pairs, size=k0, replace=False)
        rand_pairs = [pairs[int(i)] for i in rand_idx]
        rand_freqs = [f for p in rand_pairs for f in p["freqs"]]
        rand_acc = _eval_ablate(rand_freqs)
        controls["random_freq_pairs"] = {
            "k": k0,
            "freqs": rand_freqs,
            "acc": rand_acc,
            "delta_acc": float(rand_acc - baseline),
        }

        # Norm-matched: apply top ablation, measure ΔW Frobenius, then add
        # a random zero-mean perturbation of matching Frobenius norm.
        top_freqs = list(ablated_freqs)
        top_ablated = project_out_freqs_from_digit_emb(
            original, modulus=modulus, freqs=top_freqs
        )
        delta_w = (top_ablated - original.detach().cpu()).float()
        fro = float(torch.linalg.norm(delta_w[:NUM_DIGITS]).item())
        noise = torch.randn_like(delta_w[:NUM_DIGITS])
        noise = noise - noise.mean(dim=0, keepdim=True)
        nrm = float(torch.linalg.norm(noise).item()) + 1e-12
        noise = noise * (fro / nrm)
        matched = original.detach().cpu().clone()
        matched[:NUM_DIGITS] = matched[:NUM_DIGITS].float() + noise
        model.tok_emb.weight.data.copy_(
            matched.to(device=original.device, dtype=original.dtype)
        )
        matched_acc = eval_accuracy(
            model, loader, device=device, max_batches=max_batches
        )
        model.tok_emb.weight.data.copy_(original)
        controls["norm_matched_random_subspace"] = {
            "k": k0,
            "embedding_delta_frobenius_norm": fro,
            "acc": matched_acc,
            "delta_acc": float(matched_acc - baseline),
            "space": "digit_embedding",
        }

        # Magnitude control: scale digit emb to match relative Frobenius change
        # without removing Fourier directions.
        w0 = original[:NUM_DIGITS].detach().float()
        rel = fro / (float(torch.linalg.norm(w0).item()) + 1e-12)
        scaled = original.detach().cpu().clone()
        scaled[:NUM_DIGITS] = scaled[:NUM_DIGITS].float() * (1.0 - rel)
        model.tok_emb.weight.data.copy_(
            scaled.to(device=original.device, dtype=original.dtype)
        )
        mag_acc = eval_accuracy(model, loader, device=device, max_batches=max_batches)
        model.tok_emb.weight.data.copy_(original)
        controls["magnitude_scale"] = {
            "relative_parameter_change": rel,
            "acc": mag_acc,
            "delta_acc": float(mag_acc - baseline),
        }

    model.tok_emb.weight.data.copy_(original)
    return {
        "baseline_acc": baseline,
        "ablated_acc": ablated_acc,
        "delta_acc": delta_acc,
        "ablated_freqs": ablated_freqs,
        "top_k": k0,
        "fourier_source": fourier["source"],
        "top_energy_frac": fourier["top_energy_frac"],
        "freq_pairs_ranked": pairs,
        "sweep_ks": ks,
        "important_curve": important_curve,
        "unimportant_curve": unimportant_curve,
        "controls": controls,
    }


def fourier_ablation_on_unembedding(
    model: ModularTransformer,
    loader: DataLoader,
    *,
    modulus: int,
    device: torch.device,
    top_k: int = 1,
    max_batches: int | None = None,
    sweep_ks: list[int] | None = None,
) -> dict[str, Any]:
    """Ablate Fourier modes of unembedding rows ``0..p-1``; keep other classes."""
    model = model.to(device)
    baseline = eval_accuracy(model, loader, device=device, max_batches=max_batches)
    fourier = analyze_unembedding_fourier(model.head, modulus=modulus)
    pairs = _freq_pairs_by_energy(
        fourier["energy_by_freq"], modulus=modulus, skip_dc=True
    )
    n_pairs = len(pairs)
    if sweep_ks is None:
        ks = list(range(1, min(n_pairs, 6) + 1)) if n_pairs else []
    else:
        ks = sorted({int(k) for k in sweep_ks if 1 <= int(k) <= n_pairs})
        if not ks and n_pairs:
            ks = [1]

    original = model.head.weight.data.clone()

    def _eval_ablate(freq_list: list[int]) -> float:
        ablated = project_out_freqs_from_unembed(
            original, modulus=modulus, freqs=freq_list
        ).to(device=original.device, dtype=original.dtype)
        model.head.weight.data.copy_(ablated)
        acc = eval_accuracy(model, loader, device=device, max_batches=max_batches)
        model.head.weight.data.copy_(original)
        return acc

    important_curve: list[dict[str, Any]] = []
    unimportant_curve: list[dict[str, Any]] = []
    for k in ks:
        hi = pairs[:k]
        lo = pairs[-k:] if k <= n_pairs else pairs
        hi_freqs = [f for p in hi for f in p["freqs"]]
        lo_freqs = [f for p in lo for f in p["freqs"]]
        hi_acc = _eval_ablate(hi_freqs)
        lo_acc = _eval_ablate(lo_freqs)
        important_curve.append(
            {
                "k": k,
                "freqs": hi_freqs,
                "reps": [p["rep"] for p in hi],
                "energy_sum": float(sum(p["energy"] for p in hi)),
                "acc": hi_acc,
                "delta_acc": float(hi_acc - baseline),
            }
        )
        unimportant_curve.append(
            {
                "k": k,
                "freqs": lo_freqs,
                "reps": [p["rep"] for p in lo],
                "energy_sum": float(sum(p["energy"] for p in lo)),
                "acc": lo_acc,
                "delta_acc": float(lo_acc - baseline),
            }
        )

    k0 = min(max(int(top_k), 1), n_pairs) if n_pairs else 0
    if k0 and important_curve:
        point = next((c for c in important_curve if c["k"] == k0), important_curve[0])
        ablated_acc = float(point["acc"])
        ablated_freqs = list(point["freqs"])
        delta_acc = float(point["delta_acc"])
    else:
        ablated_acc = baseline
        ablated_freqs = []
        delta_acc = 0.0

    model.head.weight.data.copy_(original)
    return {
        "baseline_acc": baseline,
        "ablated_acc": ablated_acc,
        "delta_acc": delta_acc,
        "ablated_freqs": ablated_freqs,
        "top_k": k0,
        "fourier_source": fourier["source"],
        "top_energy_frac": fourier["top_energy_frac"],
        "freq_pairs_ranked": pairs,
        "sweep_ks": ks,
        "important_curve": important_curve,
        "unimportant_curve": unimportant_curve,
    }


