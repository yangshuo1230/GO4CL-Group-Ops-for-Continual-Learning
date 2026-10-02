"""Causal interventions: Fourier ablation + query-residual steering."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from go4cl.analysis.fourier import analyze_digit_embedding_fourier
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


def _class_means(vectors: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Return [C, D] class means; empty classes get zeros."""
    d = vectors.shape[-1]
    means = torch.zeros(n_classes, d, dtype=vectors.dtype, device=vectors.device)
    for c in range(n_classes):
        mask = labels == c
        if bool(mask.any()):
            means[c] = vectors[mask].mean(dim=0)
    return means


@torch.no_grad()
def steer_query_resid_sum(
    model: ModularTransformer,
    query_resid: torch.Tensor,
    sum_labels: torch.Tensor,
    *,
    modulus: int,
    delta: int = 1,
    alpha: float = 1.0,
    shuffle_means_seed: int | None = 0,
) -> dict[str, Any]:
    """Directionally edit *final* query residual (pre-head) toward another sum.

    Prefer ``steer_at_layer`` for mid-layer interventions; final-layer steering
    is nearly tautological once the unembed already decodes the sum.
    """
    device = query_resid.device
    model = model.to(device)
    model.eval()
    h = query_resid.to(device)
    y = sum_labels.to(device).long()
    means = _class_means(h, y, modulus)
    targets = (y + int(delta)) % modulus

    base_logits = model.head(h)
    base_pred = base_logits.argmax(dim=-1)
    base_acc = float((base_pred == y).float().mean().item())

    direction = means[targets] - means[y]
    h_steer = h + float(alpha) * direction
    steer_logits = model.head(h_steer)
    steer_pred = steer_logits.argmax(dim=-1)
    steer_to_target = float((steer_pred == targets).float().mean().item())
    steer_keep_true = float((steer_pred == y).float().mean().item())
    gather_t = steer_logits.gather(1, targets.view(-1, 1)).squeeze(1)
    gather_y = steer_logits.gather(1, y.view(-1, 1)).squeeze(1)
    base_t = base_logits.gather(1, targets.view(-1, 1)).squeeze(1)
    base_y = base_logits.gather(1, y.view(-1, 1)).squeeze(1)

    result: dict[str, Any] = {
        "site": "final_query_resid",
        "delta": int(delta),
        "alpha": float(alpha),
        "n": int(h.shape[0]),
        "baseline_acc_true": base_acc,
        "steered_acc_target": steer_to_target,
        "steered_acc_true": steer_keep_true,
        "mean_logit_target_gain": float((gather_t - base_t).mean().item()),
        "mean_logit_true_change": float((gather_y - base_y).mean().item()),
        "chance": 1.0 / max(modulus, 1),
    }

    if shuffle_means_seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(shuffle_means_seed))
        perm = torch.randperm(modulus, generator=g).to(device)
        means_shuf = means[perm]
        direction_s = means_shuf[targets] - means_shuf[y]
        h_s = h + float(alpha) * direction_s
        pred_s = model.head(h_s).argmax(dim=-1)
        result["shuffled_steered_acc_target"] = float(
            (pred_s == targets).float().mean().item()
        )
        result["steered_minus_shuffled"] = float(
            steer_to_target - result["shuffled_steered_acc_target"]
        )
    return result


@torch.no_grad()
def steer_at_layer(
    model: ModularTransformer,
    resid_post: torch.Tensor,
    sum_labels: torch.Tensor,
    *,
    layer_idx: int,
    modulus: int,
    delta: int = 1,
    alpha: float = 1.0,
    shuffle_means_seed: int | None = 0,
) -> dict[str, Any]:
    """Steer query residual *after* ``layer_idx``, then continue later layers + head.

    ``resid_post``: [B, T, D] cached residual after that layer's MLP.
    Intervention only edits the query position (index -1).
    """
    device = resid_post.device
    model = model.to(device)
    model.eval()
    resid = resid_post.to(device)
    y = sum_labels.to(device).long()
    h = resid[:, -1, :]
    means = _class_means(h, y, modulus)
    targets = (y + int(delta)) % modulus

    base = model.continue_from_layer(resid, layer_idx=layer_idx)
    base_pred = base["logits"].argmax(dim=-1)
    base_acc = float((base_pred == y).float().mean().item())

    def _apply(means_tbl: torch.Tensor) -> torch.Tensor:
        direction = means_tbl[targets] - means_tbl[y]
        edited = resid.clone()
        edited[:, -1, :] = h + float(alpha) * direction
        return model.continue_from_layer(edited, layer_idx=layer_idx)["logits"]

    steer_logits = _apply(means)
    steer_pred = steer_logits.argmax(dim=-1)
    steer_to_target = float((steer_pred == targets).float().mean().item())
    steer_keep_true = float((steer_pred == y).float().mean().item())
    gather_t = steer_logits.gather(1, targets.view(-1, 1)).squeeze(1)
    base_t = base["logits"].gather(1, targets.view(-1, 1)).squeeze(1)

    result: dict[str, Any] = {
        "site": f"resid_post_L{layer_idx}",
        "layer_idx": int(layer_idx),
        "delta": int(delta),
        "alpha": float(alpha),
        "n": int(resid.shape[0]),
        "baseline_acc_true": base_acc,
        "steered_acc_target": steer_to_target,
        "steered_acc_true": steer_keep_true,
        "mean_logit_target_gain": float((gather_t - base_t).mean().item()),
        "chance": 1.0 / max(modulus, 1),
    }

    if shuffle_means_seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(shuffle_means_seed))
        perm = torch.randperm(modulus, generator=g).to(device)
        pred_s = _apply(means[perm]).argmax(dim=-1)
        result["shuffled_steered_acc_target"] = float(
            (pred_s == targets).float().mean().item()
        )
        result["steered_minus_shuffled"] = float(
            steer_to_target - result["shuffled_steered_acc_target"]
        )
    return result
