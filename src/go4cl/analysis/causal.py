"""Causal interventions: Fourier ablation + query-residual steering."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from go4cl.analysis.fourier import analyze_digit_embedding_fourier, analyze_unembedding_fourier
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


def _class_means(vectors: torch.Tensor, labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Return [C, D] class means; empty classes get zeros."""
    d = vectors.shape[-1]
    means = torch.zeros(n_classes, d, dtype=vectors.dtype, device=vectors.device)
    for c in range(n_classes):
        mask = labels == c
        if bool(mask.any()):
            means[c] = vectors[mask].mean(dim=0)
    return means


def estimate_class_means(
    reference_residuals: torch.Tensor,
    reference_labels: torch.Tensor,
    *,
    n_classes: int,
) -> torch.Tensor:
    """Estimate class means on a reference split (train/val). Never use test here."""
    return _class_means(
        reference_residuals, reference_labels.long(), int(n_classes)
    )


@torch.no_grad()
def evaluate_steering(
    model: ModularTransformer,
    residuals: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_means: torch.Tensor,
    modulus: int,
    delta: int = 1,
    alpha: float = 1.0,
    site: str = "final_query_resid",
    layer_idx: int | None = None,
    shuffle_means_seed: int | None = 0,
) -> dict[str, Any]:
    """Apply precomputed class means on an evaluation split (typically test).

    ``residuals`` for final-site steering are query vectors [B, D]; for layer
    steering pass resid_post [B, T, D] and set ``layer_idx``.
    """
    device = residuals.device
    model = model.to(device)
    model.eval()
    y = labels.to(device).long()
    means = class_means.to(device)
    targets = (y + int(delta)) % modulus

    if layer_idx is None:
        h = residuals.to(device)
        if h.dim() == 3:
            h = h[:, -1, :]
        base_logits = model.head(h)
        direction = means[targets] - means[y]
        steer_logits = model.head(h + float(alpha) * direction)

        def _apply_shuf(means_tbl: torch.Tensor) -> torch.Tensor:
            d = means_tbl[targets] - means_tbl[y]
            return model.head(h + float(alpha) * d)
    else:
        resid = residuals.to(device)
        h = resid[:, -1, :]
        base = model.continue_from_layer(resid, layer_idx=layer_idx)
        base_logits = base["logits"]

        def _apply(means_tbl: torch.Tensor) -> torch.Tensor:
            direction = means_tbl[targets] - means_tbl[y]
            edited = resid.clone()
            edited[:, -1, :] = h + float(alpha) * direction
            return model.continue_from_layer(edited, layer_idx=layer_idx)["logits"]

        steer_logits = _apply(means)
        _apply_shuf = _apply

    base_pred = base_logits.argmax(dim=-1)
    base_acc = float((base_pred == y).float().mean().item())
    steer_pred = steer_logits.argmax(dim=-1)
    steer_to_target = float((steer_pred == targets).float().mean().item())
    steer_keep_true = float((steer_pred == y).float().mean().item())
    gather_t = steer_logits.gather(1, targets.view(-1, 1)).squeeze(1)
    base_t = base_logits.gather(1, targets.view(-1, 1)).squeeze(1)

    result: dict[str, Any] = {
        "site": site if layer_idx is None else f"resid_post_L{layer_idx}",
        "delta": int(delta),
        "alpha": float(alpha),
        "n": int(y.shape[0]),
        "baseline_acc_true": base_acc,
        "steered_acc_target": steer_to_target,
        "steered_acc_true": steer_keep_true,
        "mean_logit_target_gain": float((gather_t - base_t).mean().item()),
        "chance": 1.0 / max(modulus, 1),
        "direction_source": "external_class_means",
    }
    if shuffle_means_seed is not None:
        g = torch.Generator(device="cpu")
        g.manual_seed(int(shuffle_means_seed))
        perm = torch.randperm(modulus, generator=g).to(device)
        pred_s = _apply_shuf(means[perm]).argmax(dim=-1)
        result["shuffled_steered_acc_target"] = float(
            (pred_s == targets).float().mean().item()
        )
        result["steered_minus_shuffled"] = float(
            steer_to_target - result["shuffled_steered_acc_target"]
        )
    return result


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
    class_means: torch.Tensor | None = None,
    reference_residuals: torch.Tensor | None = None,
    reference_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Directionally edit *final* query residual (pre-head) toward another sum.

    Prefer ``class_means`` or train/val ``reference_*`` so directions are not
    estimated on the evaluation set. Legacy (means from ``query_resid``) is
    marked ``direction_source=eval_set_legacy``.
    """
    if class_means is not None:
        means = class_means
        source = "external_class_means"
    elif reference_residuals is not None and reference_labels is not None:
        href = reference_residuals
        if href.dim() == 3:
            href = href[:, -1, :]
        means = estimate_class_means(href, reference_labels, n_classes=modulus)
        source = "reference_split"
    else:
        means = _class_means(query_resid, sum_labels.long(), modulus)
        source = "eval_set_legacy"

    result = evaluate_steering(
        model,
        query_resid,
        sum_labels,
        class_means=means,
        modulus=modulus,
        delta=delta,
        alpha=alpha,
        site="final_query_resid",
        layer_idx=None,
        shuffle_means_seed=shuffle_means_seed,
    )
    result["direction_source"] = source
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
    class_means: torch.Tensor | None = None,
    reference_residuals: torch.Tensor | None = None,
    reference_labels: torch.Tensor | None = None,
) -> dict[str, Any]:
    """Steer query residual *after* ``layer_idx``, then continue later layers + head.

    ``resid_post``: [B, T, D] cached residual after that layer's MLP.
    Prefer reference/train means via ``class_means`` or ``reference_*``.
    """
    if class_means is not None:
        means = class_means
        source = "external_class_means"
    elif reference_residuals is not None and reference_labels is not None:
        href = reference_residuals
        if href.dim() == 3:
            href = href[:, -1, :]
        means = estimate_class_means(href, reference_labels, n_classes=modulus)
        source = "reference_split"
    else:
        means = _class_means(resid_post[:, -1, :], sum_labels.long(), modulus)
        source = "eval_set_legacy"

    result = evaluate_steering(
        model,
        resid_post,
        sum_labels,
        class_means=means,
        modulus=modulus,
        delta=delta,
        alpha=alpha,
        site=f"resid_post_L{layer_idx}",
        layer_idx=int(layer_idx),
        shuffle_means_seed=shuffle_means_seed,
    )
    result["direction_source"] = source
    result["layer_idx"] = int(layer_idx)
    return result
