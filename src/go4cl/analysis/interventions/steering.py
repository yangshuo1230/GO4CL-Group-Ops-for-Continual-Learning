"""Residual steering: add a class-mean direction at a residual site.

Class means must be estimated on a reference split (train/val), never test.
"""

from __future__ import annotations

from typing import Any

import torch

from go4cl.model.transformer import ModularTransformer

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


# ---------------------------------------------------------------------------
# Per-sample / multi-condition steering (paper figures). Does not change
# evaluate_steering / steer_at_layer return keys.
# ---------------------------------------------------------------------------

SITE_LAYER = {"L0_post": 0, "L1_post": 1, "L2_post": 2}
LAYER_SITE = {v: k for k, v in SITE_LAYER.items()}
STEERING_CONDITIONS = ("no_steering", "structured", "shuffled", "global_structured")
DEFAULT_STEER_DELTAS = (1, 2, 4, 8)
DEFAULT_STEER_ALPHA = 1.0
DEFAULT_SHUFFLE_SEED = 0


def target_label(labels: torch.Tensor, *, delta: int, modulus: int) -> torch.Tensor:
    """``(s + delta) mod p`` with the same dtype as ``labels``."""
    return (labels.long() + int(delta)) % int(modulus)


def shuffled_mean_permutation(n_classes: int, *, seed: int) -> torch.Tensor:
    """Fixed permutation of residue indices for the shuffled-mean control."""
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    return torch.randperm(int(n_classes), generator=g)


def global_shift_vector(
    class_means: torch.Tensor,
    *,
    delta: int,
    modulus: int,
) -> torch.Tensor:
    """``v_δ = mean_s( μ[(s+δ) mod p] − μ[s] )``."""
    p = int(modulus)
    idx = torch.arange(p, device=class_means.device)
    return (class_means[(idx + int(delta)) % p] - class_means[idx]).mean(dim=0)


@torch.no_grad()
def logits_after_query_add(
    model: ModularTransformer,
    resid_post: torch.Tensor,
    *,
    layer_idx: int,
    add_at_query: torch.Tensor,
) -> torch.Tensor:
    """Clone ``resid_post``, add a vector at the query token, continue forward."""
    edited = resid_post.clone()
    edited[:, -1, :] = resid_post[:, -1, :] + add_at_query
    return model.continue_from_layer(edited, layer_idx=int(layer_idx))["logits"]


@torch.no_grad()
def predict_steering_condition(
    model: ModularTransformer,
    resid_post: torch.Tensor,
    labels: torch.Tensor,
    *,
    class_means: torch.Tensor,
    layer_idx: int,
    modulus: int,
    delta: int,
    alpha: float,
    condition: str,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
) -> dict[str, torch.Tensor]:
    """Return original / target / predicted labels for one steering condition.

    ``class_means`` must come from a reference split, never the eval split.
    Only the query position is edited.
    """
    y = labels.to(resid_post.device).long()
    means = class_means.to(resid_post.device)
    targets = target_label(y, delta=delta, modulus=modulus)
    h = resid_post[:, -1, :]
    a = float(alpha)
    cond = str(condition)
    if cond == "no_steering":
        add = torch.zeros_like(h)
    elif cond == "structured":
        add = a * (means[targets] - means[y])
    elif cond == "shuffled":
        perm = shuffled_mean_permutation(modulus, seed=shuffle_seed).to(means.device)
        means_s = means[perm]
        add = a * (means_s[targets] - means_s[y])
    elif cond == "global_structured":
        add = a * global_shift_vector(means, delta=delta, modulus=modulus).unsqueeze(0)
        add = add.expand_as(h)
    else:
        raise ValueError(f"unknown steering condition: {cond}")
    logits = logits_after_query_add(
        model, resid_post, layer_idx=layer_idx, add_at_query=add
    )
    pred = logits.argmax(dim=-1)
    return {
        "original": y.detach().cpu(),
        "target": targets.detach().cpu(),
        "predicted": pred.detach().cpu(),
    }


def transition_counts(
    source: torch.Tensor | list[int],
    predicted: torch.Tensor | list[int],
    *,
    modulus: int,
) -> torch.Tensor:
    """Integer counts ``C[source, predicted]`` of shape ``[p, p]``."""
    p = int(modulus)
    src = torch.as_tensor(source, dtype=torch.long).view(-1)
    prd = torch.as_tensor(predicted, dtype=torch.long).view(-1)
    if src.numel() != prd.numel():
        raise ValueError("source and predicted must have the same length")
    counts = torch.zeros(p, p, dtype=torch.long)
    valid = (
        (src >= 0) & (src < p) & (prd >= 0) & (prd < p)
    )
    if bool(valid.any()):
        idx = src[valid] * p + prd[valid]
        counts.view(-1).index_add_(0, idx, torch.ones_like(idx, dtype=torch.long))
    return counts


def transition_matrix(
    source: torch.Tensor | list[int],
    predicted: torch.Tensor | list[int],
    *,
    modulus: int,
) -> torch.Tensor:
    """Row-normalized ``P(predicted | source)``; empty source rows stay 0."""
    counts = transition_counts(source, predicted, modulus=modulus).to(torch.float64)
    denom = counts.sum(dim=1, keepdim=True).clamp_min(1.0)
    return counts / denom


def sample_records(
    *,
    site: str,
    delta: int,
    alpha: float,
    condition: str,
    original: torch.Tensor,
    target: torch.Tensor,
    predicted: torch.Tensor,
) -> list[dict[str, Any]]:
    orig = original.long().view(-1)
    tgt = target.long().view(-1)
    pred = predicted.long().view(-1)
    rows: list[dict[str, Any]] = []
    for i in range(int(orig.numel())):
        s = int(orig[i])
        t = int(tgt[i])
        yhat = int(pred[i])
        rows.append(
            {
                "site": site,
                "delta": int(delta),
                "alpha": float(alpha),
                "condition": condition,
                "original_label": s,
                "target_label": t,
                "predicted_label": yhat,
                "original_correct": int(yhat == s),
                "target_correct": int(yhat == t),
            }
        )
    return rows


def summarize_records(
    records: list[dict[str, Any]],
    *,
    modulus: int,
    checkpoint: str,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (
            row["site"],
            int(row["delta"]),
            float(row["alpha"]),
            row["condition"],
        )
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    for site, delta, alpha, cond in sorted(groups, key=lambda k: (k[0], k[1], k[3])):
        chunk = groups[(site, delta, alpha, cond)]
        n = len(chunk)
        orig_acc = sum(int(r["original_correct"]) for r in chunk) / max(n, 1)
        tgt_acc = sum(int(r["target_correct"]) for r in chunk) / max(n, 1)
        out.append(
            {
                "modulus": int(modulus),
                "checkpoint": checkpoint,
                "site": site,
                "delta": int(delta),
                "alpha": float(alpha),
                "condition": cond,
                "n": n,
                "original_acc": orig_acc,
                "target_acc": tgt_acc,
            }
        )
    return out


def transition_rows(
    records: list[dict[str, Any]],
    *,
    modulus: int,
) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in records:
        key = (row["site"], int(row["delta"]), row["condition"])
        groups.setdefault(key, []).append(row)
    out: list[dict[str, Any]] = []
    p = int(modulus)
    for site, delta, cond in sorted(groups):
        chunk = groups[(site, delta, cond)]
        src = [int(r["original_label"]) for r in chunk]
        prd = [int(r["predicted_label"]) for r in chunk]
        counts = transition_counts(src, prd, modulus=p)
        row_sum = counts.sum(dim=1)
        for s in range(p):
            tot = int(row_sum[s])
            for k in range(p):
                c = int(counts[s, k])
                if tot == 0 and c == 0:
                    continue
                out.append(
                    {
                        "site": site,
                        "delta": int(delta),
                        "condition": cond,
                        "source_residue": s,
                        "predicted_residue": k,
                        "count": c,
                        "probability": (c / tot) if tot else 0.0,
                    }
                )
    return out

