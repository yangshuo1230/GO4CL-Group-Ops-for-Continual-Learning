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
