"""Head / component knockouts on cached residuals.

Does not import phases. These are generic causal interventions used by
composition locus analysis and by the causal-detail pipeline.
"""

from __future__ import annotations

from typing import Any

import torch

from go4cl.analysis.cache import accuracy_from_logits
from go4cl.model.transformer import ModularTransformer


@torch.no_grad()
def continue_after_edited_post(
    model: ModularTransformer,
    resid_post: torch.Tensor,
    *,
    layer_idx: int,
) -> torch.Tensor:
    """Continue from post-MLP residual of ``layer_idx`` → logits."""
    return model.continue_from_layer(resid_post, layer_idx=layer_idx)["logits"]


# Historical name kept for in-package callers during the refactor.
_continue_after_edited_post = continue_after_edited_post


@torch.no_grad()
def component_knockout(
    model: ModularTransformer,
    *,
    resid_pre: list[torch.Tensor],
    resid_mid: list[torch.Tensor],
    resid_post: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    baseline_acc: float | None = None,
) -> dict[str, Any]:
    """Per-layer zero-attn / zero-MLP write ablations on cached residuals.

    - ``zero_mlp``: skip MLP write (treat mid as post), continue later layers
    - ``zero_attn``: skip attn write (mid' = pre), then apply this layer's MLP,
      continue later layers
    """
    model = model.to(device)
    model.eval()
    n_layers = len(model.blocks)
    labels = labels.to(device)

    last = n_layers - 1
    if baseline_acc is None:
        baseline_logits = _continue_after_edited_post(
            model, resid_post[last].to(device), layer_idx=last
        )
        base = accuracy_from_logits(baseline_logits, labels)
    else:
        base = float(baseline_acc)

    by_layer: list[dict[str, Any]] = []
    worst_mlp_layer = None
    worst_mlp_delta = None

    for li in range(n_layers):
        block = model.blocks[li]
        pre = resid_pre[li].to(device)
        mid = resid_mid[li].to(device)

        # zero_mlp: post := mid (skip MLP write)
        logits_zmlp = _continue_after_edited_post(model, mid, layer_idx=li)
        acc_zmlp = accuracy_from_logits(logits_zmlp, labels)
        d_mlp = float(acc_zmlp - base)

        # zero_attn: mid' := pre, then apply this layer's MLP
        post_no_attn = pre + block.mlp(block.ln2(pre))
        logits_zattn = _continue_after_edited_post(
            model, post_no_attn, layer_idx=li
        )
        acc_zattn = accuracy_from_logits(logits_zattn, labels)
        d_attn = float(acc_zattn - base)

        row = {
            "layer": li,
            "zero_mlp_acc": float(acc_zmlp),
            "zero_mlp_delta": d_mlp,
            "zero_attn_acc": float(acc_zattn),
            "zero_attn_delta": d_attn,
        }
        by_layer.append(row)
        if worst_mlp_delta is None or d_mlp < worst_mlp_delta:
            worst_mlp_delta = d_mlp
            worst_mlp_layer = li

    return {
        "baseline_acc": float(base),
        "by_layer": by_layer,
        "compose_layer_guess": worst_mlp_layer,
        "compose_layer_zero_mlp_delta": worst_mlp_delta,
    }


@torch.no_grad()
def head_knockout(
    model: ModularTransformer,
    *,
    resid_pre: list[torch.Tensor],
    resid_post: list[torch.Tensor],
    labels: torch.Tensor,
    device: torch.device,
    baseline_acc: float | None = None,
    layers: list[int] | None = None,
) -> dict[str, Any]:
    """Per-head mean-ablation (zero head output) within each attention layer.

    For layer ℓ and head h: recompute attn with head h zeroed, apply that
    layer's MLP, then ``continue_from_layer``. Reports Δacc vs baseline.
    """
    model = model.to(device)
    model.eval()
    n_layers = len(model.blocks)
    n_heads = model.cfg.n_heads
    labels = labels.to(device)
    layers = list(range(n_layers) if layers is None else layers)

    last = n_layers - 1
    if baseline_acc is None:
        baseline_logits = _continue_after_edited_post(
            model, resid_post[last].to(device), layer_idx=last
        )
        base = accuracy_from_logits(baseline_logits, labels)
    else:
        base = float(baseline_acc)

    by_head: list[dict[str, Any]] = []
    worst = None  # (delta, layer, head)

    for li in layers:
        if not (0 <= li < n_layers):
            continue
        block = model.blocks[li]
        pre = resid_pre[li].to(device)
        x_ln = block.ln1(pre)
        for h in range(n_heads):
            attn_out, _, _ = block.attn.forward_detailed(
                x_ln, ablate_heads=[h]
            )
            mid = pre + attn_out
            post = mid + block.mlp(block.ln2(mid))
            logits = _continue_after_edited_post(model, post, layer_idx=li)
            acc = accuracy_from_logits(logits, labels)
            delta = float(acc - base)
            row = {
                "layer": li,
                "head": h,
                "acc": float(acc),
                "delta_acc": delta,
            }
            by_head.append(row)
            if worst is None or delta < worst[0]:
                worst = (delta, li, h)

    # Also: ablate all heads at once per layer (= zero full attn) for reference
    by_layer_all: list[dict[str, Any]] = []
    for li in layers:
        if not (0 <= li < n_layers):
            continue
        block = model.blocks[li]
        pre = resid_pre[li].to(device)
        attn_out, _, _ = block.attn.forward_detailed(
            block.ln1(pre), ablate_heads=list(range(n_heads))
        )
        mid = pre + attn_out
        post = mid + block.mlp(block.ln2(mid))
        logits = _continue_after_edited_post(model, post, layer_idx=li)
        acc = accuracy_from_logits(logits, labels)
        by_layer_all.append(
            {
                "layer": li,
                "acc": float(acc),
                "delta_acc": float(acc - base),
            }
        )

    return {
        "baseline_acc": float(base),
        "n_heads": n_heads,
        "by_head": by_head,
        "ablate_all_heads_by_layer": by_layer_all,
        "worst_head": (
            None
            if worst is None
            else {"layer": worst[1], "head": worst[2], "delta_acc": worst[0]}
        ),
    }


