"""Activation patching for multi-op circuit transplant.

Two families of interventions on cached residuals:

1. Operand residual patch — copy activations at key positions from a donor
   example into a receiver, re-run the patched layer, continue.
2. Query-path patch — on a same-digit query-twin, replace only the query
   position's residual / attn write / MLP write from a source forward.
"""

from __future__ import annotations

from typing import Literal

import torch

from go4cl.analysis.cache import accuracy_from_logits
from go4cl.model.transformer import ModularTransformer

QueryPatchKind = Literal[
    "resid_pre_query",
    "attn_write_query",
    "mlp_write_query",
    "resid_post_query",
]


@torch.no_grad()
def continue_from_pre(
    model: ModularTransformer,
    resid_pre: torch.Tensor,
    *,
    layer_idx: int,
) -> torch.Tensor:
    """Run block ``layer_idx`` from ``resid_pre`` then later layers → logits."""
    block = model.blocks[layer_idx]
    post = block(resid_pre)
    return model.continue_from_layer(post, layer_idx=layer_idx)["logits"]


@torch.no_grad()
def patch_token_positions(
    dest: torch.Tensor,
    source: torch.Tensor,
    positions: list[int],
) -> torch.Tensor:
    """Copy ``source[:, pos]`` into a clone of ``dest`` for each position."""
    out = dest.clone()
    for p in positions:
        out[:, p, :] = source[:, p, :]
    return out


@torch.no_grad()
def operand_patch_from_pre(
    model: ModularTransformer,
    resid_pre: torch.Tensor,
    donor_pre: torch.Tensor,
    *,
    layer_idx: int,
    positions: list[int],
) -> torch.Tensor:
    """Patch ``positions`` on ``resid_pre`` from ``donor_pre``, then continue."""
    edited = patch_token_positions(resid_pre, donor_pre, positions)
    return continue_from_pre(model, edited, layer_idx=layer_idx)


@torch.no_grad()
def query_path_patch_logits(
    model: ModularTransformer,
    *,
    layer_idx: int,
    kind: QueryPatchKind,
    recv_pre: torch.Tensor,
    recv_mid: torch.Tensor,
    recv_post: torch.Tensor,
    donor_pre: torch.Tensor,
    donor_mid: torch.Tensor,
    donor_post: torch.Tensor,
) -> torch.Tensor:
    """Path-patch the query position (index -1) of one component, then continue.

    Receiver stream is the forward we keep (typically the query-twin).
    Donor stream is the source operation on the same digits.
    """
    q = -1
    block = model.blocks[layer_idx]
    if kind == "resid_post_query":
        edited = recv_post.clone()
        edited[:, q, :] = donor_post[:, q, :]
        return model.continue_from_layer(edited, layer_idx=layer_idx)["logits"]
    if kind == "mlp_write_query":
        donor_mlp = donor_post[:, q, :] - donor_mid[:, q, :]
        edited = recv_post.clone()
        edited[:, q, :] = recv_mid[:, q, :] + donor_mlp
        return model.continue_from_layer(edited, layer_idx=layer_idx)["logits"]
    if kind == "attn_write_query":
        donor_attn = donor_mid[:, q, :] - donor_pre[:, q, :]
        mid = recv_mid.clone()
        mid[:, q, :] = recv_pre[:, q, :] + donor_attn
        post = mid + block.mlp(block.ln2(mid))
        return model.continue_from_layer(post, layer_idx=layer_idx)["logits"]
    if kind == "resid_pre_query":
        pre = recv_pre.clone()
        pre[:, q, :] = donor_pre[:, q, :]
        return continue_from_pre(model, pre, layer_idx=layer_idx)
    raise ValueError(f"unknown query patch kind: {kind}")


def split_acc(
    logits: torch.Tensor,
    labels_a: torch.Tensor,
    labels_b: torch.Tensor,
) -> dict[str, float]:
    """Accuracy vs two labelings; also restricted to rows where they differ."""
    pred = logits.argmax(-1)
    n = int(pred.shape[0])
    diff = labels_a != labels_b
    n_diff = int(diff.sum().item())
    acc_a = float((pred == labels_a).float().mean().item()) if n else 0.0
    acc_b = float((pred == labels_b).float().mean().item()) if n else 0.0
    if n_diff:
        acc_a_diff = float((pred[diff] == labels_a[diff]).float().mean().item())
        acc_b_diff = float((pred[diff] == labels_b[diff]).float().mean().item())
    else:
        acc_a_diff = acc_a
        acc_b_diff = acc_b
    return {
        "n": float(n),
        "n_label_diff": float(n_diff),
        "acc_a": acc_a,
        "acc_b": acc_b,
        "acc_a_when_diff": acc_a_diff,
        "acc_b_when_diff": acc_b_diff,
        "baseline_acc": float(accuracy_from_logits(logits, labels_a)),
    }
