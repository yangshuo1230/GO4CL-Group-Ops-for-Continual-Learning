"""Batch activation collection helpers for mech analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch.utils.data import DataLoader

from go4cl.model.transformer import ModularTransformer


@dataclass
class BatchCache:
    tokens: torch.Tensor
    labels: torch.Tensor
    logits: torch.Tensor
    query_resid: torch.Tensor
    attn: list[torch.Tensor]  # L * [B, H, T, T]
    resid_pre: list[torch.Tensor]  # L * [B, T, D]
    resid_mid: list[torch.Tensor]  # L * [B, T, D]
    resid_post: list[torch.Tensor]  # L * [B, T, D]
    tok_emb: torch.Tensor


def collect_batches(
    model: ModularTransformer,
    loader: DataLoader,
    *,
    device: torch.device,
    max_batches: int | None = None,
) -> BatchCache:
    """Run ``forward_with_cache`` over a loader and concatenate on CPU."""
    model.eval()
    toks: list[torch.Tensor] = []
    labs: list[torch.Tensor] = []
    logits: list[torch.Tensor] = []
    qres: list[torch.Tensor] = []
    attns: list[list[torch.Tensor]] | None = None
    resid_pres: list[list[torch.Tensor]] | None = None
    resid_mids: list[list[torch.Tensor]] | None = None
    resid_posts: list[list[torch.Tensor]] | None = None
    tok_embs: list[torch.Tensor] = []

    for bi, batch in enumerate(loader):
        if max_batches is not None and bi >= max_batches:
            break
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        cache = model.forward_with_cache(tokens, labels)
        toks.append(tokens.cpu())
        labs.append(labels.cpu())
        logits.append(cache["logits"].cpu())
        qres.append(cache["query_resid"].cpu())
        tok_embs.append(cache["tok_emb"].cpu())
        if attns is None:
            attns = [[] for _ in cache["attn"]]
            resid_pres = [[] for _ in cache["resid_pre"]]
            resid_mids = [[] for _ in cache["resid_mid"]]
            resid_posts = [[] for _ in cache["resid_post"]]
        assert (
            attns is not None
            and resid_pres is not None
            and resid_mids is not None
            and resid_posts is not None
        )
        for li, a in enumerate(cache["attn"]):
            attns[li].append(a.cpu())
        for li, r in enumerate(cache["resid_pre"]):
            resid_pres[li].append(r.cpu())
        for li, r in enumerate(cache["resid_mid"]):
            resid_mids[li].append(r.cpu())
        for li, r in enumerate(cache["resid_post"]):
            resid_posts[li].append(r.cpu())

    if not toks:
        raise RuntimeError("empty loader / max_batches=0")

    return BatchCache(
        tokens=torch.cat(toks, dim=0),
        labels=torch.cat(labs, dim=0),
        logits=torch.cat(logits, dim=0),
        query_resid=torch.cat(qres, dim=0),
        attn=[torch.cat(xs, dim=0) for xs in (attns or [])],
        resid_pre=[torch.cat(xs, dim=0) for xs in (resid_pres or [])],
        resid_mid=[torch.cat(xs, dim=0) for xs in (resid_mids or [])],
        resid_post=[torch.cat(xs, dim=0) for xs in (resid_posts or [])],
        tok_emb=torch.cat(tok_embs, dim=0),
    )


def operand_residues(
    tokens: torch.Tensor,
    *,
    i: int,
    j: int,
    modulus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return (xi%p, xj%p, (xi+xj)%p) from digit positions."""
    xi = tokens[:, i] % modulus
    xj = tokens[:, j] % modulus
    return xi, xj, (xi + xj) % modulus


def accuracy_from_logits(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == labels).float().mean().item())


def to_jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # noqa: BLE001
            return float(obj)
    return obj
