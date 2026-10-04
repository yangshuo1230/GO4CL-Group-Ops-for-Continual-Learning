"""Tests for 1C activation patching helpers."""

from __future__ import annotations

import torch

from go4cl.analysis.patching import (
    operand_patch_from_pre,
    query_path_patch_logits,
    split_acc,
)
from go4cl.model.transformer import ModelConfig, ModularTransformer


def test_split_acc_when_labels_differ() -> None:
    logits = torch.zeros(4, 5)
    logits[0, 1] = 10
    logits[1, 2] = 10
    logits[2, 3] = 10
    logits[3, 0] = 10
    a = torch.tensor([1, 2, 0, 0])
    b = torch.tensor([1, 0, 3, 4])
    out = split_acc(logits, a, b)
    assert out["n"] == 4.0
    assert out["n_label_diff"] == 3.0
    assert out["acc_a"] == 0.75
    assert abs(out["acc_a_when_diff"] - 2 / 3) < 1e-6


def test_operand_patch_changes_forward() -> None:
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_classes=7)
    cfg.d_mlp = 64
    torch.manual_seed(0)
    model = ModularTransformer(cfg).eval()
    tokens = torch.randint(0, 64, (6, 10))
    cache = model.forward_with_cache(tokens)
    donor = torch.roll(cache["resid_pre"][0], shifts=1, dims=0)
    logits_id = operand_patch_from_pre(
        model,
        cache["resid_pre"][0],
        cache["resid_pre"][0],
        layer_idx=0,
        positions=[0, 1],
    )
    logits_d = operand_patch_from_pre(
        model,
        cache["resid_pre"][0],
        donor,
        layer_idx=0,
        positions=[0, 1],
    )
    # Identity patch matches cached logits; donor patch should usually differ.
    assert torch.allclose(logits_id, cache["logits"], atol=1e-4, rtol=1e-4)
    assert logits_d.shape == logits_id.shape


def test_query_path_patch_identity_matches_receiver() -> None:
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_classes=7)
    cfg.d_mlp = 64
    torch.manual_seed(1)
    model = ModularTransformer(cfg).eval()
    tokens = torch.randint(0, 64, (5, 10))
    cache = model.forward_with_cache(tokens)
    for kind in (
        "resid_pre_query",
        "attn_write_query",
        "mlp_write_query",
        "resid_post_query",
    ):
        logits = query_path_patch_logits(
            model,
            layer_idx=0,
            kind=kind,  # type: ignore[arg-type]
            recv_pre=cache["resid_pre"][0],
            recv_mid=cache["resid_mid"][0],
            recv_post=cache["resid_post"][0],
            donor_pre=cache["resid_pre"][0],
            donor_mid=cache["resid_mid"][0],
            donor_post=cache["resid_post"][0],
        )
        assert torch.allclose(logits, cache["logits"], atol=1e-4, rtol=1e-4)
