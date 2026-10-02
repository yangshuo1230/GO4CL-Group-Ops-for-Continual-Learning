"""Tests for 1A-mech analysis helpers."""

from __future__ import annotations

from pathlib import Path

import torch

from go4cl.analysis.discover import discover_targets
from go4cl.analysis.fourier import fourier_energy, residue_mean_vectors
from go4cl.model.transformer import ModelConfig, ModularTransformer


def test_fourier_energy_parseval_ish() -> None:
    # Constant residue means → energy mostly in DC
    means = torch.ones(7, 4)
    out = fourier_energy(means)
    assert out["modulus"] == 7
    assert out["top_freq"] == 0
    assert out["total_energy"] > 0
    # Single frequency sinusoid along residue of dim 0
    p = 8
    r = torch.arange(p).float()
    vec = torch.stack(
        [torch.cos(2 * torch.pi * 2 * r / p), torch.zeros(p), torch.zeros(p)],
        dim=-1,
    )
    out2 = fourier_energy(vec)
    assert out2["top_freq"] in {2, p - 2}


def test_residue_mean_vectors() -> None:
    vectors = torch.tensor([[1.0, 0.0], [3.0, 0.0], [5.0, 0.0]])
    residues = torch.tensor([0, 0, 1])
    means = residue_mean_vectors(vectors, residues, modulus=3)
    assert means.shape == (3, 2)
    assert torch.allclose(means[0], torch.tensor([2.0, 0.0]))
    assert torch.allclose(means[1], torch.tensor([5.0, 0.0]))


def test_forward_with_cache_shapes() -> None:
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4)
    cfg.d_mlp = 64
    model = ModularTransformer(cfg)
    tokens = torch.randint(0, 64, (3, 10))
    cache = model.forward_with_cache(tokens)
    assert cache["query_resid"].shape == (3, 32)
    assert cache["logits"].shape[0] == 3
    assert len(cache["attn"]) == 2
    assert cache["attn"][0].shape[:3] == (3, 4, 10)


def test_discover_p31_stamp() -> None:
    root = Path("runs/phase1/scan_moduli/20260930_223737")
    if not root.is_dir():
        return  # skip if artifacts not present
    targets = discover_targets(root, moduli=[31], ckpt_kinds=["final", "best"])
    kinds = {t.ckpt_kind for t in targets}
    assert "final" in kinds
    assert all(t.modulus == 31 for t in targets)
    assert all(t.ckpt_path.is_file() for t in targets)


def test_steer_at_early_layer() -> None:
    from go4cl.analysis.causal import steer_at_layer
    from go4cl.model.transformer import ModelConfig, ModularTransformer

    cfg = ModelConfig(d_model=32, n_layers=3, n_heads=4, n_classes=7)
    cfg.d_mlp = 64
    torch.manual_seed(0)
    model = ModularTransformer(cfg)
    n, t, p, d = 70, 10, 7, 32
    tokens = torch.randint(0, 64, (n, t))
    y = torch.arange(n) % p
    cache = model.forward_with_cache(tokens)
    # Use layer-0 resid; even with random weights, shuffled control should differ
    # in structure. Just check API returns and chance bounds.
    out = steer_at_layer(
        model,
        cache["resid_post"][0],
        y,
        layer_idx=0,
        modulus=p,
        delta=1,
        alpha=1.0,
    )
    assert "steered_acc_target" in out
    assert out["site"] == "resid_post_L0"
    assert 0.0 <= out["steered_acc_target"] <= 1.0


def test_head_knockout_shapes() -> None:
    from go4cl.analysis.composition import head_knockout
    from go4cl.model.transformer import ModelConfig, ModularTransformer

    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_classes=7)
    cfg.d_mlp = 64
    torch.manual_seed(3)
    model = ModularTransformer(cfg)
    n, t, p = 32, 10, 7
    tokens = torch.randint(0, 64, (n, t))
    labels = torch.arange(n) % p
    cache = model.forward_with_cache(tokens)
    out = head_knockout(
        model,
        resid_pre=cache["resid_pre"],
        resid_post=cache["resid_post"],
        labels=labels,
        device=torch.device("cpu"),
    )
    assert out["n_heads"] == 4
    assert len(out["by_head"]) == 2 * 4
    assert out["worst_head"] is not None
    assert all(-1.05 <= r["delta_acc"] <= 1.05 for r in out["by_head"])


def test_freq_pairs_by_energy_conjugates() -> None:
    from go4cl.analysis.causal import _freq_pairs_by_energy

    # p=7 → pairs (1,6), (2,5), (3,4); energy of pair = e[f]+e[p-f]
    energy = [0.0, 10.0, 1.0, 4.0, 4.0, 1.0, 10.0]
    pairs = _freq_pairs_by_energy(energy, modulus=7, skip_dc=True)
    assert [p["rep"] for p in pairs] == [1, 3, 2]
    assert pairs[0]["freqs"] == [1, 6]
    assert pairs[0]["energy"] == 20.0
    assert pairs[-1]["rep"] == 2


def test_fourier_ablation_sweep_curves() -> None:
    from torch.utils.data import DataLoader, TensorDataset

    from go4cl.analysis.causal import fourier_ablation_on_embeddings
    from go4cl.model.transformer import ModelConfig, ModularTransformer

    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_classes=7)
    cfg.d_mlp = 64
    torch.manual_seed(1)
    model = ModularTransformer(cfg)
    n, t, p = 64, 10, 7
    tokens = torch.randint(0, 64, (n, t))
    labels = torch.randint(0, p, (n,))
    # Minimal loader mimicking dataset batches
    ds = TensorDataset(tokens, labels)

    class _Wrap(torch.utils.data.Dataset):
        def __len__(self):
            return len(ds)

        def __getitem__(self, idx):
            tok, lab = ds[idx]
            return {"tokens": tok, "labels": lab}

    loader = DataLoader(_Wrap(), batch_size=16)
    out = fourier_ablation_on_embeddings(
        model,
        loader,
        modulus=p,
        device=torch.device("cpu"),
        top_k=1,
        sweep_ks=[1, 2],
    )
    assert out["sweep_ks"] == [1, 2]
    assert len(out["important_curve"]) == 2
    assert len(out["unimportant_curve"]) == 2
    assert out["important_curve"][0]["k"] == 1
    assert "delta_acc" in out["important_curve"][0]
    assert "delta_acc" in out["unimportant_curve"][0]
    # Full ablation of all non-DC pairs should be k=n_pairs=3; k=2 still valid
    assert all(0.0 <= c["acc"] <= 1.0 for c in out["important_curve"])


def test_composition_sites_and_knockout() -> None:
    from go4cl.analysis.composition import (
        component_knockout,
        composition_sites,
        freq_harmonic_fit,
    )
    from go4cl.model.transformer import ModelConfig, ModularTransformer

    assert composition_sites(3) == [
        "L0_mid",
        "L0_post",
        "L1_mid",
        "L1_post",
        "L2_mid",
        "L2_post",
    ]

    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, n_classes=7)
    cfg.d_mlp = 64
    torch.manual_seed(2)
    model = ModularTransformer(cfg)
    n, t, p = 48, 10, 7
    tokens = torch.randint(0, 64, (n, t))
    labels = torch.arange(n) % p
    cache = model.forward_with_cache(tokens)
    ko = component_knockout(
        model,
        resid_pre=cache["resid_pre"],
        resid_mid=cache["resid_mid"],
        resid_post=cache["resid_post"],
        labels=labels,
        device=torch.device("cpu"),
    )
    assert len(ko["by_layer"]) == 2
    assert "compose_layer_guess" in ko
    assert all(
        -1.0 <= row["zero_mlp_delta"] <= 1.0 for row in ko["by_layer"]
    )

    # Harmonic fit API: R2 in a plausible range
    harm = freq_harmonic_fit(
        train_resid_mid=cache["resid_mid"],
        train_resid_post=cache["resid_post"],
        train_tokens=tokens,
        test_resid_mid=cache["resid_mid"],
        test_resid_post=cache["resid_post"],
        test_tokens=tokens,
        operand_i=0,
        operand_j=1,
        modulus=p,
        n_layers=2,
        tok_emb_weight=model.tok_emb.weight.detach(),
        top_k_pairs=2,
        device=torch.device("cpu"),
    )
    assert set(harm["sites"]) == set(composition_sites(2))
    assert len(harm["freqs"]) == 2
    for site in harm["sites"]:
        r2 = harm["by_site"][site]["R2_sum_mean"]
        assert -1.0 <= r2 <= 1.0
