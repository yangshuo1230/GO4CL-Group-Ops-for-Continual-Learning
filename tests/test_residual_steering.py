"""Residual steering helpers, CSV schema, and CPU plot smoke tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from go4cl.analysis.interventions.steering import (
    estimate_class_means,
    global_shift_vector,
    predict_steering_condition,
    sample_records,
    shuffled_mean_permutation,
    summarize_records,
    target_label,
    transition_matrix,
    transition_rows,
)
from go4cl.analysis.pipelines.plot_residual_steering import (
    matrix_from_prediction_rows,
    plot_fig6,
    plot_fig6b,
)
from go4cl.analysis.pipelines.residual_steering import PREDICTION_FIELDS, SUMMARY_FIELDS
from go4cl.analysis.reporting import write_csv_rows
from go4cl.model.transformer import ModelConfig, ModularTransformer


def test_target_label_mod_and_wrap() -> None:
    y = torch.tensor([0, 29, 30, 4])
    t = target_label(y, delta=4, modulus=31)
    assert t.tolist() == [4, 2, 3, 8]
    t2 = target_label(y, delta=1, modulus=31)
    assert int(t2[2]) == 0


def test_transition_matrix_row_normalized_and_wrap() -> None:
    source = torch.tensor([0, 0, 30, 30, 30])
    pred = torch.tensor([0, 1, 3, 3, 2])  # 30 -> 3 wraps conceptually as a class
    m = transition_matrix(source, pred, modulus=31)
    assert m.shape == (31, 31)
    assert torch.allclose(m[0].sum(), torch.tensor(1.0, dtype=torch.float64))
    assert abs(float(m[0, 0] - 0.5)) < 1e-12
    assert abs(float(m[0, 1] - 0.5)) < 1e-12
    assert abs(float(m[30, 3] - 2 / 3)) < 1e-12
    assert abs(float(m[30, 2] - 1 / 3)) < 1e-12
    assert float(m[1].sum()) == 0.0  # empty source row


def test_shuffled_mapping_reproducible() -> None:
    a = shuffled_mean_permutation(31, seed=0)
    b = shuffled_mean_permutation(31, seed=0)
    c = shuffled_mean_permutation(31, seed=1)
    assert torch.equal(a, b)
    assert not torch.equal(a, c)
    assert int(a.unique().numel()) == 31


def test_reference_means_not_from_eval() -> None:
    p, d = 7, 8
    ref = torch.zeros(14, d)
    y_ref = torch.arange(14) % p
    for c in range(p):
        ref[y_ref == c] = float(c)
    eval_h = torch.full((14, d), 99.0)
    y_eval = y_ref.clone()
    mu_ref = estimate_class_means(ref, y_ref, n_classes=p)
    mu_eval = estimate_class_means(eval_h, y_eval, n_classes=p)
    assert not torch.allclose(mu_ref, mu_eval)
    # Structured add uses reference μ[(s+δ)] − μ[s], independent of eval activations.
    delta = 1
    add = mu_ref[(y_eval + delta) % p] - mu_ref[y_eval]
    # Non-wrapping classes shift by +1; class 6 wraps 6→0 so 0-6 = -6.
    assert torch.allclose(add[:6, 0], torch.ones(6))
    assert torch.allclose(add[y_eval == 6][:, 0], torch.full((2,), -6.0))


def test_csv_fields_complete(tmp_path: Path) -> None:
    orig = torch.tensor([0, 1, 30])
    tgt = target_label(orig, delta=4, modulus=31)
    pred = torch.tensor([4, 5, 3])
    rec = sample_records(
        site="L0_post",
        delta=4,
        alpha=1.0,
        condition="structured",
        original=orig,
        target=tgt,
        predicted=pred,
    )
    summary = summarize_records(rec, modulus=31, checkpoint="ckpt.pt")
    trans = transition_rows(rec, modulus=31)
    s_path = write_csv_rows(tmp_path / "steering_summary.csv", summary, SUMMARY_FIELDS)
    p_path = write_csv_rows(
        tmp_path / "steering_predictions.csv", trans, PREDICTION_FIELDS
    )
    with s_path.open() as handle:
        header = handle.readline().strip().split(",")
    assert header == list(SUMMARY_FIELDS)
    with p_path.open() as handle:
        header = handle.readline().strip().split(",")
    assert header == list(PREDICTION_FIELDS)
    # wrap-around source 30 → pred 3 in the transition table
    assert any(
        int(r["source_residue"]) == 30
        and int(r["predicted_residue"]) == 3
        and abs(float(r["probability"]) - 1.0) < 1e-12
        for r in trans
    )
    # probabilities sum to 1 per source that appears
    by_src = {}
    for r in trans:
        if r["source_residue"] not in (0, 1, 30):
            continue
        by_src.setdefault(r["source_residue"], 0.0)
        by_src[r["source_residue"]] += float(r["probability"])
    assert all(abs(v - 1.0) < 1e-12 for v in by_src.values())


def test_global_shift_is_class_independent() -> None:
    p, d = 5, 3
    means = torch.randn(p, d)
    v = global_shift_vector(means, delta=2, modulus=p)
    assert v.shape == (d,)
    # mean of cyclic differences is independent of starting class
    idx = torch.arange(p)
    again = (means[(idx + 2) % p] - means[idx]).mean(0)
    assert torch.allclose(v, again)


def test_predict_conditions_cpu_tiny_model() -> None:
    cfg = ModelConfig(d_model=32, n_layers=3, n_heads=4, n_classes=7, d_mlp=64)
    torch.manual_seed(0)
    model = ModularTransformer(cfg)
    model.eval()
    n, t, p = 21, 10, 7
    tokens = torch.randint(0, 64, (n, t))
    y = torch.arange(n) % p
    cache = model.forward_with_cache(tokens)
    # Reference means from a different tensor so eval is not the source of μ.
    ref = torch.randn(n, 32)
    means = estimate_class_means(ref, y, n_classes=p)
    resid = cache["resid_post"][0]
    out_s = predict_steering_condition(
        model,
        resid,
        y,
        class_means=means,
        layer_idx=0,
        modulus=p,
        delta=1,
        alpha=1.0,
        condition="structured",
        shuffle_seed=0,
    )
    out_z = predict_steering_condition(
        model,
        resid,
        y,
        class_means=means,
        layer_idx=0,
        modulus=p,
        delta=1,
        alpha=1.0,
        condition="shuffled",
        shuffle_seed=0,
    )
    out_g = predict_steering_condition(
        model,
        resid,
        y,
        class_means=means,
        layer_idx=0,
        modulus=p,
        delta=1,
        alpha=1.0,
        condition="global_structured",
        shuffle_seed=0,
    )
    assert out_s["target"].tolist() == ((y + 1) % p).tolist()
    assert out_s["predicted"].shape == y.shape
    assert out_z["predicted"].shape == y.shape
    assert out_g["predicted"].shape == y.shape
    # Query-only: other token rows of a clone used internally; original cache unchanged.
    before = resid[:, 0, :].clone()
    predict_steering_condition(
        model,
        resid,
        y,
        class_means=means,
        layer_idx=0,
        modulus=p,
        delta=1,
        alpha=10.0,
        condition="structured",
    )
    assert torch.equal(resid[:, 0, :], before)


def test_plot_smoke_cpu_no_display(tmp_path: Path) -> None:
    modulus = 8
    summary = []
    pred_rows = []
    rng = np.random.default_rng(0)
    for site in ("L0_post", "L1_post", "L2_post"):
        for delta in (1, 2, 4, 8):
            for cond, t_acc, o_acc in (
                ("no_steering", 0.05, 0.9),
                ("structured", 0.8, 0.1),
                ("shuffled", 0.1, 0.2),
                ("global_structured", 0.4, 0.3),
            ):
                summary.append(
                    {
                        "modulus": modulus,
                        "checkpoint": "dummy",
                        "site": site,
                        "delta": delta,
                        "alpha": 1.0,
                        "condition": cond,
                        "n": modulus,
                        "original_acc": o_acc,
                        "target_acc": t_acc,
                    }
                )
                if site == "L0_post" and delta == 4:
                    for s in range(modulus):
                        logits = rng.random(modulus)
                        if cond == "no_steering":
                            logits = np.zeros(modulus)
                            logits[s] = 1.0
                        elif cond == "structured":
                            logits = np.zeros(modulus)
                            logits[(s + 4) % modulus] = 1.0
                        probs = logits / logits.sum()
                        for k, pr in enumerate(probs):
                            pred_rows.append(
                                {
                                    "site": site,
                                    "delta": delta,
                                    "condition": cond,
                                    "source_residue": s,
                                    "predicted_residue": k,
                                    "count": int(round(pr * 10)),
                                    "probability": float(pr),
                                }
                            )
    fig_dir = tmp_path / "figures"
    written = plot_fig6(
        summary_rows=[{k: str(v) for k, v in r.items()} for r in summary],
        prediction_rows=[{k: str(v) for k, v in r.items()} for r in pred_rows],
        out_stem=fig_dir / "fig6_residual_steering",
        delta=4,
        matrix_site="L0_post",
    )
    written += plot_fig6b(
        summary_rows=[{k: str(v) for k, v in r.items()} for r in summary],
        out_stem=fig_dir / "fig6b_steering_across_deltas",
    )
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
    rebuilt = matrix_from_prediction_rows(
        [{k: str(v) for k, v in r.items()} for r in pred_rows],
        site="L0_post",
        delta=4,
        condition="structured",
        modulus=modulus,
    )
    assert rebuilt.shape == (modulus, modulus)
    assert np.allclose(rebuilt.sum(axis=1), 1.0)
    assert abs(rebuilt[7, 3] - 1.0) < 1e-9  # (7+4)%8 = 3
