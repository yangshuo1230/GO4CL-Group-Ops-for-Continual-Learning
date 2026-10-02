"""Composition locus: where / how modular addition is computed.

Three complementary analyses on query residuals:
  1. probe_info_ladder — linear readability of xi, xj, sum at mid/post sites
  2. component_knockout — causal necessity of attn vs MLP writes per layer
  3. head_knockout — per-head zero-ablation within each attention layer
  4. freq_harmonic_fit — operand vs sum Fourier-harmonic R^2 signature
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from go4cl.analysis.cache import accuracy_from_logits
from go4cl.analysis.causal import _freq_pairs_by_energy
from go4cl.analysis.fourier import analyze_digit_embedding_fourier
from go4cl.analysis.probes import run_operand_probes
from go4cl.model.transformer import ModularTransformer


def composition_sites(n_layers: int) -> list[str]:
    """Ordered site names: L0_mid, L0_post, ..., L{n-1}_mid, L{n-1}_post."""
    sites: list[str] = []
    for li in range(n_layers):
        sites.append(f"L{li}_mid")
        sites.append(f"L{li}_post")
    return sites


def _query_at_site(
    resid_mid: list[torch.Tensor],
    resid_post: list[torch.Tensor],
    site: str,
    device: torch.device,
) -> torch.Tensor:
    """Extract query-position vectors [N, D] for a site name."""
    if site.endswith("_mid"):
        li = int(site[1 : -len("_mid")])
        return resid_mid[li][:, -1, :].to(device)
    if site.endswith("_post"):
        li = int(site[1 : -len("_post")])
        return resid_post[li][:, -1, :].to(device)
    raise ValueError(f"unknown site: {site}")


def probe_info_ladder(
    *,
    train_resid_mid: list[torch.Tensor],
    train_resid_post: list[torch.Tensor],
    train_tokens: torch.Tensor,
    eval_resid_mid: dict[str, list[torch.Tensor]],
    eval_resid_post: dict[str, list[torch.Tensor]],
    eval_tokens: dict[str, torch.Tensor],
    operand_i: int,
    operand_j: int,
    modulus: int,
    n_layers: int,
    steps: int = 400,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Probe xi / xj / sum at every resid_mid and resid_post query site."""
    if device is None:
        device = train_resid_post[0].device
    sites = composition_sites(n_layers)
    by_site: dict[str, Any] = {}
    for site in sites:
        train_q = _query_at_site(
            train_resid_mid, train_resid_post, site, device
        )
        eval_qs = {
            name: _query_at_site(
                eval_resid_mid[name], eval_resid_post[name], site, device
            )
            for name in eval_tokens
        }
        probes = run_operand_probes(
            train_resid=train_q,
            train_tokens=train_tokens,
            eval_resids=eval_qs,
            eval_tokens=eval_tokens,
            operand_i=operand_i,
            operand_j=operand_j,
            modulus=modulus,
            steps=steps,
        )
        xi = probes["xi_mod_p"]["eval_acc"].get("test")
        xj = probes["xj_mod_p"]["eval_acc"].get("test")
        sm = probes["sum_mod_p"]["eval_acc"].get("test")
        by_site[site] = {
            "probe_xi": float(xi) if xi is not None else None,
            "probe_xj": float(xj) if xj is not None else None,
            "probe_sum": float(sm) if sm is not None else None,
            "eval_acc": {
                "xi_mod_p": probes["xi_mod_p"]["eval_acc"],
                "xj_mod_p": probes["xj_mod_p"]["eval_acc"],
                "sum_mod_p": probes["sum_mod_p"]["eval_acc"],
            },
        }

    # Largest mid→post jump in probe_sum identifies composition layer guess
    jump_layer = None
    jump_delta = None
    for li in range(n_layers):
        mid_s = by_site[f"L{li}_mid"]["probe_sum"]
        post_s = by_site[f"L{li}_post"]["probe_sum"]
        if mid_s is None or post_s is None:
            continue
        d = float(post_s) - float(mid_s)
        if jump_delta is None or d > jump_delta:
            jump_delta = d
            jump_layer = li

    return {
        "sites": sites,
        "by_site": by_site,
        "ladder_sum_jump_layer": jump_layer,
        "ladder_sum_jump_delta": jump_delta,
    }


@torch.no_grad()
def _continue_after_edited_post(
    model: ModularTransformer,
    resid_post: torch.Tensor,
    *,
    layer_idx: int,
) -> torch.Tensor:
    """Continue from post-MLP residual of ``layer_idx`` → logits."""
    return model.continue_from_layer(resid_post, layer_idx=layer_idx)["logits"]


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


def _ridge_r2_holdout(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_test: torch.Tensor,
    y_test: torch.Tensor,
    *,
    ridge: float = 1e-2,
) -> float:
    n, d = x_train.shape
    xtx = x_train.T @ x_train + ridge * torch.eye(
        d, device=x_train.device, dtype=x_train.dtype
    )
    xty = x_train.T @ y_train
    try:
        w = torch.linalg.solve(xtx, xty)
    except Exception:  # noqa: BLE001
        w = torch.linalg.lstsq(xtx, xty).solution
    pred = x_test @ w
    resid = y_test - pred
    sse = float((resid ** 2).sum().item())
    y_mean = y_test.mean()
    sst = float(((y_test - y_mean) ** 2).sum().item())
    if sst < 1e-12:
        return 0.0
    return float(max(1.0 - sse / sst, -1.0))  # clip absurd negatives lightly


def _trig_templates(
    residues: torch.Tensor,
    *,
    freq: int,
    modulus: int,
) -> torch.Tensor:
    """Return [N, 2] = [cos(2π f r / p), sin(2π f r / p)]."""
    ang = 2.0 * torch.pi * float(freq) * residues.float() / float(modulus)
    return torch.stack([torch.cos(ang), torch.sin(ang)], dim=-1)


def freq_harmonic_fit(
    *,
    train_resid_mid: list[torch.Tensor],
    train_resid_post: list[torch.Tensor],
    train_tokens: torch.Tensor,
    test_resid_mid: list[torch.Tensor],
    test_resid_post: list[torch.Tensor],
    test_tokens: torch.Tensor,
    operand_i: int,
    operand_j: int,
    modulus: int,
    n_layers: int,
    tok_emb_weight: torch.Tensor,
    top_k_pairs: int = 3,
    device: torch.device | None = None,
    ridge: float = 1e-2,
) -> dict[str, Any]:
    """Ridge-fit query vectors to operand / sum Fourier harmonics.

    For each top energy conjugate-pair representative frequency f and each
    mid/post site, report holdout R^2 for predicting cos/sin of f·xi, f·xj,
    and f·sum from the residual (mean R^2 over the two trig components for
    operand = mean of xi and xj; sum = mean of cos/sin of sum).
    """
    if device is None:
        device = train_resid_post[0].device

    emb = nn.Embedding(tok_emb_weight.shape[0], tok_emb_weight.shape[1])
    with torch.no_grad():
        emb.weight.copy_(tok_emb_weight)
    fourier = analyze_digit_embedding_fourier(emb, modulus=modulus)
    pairs = _freq_pairs_by_energy(
        fourier["energy_by_freq"], modulus=modulus, skip_dc=True
    )
    ks = pairs[: max(1, min(top_k_pairs, len(pairs)))]
    freqs = [int(p["rep"]) for p in ks]

    xi_tr = (train_tokens[:, operand_i] % modulus).to(device)
    xj_tr = (train_tokens[:, operand_j] % modulus).to(device)
    sum_tr = (xi_tr + xj_tr) % modulus
    xi_te = (test_tokens[:, operand_i] % modulus).to(device)
    xj_te = (test_tokens[:, operand_j] % modulus).to(device)
    sum_te = (xi_te + xj_te) % modulus

    sites = composition_sites(n_layers)
    by_site: dict[str, Any] = {}

    for site in sites:
        q_tr = _query_at_site(
            train_resid_mid, train_resid_post, site, device
        ).float()
        q_te = _query_at_site(
            test_resid_mid, test_resid_post, site, device
        ).float()
        # Center features for stable ridge
        mu = q_tr.mean(dim=0, keepdim=True)
        q_tr_c = q_tr - mu
        q_te_c = q_te - mu

        per_freq: list[dict[str, Any]] = []
        for f in freqs:
            r2_op_parts: list[float] = []
            r2_sum_parts: list[float] = []
            for residues_tr, residues_te, bucket in (
                (xi_tr, xi_te, "op"),
                (xj_tr, xj_te, "op"),
                (sum_tr, sum_te, "sum"),
            ):
                tmpl_tr = _trig_templates(residues_tr, freq=f, modulus=modulus)
                tmpl_te = _trig_templates(residues_te, freq=f, modulus=modulus)
                for c in range(2):
                    r2 = _ridge_r2_holdout(
                        q_tr_c,
                        tmpl_tr[:, c],
                        q_te_c,
                        tmpl_te[:, c],
                        ridge=ridge,
                    )
                    if bucket == "op":
                        r2_op_parts.append(r2)
                    else:
                        r2_sum_parts.append(r2)
            r2_op = float(sum(r2_op_parts) / max(len(r2_op_parts), 1))
            r2_sum = float(sum(r2_sum_parts) / max(len(r2_sum_parts), 1))
            per_freq.append(
                {
                    "freq": f,
                    "R2_operand": r2_op,
                    "R2_sum": r2_sum,
                    "sum_minus_operand_R2": float(r2_sum - r2_op),
                }
            )

        mean_op = float(
            sum(e["R2_operand"] for e in per_freq) / max(len(per_freq), 1)
        )
        mean_sum = float(
            sum(e["R2_sum"] for e in per_freq) / max(len(per_freq), 1)
        )
        by_site[site] = {
            "per_freq": per_freq,
            "R2_operand_mean": mean_op,
            "R2_sum_mean": mean_sum,
            "sum_minus_operand_R2_mean": float(mean_sum - mean_op),
        }

    # Mid→post jump in sum−operand R2
    jump_layer = None
    jump_delta = None
    for li in range(n_layers):
        mid = by_site[f"L{li}_mid"]["sum_minus_operand_R2_mean"]
        post = by_site[f"L{li}_post"]["sum_minus_operand_R2_mean"]
        d = float(post) - float(mid)
        if jump_delta is None or d > jump_delta:
            jump_delta = d
            jump_layer = li

    return {
        "freqs": freqs,
        "freq_pairs": ks,
        "sites": sites,
        "by_site": by_site,
        "harmonic_jump_layer": jump_layer,
        "harmonic_jump_delta": jump_delta,
    }


def run_composition_analysis(
    model: ModularTransformer,
    *,
    train_cache,
    test_cache,
    val_cache=None,
    operand_i: int,
    operand_j: int,
    modulus: int,
    device: torch.device,
    probe_steps: int = 400,
    top_k_pairs: int = 3,
    baseline_acc: float | None = None,
) -> dict[str, Any]:
    """Run ladder + knockout + harmonic fit; return combined report."""
    n_layers = len(model.blocks)
    eval_tokens = {"test": test_cache.tokens}
    eval_mid = {"test": test_cache.resid_mid}
    eval_post = {"test": test_cache.resid_post}
    if val_cache is not None:
        eval_tokens["val"] = val_cache.tokens
        eval_mid["val"] = val_cache.resid_mid
        eval_post["val"] = val_cache.resid_post

    ladder = probe_info_ladder(
        train_resid_mid=train_cache.resid_mid,
        train_resid_post=train_cache.resid_post,
        train_tokens=train_cache.tokens,
        eval_resid_mid=eval_mid,
        eval_resid_post=eval_post,
        eval_tokens=eval_tokens,
        operand_i=operand_i,
        operand_j=operand_j,
        modulus=modulus,
        n_layers=n_layers,
        steps=probe_steps,
        device=device,
    )

    knockout = component_knockout(
        model,
        resid_pre=test_cache.resid_pre,
        resid_mid=test_cache.resid_mid,
        resid_post=test_cache.resid_post,
        labels=test_cache.labels,
        device=device,
        baseline_acc=baseline_acc,
    )

    heads = head_knockout(
        model,
        resid_pre=test_cache.resid_pre,
        resid_post=test_cache.resid_post,
        labels=test_cache.labels,
        device=device,
        baseline_acc=baseline_acc,
    )

    harmonic = freq_harmonic_fit(
        train_resid_mid=train_cache.resid_mid,
        train_resid_post=train_cache.resid_post,
        train_tokens=train_cache.tokens,
        test_resid_mid=test_cache.resid_mid,
        test_resid_post=test_cache.resid_post,
        test_tokens=test_cache.tokens,
        operand_i=operand_i,
        operand_j=operand_j,
        modulus=modulus,
        n_layers=n_layers,
        tok_emb_weight=model.tok_emb.weight.detach().cpu(),
        top_k_pairs=top_k_pairs,
        device=device,
    )

    # Prefer knockout for compose_layer_guess; fall back to ladder jump
    compose_guess = knockout.get("compose_layer_guess")
    if compose_guess is None:
        compose_guess = ladder.get("ladder_sum_jump_layer")

    return {
        "ladder": ladder,
        "knockout": knockout,
        "head_knockout": heads,
        "harmonic": harmonic,
        "compose_layer_guess": compose_guess,
        "ladder_sum_jump_layer": ladder.get("ladder_sum_jump_layer"),
        "harmonic_jump_layer": harmonic.get("harmonic_jump_layer"),
        "worst_attn_head": heads.get("worst_head"),
    }
