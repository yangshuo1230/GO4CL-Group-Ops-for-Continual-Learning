"""Linear probes on residual stream for modular features."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from go4cl.model.transformer import ModelConfig, ModularTransformer


@torch.no_grad()
def _accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    return float((logits.argmax(dim=-1) == y).float().mean().item())


def fit_linear_probe(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    x_eval: dict[str, torch.Tensor],
    y_eval: dict[str, torch.Tensor],
    *,
    n_classes: int,
    steps: int = 400,
    lr: float = 0.05,
    weight_decay: float = 1e-2,
) -> dict[str, Any]:
    """Fit a linear classifier with AdamW; report train/eval accuracies."""
    d = x_train.shape[-1]
    device = x_train.device
    probe = nn.Linear(d, n_classes).to(device)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=weight_decay)
    probe.train()
    for _ in range(steps):
        logits = probe(x_train)
        loss = F.cross_entropy(logits, y_train)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    probe.eval()
    with torch.no_grad():
        train_acc = _accuracy(probe(x_train), y_train)
        eval_acc = {
            name: _accuracy(probe(x_eval[name]), y_eval[name]) for name in x_eval
        }
        weight = probe.weight.detach().cpu().clone()
        bias = probe.bias.detach().cpu().clone() if probe.bias is not None else None
    return {
        "train_acc": train_acc,
        "eval_acc": eval_acc,
        "n_train": int(x_train.shape[0]),
        "n_classes": n_classes,
        "weight": weight,
        "bias": bias,
    }


def _probe_targets(
    train_tokens: torch.Tensor,
    eval_tokens: dict[str, torch.Tensor],
    *,
    operand_i: int,
    operand_j: int,
    modulus: int,
    device: torch.device,
) -> dict[str, tuple[torch.Tensor, dict[str, torch.Tensor]]]:
    xi_tr = (train_tokens[:, operand_i] % modulus).to(device)
    xj_tr = (train_tokens[:, operand_j] % modulus).to(device)
    sum_tr = (xi_tr + xj_tr) % modulus
    return {
        "xi_mod_p": (
            xi_tr,
            {
                name: (tok[:, operand_i] % modulus).to(device)
                for name, tok in eval_tokens.items()
            },
        ),
        "xj_mod_p": (
            xj_tr,
            {
                name: (tok[:, operand_j] % modulus).to(device)
                for name, tok in eval_tokens.items()
            },
        ),
        "sum_mod_p": (
            sum_tr,
            {
                name: (
                    (tok[:, operand_i] % modulus) + (tok[:, operand_j] % modulus)
                )
                .remainder(modulus)
                .to(device)
                for name, tok in eval_tokens.items()
            },
        ),
    }


def run_operand_probes(
    *,
    train_resid: torch.Tensor,
    train_tokens: torch.Tensor,
    eval_resids: dict[str, torch.Tensor],
    eval_tokens: dict[str, torch.Tensor],
    operand_i: int,
    operand_j: int,
    modulus: int,
    steps: int = 400,
) -> dict[str, Any]:
    """Probe xi%p, xj%p, (xi+xj)%p from a residual representation."""
    device = train_resid.device
    targets = _probe_targets(
        train_tokens,
        eval_tokens,
        operand_i=operand_i,
        operand_j=operand_j,
        modulus=modulus,
        device=device,
    )

    out: dict[str, Any] = {}
    for key, (y_tr, y_ev) in targets.items():
        fitted = fit_linear_probe(
            train_resid,
            y_tr,
            eval_resids,
            y_ev,
            n_classes=modulus,
            steps=steps,
        )
        # Drop bulky tensors from JSON-facing report; keep for optional callers.
        out[key] = {
            "train_acc": fitted["train_acc"],
            "eval_acc": fitted["eval_acc"],
            "n_train": fitted["n_train"],
            "n_classes": fitted["n_classes"],
        }
        out[f"_{key}_weight"] = fitted["weight"]
        out[f"_{key}_bias"] = fitted["bias"]
    sum_test = out["sum_mod_p"]["eval_acc"].get("test")
    if sum_test is None and out["sum_mod_p"]["eval_acc"]:
        sum_test = next(iter(out["sum_mod_p"]["eval_acc"].values()))
    out["probe_sum_acc"] = float(sum_test) if sum_test is not None else None
    return out


def run_operand_probes_with_random_control(
    *,
    trained_model: ModularTransformer,
    train_resid: torch.Tensor,
    train_tokens: torch.Tensor,
    eval_resids: dict[str, torch.Tensor],
    eval_tokens: dict[str, torch.Tensor],
    operand_i: int,
    operand_j: int,
    modulus: int,
    steps: int = 400,
    random_seed: int = 0,
    collect_random_fn,
) -> dict[str, Any]:
    """Trained-model probes + same probes on a fresh random-init model.

    The random-init control checks that high probe accuracy is not merely
    due to an overpowered linear classifier on arbitrary activations.
    ``collect_random_fn(random_model) -> (train_resid, eval_resids)`` should
    run the same token batches through the random model.
    """
    trained = run_operand_probes(
        train_resid=train_resid,
        train_tokens=train_tokens,
        eval_resids=eval_resids,
        eval_tokens=eval_tokens,
        operand_i=operand_i,
        operand_j=operand_j,
        modulus=modulus,
        steps=steps,
    )

    cfg = ModelConfig.from_dict(trained_model.cfg.to_dict())
    torch.manual_seed(int(random_seed))
    random_model = ModularTransformer(cfg).to(train_resid.device)
    random_model.eval()
    rand_train, rand_eval = collect_random_fn(random_model)
    random = run_operand_probes(
        train_resid=rand_train,
        train_tokens=train_tokens,
        eval_resids=rand_eval,
        eval_tokens=eval_tokens,
        operand_i=operand_i,
        operand_j=operand_j,
        modulus=modulus,
        steps=steps,
    )

    # Strip private weight keys from nested reports for JSON
    def _public(d: dict[str, Any]) -> dict[str, Any]:
        return {k: v for k, v in d.items() if not str(k).startswith("_")}

    trained_pub = _public(trained)
    random_pub = _public(random)
    t_sum = trained_pub.get("probe_sum_acc")
    r_sum = random_pub.get("probe_sum_acc")
    return {
        "trained": trained_pub,
        "random_init": random_pub,
        "probe_sum_acc": t_sum,
        "probe_sum_acc_random": r_sum,
        "probe_sum_acc_delta": (
            float(t_sum) - float(r_sum)
            if t_sum is not None and r_sum is not None
            else None
        ),
        # keep trained sum weights for optional steering
        "_sum_weight": trained.get("_sum_mod_p_weight"),
        "_sum_bias": trained.get("_sum_mod_p_bias"),
    }


def run_layer_probes_with_random_control(
    *,
    trained_model: ModularTransformer,
    train_layer_resids: dict[int, torch.Tensor],
    eval_layer_resids: dict[str, dict[int, torch.Tensor]],
    train_tokens: torch.Tensor,
    eval_tokens: dict[str, torch.Tensor],
    operand_i: int,
    operand_j: int,
    modulus: int,
    layers: list[int],
    steps: int = 400,
    random_seed: int = 0,
    collect_random_layers_fn,
) -> dict[str, Any]:
    """Per-layer probes on query-position residuals + random-init control.

    ``train_layer_resids[ℓ]`` / ``eval_layer_resids[split][ℓ]`` are [N, D]
    vectors at the query position after layer ``ℓ``.
    ``collect_random_layers_fn(random_model) -> (train_dict, eval_dict)`` with
    the same nesting.
    """
    cfg = ModelConfig.from_dict(trained_model.cfg.to_dict())
    torch.manual_seed(int(random_seed))
    random_model = ModularTransformer(cfg).to(next(iter(train_layer_resids.values())).device)
    random_model.eval()
    rand_train, rand_eval = collect_random_layers_fn(random_model)

    by_layer: dict[str, Any] = {}
    for li in layers:
        trained = run_operand_probes(
            train_resid=train_layer_resids[li],
            train_tokens=train_tokens,
            eval_resids={
                name: eval_layer_resids[name][li] for name in eval_layer_resids
            },
            eval_tokens=eval_tokens,
            operand_i=operand_i,
            operand_j=operand_j,
            modulus=modulus,
            steps=steps,
        )
        random = run_operand_probes(
            train_resid=rand_train[li],
            train_tokens=train_tokens,
            eval_resids={name: rand_eval[name][li] for name in rand_eval},
            eval_tokens=eval_tokens,
            operand_i=operand_i,
            operand_j=operand_j,
            modulus=modulus,
            steps=steps,
        )

        def _public(d: dict[str, Any]) -> dict[str, Any]:
            return {k: v for k, v in d.items() if not str(k).startswith("_")}

        t_pub, r_pub = _public(trained), _public(random)
        t_sum, r_sum = t_pub.get("probe_sum_acc"), r_pub.get("probe_sum_acc")
        by_layer[f"L{li}"] = {
            "layer_idx": li,
            "trained": t_pub,
            "random_init": r_pub,
            "probe_sum_acc": t_sum,
            "probe_sum_acc_random": r_sum,
            "probe_sum_acc_delta": (
                float(t_sum) - float(r_sum)
                if t_sum is not None and r_sum is not None
                else None
            ),
        }
    return by_layer
