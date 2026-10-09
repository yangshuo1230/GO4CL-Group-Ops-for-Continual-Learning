"""Evaluate layer-wise parameter patches without training.

Each intervention starts from a fresh clone of theta_AB and copies the named
groups from theta_A. Empty patch equals theta_AB. Patching every coarse group
equals theta_A on the registry parameters. Attention-mask buffers are not
trainable and stay with theta_AB; they are reported, not silently mixed.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn

from go4cl.model.transformer import ModularTransformer
from go4cl.phases.transfer_mechanism.checkpoint_mix import clone_state_dict
from go4cl.phases.transfer_mechanism.parameter_groups import (
    COARSE_GROUPS,
    ParamSlice,
    assert_partition,
    attention_layer_group,
    build_registry,
    mlp_layer_group,
    slice_selected,
)


def load_theta(path: Path | str) -> dict[str, torch.Tensor]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    state = payload.get("model_state") or payload.get("model_state_dict")
    if not isinstance(state, dict):
        raise ValueError(f"{path} has no model_state")
    return clone_state_dict(state)


def _copy_slice(dst: dict[str, torch.Tensor], src: dict[str, torch.Tensor], sl: ParamSlice) -> None:
    if sl.rows is None:
        dst[sl.name] = src[sl.name].detach().clone()
        return
    start, stop = sl.rows
    dst[sl.name][start:stop] = src[sl.name][start:stop]


def apply_patch(
    theta_ab: dict[str, torch.Tensor],
    theta_a: dict[str, torch.Tensor],
    groups: list[str] | tuple[str, ...],
    registry: list[ParamSlice],
) -> dict[str, torch.Tensor]:
    """Clone theta_AB, then restore the selected slices from theta_A."""
    patched = clone_state_dict(theta_ab)
    selected = set(groups)
    if not selected:
        return patched
    for sl in registry:
        if slice_selected(sl, selected):
            _copy_slice(patched, theta_a, sl)
    return patched


def patched_numel(groups: list[str] | tuple[str, ...], registry: list[ParamSlice]) -> int:
    selected = set(groups)
    return int(sum(sl.numel for sl in registry if slice_selected(sl, selected)))


def change_norm(
    patched: dict[str, torch.Tensor], theta_ab: dict[str, torch.Tensor]
) -> float:
    total = 0.0
    for key, value in patched.items():
        if not torch.is_floating_point(value):
            continue
        delta = value.detach() - theta_ab[key].detach()
        total += float(torch.sum(delta * delta).item())
    return total ** 0.5


def default_patch_specs(n_layers: int, *, mlp_layer: int = 0) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [{"name": "empty", "groups": []}]
    specs.append({"name": "digit_embedding", "groups": ["digit_embedding"]})
    specs.append({"name": "control_embedding", "groups": ["control_embedding"]})
    for layer in range(int(n_layers)):
        specs.append(
            {"name": attention_layer_group(layer), "groups": [attention_layer_group(layer)]}
        )
    for layer in range(int(n_layers)):
        specs.append({"name": mlp_layer_group(layer), "groups": [mlp_layer_group(layer)]})
    specs.append({"name": "output", "groups": ["output"]})
    specs.append(
        {
            "name": f"digit_embedding+{mlp_layer_group(mlp_layer)}",
            "groups": ["digit_embedding", mlp_layer_group(mlp_layer)],
        }
    )
    specs.append({"name": "all_mlp", "groups": ["mlp"]})
    specs.append({"name": "all_attention", "groups": ["attention"]})
    specs.append({"name": "full_theta_A", "groups": list(COARSE_GROUPS)})
    return specs


def registry_equal(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
    registry: list[ParamSlice],
) -> bool:
    for sl in registry:
        a = left[sl.name]
        b = right[sl.name]
        if sl.rows is None:
            if not torch.equal(a, b):
                return False
        else:
            start, stop = sl.rows
            if not torch.equal(a[start:stop], b[start:stop]):
                return False
    return True


def unclassified_parameters(model: nn.Module, registry: list[ParamSlice]) -> dict[str, list[str]]:
    named = {sl.name for sl in registry}
    trainable = [
        name for name, param in model.named_parameters() if param.requires_grad and name not in named
    ]
    buffers = [name for name, _buf in model.named_buffers()]
    return {
        "unclassified_trainable": trainable,
        "buffers_not_patched": buffers,
    }


def assert_patch_sanity(
    model: ModularTransformer,
    theta_a: dict[str, torch.Tensor],
    theta_ab: dict[str, torch.Tensor],
) -> dict[str, Any]:
    registry = build_registry(model)
    assert_partition(model, registry)
    empty = apply_patch(theta_ab, theta_a, [], registry)
    if not registry_equal(empty, theta_ab, registry):
        raise RuntimeError("empty patch does not equal theta_AB")
    full = apply_patch(theta_ab, theta_a, list(COARSE_GROUPS), registry)
    if not registry_equal(full, theta_a, registry):
        raise RuntimeError("full patch does not equal theta_A on registry parameters")
    report = unclassified_parameters(model, registry)
    if report["unclassified_trainable"]:
        raise RuntimeError(
            "trainable parameters are unclassified: "
            + ", ".join(report["unclassified_trainable"])
        )
    report["empty_equals_theta_ab"] = True
    report["full_equals_theta_a"] = True
    report["partition_ok"] = True
    report["row_split_note"] = (
        "tok_emb rows are split: digit rows and control rows are different groups, "
        "and each row belongs to exactly one of them"
    )
    return report


def _flatten_ops(prefix: str, by_operation: dict[str, dict[str, Any]]) -> dict[str, float]:
    out = {}
    for key, rec in by_operation.items():
        out[f"{prefix}/{key}"] = float(rec.get("accuracy", float("nan")))
    return out


def evaluate_patches(
    model: ModularTransformer,
    theta_a: dict[str, torch.Tensor],
    theta_ab: dict[str, torch.Tensor],
    loader_a,
    loader_b,
    *,
    device: torch.device | None = None,
    mlp_layer: int = 0,
) -> list[dict[str, Any]]:
    """Evaluate every catalog patch. Each one reloads a clone, so order cannot leak."""
    from go4cl.metrics.behavioral import evaluate

    device = device or torch.device("cpu")
    registry = build_registry(model)
    total = int(sum(sl.numel for sl in registry))
    sanity = assert_patch_sanity(model, theta_a, theta_ab)
    base_ab = apply_patch(theta_ab, theta_a, [], registry)
    model.load_state_dict(base_ab, strict=True)
    base_a = evaluate(model, loader_a, device)
    base_b = evaluate(model, loader_b, device)
    rows = []
    for spec in default_patch_specs(len(model.blocks), mlp_layer=mlp_layer):
        patched = apply_patch(theta_ab, theta_a, spec["groups"], registry)
        model.load_state_dict(clone_state_dict(patched), strict=True)
        ev_a = evaluate(model, loader_a, device)
        ev_b = evaluate(model, loader_b, device)
        n_patched = patched_numel(spec["groups"], registry)
        row = {
            "patch": spec["name"],
            "groups": ",".join(spec["groups"]),
            "A_acc": ev_a.accuracy,
            "B_acc": ev_b.accuracy,
            "A_recovery": ev_a.accuracy - base_a.accuracy,
            "B_loss": base_b.accuracy - ev_b.accuracy,
            "param_change_norm": change_norm(patched, theta_ab),
            "n_patched": n_patched,
            "frac_patched": n_patched / total if total else 0.0,
            "empty_equals_theta_ab": sanity["empty_equals_theta_ab"],
            "full_equals_theta_a": sanity["full_equals_theta_a"],
            "unclassified_trainable": ",".join(sanity["unclassified_trainable"]),
            "buffers_not_patched": ",".join(sanity["buffers_not_patched"]),
        }
        row.update(_flatten_ops("A_op", ev_a.by_operation))
        row.update(_flatten_ops("B_op", ev_b.by_operation))
        rows.append(row)
    return rows


def write_patch_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def smoke_param_patch(out_dir: Path) -> dict[str, Any]:
    """CPU sanity on a tiny random model. Does not read or write training checkpoints."""
    from go4cl.constants import CONTEXT_LENGTH
    from go4cl.model.transformer import ModelConfig

    torch.manual_seed(0)
    model = ModularTransformer(ModelConfig(n_layers=2, d_model=32, n_heads=4, d_mlp=64))
    theta_a = clone_state_dict(model.state_dict())
    with torch.no_grad():
        for value in model.parameters():
            value.add_(0.1)
    theta_ab = clone_state_dict(model.state_dict())
    tokens = torch.randint(0, 10, (4, CONTEXT_LENGTH))
    labels = torch.randint(0, 5, (4,))
    batch = {
        "tokens": tokens,
        "labels": labels,
        "slots": torch.tensor([0, 1, 2, 3]),
        "moduli": torch.tensor([23, 23, 41, 41]),
        "task_ids": torch.tensor([0, 0, 1, 1]),
        "latent_ids": torch.tensor([0, 1, 0, 1]),
    }
    rows = evaluate_patches(model, theta_a, theta_ab, [batch], [batch], device=torch.device("cpu"))
    out_dir.mkdir(parents=True, exist_ok=True)
    write_patch_csv(out_dir / "param_patch.csv", rows)
    (out_dir / "sanity.json").write_text(
        json.dumps({"n_patches": len(rows), "formal_gpu_started": False}, indent=2) + "\n",
        encoding="utf-8",
    )
    return {"n_patches": len(rows), "out_dir": str(out_dir)}
