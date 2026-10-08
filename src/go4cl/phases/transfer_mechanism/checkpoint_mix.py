"""Hybrid checkpoints for necessity (reset-one) and sufficiency (keep-only).

``reset_*`` starts from theta_A and writes the named group from theta_0.
``keep_*`` starts from theta_0 and writes the named group from theta_A.
Inputs are cloned; theta_0 and theta_A tensors are not written in place.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from go4cl.model.transformer import ModularTransformer
from go4cl.phases.transfer_mechanism import PARAMETER_GROUP_SCHEMA_VERSION
from go4cl.phases.transfer_mechanism.parameter_groups import (
    COARSE_GROUPS,
    ParamSlice,
    build_registry,
    known_group_names,
    logical_units,
    slice_selected,
    unit_selected,
)
from go4cl.train.loop import TrainConfig, build_optimizer

REPLAY_ERROR = (
    "Checkpoint-mixing source localization must run without replay, "
    "because replay can relearn reset A components during B training."
)

DEFAULT_INTERVENTIONS: tuple[str, ...] = (
    "full_A",
    "reset_digit_embedding",
    "reset_control_embedding",
    "reset_attention",
    "reset_mlp",
    "reset_output",
    "full_fresh",
    "keep_digit_embedding",
    "keep_control_embedding",
    "keep_attention",
    "keep_mlp",
    "keep_output",
)

KEEP_MULTIPLE: dict[str, tuple[str, ...]] = {
    "keep_digit_embedding+mlp": ("digit_embedding", "mlp"),
    "keep_digit_embedding+output": ("digit_embedding", "output"),
    "keep_digit_embedding+mlp+output": ("digit_embedding", "mlp", "output"),
}


def reject_nonzero_replay(replay_ratio: float) -> None:
    if float(replay_ratio) != 0.0:
        raise ValueError(REPLAY_ERROR)


def reserved_interventions(n_layers: int = 3) -> tuple[str, ...]:
    """Named for later grids. This round does not launch them."""
    names = [f"reset_attention_layer_{i}" for i in range(n_layers)]
    names += [f"reset_mlp_layer_{i}" for i in range(n_layers)]
    names += list(KEEP_MULTIPLE)
    return tuple(names)


@dataclass(frozen=True)
class InterventionSpec:
    name: str
    kind: str  # full_A | full_fresh | reset | keep
    base: str  # theta_A | theta_0
    donor: str | None
    donor_groups: tuple[str, ...]


def parse_intervention(name: str, *, n_layers: int) -> InterventionSpec:
    if name == "full_A":
        return InterventionSpec(name, "full_A", "theta_A", None, ())
    if name == "full_fresh":
        return InterventionSpec(name, "full_fresh", "theta_0", None, ())
    if name in KEEP_MULTIPLE:
        groups = KEEP_MULTIPLE[name]
        _validate_groups(groups, n_layers=n_layers)
        return InterventionSpec(name, "keep", "theta_0", "theta_A", groups)
    if name.startswith("reset_"):
        groups = (name[len("reset_") :],)
        _validate_groups(groups, n_layers=n_layers)
        return InterventionSpec(name, "reset", "theta_A", "theta_0", groups)
    if name.startswith("keep_"):
        groups = (name[len("keep_") :],)
        _validate_groups(groups, n_layers=n_layers)
        return InterventionSpec(name, "keep", "theta_0", "theta_A", groups)
    raise ValueError(
        f"unknown intervention {name!r}. Expected full_A, full_fresh, "
        "reset_<group>, keep_<group>, or a keep_multiple preset."
    )


def _validate_groups(groups: tuple[str, ...], *, n_layers: int) -> None:
    known = known_group_names(n_layers)
    unknown = [g for g in groups if g not in known]
    if unknown:
        raise ValueError(
            f"unknown parameter groups {unknown}. Known for n_layers={n_layers}: "
            f"{sorted(known)}"
        )
    if len(set(groups)) != len(groups):
        raise ValueError(f"duplicate groups in intervention: {groups}")


def origin_labels(spec: InterventionSpec, n_layers: int) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return ``(groups_from_theta_0, groups_from_theta_A)`` at the right granularity."""
    selected = set(spec.donor_groups)
    units = logical_units(selected, n_layers)
    if spec.kind == "full_A":
        return (), tuple(units)
    if spec.kind == "full_fresh":
        return tuple(units), ()
    from_init: list[str] = []
    from_a: list[str] = []
    donor_is_init = spec.donor == "theta_0"
    for unit in units:
        taken = unit_selected(unit, selected)
        comes_from_donor = taken
        if comes_from_donor and donor_is_init:
            from_init.append(unit)
        elif comes_from_donor and not donor_is_init:
            from_a.append(unit)
        elif donor_is_init:
            from_a.append(unit)
        else:
            from_init.append(unit)
    return tuple(from_init), tuple(from_a)


def clone_state_dict(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.detach().clone() for key, value in state.items()}


def state_dict_from_model(model: ModularTransformer) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def save_theta_checkpoint(
    path: Path | str,
    model: ModularTransformer,
    *,
    task_pair: dict[str, Any],
    dataset_hash: str,
    task_seed: int,
    model_seed: int,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    carry_optimizer_state: bool = False,
) -> None:
    """Write theta_0 / theta_A. Optimizer state is omitted unless carry is set."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    canonical = state_dict_from_model(model)
    payload: dict[str, Any] = {
        "model_state_dict": canonical,
        "model_state": clone_state_dict(canonical),
        "model_config": model.cfg.to_dict(),
        "task_pair": task_pair,
        "dataset_hash": dataset_hash,
        "task_seed": int(task_seed),
        "model_seed": int(model_seed),
        "parameter_group_schema_version": PARAMETER_GROUP_SCHEMA_VERSION,
        "step": int(step),
        "carry_optimizer_state": bool(carry_optimizer_state),
    }
    if carry_optimizer_state:
        if optimizer is None:
            raise ValueError("carry_optimizer_state=true requires an optimizer to save")
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_theta_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Any], dict[str, torch.Tensor]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    if "model_state_dict" in payload:
        raw = payload["model_state_dict"]
    elif "model_state" in payload:
        raw = payload["model_state"]
    else:
        raise KeyError(f"{path} has neither model_state_dict nor model_state")
    state = {key: value.detach().clone() for key, value in raw.items()}
    return payload, state


def optimizer_has_adam_moments(optimizer: torch.optim.Optimizer) -> bool:
    for state in optimizer.state.values():
        if "exp_avg" in state or "exp_avg_sq" in state:
            return True
    return False


def build_b_optimizer(
    model: ModularTransformer,
    cfg: TrainConfig,
    *,
    carry_optimizer_state: bool,
    payload: dict[str, Any] | None,
) -> torch.optim.Optimizer:
    """B-phase AdamW. Default does not load A moments or step-dependent state.

    This codebase has no LR scheduler. A fresh optimizer plus B-local step 0
    is what ``carry_optimizer_state=false`` means.
    """
    opt = build_optimizer(model, cfg)
    if not carry_optimizer_state:
        if optimizer_has_adam_moments(opt):
            raise RuntimeError("fresh B optimizer already contains Adam moments")
        return opt
    if not payload or "optimizer_state" not in payload:
        raise ValueError(
            "carry_optimizer_state=true but the source checkpoint has no optimizer_state"
        )
    opt.load_state_dict(payload["optimizer_state"])
    return opt


def build_hybrid(
    theta_0: dict[str, torch.Tensor],
    theta_A: dict[str, torch.Tensor],
    intervention: str,
    registry: list[ParamSlice],
    *,
    n_layers: int,
    task_seed: int,
    model_seed: int,
    dataset_hash: str,
) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
    """Return a new state dict and a hybrid manifest. Inputs are not mutated."""
    spec = parse_intervention(intervention, n_layers=n_layers)
    _require_compatible(theta_0, theta_A)
    base_src = theta_A if spec.base == "theta_A" else theta_0
    donor_src = None
    if spec.donor == "theta_0":
        donor_src = theta_0
    elif spec.donor == "theta_A":
        donor_src = theta_A
    hybrid = clone_state_dict(base_src)
    selected = set(spec.donor_groups)
    if donor_src is not None and selected:
        for sl in registry:
            if not slice_selected(sl, selected):
                continue
            _copy_slice(hybrid, donor_src, sl)
    manifest = _manifest(
        hybrid,
        base_src,
        spec,
        registry,
        n_layers=n_layers,
        task_seed=task_seed,
        model_seed=model_seed,
        dataset_hash=dataset_hash,
    )
    _validate_hybrid(hybrid, theta_0, theta_A, spec, registry)
    return hybrid, manifest


def save_hybrid_checkpoint(
    path: Path | str,
    state: dict[str, torch.Tensor],
    *,
    model_config: dict[str, Any],
    manifest: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cpu = {key: value.detach().cpu().clone() for key, value in state.items()}
    torch.save(
        {
            "model_state_dict": cpu,
            "model_state": clone_state_dict(cpu),
            "model_config": model_config,
            "hybrid_manifest": manifest,
            "task_seed": manifest.get("task_seed"),
            "model_seed": manifest.get("model_seed"),
            "dataset_hash": manifest.get("dataset_hash"),
            "parameter_group_schema_version": PARAMETER_GROUP_SCHEMA_VERSION,
        },
        path,
    )


def load_hybrid_into(
    model: ModularTransformer, state: dict[str, torch.Tensor]
) -> ModularTransformer:
    incompatible = model.load_state_dict(state, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "hybrid state did not load strictly: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    return model


def _copy_slice(
    dst: dict[str, torch.Tensor],
    src: dict[str, torch.Tensor],
    sl: ParamSlice,
) -> None:
    if sl.rows is None:
        dst[sl.name] = src[sl.name].detach().clone()
        return
    start, stop = sl.rows
    # Index assignment writes into the cloned destination only.
    dst[sl.name][start:stop] = src[sl.name][start:stop]


def _slice_view(tensor: torch.Tensor, sl: ParamSlice) -> torch.Tensor:
    if sl.rows is None:
        return tensor
    start, stop = sl.rows
    return tensor[start:stop]


def _require_compatible(
    theta_0: dict[str, torch.Tensor], theta_A: dict[str, torch.Tensor]
) -> None:
    if set(theta_0) != set(theta_A):
        raise ValueError(
            "theta_0 and theta_A keys differ: "
            f"only_0={sorted(set(theta_0) - set(theta_A))[:8]} "
            f"only_A={sorted(set(theta_A) - set(theta_0))[:8]}"
        )
    for key in theta_0:
        a = theta_0[key]
        b = theta_A[key]
        if a.shape != b.shape or a.dtype != b.dtype or a.device != b.device:
            raise ValueError(
                f"{key} mismatch shape/dtype/device "
                f"{tuple(a.shape)} {a.dtype} {a.device} vs "
                f"{tuple(b.shape)} {b.dtype} {b.device}"
            )


def _validate_hybrid(
    hybrid: dict[str, torch.Tensor],
    theta_0: dict[str, torch.Tensor],
    theta_A: dict[str, torch.Tensor],
    spec: InterventionSpec,
    registry: list[ParamSlice],
) -> None:
    _require_compatible(hybrid, theta_A)
    _require_compatible(hybrid, theta_0)
    selected = set(spec.donor_groups)
    base = theta_A if spec.base == "theta_A" else theta_0
    donor = theta_0 if spec.donor == "theta_0" else theta_A
    param_names = {sl.name for sl in registry}
    for key in hybrid:
        if key in param_names:
            continue
        if not torch.equal(hybrid[key], base[key]):
            raise RuntimeError(f"buffer {key} changed relative to the base checkpoint")
    for sl in registry:
        got = _slice_view(hybrid[sl.name], sl)
        if slice_selected(sl, selected):
            if donor is None:
                raise RuntimeError(f"{spec.name} selects {sl.group} but has no donor")
            expected = _slice_view(donor[sl.name], sl)
            role = "donor"
        else:
            expected = _slice_view(base[sl.name], sl)
            role = "base"
        if got.shape != expected.shape or got.dtype != expected.dtype or got.device != expected.device:
            raise RuntimeError(f"{sl.name} {sl.group} shape/dtype/device drifted")
        if not torch.equal(got, expected):
            raise RuntimeError(
                f"{spec.name}: {sl.name} group={sl.group} rows={sl.rows} "
                f"does not match {role}"
            )


def _manifest(
    hybrid: dict[str, torch.Tensor],
    base: dict[str, torch.Tensor],
    spec: InterventionSpec,
    registry: list[ParamSlice],
    *,
    n_layers: int,
    task_seed: int,
    model_seed: int,
    dataset_hash: str,
) -> dict[str, Any]:
    changed_names: list[str] = []
    changed = 0
    total = 0
    seen: set[str] = set()
    for sl in registry:
        if sl.name in seen:
            continue
        seen.add(sl.name)
        diff = hybrid[sl.name] != base[sl.name]
        n = int(diff.sum().item())
        total += int(diff.numel())
        changed += n
        if n:
            changed_names.append(sl.name)
    from_init, from_a = origin_labels(spec, n_layers)
    fraction = (changed / total) if total else 0.0
    return {
        "intervention": spec.name,
        "kind": spec.kind,
        "base_checkpoint": spec.base,
        "donor_checkpoint": spec.donor,
        "reset_groups": list(from_init),
        "kept_groups": list(from_a),
        "donor_groups": list(spec.donor_groups),
        "changed_parameter_names": changed_names,
        "changed_numel": changed,
        "changed_fraction": fraction,
        "task_seed": int(task_seed),
        "model_seed": int(model_seed),
        "dataset_hash": dataset_hash,
        "parameter_group_schema_version": PARAMETER_GROUP_SCHEMA_VERSION,
        "coarse_groups": list(COARSE_GROUPS),
    }


def registry_for_model(model: ModularTransformer) -> list[ParamSlice]:
    return build_registry(model)
