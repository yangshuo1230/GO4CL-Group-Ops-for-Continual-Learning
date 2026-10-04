"""Thin Weights & Biases helpers."""

from __future__ import annotations

from typing import Any


def init_wandb(
    *,
    enabled: bool,
    project: str,
    name: str,
    config: dict[str, Any] | None = None,
    dir: str | None = None,
    mode: str | None = None,
    group: str | None = None,
    tags: list[str] | None = None,
) -> Any | None:
    if not enabled:
        return None
    import wandb

    kwargs: dict[str, Any] = {
        "project": project,
        "name": name,
        "config": config or {},
        "reinit": True,
    }
    if dir is not None:
        kwargs["dir"] = dir
    if mode is not None:
        kwargs["mode"] = mode
    if group is not None:
        kwargs["group"] = group
    if tags is not None:
        kwargs["tags"] = tags
    return wandb.init(**kwargs)


def define_train_metrics() -> None:
    """Optional metric grouping. X-axis is W&B's built-in log step (not a metric)."""
    import wandb

    if wandb.run is None:
        return
    # No custom step_metric — avoid logging a redundant ``step`` series/chart.
    wandb.define_metric("train_*")
    wandb.define_metric("A_val_*")
    wandb.define_metric("B_val_*")
    wandb.define_metric("A_val_acc/*")
    wandb.define_metric("A_test_acc/*")
    wandb.define_metric("B_val_acc/*")
    wandb.define_metric("B_test_acc/*")
    wandb.define_metric("best/*")
    wandb.define_metric("final/*")
    wandb.define_metric("charts/*")


def modulus_acc_metrics(
    by_modulus: dict[int, dict[str, float]],
    *,
    prefix: str,
) -> dict[str, float]:
    """Flatten EvalResult.by_modulus into ``{prefix}/p{m}`` scalars for W&B."""
    return {
        f"{prefix}/p{int(m)}": float(stats["accuracy"])
        for m, stats in sorted(by_modulus.items())
    }


def slot_acc_metrics(
    by_slot: dict[int, dict[str, float]],
    *,
    prefix: str,
) -> dict[str, float]:
    """Flatten EvalResult.by_slot into ``{prefix}/s{slot}`` scalars."""
    return {
        f"{prefix}/s{int(s)}": float(stats["accuracy"])
        for s, stats in sorted(by_slot.items())
    }


def line_series_by_modulus(
    history: dict[int, list[tuple[int, float]]],
    *,
    title: str,
    xname: str = "step",
) -> Any | None:
    """Multi-line W&B chart: one series per modulus."""
    if not history:
        return None
    import wandb

    keys = sorted(history)
    xs = [[t for t, _ in history[m]] for m in keys]
    ys = [[a for _, a in history[m]] for m in keys]
    return wandb.plot.line_series(
        xs=xs,
        ys=ys,
        keys=[f"p{m}" for m in keys],
        title=title,
        xname=xname,
    )


def _is_wandb_media(value: Any) -> bool:
    mod = type(value).__module__
    return isinstance(mod, str) and mod.startswith("wandb")


def _wandb_visible(key: str) -> bool:
    """Drop diagnostic series that flood the W&B panel.

    Per-modulus accuracy stays as ``*/p{m}`` scalars. The repeated
    ``charts/val_acc_by_modulus`` table, nuisance controls, logit margin,
    and normalized CE are kept in local metrics only.
    """
    if key.startswith("charts/"):
        return False
    if "nuisance" in key:
        return False
    if key.endswith("_margin") or "/margin" in key:
        return False
    if key.endswith("_nce") or "/nce" in key:
        return False
    return True


def log_wandb(metrics: dict[str, Any], *, step: int | None = None) -> None:
    import wandb

    if wandb.run is None:
        return
    if step is None and "step" in metrics and isinstance(metrics["step"], (int, float)):
        step = int(metrics["step"])
    scalars = {
        k: v
        for k, v in metrics.items()
        if k != "step" and isinstance(v, (int, float)) and _wandb_visible(k)
    }
    media = {
        k: v
        for k, v in metrics.items()
        if k != "step" and k not in scalars and _is_wandb_media(v) and _wandb_visible(k)
    }
    payload = {**scalars, **media}
    if not payload:
        return
    wandb.log(payload, step=step)
    # Keep latest scalar values visible as run-table columns.
    if wandb.run is not None:
        for k, v in scalars.items():
            wandb.run.summary[k] = v


def finish_wandb() -> None:
    import wandb

    if wandb.run is not None:
        wandb.finish()
