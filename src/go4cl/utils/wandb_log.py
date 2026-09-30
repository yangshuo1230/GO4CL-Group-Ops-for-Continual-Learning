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
    wandb.define_metric("final/*")


def log_wandb(metrics: dict[str, Any], *, step: int | None = None) -> None:
    import wandb

    if wandb.run is None:
        return
    if step is None and "step" in metrics and isinstance(metrics["step"], (int, float)):
        step = int(metrics["step"])
    payload = {
        k: v
        for k, v in metrics.items()
        if k != "step" and isinstance(v, (int, float))
    }
    if not payload:
        return
    wandb.log(payload, step=step)
    # Keep latest values visible as run-table columns.
    if wandb.run is not None:
        for k, v in payload.items():
            wandb.run.summary[k] = v


def finish_wandb() -> None:
    import wandb

    if wandb.run is not None:
        wandb.finish()
