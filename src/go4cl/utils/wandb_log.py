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
    """Make W&B charts share a common step axis.

    W&B only allows glob *suffixes* (e.g. ``train_*``), not mid-string globs
    like ``*_val_*``.
    """
    import wandb

    if wandb.run is None:
        return
    wandb.define_metric("step")
    wandb.define_metric("train_*", step_metric="step")
    wandb.define_metric("A_val_*", step_metric="step")
    wandb.define_metric("B_val_*", step_metric="step")
    wandb.define_metric("final/*", step_metric="step")


def log_wandb(metrics: dict[str, Any], *, step: int | None = None) -> None:
    import wandb

    if wandb.run is None:
        return
    payload = {k: v for k, v in metrics.items() if isinstance(v, (int, float))}
    if not payload:
        return
    if step is None and "step" in metrics:
        step = int(metrics["step"])
    if step is not None:
        payload.setdefault("step", int(step))
    wandb.log(payload, step=step)
    # Keep latest values visible as run-table columns.
    if wandb.run is not None:
        for k, v in payload.items():
            if k != "step":
                wandb.run.summary[k] = v


def finish_wandb() -> None:
    import wandb

    if wandb.run is not None:
        wandb.finish()
