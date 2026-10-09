"""High-level training protocols from the experiment plan.

``run_protocol`` initializes the model and loaders, dispatches to a named
implementation, then writes metrics / history / config_resolved.json.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from go4cl.defaults import PHASE2
from go4cl.metrics.continual import summarize_behavior
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.train.loaders import task_loaders
from go4cl.train.loop import TrainConfig
from go4cl.train.protocol_common import ProtocolName, ProtocolResult, ProtocolSession, write_history
from go4cl.train.protocol_impl import (
    canonical_optimizer_transition,
    run_continued_control,
    run_interleaved,
    run_joint,
    run_sequential_ab,
    run_sequential_ba,
    run_single_task,
)

NULL_TASK_UNDEFINED = (
    "null_task_tokens is undefined for replay and interleaved. Replay builds "
    "its own loader and bypasses the null-task wrapper, and interleaved can "
    "treat TASK_B as both a negative and a real task. Defined combinations "
    "are sequential A→B with no replay, and joint negatives."
)


def assert_null_task_protocol(
    protocol: str,
    null_task_tokens: bool,
    replay_ratio: float = 0.0,
) -> None:
    """Reject combinations whose negative-task sampling is not defined."""
    if not null_task_tokens:
        return
    if protocol in {"interleaved", "sequential_ab_replay"} or float(replay_ratio) > 0:
        raise ValueError(NULL_TASK_UNDEFINED)
from go4cl.utils.checkpoint import save_checkpoint, write_json
from go4cl.utils.seed import seed_everything
from go4cl.utils.wandb_log import (
    define_train_metrics,
    finish_wandb,
    init_wandb,
    log_wandb,
)

# Re-export names used by tests / older callers.
from go4cl.train.loaders import _task_loaders  # noqa: F401
from go4cl.train.protocol_common import (  # noqa: F401
    _eval_bundle,
    _flatten_block,
    _switch_stop,
    _write_history,
)


def run_protocol(
    protocol: ProtocolName,
    data_root: Path | str,
    out_dir: Path | str,
    *,
    model_cfg: ModelConfig | None = None,
    train_cfg: TrainConfig | None = None,
    model_seed: int = 0,
    sampler_seed: int | None = None,
    phase_steps: int | None = None,
    wandb_enabled: bool = True,
    wandb_project: str = "go4cl",
    wandb_name: str | None = None,
    wandb_mode: str | None = None,
    wandb_config: dict[str, Any] | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    include_test: bool = False,
    switch_on: str = "fixed",
    eval_n_per_operation: int = 256,
    theta_a_ckpt: str | None = None,
    optimizer_transition: str = "preserve",
    a_mastery: dict | None = None,
    replay_coverage: dict | None = None,
) -> ProtocolResult:
    """
    Run one of the plan's training protocols on a fixed dataset root.

    Metrics stream to W&B when ``wandb_enabled`` is true (default).
    ``sampler_seed`` controls online/minibatch streams; defaults to ``model_seed``
    only for backward compatibility — prefer setting it explicitly.
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ModelConfig()
    steps = phase_steps or train_cfg.max_steps
    resolved_sampler_seed = int(sampler_seed if sampler_seed is not None else model_seed)
    optimizer_mode = canonical_optimizer_transition(optimizer_transition)
    replay_ratio = (
        float(PHASE2.sequential_ab_replay_ratio)
        if protocol == "sequential_ab_replay"
        else 0.0
    )
    assert_null_task_protocol(
        protocol,
        bool(train_cfg.null_task_tokens),
        replay_ratio=replay_ratio,
    )

    seed_everything(model_seed)
    device = torch.device(train_cfg.device)
    model = ModularTransformer(model_cfg).to(device)
    loaders = task_loaders(
        data_root,
        batch_size=train_cfg.batch_size,
        train_replacement=bool(train_cfg.train_replacement),
        train_seed=model_seed,
        sampler_seed=resolved_sampler_seed,
        eval_n_per_operation=int(eval_n_per_operation),
    )
    packed_a = getattr(loaders["A"]["train"], "n_packs", None)
    packed_b = getattr(loaders["B"]["train"], "n_packs", None)
    if train_cfg.null_task_tokens:
        from go4cl.data.null_task import wrap_null_task_loader

        loaders["A"]["train"] = wrap_null_task_loader(
            loaders["A"]["train"],
            phase="a",
            ratio=float(train_cfg.null_task_ratio),
            label=int(train_cfg.null_task_label),
            seed=resolved_sampler_seed + 901,
            enabled=True,
        )
        loaders["B"]["train"] = wrap_null_task_loader(
            loaders["B"]["train"],
            phase="b",
            ratio=float(train_cfg.null_task_ratio),
            label=int(train_cfg.null_task_label),
            seed=resolved_sampler_seed + 902,
            enabled=True,
        )
    effective_bs = {
        "A_train": loaders["A"]["train"].batch_size,
        "B_train": loaders["B"]["train"].batch_size,
        "A_train_n": len(loaders["A"]["train"].dataset),  # type: ignore[arg-type]
        "B_train_n": len(loaders["B"]["train"].dataset),  # type: ignore[arg-type]
        "train_replacement": bool(train_cfg.train_replacement),
        "train_mode": "packed_online" if packed_a is not None else "fixed",
        "n_packs_per_step_A": packed_a,
        "n_packs_per_step_B": packed_b,
    }

    run_name = wandb_name or f"{protocol}_ms{model_seed}"
    cfg_payload = {
        "protocol": protocol,
        "model_seed": model_seed,
        "sampler_seed": resolved_sampler_seed,
        "phase_steps": steps,
        "include_test": bool(include_test),
        "switch_on": switch_on,
        "data_root": str(data_root),
        "out_dir": str(out_dir),
        "model": model_cfg.to_dict(),
        "train": {
            "lr": train_cfg.lr,
            "weight_decay": train_cfg.weight_decay,
            "batch_size": train_cfg.batch_size,
            "full_batch": train_cfg.batch_size is None or train_cfg.batch_size <= 0,
            "train_replacement": bool(train_cfg.train_replacement),
            "effective_batch": effective_bs,
            "eval_every": train_cfg.eval_every,
            "device": train_cfg.device,
            "null_task_tokens": bool(train_cfg.null_task_tokens),
            "null_task_ratio": float(train_cfg.null_task_ratio),
            "null_task_label": int(train_cfg.null_task_label),
            "replay_ratio": replay_ratio,
            "theta_a_ckpt": theta_a_ckpt,
            "optimizer_transition": optimizer_mode,
        },
        **(wandb_config or {}),
    }
    metrics: dict[str, Any] = {
        "protocol": protocol,
        "model_seed": model_seed,
        "sampler_seed": resolved_sampler_seed,
        "optimizer_transition": optimizer_mode,
    }
    wb = init_wandb(
        enabled=wandb_enabled,
        project=wandb_project,
        name=run_name,
        config=cfg_payload,
        dir=str(out_dir),
        mode=wandb_mode,
        group=wandb_group,
        tags=wandb_tags,
    )
    wandb_url = getattr(wb, "url", None) if wb is not None else None
    if wb is not None:
        define_train_metrics()

    session = ProtocolSession(
        protocol=protocol,
        model=model,
        loaders=loaders,
        device=device,
        train_cfg=train_cfg,
        out_dir=out_dir,
        data_root=data_root,
        steps=steps,
        metrics=metrics,
        include_test=bool(include_test),
        switch_on=switch_on,
        packed_a=packed_a,
        resolved_sampler_seed=resolved_sampler_seed,
        replay_coverage=replay_coverage,
    )
    metrics["switch_on"] = switch_on

    try:
        if protocol == "a_only":
            run_single_task(session, task="A")
        elif protocol == "b_only":
            run_single_task(session, task="B")
        elif protocol == "joint":
            run_joint(session)
        elif protocol == "sequential_ab":
            run_sequential_ab(
                session,
                theta_a_ckpt=theta_a_ckpt,
                optimizer_transition=optimizer_mode,
                a_mastery=a_mastery,
            )
        elif protocol == "sequential_ab_replay":
            run_sequential_ab(
                session,
                replay_ratio=replay_ratio,
                theta_a_ckpt=theta_a_ckpt,
                optimizer_transition=optimizer_mode,
            )
            if replay_coverage:
                from go4cl.phases.phase2.replay_coverage import attach_coverage_metrics

                attach_coverage_metrics(session, replay_coverage)
        elif protocol == "a_only_continued":
            run_continued_control(session)
        elif protocol == "sequential_ba":
            run_sequential_ba(session)
        elif protocol == "interleaved":
            run_interleaved(session)
        else:
            raise ValueError(f"unknown protocol: {protocol}")

        flat_final: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                flat_final[f"final/{k}"] = v
            elif isinstance(v, dict):
                for kk, vv in v.items():
                    if isinstance(vv, (int, float)):
                        flat_final[f"final/{k}/{kk}"] = vv
        if flat_final:
            log_wandb(flat_final, step=session.final_step)

    finally:
        behavior = summarize_behavior(
            protocol,
            session.history,
            switch_step=metrics.get("switch_step"),
        )
        # A shared theta_A checkpoint may not appear in session.history.
        # History-only retention can therefore look high after A has already
        # collapsed. The explicit switch endpoints are authoritative.
        if metrics.get("a_mastery_valid") is False:
            behavior["retention_A"] = None
            behavior["retention_note"] = (
                "invalid A mastery; retention_A_from_switch was not computed"
            )
        elif metrics.get("retention_A_from_switch") is not None:
            behavior["retention_A"] = metrics["retention_A_from_switch"]
        metrics["behavior"] = behavior
        if "forgetting_A" in behavior:
            metrics["forgetting_A"] = behavior["forgetting_A"]
        elif "forgetting_A_from_switch" in metrics:
            metrics["forgetting_A"] = metrics["forgetting_A_from_switch"]
        if "forgetting_B" in behavior:
            metrics["forgetting_B"] = behavior["forgetting_B"]
        elif "forgetting_B_from_switch" in metrics:
            metrics["forgetting_B"] = metrics["forgetting_B_from_switch"]
        write_history(out_dir / "eval_history.jsonl", session.history)
        behavior_log = {
            f"behavior/{k}": v
            for k, v in behavior.items()
            if isinstance(v, (int, float))
        }
        if behavior_log:
            log_wandb(behavior_log, step=session.final_step)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "config_resolved.json", cfg_payload)
        finish_wandb()

    save_checkpoint(out_dir / "ckpts" / "final.pt", model, step=session.final_step)
    return ProtocolResult(protocol=protocol, metrics=metrics, wandb_url=wandb_url)
