"""Named training-protocol implementations.

Sequential A→B continues the global step counter across the task switch.
``optimizer_transition`` is ``preserve`` (keep AdamW moments; the historical
Phase 2 default) or ``fresh`` (load θ_A weights and build a new AdamW).
"""

from __future__ import annotations

from go4cl.data.dataset import (
    ModularAdditionDataset,
    make_balanced_joint_loader,
)
from go4cl.data.manifest import DataManifest
from go4cl.data.packed import AlternatingTaskLoader, BalancedPackedJointLoader, MixedPackedReplayLoader
from go4cl.metrics.behavioral import forgetting
from go4cl.train.loop import TrainConfig, build_optimizer, train_segment, train_steps
from go4cl.train.protocol_common import ProtocolSession, flatten_block, switch_stop
from go4cl.utils.checkpoint import load_checkpoint, save_checkpoint
from go4cl.utils.wandb_log import log_wandb


def run_single_task(session: ProtocolSession, *, task: str) -> None:
    cfg = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
    run_name = "a_only" if task == "A" else "b_only"
    state = train_steps(
        session.model,
        session.loaders[task]["train"],
        cfg=cfg,
        eval_loaders=session.eval_loaders(task),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name=run_name,
    )
    session.absorb(state)
    session.final_step = state.step
    tag = "after_a" if task == "A" else "after_b"
    session.metrics.update(session.eval_both(tag))
    session.metrics["events"] = dict(state.events)
    if task == "A" and state.first_stable_ckpt_path:
        session.metrics["first_stable_ckpt"] = state.first_stable_ckpt_path
    session.attach_best_by_val(state, final_tag=run_name)


def run_joint(session: ProtocolSession) -> None:
    if session.packed_a is not None:
        manifest = DataManifest.load(session.data_root / "manifest.json")
        joint_loader = BalancedPackedJointLoader(
            manifest.task_pair.task_a,
            manifest.task_pair.task_b,
            manifest.residue_splits,
            batch_size=int(session.train_cfg.batch_size or 0),
            seed=session.resolved_sampler_seed,
        )
    else:
        ds_a = ModularAdditionDataset.from_disk(session.data_root, "A", "train", 0)
        ds_b = ModularAdditionDataset.from_disk(session.data_root, "B", "train", 1)
        joint_loader = make_balanced_joint_loader(
            ds_a, ds_b, batch_size=session.train_cfg.batch_size
        )
    if session.train_cfg.null_task_tokens:
        from go4cl.data.null_task import wrap_null_task_loader

        joint_loader = wrap_null_task_loader(
            joint_loader,
            phase="joint",
            ratio=float(session.train_cfg.null_task_ratio),
            label=int(session.train_cfg.null_task_label),
            seed=session.resolved_sampler_seed + 903,
            enabled=True,
        )
    cfg = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": 2 * session.steps})
    state = train_steps(
        session.model,
        joint_loader,
        cfg=cfg,
        eval_loaders=session.eval_loaders("A", "B"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="joint",
    )
    session.absorb(state)
    session.final_step = state.step
    session.metrics.update(session.eval_both("after_joint"))
    session.metrics["events"] = dict(state.events)
    session.attach_best_by_val(state, final_tag="joint")


def run_interleaved(session: ProtocolSession) -> None:
    mixed = AlternatingTaskLoader(
        session.loaders["A"]["train"], session.loaders["B"]["train"]
    )
    cfg = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": 2 * session.steps})
    state = train_steps(
        session.model,
        mixed,
        cfg=cfg,
        eval_loaders=session.eval_loaders("A", "B"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="interleaved",
    )
    session.absorb(state)
    session.final_step = state.step
    session.metrics.update(session.eval_both("after_interleaved"))
    session.metrics["events"] = dict(state.events)
    session.attach_best_by_val(state, final_tag="interleaved")


def _phase_b_loader(session: ProtocolSession, replay_ratio: float):
    """B-phase train stream. ``replay_ratio=0`` is B-only (plan sequential_ab)."""
    if replay_ratio <= 0:
        return session.loaders["B"]["train"]
    if session.packed_a is None:
        raise ValueError("sequential_ab_replay requires packed_online train loaders")
    manifest = DataManifest.load(session.data_root / "manifest.json")
    return MixedPackedReplayLoader(
        manifest.task_pair.task_a,
        manifest.task_pair.task_b,
        manifest.residue_splits,
        batch_size=int(session.train_cfg.batch_size or 0),
        seed=session.resolved_sampler_seed + 11_017,
        frac_a=float(replay_ratio),
    )


def canonical_optimizer_transition(name: str) -> str:
    """Public names are ``fresh`` and ``preserve``. ``reset`` means ``fresh``."""
    if name in {"fresh", "reset"}:
        return "fresh"
    if name == "preserve":
        return "preserve"
    raise ValueError(
        "optimizer_transition must be fresh or preserve "
        f"(reset is an alias of fresh), got {name}"
    )


def phase_b_optimizer(
    model,
    train_cfg: TrainConfig,
    *,
    transition: str,
    carried: torch.optim.Optimizer | None = None,
    payload: dict | None = None,
    checkpoint_path: str | None = None,
):
    """Build the AdamW that trains B.

    ``fresh`` ignores any carried moments. ``preserve`` restores them from
    ``carried`` or from ``payload['optimizer_state']``.
    """
    import torch

    mode = canonical_optimizer_transition(transition)
    if mode == "fresh":
        return build_optimizer(model, train_cfg), mode
    if carried is not None and (payload is None or payload.get("optimizer_state") is None):
        return carried, mode
    opt = build_optimizer(model, train_cfg)
    state = None if payload is None else payload.get("optimizer_state")
    if state is None and checkpoint_path:
        from pathlib import Path

        alt = Path(checkpoint_path).parent / "a_only_final.pt"
        if alt.is_file() and alt.resolve() != Path(checkpoint_path).resolve():
            alt_payload = torch.load(alt, map_location="cpu", weights_only=False)
            state = alt_payload.get("optimizer_state")
    if state is None:
        raise RuntimeError(
            "optimizer_transition=preserve requires the A checkpoint's AdamW "
            "state (moments and step). Refusing to start B with a new optimizer."
        )
    opt.load_state_dict(state)
    return opt, mode


def run_sequential_ab(
    session: ProtocolSession,
    *,
    replay_ratio: float = 0.0,
    theta_a_ckpt: str | None = None,
    optimizer_transition: str = "preserve",
) -> None:
    """A→B sequential. If ``theta_a_ckpt`` is set, skip phase A and load that checkpoint.

    The default ``preserve`` keeps the historical Phase 2 optimizer handoff.
    ``fresh`` loads θ_A weights and creates a new AdamW for B.
    """
    import shutil
    from pathlib import Path

    mode = canonical_optimizer_transition(optimizer_transition)
    opt_a = None
    start_step = 0
    if theta_a_ckpt:
        ckpt_path = Path(theta_a_ckpt)
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"theta_a_ckpt not found: {ckpt_path}")
        _model, payload = load_checkpoint(
            ckpt_path, model=session.model, map_location=session.device
        )
        start_step = int(payload.get("step", session.steps))
        cfg_opt = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
        opt_a, mode = phase_b_optimizer(
            session.model,
            cfg_opt,
            transition=mode,
            payload=payload,
            checkpoint_path=str(ckpt_path),
        )
        out_theta = session.out_dir / "ckpts" / "theta_A.pt"
        out_theta.parent.mkdir(parents=True, exist_ok=True)
        if ckpt_path.resolve() != out_theta.resolve():
            shutil.copy2(ckpt_path, out_theta)
        session.metrics["after_a"] = session.eval_both("after_a")
        session.metrics["switch_step"] = start_step
        session.metrics["events_phase_a"] = {"reused_theta_a": True}
        session.metrics["theta_a_ckpt"] = str(ckpt_path)
        session.metrics["A_test_acc_at_switch"] = session.metrics["after_a"]["A_test_acc"]
        session.metrics["A_val_acc_at_switch"] = session.metrics["after_a"]["A_val_acc"]
        log_wandb(
            {
                f"after_a/{k}": v
                for k, v in session.metrics["after_a"].items()
                if isinstance(v, (int, float))
            },
            step=start_step,
        )
    else:
        stop_first = switch_stop(session.switch_on)
        cfg_a = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
        seg_a = train_segment(
            session.model,
            session.loaders["A"]["train"],
            cfg=cfg_a,
            optimizer_transition="preserve",
            eval_loaders=session.eval_loaders("A"),
            ckpt_dir=str(session.out_dir / "ckpts"),
            run_name="phase_a",
            stop_fn=stop_first,
        )
        session.absorb(seg_a.state)
        start_step = seg_a.state.step
        opt_a = seg_a.optimizer
        save_checkpoint(
            session.out_dir / "ckpts" / "theta_A.pt",
            session.model,
            optimizer=opt_a,
            step=start_step,
            meta={
                "task_switch": "A_done",
                "role": "theta_A",
                "optimizer_transition": mode,
                "switch_on": session.switch_on,
            },
        )
        if mode == "fresh":
            opt_a = None
        session.metrics["after_a"] = session.eval_both("after_a")
        session.metrics["switch_step"] = start_step
        session.metrics["events_phase_a"] = dict(seg_a.state.events)
        session.metrics["A_test_acc_at_switch"] = session.metrics["after_a"]["A_test_acc"]
        session.metrics["A_val_acc_at_switch"] = session.metrics["after_a"]["A_val_acc"]
        log_wandb(
            {
                f"after_a/{k}": v
                for k, v in session.metrics["after_a"].items()
                if isinstance(v, (int, float))
            },
            step=start_step,
        )

    max_acc_a = session.metrics["after_a"]["A_test_acc"]
    loader_b = _phase_b_loader(session, replay_ratio)
    if replay_ratio > 0:
        session.metrics["replay_ratio"] = float(replay_ratio)
        session.metrics["replay_frac_a"] = float(getattr(loader_b, "frac_a", replay_ratio))
        session.metrics["replay_n_packs_a"] = int(getattr(loader_b, "n_packs_a", 0))
        session.metrics["replay_n_packs_b"] = int(getattr(loader_b, "n_packs_b", 0))
    cfg_b = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
    seg_b = train_segment(
        session.model,
        loader_b,
        cfg=cfg_b,
        optimizer=opt_a,
        start_step=start_step,
        optimizer_transition="fresh" if mode == "fresh" else "preserve",
        eval_loaders=session.eval_loaders("A", "B"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="phase_b",
    )
    session.absorb(seg_b.state)
    state_b = seg_b.state
    session.final_step = state_b.step
    session.metrics["after_b"] = session.eval_both("after_b")
    flatten_block(session.metrics, session.metrics["after_b"])
    session.metrics["forgetting_A_from_switch"] = forgetting(
        max_acc_a, session.metrics["after_b"]["A_test_acc"]
    )
    session.metrics["optimizer_transition"] = mode
    session.metrics["primary_event"] = "B_t_gen"
    session.metrics["events_phase_b"] = dict(state_b.events)
    session.attach_best_by_val(state_b, final_tag="phase_b")


def run_continued_control(session: ProtocolSession) -> None:
    stop_first = switch_stop(session.switch_on)
    cfg_a = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
    seg_a = train_segment(
        session.model,
        session.loaders["A"]["train"],
        cfg=cfg_a,
        optimizer_transition="preserve",
        eval_loaders=session.eval_loaders("A"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="phase_a",
        stop_fn=stop_first,
    )
    session.absorb(seg_a.state)
    state_a = seg_a.state
    save_checkpoint(
        session.out_dir / "ckpts" / "theta_A.pt",
        session.model,
        optimizer=seg_a.optimizer,
        step=state_a.step,
        meta={
            "task_switch": "A_done",
            "optimizer_transition": "preserve",
            "switch_on": session.switch_on,
        },
    )
    session.metrics["after_a"] = session.eval_both("after_a")
    session.metrics["switch_step"] = state_a.step
    session.metrics["events_phase_a"] = dict(state_a.events)
    session.metrics["A_test_acc_at_switch"] = session.metrics["after_a"]["A_test_acc"]
    session.metrics["A_val_acc_at_switch"] = session.metrics["after_a"]["A_val_acc"]
    log_wandb(
        {
            f"after_a/{k}": v
            for k, v in session.metrics["after_a"].items()
            if isinstance(v, (int, float))
        },
        step=state_a.step,
    )
    max_acc_a = session.metrics["after_a"]["A_test_acc"]
    cfg_c = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
    seg_c = train_segment(
        session.model,
        session.loaders["A"]["train"],
        cfg=cfg_c,
        optimizer=seg_a.optimizer,
        start_step=state_a.step,
        optimizer_transition="preserve",
        eval_loaders=session.eval_loaders("A"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="phase_a_continued",
    )
    session.absorb(seg_c.state)
    state_c = seg_c.state
    session.final_step = state_c.step
    session.metrics["after_continued"] = session.eval_both("after_continued")
    flatten_block(session.metrics, session.metrics["after_continued"])
    session.metrics["drift_A"] = session.metrics["after_continued"]["A_test_acc"] - max_acc_a
    session.attach_best_by_val(state_c, final_tag="phase_a_continued")


def run_sequential_ba(session: ProtocolSession) -> None:
    stop_first = switch_stop(session.switch_on)
    cfg_b = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
    seg_b = train_segment(
        session.model,
        session.loaders["B"]["train"],
        cfg=cfg_b,
        optimizer_transition="preserve",
        eval_loaders=session.eval_loaders("B"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="phase_b",
        stop_fn=stop_first,
    )
    session.absorb(seg_b.state)
    save_checkpoint(
        session.out_dir / "ckpts" / "theta_B.pt",
        session.model,
        optimizer=seg_b.optimizer,
        step=seg_b.state.step,
        meta={
            "task_switch": "B_done",
            "optimizer_transition": "preserve",
            "switch_on": session.switch_on,
        },
    )
    session.metrics["after_first"] = session.eval_both("after_b")
    session.metrics["switch_step"] = seg_b.state.step
    session.metrics["B_test_acc_at_switch"] = session.metrics["after_first"]["B_test_acc"]
    session.metrics["events_phase_b"] = dict(seg_b.state.events)
    max_acc_b = session.metrics["after_first"]["B_test_acc"]
    cfg_a = TrainConfig(**{**session.train_cfg.__dict__, "max_steps": session.steps})
    seg_a = train_segment(
        session.model,
        session.loaders["A"]["train"],
        cfg=cfg_a,
        optimizer=seg_b.optimizer,
        start_step=seg_b.state.step,
        optimizer_transition="preserve",
        eval_loaders=session.eval_loaders("A", "B"),
        ckpt_dir=str(session.out_dir / "ckpts"),
        run_name="phase_a",
    )
    session.absorb(seg_a.state)
    session.final_step = seg_a.state.step
    session.metrics["after_ba"] = session.eval_both("after_ba")
    flatten_block(session.metrics, session.metrics["after_ba"])
    session.metrics["forgetting_B_from_switch"] = forgetting(
        max_acc_b, session.metrics["after_ba"]["B_test_acc"]
    )
    session.metrics["optimizer_transition"] = "preserve"
    session.metrics["events_phase_a"] = dict(seg_a.state.events)
    session.attach_best_by_val(seg_a.state, final_tag="phase_a")
