"""High-level training protocols from the experiment plan."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from go4cl.data.dataset import (
    ModularAdditionDataset,
    make_balanced_joint_loader,
    make_loader,
)
from go4cl.data.manifest import DataManifest
from go4cl.data.packed import (
    AlternatingTaskLoader,
    BalancedPackedJointLoader,
    make_packed_multi_op_train_loader,
)
from go4cl.metrics.behavioral import evaluate, forgetting
from go4cl.metrics.continual import summarize_behavior
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.train.loop import TrainConfig, TrainState, train_steps
from go4cl.utils.checkpoint import load_checkpoint, save_checkpoint, write_json
from go4cl.utils.seed import seed_everything
from go4cl.utils.wandb_log import (
    define_train_metrics,
    finish_wandb,
    init_wandb,
    log_wandb,
    modulus_acc_metrics,
    slot_acc_metrics,
)

ProtocolName = Literal[
    "a_only",
    "b_only",
    "joint",
    "interleaved",
    "sequential_ab",
    "sequential_ba",
    "a_only_continued",
]


@dataclass
class ProtocolResult:
    protocol: str
    metrics: dict[str, Any]
    wandb_url: str | None = None


def _eval_bundle(
    loaders: dict[str, dict[str, DataLoader]],
    *tasks: str,
    include_test: bool = False,
) -> dict[str, DataLoader]:
    """Primary val (+ optional train_eval / iid / nuisance / test) for train_steps.

    Test loaders are logged only. Checkpoint selection still uses ``*_val``.
    """
    out: dict[str, DataLoader] = {}
    for task in tasks:
        out[f"{task}_val"] = loaders[task]["val"]
        if "train_eval" in loaders[task]:
            out[f"{task}_train_eval"] = loaders[task]["train_eval"]
        if "iid" in loaders[task]:
            out[f"{task}_iid"] = loaders[task]["iid"]
        if "val_nuisance" in loaders[task]:
            out[f"{task}_val_nuisance"] = loaders[task]["val_nuisance"]
        if include_test:
            out[f"{task}_test"] = loaders[task]["test"]
            if "test_nuisance" in loaders[task]:
                out[f"{task}_test_nuisance"] = loaders[task]["test_nuisance"]
    return out


def _switch_stop(switch_on: str):
    """Stop the first phase once a stable event fires. ``fixed`` never stops early."""
    if switch_on == "fixed":
        return None
    if switch_on not in {"t_mem", "t_gen"}:
        raise ValueError(f"switch_on must be fixed|t_mem|t_gen, got {switch_on}")

    def _stop(state) -> bool:
        return state.events.get(switch_on) is not None

    return _stop


def _write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _flatten_block(metrics: dict[str, Any], block: dict[str, Any]) -> None:
    for key, value in block.items():
        if key != "tag":
            metrics[key] = value


def _task_loaders(
    data_root: Path,
    *,
    batch_size: int | None,
    train_replacement: bool = False,
    train_seed: int = 0,
    sampler_seed: int | None = None,
    eval_n_per_operation: int = 256,
    eval_seed: int = 0,
) -> dict[str, dict[str, DataLoader]]:
    """Build train/val/test loaders.

    For ``packed_online`` manifests:
      - train: replayable packed stream
      - val/test: **packed_id** (primary; distractors from train pool)
      - val_nuisance / test_nuisance: on-disk nuisance_random control
      - train_eval: fixed packed_id analysis set from train pools (t_mem)
      - iid: packed_id with all ops from train pools (t_iid)
    """
    from go4cl.data.context import build_analysis_dataset
    from go4cl.data.eval_contexts import make_eval_context_loader

    manifest_path = data_root / "manifest.json"
    manifest = DataManifest.load(manifest_path) if manifest_path.is_file() else None
    packed = bool(manifest and manifest.train_mode == "packed_online")
    stream_seed = int(sampler_seed if sampler_seed is not None else train_seed)

    out: dict[str, dict[str, DataLoader]] = {}
    for task_name, task_id in (("A", 0), ("B", 1)):
        out[task_name] = {}
        for split in ("train", "val", "test"):
            if packed and split == "train":
                continue
            ds = ModularAdditionDataset.from_disk(data_root, task_name, split, task_id)  # type: ignore[arg-type]
            is_train = split == "train"
            out[task_name][split] = make_loader(
                ds,
                batch_size=batch_size if is_train else None,
                shuffle=is_train and not (train_replacement and batch_size),
                replacement=bool(is_train and train_replacement and batch_size),
            )
        if packed:
            assert manifest is not None
            if batch_size is None or batch_size <= 0:
                raise ValueError("packed_online train requires a positive batch_size")
            task = (
                manifest.task_pair.task_a
                if task_name == "A"
                else manifest.task_pair.task_b
            )
            out[task_name]["train"] = make_packed_multi_op_train_loader(
                task,
                manifest.residue_splits,
                batch_size=int(batch_size),
                seed=int(stream_seed) + 10_007 * int(task_id),
            )
            # Keep disk splits as nuisance controls
            out[task_name]["val_nuisance"] = out[task_name]["val"]
            out[task_name]["test_nuisance"] = out[task_name]["test"]
            # Primary eval = packed_id
            for split_name, key in (("val", "val"), ("test", "test")):
                out[task_name][key] = make_eval_context_loader(
                    task,
                    manifest.residue_splits,
                    target_split=split_name,  # type: ignore[arg-type]
                    context_mode="packed_id",
                    distractor_split="train",
                    n_per_operation=int(eval_n_per_operation),
                    seed=int(eval_seed) + 1009 * task_id + (0 if split_name == "val" else 1),
                )
            # Fixed train-eval for t_mem (not minibatch)
            train_eval_ex = build_analysis_dataset(
                task,
                manifest.residue_splits,
                split="train",
                context_mode="packed_id",
                analysis_seed=int(eval_seed) + 17 + task_id,
                aliases_per_pair=2,
                contexts_per_pair=1,
            )
            out[task_name]["train_eval"] = make_loader(
                ModularAdditionDataset.from_examples(train_eval_ex, task_id=task_id),
                batch_size=None,
                shuffle=False,
            )
            # t_iid: all ops from train pool, packed_id
            out[task_name]["iid"] = make_eval_context_loader(
                task,
                manifest.residue_splits,
                target_split="train",
                context_mode="packed_id",
                distractor_split="train",
                n_per_operation=int(eval_n_per_operation),
                seed=int(eval_seed) + 409 + task_id,
            )
    return out


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

    seed_everything(model_seed)
    device = torch.device(train_cfg.device)
    model = ModularTransformer(model_cfg).to(device)
    loaders = _task_loaders(
        data_root,
        batch_size=train_cfg.batch_size,
        train_replacement=bool(train_cfg.train_replacement),
        train_seed=model_seed,
        sampler_seed=resolved_sampler_seed,
        eval_n_per_operation=int(eval_n_per_operation),
    )
    packed_a = getattr(loaders["A"]["train"], "n_packs", None)
    packed_b = getattr(loaders["B"]["train"], "n_packs", None)
    # Resolve effective train batch sizes for logging
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
        },
        **(wandb_config or {}),
    }
    metrics: dict[str, Any] = {
        "protocol": protocol,
        "model_seed": model_seed,
        "sampler_seed": resolved_sampler_seed,
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

    final_step = 0
    history: list[dict[str, Any]] = []
    bundle_kw = {"include_test": bool(include_test)}

    def _absorb(state) -> None:
        history.extend(state.eval_history)

    def _eval_both(tag: str) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for task in ("A", "B"):
            for split in ("val", "test"):
                r = evaluate(model, loaders[task][split], device)
                res[f"{task}_{split}_acc"] = r.accuracy
                res[f"{task}_{split}_loss"] = r.loss
                res[f"{task}_{split}_macro_op_acc"] = r.macro_operation_accuracy
                res[f"{task}_{split}_macro_op_loss"] = r.macro_operation_loss
                res[f"{task}_{split}_margin"] = r.mean_correct_logit_margin
                res[f"{task}_{split}_nce"] = r.normalized_cross_entropy
                res.update(
                    modulus_acc_metrics(
                        r.by_modulus, prefix=f"{task}_{split}_acc"
                    )
                )
                res.update(
                    slot_acc_metrics(r.by_slot, prefix=f"{task}_{split}_acc")
                )
            for ctrl in ("val_nuisance", "test_nuisance"):
                if ctrl in loaders[task]:
                    r = evaluate(model, loaders[task][ctrl], device)
                    res[f"{task}_{ctrl}_acc"] = r.accuracy
                    res[f"{task}_{ctrl}_macro_op_acc"] = r.macro_operation_accuracy
        res["tag"] = tag
        return res

    def _attach_best_by_val(state: TrainState, *, final_tag: str) -> None:
        """Evaluate best-by-val ckpt; keep top-level metrics as final weights.

        Adds best_* keys selected by mean val accuracy during training.
        Restores final weights afterward so trailing final.pt stays final.
        """
        metrics["final_step"] = state.step
        metrics["best_step"] = state.best_step
        metrics["best_val_score"] = (
            float(state.best_val_acc) if state.best_val_acc >= 0 else None
        )
        best_path = state.best_ckpt_path
        if not best_path or not Path(best_path).is_file():
            return
        # Snapshot final weights path written by train_steps.
        final_path = Path(out_dir) / "ckpts" / f"{final_tag}_final.pt"
        load_checkpoint(best_path, model=model, map_location=device)
        best = _eval_both("best_by_val")
        metrics["best_A_val_acc"] = best["A_val_acc"]
        metrics["best_A_test_acc"] = best["A_test_acc"]
        metrics["best_A_val_loss"] = best["A_val_loss"]
        metrics["best_A_test_loss"] = best["A_test_loss"]
        metrics["best_B_val_acc"] = best["B_val_acc"]
        metrics["best_B_test_acc"] = best["B_test_acc"]
        metrics["best_ckpt"] = str(best_path)
        best_log: dict[str, Any] = {
            "best/A_val_acc": best["A_val_acc"],
            "best/A_test_acc": best["A_test_acc"],
            "best/B_val_acc": best["B_val_acc"],
            "best/B_test_acc": best["B_test_acc"],
            "best/step": state.best_step,
            "best/val_score": state.best_val_acc,
        }
        for k, v in best.items():
            if isinstance(v, (int, float)) and (
                k.startswith("A_val_acc/")
                or k.startswith("A_test_acc/")
                or k.startswith("B_val_acc/")
                or k.startswith("B_test_acc/")
            ):
                best_log[f"best/{k}"] = v
                metrics[f"best_{k.replace('/', '_')}"] = v
        log_wandb(best_log, step=state.step)
        if final_path.is_file():
            load_checkpoint(final_path, model=model, map_location=device)

    stop_first = _switch_stop(switch_on)
    metrics["switch_on"] = switch_on

    try:
        if protocol == "a_only":
            cfg = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state = train_steps(
                model,
                loaders["A"]["train"],
                cfg=cfg,
                eval_loaders=_eval_bundle(loaders, "A", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="a_only",
            )
            _absorb(state)
            final_step = state.step
            metrics.update(_eval_both("after_a"))
            metrics["events"] = dict(state.events)
            if state.first_stable_ckpt_path:
                metrics["first_stable_ckpt"] = state.first_stable_ckpt_path
            _attach_best_by_val(state, final_tag="a_only")

        elif protocol == "b_only":
            cfg = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state = train_steps(
                model,
                loaders["B"]["train"],
                cfg=cfg,
                eval_loaders=_eval_bundle(loaders, "B", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="b_only",
            )
            _absorb(state)
            final_step = state.step
            metrics.update(_eval_both("after_b"))
            metrics["events"] = dict(state.events)
            _attach_best_by_val(state, final_tag="b_only")

        elif protocol == "joint":
            if packed_a is not None:
                manifest = DataManifest.load(data_root / "manifest.json")
                joint_loader = BalancedPackedJointLoader(
                    manifest.task_pair.task_a,
                    manifest.task_pair.task_b,
                    manifest.residue_splits,
                    batch_size=int(train_cfg.batch_size or 0),
                    seed=resolved_sampler_seed,
                )
            else:
                ds_a = ModularAdditionDataset.from_disk(data_root, "A", "train", 0)
                ds_b = ModularAdditionDataset.from_disk(data_root, "B", "train", 1)
                joint_loader = make_balanced_joint_loader(
                    ds_a, ds_b, batch_size=train_cfg.batch_size
                )
            cfg = TrainConfig(**{**train_cfg.__dict__, "max_steps": 2 * steps})
            state = train_steps(
                model,
                joint_loader,
                cfg=cfg,
                eval_loaders=_eval_bundle(loaders, "A", "B", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="joint",
            )
            _absorb(state)
            final_step = state.step
            metrics.update(_eval_both("after_joint"))
            metrics["events"] = dict(state.events)
            _attach_best_by_val(state, final_tag="joint")

        elif protocol in {"sequential_ab", "a_only_continued"}:
            from go4cl.train.loop import train_segment

            cfg_a = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            seg_a = train_segment(
                model,
                loaders["A"]["train"],
                cfg=cfg_a,
                optimizer_transition="preserve",
                eval_loaders=_eval_bundle(loaders, "A", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="phase_a",
                stop_fn=stop_first,
            )
            _absorb(seg_a.state)
            state_a = seg_a.state
            save_checkpoint(
                out_dir / "ckpts" / "theta_A.pt",
                model,
                optimizer=seg_a.optimizer,
                step=state_a.step,
                meta={
                    "task_switch": "A_done",
                    "optimizer_transition": "preserve",
                    "switch_on": switch_on,
                },
            )
            metrics["after_a"] = _eval_both("after_a")
            metrics["switch_step"] = state_a.step
            metrics["events_phase_a"] = dict(state_a.events)
            metrics["A_test_acc_at_switch"] = metrics["after_a"]["A_test_acc"]
            metrics["A_val_acc_at_switch"] = metrics["after_a"]["A_val_acc"]
            log_wandb(
                {
                    f"after_a/{k}": v
                    for k, v in metrics["after_a"].items()
                    if isinstance(v, (int, float))
                },
                step=state_a.step,
            )
            max_acc_a = metrics["after_a"]["A_test_acc"]

            if protocol == "a_only_continued":
                cfg_c = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
                seg_c = train_segment(
                    model,
                    loaders["A"]["train"],
                    cfg=cfg_c,
                    optimizer=seg_a.optimizer,
                    start_step=state_a.step,
                    optimizer_transition="preserve",
                    eval_loaders=_eval_bundle(loaders, "A", **bundle_kw),
                    ckpt_dir=str(out_dir / "ckpts"),
                    run_name="phase_a_continued",
                )
                _absorb(seg_c.state)
                state_c = seg_c.state
                final_step = state_c.step
                metrics["after_continued"] = _eval_both("after_continued")
                _flatten_block(metrics, metrics["after_continued"])
                metrics["drift_A"] = (
                    metrics["after_continued"]["A_test_acc"] - max_acc_a
                )
                _attach_best_by_val(state_c, final_tag="phase_a_continued")
            else:
                cfg_b = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
                seg_b = train_segment(
                    model,
                    loaders["B"]["train"],
                    cfg=cfg_b,
                    optimizer=seg_a.optimizer,
                    start_step=state_a.step,
                    optimizer_transition="preserve",
                    eval_loaders=_eval_bundle(loaders, "A", "B", **bundle_kw),
                    ckpt_dir=str(out_dir / "ckpts"),
                    run_name="phase_b",
                )
                _absorb(seg_b.state)
                state_b = seg_b.state
                final_step = state_b.step
                metrics["after_b"] = _eval_both("after_b")
                _flatten_block(metrics, metrics["after_b"])
                metrics["forgetting_A_from_switch"] = forgetting(
                    max_acc_a, metrics["after_b"]["A_test_acc"]
                )
                metrics["optimizer_transition"] = "preserve"
                metrics["events_phase_b"] = dict(state_b.events)
                _attach_best_by_val(state_b, final_tag="phase_b")

        elif protocol == "sequential_ba":
            from go4cl.train.loop import train_segment

            cfg_b = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            seg_b = train_segment(
                model,
                loaders["B"]["train"],
                cfg=cfg_b,
                optimizer_transition="preserve",
                eval_loaders=_eval_bundle(loaders, "B", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="phase_b",
                stop_fn=stop_first,
            )
            _absorb(seg_b.state)
            save_checkpoint(
                out_dir / "ckpts" / "theta_B.pt",
                model,
                optimizer=seg_b.optimizer,
                step=seg_b.state.step,
                meta={
                    "task_switch": "B_done",
                    "optimizer_transition": "preserve",
                    "switch_on": switch_on,
                },
            )
            metrics["after_first"] = _eval_both("after_b")
            metrics["switch_step"] = seg_b.state.step
            metrics["B_test_acc_at_switch"] = metrics["after_first"]["B_test_acc"]
            metrics["events_phase_b"] = dict(seg_b.state.events)
            max_acc_b = metrics["after_first"]["B_test_acc"]
            cfg_a = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            seg_a = train_segment(
                model,
                loaders["A"]["train"],
                cfg=cfg_a,
                optimizer=seg_b.optimizer,
                start_step=seg_b.state.step,
                optimizer_transition="preserve",
                eval_loaders=_eval_bundle(loaders, "A", "B", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="phase_a",
            )
            _absorb(seg_a.state)
            final_step = seg_a.state.step
            metrics["after_ba"] = _eval_both("after_ba")
            _flatten_block(metrics, metrics["after_ba"])
            metrics["forgetting_B_from_switch"] = forgetting(
                max_acc_b, metrics["after_ba"]["B_test_acc"]
            )
            metrics["optimizer_transition"] = "preserve"
            metrics["events_phase_a"] = dict(seg_a.state.events)
            _attach_best_by_val(seg_a.state, final_tag="phase_a")

        elif protocol == "interleaved":
            mixed = AlternatingTaskLoader(loaders["A"]["train"], loaders["B"]["train"])
            cfg = TrainConfig(**{**train_cfg.__dict__, "max_steps": 2 * steps})
            state = train_steps(
                model,
                mixed,
                cfg=cfg,
                eval_loaders=_eval_bundle(loaders, "A", "B", **bundle_kw),
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="interleaved",
            )
            _absorb(state)
            final_step = state.step
            metrics.update(_eval_both("after_interleaved"))
            metrics["events"] = dict(state.events)
            _attach_best_by_val(state, final_tag="interleaved")

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
            log_wandb(flat_final, step=final_step)

    finally:
        behavior = summarize_behavior(
            protocol,
            history,
            switch_step=metrics.get("switch_step"),
        )
        metrics["behavior"] = behavior
        if "forgetting_A" in behavior:
            metrics["forgetting_A"] = behavior["forgetting_A"]
        elif "forgetting_A_from_switch" in metrics:
            metrics["forgetting_A"] = metrics["forgetting_A_from_switch"]
        if "forgetting_B" in behavior:
            metrics["forgetting_B"] = behavior["forgetting_B"]
        elif "forgetting_B_from_switch" in metrics:
            metrics["forgetting_B"] = metrics["forgetting_B_from_switch"]
        _write_history(out_dir / "eval_history.jsonl", history)
        behavior_log = {
            f"behavior/{k}": v
            for k, v in behavior.items()
            if isinstance(v, (int, float))
        }
        if behavior_log:
            log_wandb(behavior_log, step=final_step)
        write_json(out_dir / "metrics.json", metrics)
        write_json(out_dir / "config_resolved.json", cfg_payload)
        finish_wandb()

    save_checkpoint(out_dir / "ckpts" / "final.pt", model, step=final_step)
    return ProtocolResult(protocol=protocol, metrics=metrics, wandb_url=wandb_url)
