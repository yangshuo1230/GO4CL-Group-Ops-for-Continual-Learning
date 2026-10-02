"""High-level training protocols from the experiment plan."""

from __future__ import annotations

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
from go4cl.data.packed import make_packed_multi_op_train_loader
from go4cl.metrics.behavioral import evaluate, forgetting
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


def _task_loaders(
    data_root: Path,
    *,
    batch_size: int | None,
    train_replacement: bool = False,
    train_seed: int = 0,
) -> dict[str, dict[str, DataLoader]]:
    manifest_path = data_root / "manifest.json"
    manifest = DataManifest.load(manifest_path) if manifest_path.is_file() else None
    packed = bool(manifest and manifest.train_mode == "packed_online")

    out: dict[str, dict[str, DataLoader]] = {}
    for task_name, task_id in (("A", 0), ("B", 1)):
        out[task_name] = {}
        for split in ("train", "val", "test"):
            if packed and split == "train":
                continue
            ds = ModularAdditionDataset.from_disk(data_root, task_name, split, task_id)  # type: ignore[arg-type]
            is_train = split == "train"
            # Train: optional fixed-size with-replacement batches.
            # Eval: full split, no replacement (stable metrics).
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
                seed=int(train_seed) + 10_007 * int(task_id),
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
    phase_steps: int | None = None,
    wandb_enabled: bool = True,
    wandb_project: str = "go4cl",
    wandb_name: str | None = None,
    wandb_mode: str | None = None,
    wandb_config: dict[str, Any] | None = None,
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
) -> ProtocolResult:
    """
    Run one of the plan's training protocols on a fixed dataset root.

    Metrics stream to W&B when ``wandb_enabled`` is true (default).
    """
    data_root = Path(data_root)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_cfg = train_cfg or TrainConfig()
    model_cfg = model_cfg or ModelConfig()
    steps = phase_steps or train_cfg.max_steps

    seed_everything(model_seed)
    device = torch.device(train_cfg.device)
    model = ModularTransformer(model_cfg).to(device)
    loaders = _task_loaders(
        data_root,
        batch_size=train_cfg.batch_size,
        train_replacement=bool(train_cfg.train_replacement),
        train_seed=model_seed,
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
        "phase_steps": steps,
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

    metrics: dict[str, Any] = {"protocol": protocol, "model_seed": model_seed}
    final_step = 0

    def _eval_both(tag: str) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for task in ("A", "B"):
            for split in ("val", "test"):
                r = evaluate(model, loaders[task][split], device)
                res[f"{task}_{split}_acc"] = r.accuracy
                res[f"{task}_{split}_loss"] = r.loss
                # Per-modulus held-out accuracies for W&B / metrics.json
                res.update(
                    modulus_acc_metrics(
                        r.by_modulus, prefix=f"{task}_{split}_acc"
                    )
                )
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

    try:
        if protocol == "a_only":
            cfg = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state = train_steps(
                model,
                loaders["A"]["train"],
                cfg=cfg,
                eval_loaders={"A_val": loaders["A"]["val"]},
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="a_only",
            )
            final_step = state.step
            metrics.update(_eval_both("after_a"))
            _attach_best_by_val(state, final_tag="a_only")

        elif protocol == "b_only":
            cfg = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state = train_steps(
                model,
                loaders["B"]["train"],
                cfg=cfg,
                eval_loaders={"B_val": loaders["B"]["val"]},
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="b_only",
            )
            final_step = state.step
            metrics.update(_eval_both("after_b"))
            _attach_best_by_val(state, final_tag="b_only")

        elif protocol == "joint":
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
                eval_loaders={
                    "A_val": loaders["A"]["val"],
                    "B_val": loaders["B"]["val"],
                },
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="joint",
            )
            final_step = state.step
            metrics.update(_eval_both("after_joint"))
            _attach_best_by_val(state, final_tag="joint")

        elif protocol in {"sequential_ab", "a_only_continued"}:
            cfg_a = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state_a = train_steps(
                model,
                loaders["A"]["train"],
                cfg=cfg_a,
                eval_loaders={"A_val": loaders["A"]["val"]},
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="phase_a",
            )
            save_checkpoint(out_dir / "ckpts" / "theta_A.pt", model, step=state_a.step)
            metrics["after_a"] = _eval_both("after_a")
            log_wandb({f"after_a/{k}": v for k, v in metrics["after_a"].items() if isinstance(v, (int, float))}, step=state_a.step)
            max_acc_a = metrics["after_a"]["A_test_acc"]

            if protocol == "a_only_continued":
                cfg_c = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
                state_c = train_steps(
                    model,
                    loaders["A"]["train"],
                    cfg=cfg_c,
                    eval_loaders={"A_val": loaders["A"]["val"]},
                    ckpt_dir=str(out_dir / "ckpts"),
                    run_name="phase_a_continued",
                )
                final_step = state_a.step + state_c.step
                metrics["after_continued"] = _eval_both("after_continued")
                _attach_best_by_val(state_c, final_tag="phase_a_continued")
            else:
                cfg_b = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
                state_b = train_steps(
                    model,
                    loaders["B"]["train"],
                    cfg=cfg_b,
                    eval_loaders={
                        "A_val": loaders["A"]["val"],
                        "B_val": loaders["B"]["val"],
                    },
                    ckpt_dir=str(out_dir / "ckpts"),
                    run_name="phase_b",
                )
                final_step = state_a.step + state_b.step
                metrics["after_b"] = _eval_both("after_b")
                metrics["forgetting_A"] = forgetting(
                    max_acc_a, metrics["after_b"]["A_test_acc"]
                )
                _attach_best_by_val(state_b, final_tag="phase_b")

        elif protocol == "sequential_ba":
            cfg_b = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state_b = train_steps(
                model,
                loaders["B"]["train"],
                cfg=cfg_b,
                eval_loaders={"B_val": loaders["B"]["val"]},
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="phase_b",
            )
            cfg_a = TrainConfig(**{**train_cfg.__dict__, "max_steps": steps})
            state_a = train_steps(
                model,
                loaders["A"]["train"],
                cfg=cfg_a,
                eval_loaders={
                    "A_val": loaders["A"]["val"],
                    "B_val": loaders["B"]["val"],
                },
                ckpt_dir=str(out_dir / "ckpts"),
                run_name="phase_a",
            )
            final_step = state_b.step + state_a.step
            metrics.update(_eval_both("after_ba"))
            _attach_best_by_val(state_a, final_tag="phase_a")

        elif protocol == "interleaved":
            from go4cl.train.loop import build_optimizer, infinite_loader

            opt = build_optimizer(model, train_cfg)
            it_a = infinite_loader(loaders["A"]["train"])
            it_b = infinite_loader(loaders["B"]["train"])
            total = 2 * steps
            for step in range(1, total + 1):
                model.train()
                batch = next(it_a if step % 2 == 1 else it_b)
                tokens = batch["tokens"].to(device)
                labels = batch["labels"].to(device)
                loss = model(tokens, labels)["loss"]
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if train_cfg.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                opt.step()
                if step % 50 == 0 or step % train_cfg.eval_every == 0:
                    record: dict[str, Any] = {"step": step, "train_loss": float(loss.item())}
                    if step % train_cfg.eval_every == 0:
                        for name, loader in (
                            ("A_val", loaders["A"]["val"]),
                            ("B_val", loaders["B"]["val"]),
                        ):
                            r = evaluate(model, loader, device)
                            record[f"{name}_acc"] = r.accuracy
                            record[f"{name}_loss"] = r.loss
                    log_wandb(record, step=step)
            final_step = total
            metrics.update(_eval_both("after_interleaved"))

        else:
            raise ValueError(f"unknown protocol: {protocol}")

        # Final scalar summary on W&B
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
        finish_wandb()

    write_json(out_dir / "metrics.json", metrics)
    save_checkpoint(out_dir / "ckpts" / "final.pt", model, step=final_step)
    return ProtocolResult(protocol=protocol, metrics=metrics, wandb_url=wandb_url)
