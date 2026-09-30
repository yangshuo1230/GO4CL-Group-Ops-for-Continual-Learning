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
from go4cl.metrics.behavioral import evaluate, forgetting
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.train.loop import TrainConfig, TrainState, train_steps
from go4cl.utils.checkpoint import save_checkpoint, write_json
from go4cl.utils.seed import seed_everything

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
    history: list[dict[str, Any]]
    metrics: dict[str, Any]


def _task_loaders(
    data_root: Path,
    *,
    batch_size: int,
) -> dict[str, dict[str, DataLoader]]:
    out: dict[str, dict[str, DataLoader]] = {}
    for task_name, task_id in (("A", 0), ("B", 1)):
        out[task_name] = {}
        for split in ("train", "val", "test"):
            ds = ModularAdditionDataset.from_disk(data_root, task_name, split, task_id)  # type: ignore[arg-type]
            out[task_name][split] = make_loader(
                ds, batch_size=batch_size, shuffle=(split == "train")
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
) -> ProtocolResult:
    """
    Run one of the plan's training protocols on a fixed dataset root.

    Sequential protocols train A for `phase_steps` then B for `phase_steps`.
    Joint trains for 2 * phase_steps with 50/50 mixture.
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
    loaders = _task_loaders(data_root, batch_size=train_cfg.batch_size)

    history: list[dict[str, Any]] = []
    metrics: dict[str, Any] = {"protocol": protocol, "model_seed": model_seed}

    def _eval_both(tag: str) -> dict[str, Any]:
        res = {}
        for task in ("A", "B"):
            for split in ("val", "test"):
                r = evaluate(model, loaders[task][split], device)
                res[f"{task}_{split}_acc"] = r.accuracy
                res[f"{task}_{split}_loss"] = r.loss
        res["tag"] = tag
        return res

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
        history = state.history
        metrics.update(_eval_both("after_a"))

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
        history = state.history
        metrics.update(_eval_both("after_b"))

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
        history = state.history
        metrics.update(_eval_both("after_joint"))

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
            history = state_a.history + state_c.history
            metrics["after_continued"] = _eval_both("after_continued")
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
            history = state_a.history + [
                {**h, "step": h["step"] + state_a.step} for h in state_b.history
            ]
            metrics["after_b"] = _eval_both("after_b")
            metrics["forgetting_A"] = forgetting(
                max_acc_a, metrics["after_b"]["A_test_acc"]
            )

    elif protocol == "sequential_ba":
        # Train B then A (order control)
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
        history = state_b.history + [
            {**h, "step": h["step"] + state_b.step} for h in state_a.history
        ]
        metrics.update(_eval_both("after_ba"))

    elif protocol == "interleaved":
        # Alternate batches from A and B
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
            if step % train_cfg.eval_every == 0:
                record = {"step": step, "train_loss": float(loss.item())}
                for name, loader in (
                    ("A_val", loaders["A"]["val"]),
                    ("B_val", loaders["B"]["val"]),
                ):
                    r = evaluate(model, loader, device)
                    record[f"{name}_acc"] = r.accuracy
                    record[f"{name}_loss"] = r.loss
                history.append(record)
        metrics.update(_eval_both("after_interleaved"))

    else:
        raise ValueError(f"unknown protocol: {protocol}")

    write_json(out_dir / "metrics.json", metrics)
    write_json(out_dir / "history.json", history)
    save_checkpoint(out_dir / "ckpts" / "final.pt", model, step=len(history))
    return ProtocolResult(protocol=protocol, history=history, metrics=metrics)
