"""Training loop primitives."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from go4cl.metrics.behavioral import EvalResult, evaluate
from go4cl.model.transformer import ModularTransformer
from go4cl.utils.checkpoint import save_checkpoint


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1.0
    batch_size: int = 128
    max_steps: int = 2000
    eval_every: int = 100
    ckpt_every: int = 500
    grad_clip: float | None = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainState:
    step: int = 0
    history: list[dict[str, Any]] = field(default_factory=list)
    best_val_acc: float = 0.0


def build_optimizer(model: ModularTransformer, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
    )


def infinite_loader(loader: DataLoader) -> Iterator[dict[str, torch.Tensor]]:
    while True:
        for batch in loader:
            yield batch


def train_steps(
    model: ModularTransformer,
    train_loader: DataLoader,
    *,
    cfg: TrainConfig,
    eval_loaders: dict[str, DataLoader] | None = None,
    ckpt_dir: str | None = None,
    run_name: str = "run",
    stop_fn: Callable[[TrainState], bool] | None = None,
    log_every: int = 50,
) -> TrainState:
    device = torch.device(cfg.device)
    model.to(device)
    opt = build_optimizer(model, cfg)
    state = TrainState()
    batches = infinite_loader(train_loader)
    pbar = tqdm(range(1, cfg.max_steps + 1), desc=run_name, leave=False)

    for step in pbar:
        model.train()
        batch = next(batches)
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        out = model(tokens, labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        state.step = step

        if step % log_every == 0:
            pbar.set_postfix(loss=float(loss.item()))

        record: dict[str, Any] = {"step": step, "train_loss": float(loss.item())}
        if eval_loaders and step % cfg.eval_every == 0:
            for name, loader in eval_loaders.items():
                result: EvalResult = evaluate(model, loader, device)
                record[f"{name}_loss"] = result.loss
                record[f"{name}_acc"] = result.accuracy
                if name.endswith("val") or name == "val":
                    state.best_val_acc = max(state.best_val_acc, result.accuracy)
            state.history.append(record)

        if ckpt_dir and step % cfg.ckpt_every == 0:
            save_checkpoint(
                f"{ckpt_dir}/{run_name}_step{step}.pt",
                model,
                optimizer=opt,
                step=step,
                meta={"run_name": run_name},
            )

        if stop_fn is not None and stop_fn(state):
            break

    if ckpt_dir:
        save_checkpoint(
            f"{ckpt_dir}/{run_name}_final.pt",
            model,
            optimizer=opt,
            step=state.step,
            meta={"run_name": run_name, "final": True},
        )
    return state
