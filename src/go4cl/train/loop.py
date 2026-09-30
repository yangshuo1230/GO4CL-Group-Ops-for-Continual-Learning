"""Training loop primitives."""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from go4cl.metrics.behavioral import EvalResult, evaluate
from go4cl.model.transformer import ModularTransformer
from go4cl.utils.checkpoint import save_checkpoint
from go4cl.utils.wandb_log import log_wandb


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1.0
    # None / <=0 => full-batch (batch_size = dataset length)
    batch_size: int | None = None
    # If True and batch_size is set: train loader draws fixed-size batches
    # with replacement (Phase 1A uses this for batch_size=2048).
    train_replacement: bool = False
    max_steps: int = 2000
    eval_every: int = 100
    ckpt_every: int = 500
    grad_clip: float | None = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainState:
    step: int = 0
    # Selection score over eval loaders whose names end with "val" (mean if several).
    # Starts at -1 so the first val eval always writes a best checkpoint.
    best_val_acc: float = -1.0
    best_step: int = 0
    best_ckpt_path: str | None = None


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

        record: dict[str, Any] | None = None
        do_eval = bool(eval_loaders) and step % cfg.eval_every == 0
        if step % log_every == 0 or do_eval:
            with torch.no_grad():
                preds = out["logits"].argmax(dim=-1)
                train_acc = float((preds == labels).float().mean().item())
            record = {
                "step": step,
                "train_loss": float(loss.item()),
                "train_acc": train_acc,
            }

        if do_eval:
            assert record is not None
            val_accs: dict[str, float] = {}
            for name, loader in eval_loaders.items():  # type: ignore[union-attr]
                result: EvalResult = evaluate(model, loader, device)
                record[f"{name}_loss"] = result.loss
                record[f"{name}_acc"] = result.accuracy
                if name.endswith("val") or name == "val":
                    val_accs[name] = result.accuracy
            if val_accs:
                score = sum(val_accs.values()) / len(val_accs)
                if score > state.best_val_acc:
                    state.best_val_acc = score
                    state.best_step = step
                    if ckpt_dir:
                        best_path = f"{ckpt_dir}/{run_name}_best.pt"
                        save_checkpoint(
                            best_path,
                            model,
                            optimizer=opt,
                            step=step,
                            meta={
                                "run_name": run_name,
                                "best": True,
                                "best_val_acc": score,
                                "val_accs": val_accs,
                            },
                        )
                        state.best_ckpt_path = best_path
                    record["best_val_acc"] = score
                    record["best_step"] = step
                else:
                    record["best_val_acc"] = state.best_val_acc
                    record["best_step"] = state.best_step

        if record is not None:
            log_wandb(record, step=step)

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
        # Stable alias pointing at the latest best-by-val weights.
        if state.best_ckpt_path is not None:
            src = Path(state.best_ckpt_path)
            if src.is_file():
                shutil.copy2(src, Path(ckpt_dir) / "best.pt")
    return state
