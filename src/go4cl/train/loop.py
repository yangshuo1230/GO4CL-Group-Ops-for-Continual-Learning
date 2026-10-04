"""Training loop primitives."""

from __future__ import annotations

import re
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterator

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from go4cl.metrics.behavioral import EvalResult, evaluate
from go4cl.model.transformer import ModularTransformer
from go4cl.train.checkpoint_selection import (
    CheckpointSelector,
    StableEventDetector,
    default_event_detectors,
)
from go4cl.utils.checkpoint import save_checkpoint
from go4cl.utils.wandb_log import (
    log_wandb,
    modulus_acc_metrics,
    slot_acc_metrics,
)

_PRIMARY_VAL = re.compile(r"^(?:A|B)_val$|^val$")


def is_primary_val_loader(name: str) -> bool:
    """Only A_val / B_val / val drive checkpoint selection (not nuisance/iid)."""
    return bool(_PRIMARY_VAL.match(name))


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
    # Selection score: macro_operation_accuracy over primary val loaders.
    best_val_acc: float = -1.0
    best_val_loss: float = float("inf")
    best_step: int = 0
    best_ckpt_path: str | None = None
    first_stable_ckpt_path: str | None = None
    events: dict[str, int | None] = field(default_factory=dict)
    optimizer: torch.optim.Optimizer | None = None
    # Numeric eval rows (no W&B media). Phase 2 reads this for forgetting curves.
    eval_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class TrainSegmentResult:
    state: TrainState
    optimizer: torch.optim.Optimizer
    model: ModularTransformer


def build_optimizer(model: ModularTransformer, cfg: TrainConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.98),
    )


def _train_callable(model: ModularTransformer, device: torch.device):
    """Use a Triton-compiled forward on CUDA. Cached on the module across segments.

    ``False`` means compile was attempted and failed, so later segments stay eager.
    """
    if device.type != "cuda":
        return model
    cached = getattr(model, "_go4cl_compiled", None)
    if cached is False:
        return model
    if cached is None:
        print(
            "[train] compiling with Triton (mode=reduce-overhead); first step is slow",
            flush=True,
        )
        cached = torch.compile(model, mode="reduce-overhead")
        # Bypass Module.__setattr__ so the wrapper is not registered as a child.
        object.__setattr__(model, "_go4cl_compiled", cached)
    return cached


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
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    track_events: bool = True,
) -> TrainState:
    """Run ``cfg.max_steps`` optimization steps.

    If ``optimizer`` is passed, moments are preserved (Phase 2 ``preserve`` mode).
    ``start_step`` continues the global step counter across segments.
    """
    device = torch.device(cfg.device)
    model.to(device)
    opt = optimizer if optimizer is not None else build_optimizer(model, cfg)
    state = TrainState(step=int(start_step), optimizer=opt)
    selector = CheckpointSelector()
    detectors = default_event_detectors() if track_events else {}
    for name in detectors:
        state.events[name] = None
    batches = infinite_loader(train_loader)
    total = int(cfg.max_steps)
    pbar = tqdm(range(1, total + 1), desc=run_name, leave=False)
    train_model = _train_callable(model, device)

    for local_step in pbar:
        model.train()
        batch = next(batches)
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        try:
            out = train_model(tokens, labels)
        except Exception as exc:
            if train_model is model:
                raise
            print(
                f"[{run_name}] Triton compile failed ({type(exc).__name__}: {exc}); "
                "continuing eager",
                flush=True,
            )
            object.__setattr__(model, "_go4cl_compiled", False)
            train_model = model
            out = model(tokens, labels)
        loss = out["loss"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step = int(start_step) + local_step
        state.step = step

        if local_step % log_every == 0:
            pbar.set_postfix(loss=float(loss.item()))

        record: dict[str, Any] | None = None
        do_eval = bool(eval_loaders) and local_step % cfg.eval_every == 0
        if local_step % log_every == 0 or do_eval:
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
            val_macros: dict[str, float] = {}
            val_losses: dict[str, float] = {}
            metrics_by_name: dict[str, EvalResult] = {}
            for name, loader in eval_loaders.items():  # type: ignore[union-attr]
                result: EvalResult = evaluate(model, loader, device)
                metrics_by_name[name] = result
                record[f"{name}_loss"] = result.loss
                record[f"{name}_acc"] = result.accuracy
                record[f"{name}_macro_op_acc"] = result.macro_operation_accuracy
                record[f"{name}_macro_op_loss"] = result.macro_operation_loss
                record[f"{name}_margin"] = result.mean_correct_logit_margin
                record[f"{name}_nce"] = result.normalized_cross_entropy
                record.update(
                    modulus_acc_metrics(result.by_modulus, prefix=f"{name}_acc")
                )
                record.update(slot_acc_metrics(result.by_slot, prefix=f"{name}_acc"))
                if is_primary_val_loader(name):
                    val_macros[name] = result.macro_operation_accuracy
                    val_losses[name] = result.macro_operation_loss

            # Stable events (fixed eval sets only — never minibatch train_acc)
            if detectors:
                t_mem_val = _first_macro(
                    metrics_by_name, suffixes=("_train_eval",), exact=("train_eval",)
                )
                t_gen_val = (
                    sum(val_macros.values()) / len(val_macros) if val_macros else None
                )
                t_iid_val = _first_macro(
                    metrics_by_name, suffixes=("_iid",), exact=("iid",)
                )
                _maybe_fire(detectors, "t_mem", step, t_mem_val, state, record)
                _maybe_fire(detectors, "t_gen", step, t_gen_val, state, record)
                _maybe_fire(detectors, "t_iid", step, t_iid_val, state, record)
                if (
                    detectors["t_gen"].triggered_step == step
                    and ckpt_dir
                    and state.first_stable_ckpt_path is None
                ):
                    path = f"{ckpt_dir}/{run_name}_first_stable_threshold.pt"
                    save_checkpoint(
                        path,
                        model,
                        optimizer=opt,
                        step=step,
                        meta={
                            "run_name": run_name,
                            "event": "t_gen",
                            "macro_operation_accuracy": t_gen_val,
                        },
                    )
                    state.first_stable_ckpt_path = path
                    shutil.copy2(path, Path(ckpt_dir) / "first_stable_threshold.pt")

            if val_macros:
                score = sum(val_macros.values()) / len(val_macros)
                score_loss = sum(val_losses.values()) / len(val_losses)
                if selector.observe(
                    step,
                    macro_operation_accuracy=score,
                    macro_operation_loss=score_loss,
                ):
                    state.best_val_acc = score
                    state.best_val_loss = score_loss
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
                                "best_val_macro_op_acc": score,
                                "best_val_macro_op_loss": score_loss,
                                "val_macros": val_macros,
                                "selection": "macro_operation_accuracy",
                            },
                        )
                        state.best_ckpt_path = best_path
                    record["best_val_acc"] = score
                    record["best_step"] = step
                else:
                    record["best_val_acc"] = state.best_val_acc
                    record["best_step"] = state.best_step
            _append_eval_history(state, record, segment=run_name)

        if record is not None:
            log_wandb(record, step=step)

        if ckpt_dir and local_step % cfg.ckpt_every == 0:
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
        if state.best_ckpt_path is not None:
            src = Path(state.best_ckpt_path)
            if src.is_file():
                shutil.copy2(src, Path(ckpt_dir) / "best.pt")
    state.optimizer = opt
    return state


def train_segment(
    model: ModularTransformer,
    train_loader: DataLoader,
    *,
    cfg: TrainConfig,
    optimizer: torch.optim.Optimizer | None = None,
    start_step: int = 0,
    optimizer_transition: str = "preserve",
    eval_loaders: dict[str, DataLoader] | None = None,
    ckpt_dir: str | None = None,
    run_name: str = "segment",
    track_events: bool = True,
    stop_fn: Callable[[TrainState], bool] | None = None,
) -> TrainSegmentResult:
    """Optimizer-aware training segment for Phase 2 sequential protocols.

    ``optimizer_transition``:
      - ``preserve``: reuse ``optimizer`` (or create once and keep)
      - ``reset``: always build a fresh AdamW
    """
    if optimizer_transition not in {"preserve", "reset"}:
        raise ValueError(f"unknown optimizer_transition: {optimizer_transition}")
    opt = None if optimizer_transition == "reset" else optimizer
    state = train_steps(
        model,
        train_loader,
        cfg=cfg,
        eval_loaders=eval_loaders,
        ckpt_dir=ckpt_dir,
        run_name=run_name,
        optimizer=opt,
        start_step=start_step,
        track_events=track_events,
        stop_fn=stop_fn,
    )
    assert state.optimizer is not None
    return TrainSegmentResult(state=state, optimizer=state.optimizer, model=model)


def _append_eval_history(
    state: TrainState, record: dict[str, Any], *, segment: str
) -> None:
    """Keep a JSON-safe copy of one eval row for offline behavioral metrics."""
    row: dict[str, Any] = {"step": int(record.get("step", state.step)), "segment": segment}
    for key, value in record.items():
        if key == "step":
            continue
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            row[key] = value
    state.eval_history.append(row)


def _first_macro(
    metrics: dict[str, EvalResult],
    *,
    suffixes: tuple[str, ...],
    exact: tuple[str, ...],
) -> float | None:
    for name, result in metrics.items():
        if name in exact or any(name.endswith(s) for s in suffixes):
            return float(result.macro_operation_accuracy)
    return None


def _maybe_fire(
    detectors: dict[str, StableEventDetector],
    name: str,
    step: int,
    value: float | None,
    state: TrainState,
    record: dict[str, Any],
) -> None:
    if value is None or name not in detectors:
        return
    fired = detectors[name].observe(step, float(value))
    record[f"event/{name}_value"] = float(value)
    record[f"event/{name}_streak"] = detectors[name]._streak
    if fired:
        state.events[name] = step
        record[f"event/{name}_step"] = step
