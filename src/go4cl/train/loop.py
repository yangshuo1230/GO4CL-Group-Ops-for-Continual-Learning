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
    task_event_values,
    task_scoped_detectors,
)
from go4cl.utils.checkpoint import save_checkpoint
from go4cl.utils.wandb_log import (
    log_wandb,
    modulus_acc_metrics,
    operation_acc_metrics,
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
    # Off by default. When set, train loaders mix TASK-slot reject rows
    # (see go4cl.data.null_task). Eval is unchanged; no A replay on phase B.
    null_task_tokens: bool = False
    null_task_ratio: float = 0.25
    null_task_label: int = 0
    # Task-partition smoke sets this false. Default keeps the CUDA compile path.
    compile_model: bool = True
    # Task-partition turns this off. Other protocols keep logging when a run exists.
    log_to_wandb: bool = True


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


def _train_callable(
    model: ModularTransformer, device: torch.device, *, compile_model: bool = True
):
    """Use a Triton-compiled forward on CUDA. Cached on the module across segments.

    ``False`` means compile was attempted and failed, so later segments stay eager.
    """
    if device.type != "cuda" or not compile_model:
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
    after_eval: Callable[[int, dict[str, Any], ModularTransformer, torch.optim.Optimizer], None]
    | None = None,
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
    b_selector = CheckpointSelector()
    detectors = task_scoped_detectors() if track_events else {}
    for name in detectors:
        state.events[name] = None
    batches = infinite_loader(train_loader)
    total = int(cfg.max_steps)
    pbar = tqdm(range(1, total + 1), desc=run_name, leave=False)
    train_model = _train_callable(model, device, compile_model=cfg.compile_model)

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
                record.update(
                    operation_acc_metrics(result.by_operation, prefix=f"{name}_acc")
                )
                if is_primary_val_loader(name):
                    val_macros[name] = result.macro_operation_accuracy
                    val_losses[name] = result.macro_operation_loss

            # Stable events are task-scoped. A/B validation is never averaged
            # into the event that means "B has learned".
            if detectors:
                macros = {
                    name: float(result.macro_operation_accuracy)
                    for name, result in metrics_by_name.items()
                }
                event_values, compat_task = task_event_values(macros)
                if compat_task is not None:
                    record["event/compat_task"] = compat_task
                for event_name, event_value in event_values.items():
                    _maybe_fire(
                        detectors, event_name, step, event_value, state, record
                    )
                b_gen = detectors.get("B_t_gen")
                if (
                    b_gen is not None
                    and b_gen.triggered_step == step
                    and ckpt_dir
                ):
                    path = f"{ckpt_dir}/B_first_stable.pt"
                    save_checkpoint(
                        path,
                        model,
                        optimizer=opt,
                        step=step,
                        meta={
                            "run_name": run_name,
                            "role": "B_first_stable",
                            "event": "B_t_gen",
                            "macro_operation_accuracy": event_values.get("B_t_gen"),
                            "definition": (
                                "five consecutive B validation macro-operation "
                                "accuracies >= 0.9"
                            ),
                        },
                    )
                    state.first_stable_ckpt_path = path
                    state.events["primary_event"] = "B_t_gen"
                elif (
                    compat_task is not None
                    and detectors["t_gen"].triggered_step == step
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
                            "compat_task": compat_task,
                            "macro_operation_accuracy": event_values.get("t_gen"),
                            "note": f"t_gen aliases task {compat_task}",
                        },
                    )
                    state.first_stable_ckpt_path = path
                    shutil.copy2(path, Path(ckpt_dir) / "first_stable_threshold.pt")

            if "B_val" in val_macros and ckpt_dir:
                if b_selector.observe(
                    step,
                    macro_operation_accuracy=val_macros["B_val"],
                    macro_operation_loss=val_losses["B_val"],
                ):
                    save_checkpoint(
                        f"{ckpt_dir}/B_best_val.pt",
                        model,
                        optimizer=opt,
                        step=step,
                        meta={
                            "run_name": run_name,
                            "role": "B_best_val",
                            "selection": "B_val_macro_operation_accuracy",
                            "best_val_macro_op_acc": val_macros["B_val"],
                            "best_val_macro_op_loss": val_losses["B_val"],
                        },
                    )
            if val_macros:
                both = "A_val" in val_macros and "B_val" in val_macros
                if both:
                    legacy_role = "AB_tradeoff_best"
                elif "B_val" in val_macros:
                    legacy_role = "B_best_val"
                elif "A_val" in val_macros:
                    legacy_role = "A_best_val"
                else:
                    legacy_role = "single_task_best"
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
                                "legacy_role": legacy_role,
                                "best_val_acc": score,
                                "best_val_macro_op_acc": score,
                                "best_val_macro_op_loss": score_loss,
                                "val_macros": val_macros,
                                "selection": (
                                    "mean_A_val_and_B_val"
                                    if both
                                    else "macro_operation_accuracy"
                                ),
                                "note": (
                                    "best.pt is the legacy A/B tradeoff checkpoint"
                                    if both
                                    else f"best.pt matches role {legacy_role}"
                                ),
                            },
                        )
                        state.best_ckpt_path = best_path
                        if both:
                            shutil.copy2(
                                best_path, Path(ckpt_dir) / "AB_tradeoff_best.pt"
                            )
                    record["best_val_acc"] = score
                    record["best_step"] = step
                    record["best_legacy_role"] = legacy_role
                else:
                    record["best_val_acc"] = state.best_val_acc
                    record["best_step"] = state.best_step
            _append_eval_history(state, record, segment=run_name)
            if after_eval is not None:
                after_eval(step, record, model, opt)

        if record is not None and cfg.log_to_wandb:
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
        final_meta: dict[str, Any] = {"run_name": run_name, "final": True}
        if run_name == "phase_b":
            final_meta["role"] = "phase_b_final"
        save_checkpoint(
            f"{ckpt_dir}/{run_name}_final.pt",
            model,
            optimizer=opt,
            step=state.step,
            meta=final_meta,
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
    after_eval: Callable[[int, dict[str, Any], ModularTransformer, torch.optim.Optimizer], None]
    | None = None,
) -> TrainSegmentResult:
    """Optimizer-aware training segment for Phase 2 sequential protocols.

    ``optimizer_transition``:
      - ``preserve``: reuse ``optimizer`` (or create once and keep)
      - ``reset``: always build a fresh AdamW
    """
    if optimizer_transition not in {"preserve", "reset", "fresh"}:
        raise ValueError(
            "optimizer_transition must be preserve, fresh, or reset "
            f"(reset is an alias of fresh), got {optimizer_transition}"
        )
    # fresh and the older name reset both build a new AdamW.
    opt = None if optimizer_transition in {"reset", "fresh"} else optimizer
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
        after_eval=after_eval,
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
