"""Shared protocol evaluation, history, and best-ckpt reporting.

Val loaders drive checkpoint selection. Test loaders are report-only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
from torch.utils.data import DataLoader

from go4cl.metrics.behavioral import evaluate
from go4cl.train.loop import TrainConfig, TrainState
from go4cl.utils.checkpoint import load_checkpoint
from go4cl.utils.wandb_log import log_wandb, modulus_acc_metrics, slot_acc_metrics

ProtocolName = Literal[
    "a_only",
    "b_only",
    "joint",
    "interleaved",
    "sequential_ab",
    "sequential_ab_replay",
    "sequential_ba",
    "a_only_continued",
]


@dataclass
class ProtocolResult:
    protocol: str
    metrics: dict[str, Any]
    wandb_url: str | None = None


def eval_bundle(
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


def switch_stop(switch_on: str):
    """Stop the first phase once a stable event fires. ``fixed`` never stops early."""
    if switch_on == "fixed":
        return None
    if switch_on not in {"t_mem", "t_gen"}:
        raise ValueError(f"switch_on must be fixed|t_mem|t_gen, got {switch_on}")

    def _stop(state) -> bool:
        return state.events.get(switch_on) is not None

    return _stop


def write_history(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def flatten_block(metrics: dict[str, Any], block: dict[str, Any]) -> None:
    for key, value in block.items():
        if key != "tag":
            metrics[key] = value


@dataclass
class ProtocolSession:
    """Mutable run state shared by protocol implementations."""

    protocol: str
    model: Any
    loaders: dict[str, dict[str, DataLoader]]
    device: torch.device
    train_cfg: TrainConfig
    out_dir: Path
    data_root: Path
    steps: int
    metrics: dict[str, Any]
    include_test: bool
    switch_on: str
    packed_a: Any
    resolved_sampler_seed: int
    # Set only by the fixed-coverage replay grid. None keeps the normal replay loader.
    replay_coverage: dict | None = None
    history: list[dict[str, Any]] = field(default_factory=list)
    final_step: int = 0

    def absorb(self, state) -> None:
        self.history.extend(state.eval_history)

    def eval_loaders(self, *tasks: str) -> dict[str, DataLoader]:
        return eval_bundle(self.loaders, *tasks, include_test=self.include_test)

    def eval_both(self, tag: str) -> dict[str, Any]:
        res: dict[str, Any] = {}
        for task in ("A", "B"):
            for split in ("val", "test"):
                r = evaluate(self.model, self.loaders[task][split], self.device)
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
                if ctrl in self.loaders[task]:
                    r = evaluate(self.model, self.loaders[task][ctrl], self.device)
                    res[f"{task}_{ctrl}_acc"] = r.accuracy
                    res[f"{task}_{ctrl}_macro_op_acc"] = r.macro_operation_accuracy
        res["tag"] = tag
        return res

    def attach_best_by_val(self, state: TrainState, *, final_tag: str) -> None:
        """Evaluate best-by-val ckpt; keep top-level metrics as final weights."""
        self.metrics["final_step"] = state.step
        self.metrics["best_step"] = state.best_step
        self.metrics["best_val_score"] = (
            float(state.best_val_acc) if state.best_val_acc >= 0 else None
        )
        best_path = state.best_ckpt_path
        if not best_path or not Path(best_path).is_file():
            return
        final_path = Path(self.out_dir) / "ckpts" / f"{final_tag}_final.pt"
        load_checkpoint(best_path, model=self.model, map_location=self.device)
        best = self.eval_both("best_by_val")
        self.metrics["best_A_val_acc"] = best["A_val_acc"]
        self.metrics["best_A_test_acc"] = best["A_test_acc"]
        self.metrics["best_A_val_loss"] = best["A_val_loss"]
        self.metrics["best_A_test_loss"] = best["A_test_loss"]
        self.metrics["best_B_val_acc"] = best["B_val_acc"]
        self.metrics["best_B_test_acc"] = best["B_test_acc"]
        self.metrics["best_ckpt"] = str(best_path)
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
                self.metrics[f"best_{k.replace('/', '_')}"] = v
        log_wandb(best_log, step=state.step)
        if final_path.is_file():
            load_checkpoint(final_path, model=self.model, map_location=self.device)


# Compatibility names used by older imports / tests.
_eval_bundle = eval_bundle
_switch_stop = switch_stop
_write_history = write_history
_flatten_block = flatten_block
