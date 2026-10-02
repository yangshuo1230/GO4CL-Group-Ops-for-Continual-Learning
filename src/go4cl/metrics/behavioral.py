"""Behavioral metrics for continual learning experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from go4cl.model.transformer import ModularTransformer


@dataclass
class EvalResult:
    loss: float
    accuracy: float
    n: int
    by_slot: dict[int, dict[str, float]] = field(default_factory=dict)
    by_modulus: dict[int, dict[str, float]] = field(default_factory=dict)
    by_operation: dict[str, dict[str, float]] = field(default_factory=dict)
    macro_operation_accuracy: float = 0.0
    macro_operation_loss: float = 0.0
    mean_correct_logit_margin: float = 0.0
    normalized_cross_entropy: float = 0.0

    # Back-compat aliases used by older callers
    @property
    def micro_accuracy(self) -> float:
        return self.accuracy

    @property
    def micro_loss(self) -> float:
        return self.loss

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "micro_loss": self.loss,
            "micro_accuracy": self.accuracy,
            "n": self.n,
            "n_examples": self.n,
            "by_slot": {str(k): v for k, v in self.by_slot.items()},
            "by_modulus": {str(k): v for k, v in self.by_modulus.items()},
            "by_operation": dict(self.by_operation),
            "macro_operation_accuracy": self.macro_operation_accuracy,
            "macro_operation_loss": self.macro_operation_loss,
            "mean_correct_logit_margin": self.mean_correct_logit_margin,
            "normalized_cross_entropy": self.normalized_cross_entropy,
        }


def _op_key(task_id: int, latent_id: int, slot: int) -> str:
    if latent_id >= 0:
        return f"task{task_id}/lat{latent_id}/slot{slot}"
    return f"task{task_id}/slot{slot}"


@torch.no_grad()
def evaluate(
    model: ModularTransformer,
    loader: DataLoader,
    device: torch.device,
) -> EvalResult:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_n = 0
    margin_sum = 0.0
    nce_sum = 0.0
    slot_stats: dict[int, list[float]] = {}
    mod_stats: dict[int, list[float]] = {}
    op_correct: dict[str, list[float]] = {}
    op_loss: dict[str, list[float]] = {}

    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        slots = batch["slots"]
        moduli = batch["moduli"]
        task_ids = batch.get("task_ids")
        latent_ids = batch.get("latent_ids")
        out = model(tokens, labels)
        logits = out["logits"]
        loss = out["loss"]
        preds = logits.argmax(dim=-1)
        correct = preds == labels
        bs = labels.shape[0]
        # per-example CE for macro loss
        ce = F.cross_entropy(logits, labels, reduction="none")
        total_loss += float(loss.item()) * bs
        total_correct += int(correct.sum().item())
        total_n += bs
        margin_sum += mean_margin(logits, labels) * bs
        if moduli is not None:
            nce_sum += normalized_ce(logits, labels, moduli.to(device)) * bs

        for i in range(bs):
            s = int(slots[i].item())
            m = int(moduli[i].item())
            tid = int(task_ids[i].item()) if task_ids is not None else 0
            lid = int(latent_ids[i].item()) if latent_ids is not None else -1
            ok = float(correct[i].item())
            slot_stats.setdefault(s, []).append(ok)
            mod_stats.setdefault(m, []).append(ok)
            key = _op_key(tid, lid, s)
            op_correct.setdefault(key, []).append(ok)
            op_loss.setdefault(key, []).append(float(ce[i].item()))

    def _agg_acc(stats: dict[int, list[float]]) -> dict[int, dict[str, float]]:
        return {
            k: {"accuracy": sum(v) / max(len(v), 1), "n": float(len(v))}
            for k, v in sorted(stats.items())
        }

    by_operation: dict[str, dict[str, float]] = {}
    for key in sorted(op_correct.keys()):
        accs = op_correct[key]
        losses = op_loss[key]
        by_operation[key] = {
            "accuracy": sum(accs) / max(len(accs), 1),
            "loss": sum(losses) / max(len(losses), 1),
            "n": float(len(accs)),
        }

    if by_operation:
        macro_acc = sum(v["accuracy"] for v in by_operation.values()) / len(by_operation)
        macro_loss = sum(v["loss"] for v in by_operation.values()) / len(by_operation)
    else:
        macro_acc = total_correct / max(total_n, 1)
        macro_loss = total_loss / max(total_n, 1)

    return EvalResult(
        loss=total_loss / max(total_n, 1),
        accuracy=total_correct / max(total_n, 1),
        n=total_n,
        by_slot=_agg_acc(slot_stats),
        by_modulus=_agg_acc(mod_stats),
        by_operation=by_operation,
        macro_operation_accuracy=macro_acc,
        macro_operation_loss=macro_loss,
        mean_correct_logit_margin=margin_sum / max(total_n, 1),
        normalized_cross_entropy=nce_sum / max(total_n, 1),
    )


def forgetting(max_acc_a: float, acc_a_after_b: float) -> float:
    """F_A = max_t Acc_A(t) - Acc_A(after B)."""
    return max_acc_a - acc_a_after_b


def mean_margin(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Mean logit margin: correct logit minus max incorrect logit."""
    with torch.no_grad():
        gather = logits.gather(1, labels.view(-1, 1)).squeeze(1)
        mask = torch.ones_like(logits, dtype=torch.bool)
        mask.scatter_(1, labels.view(-1, 1), False)
        max_other = logits.masked_fill(~mask, float("-inf")).max(dim=1).values
        return float((gather - max_other).mean().item())


def normalized_ce(logits: torch.Tensor, labels: torch.Tensor, modulus: torch.Tensor) -> float:
    """Cross-entropy normalized by log p, averaged over the batch."""
    with torch.no_grad():
        ce = F.cross_entropy(logits, labels, reduction="none")
        return float((ce / modulus.float().clamp_min(2).log()).mean().item())
