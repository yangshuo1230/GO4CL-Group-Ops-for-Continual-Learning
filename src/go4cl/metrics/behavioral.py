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

    def to_dict(self) -> dict[str, Any]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "n": self.n,
            "by_slot": {str(k): v for k, v in self.by_slot.items()},
            "by_modulus": {str(k): v for k, v in self.by_modulus.items()},
        }


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
    slot_stats: dict[int, list[float]] = {}
    mod_stats: dict[int, list[float]] = {}

    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        slots = batch["slots"]
        moduli = batch["moduli"]
        out = model(tokens, labels)
        logits = out["logits"]
        loss = out["loss"]
        preds = logits.argmax(dim=-1)
        correct = preds == labels
        bs = labels.shape[0]
        total_loss += float(loss.item()) * bs
        total_correct += int(correct.sum().item())
        total_n += bs
        for i in range(bs):
            s = int(slots[i].item())
            m = int(moduli[i].item())
            slot_stats.setdefault(s, []).append(float(correct[i].item()))
            mod_stats.setdefault(m, []).append(float(correct[i].item()))

    def _agg(stats: dict[int, list[float]]) -> dict[int, dict[str, float]]:
        return {
            k: {"accuracy": sum(v) / max(len(v), 1), "n": float(len(v))}
            for k, v in sorted(stats.items())
        }

    return EvalResult(
        loss=total_loss / max(total_n, 1),
        accuracy=total_correct / max(total_n, 1),
        n=total_n,
        by_slot=_agg(slot_stats),
        by_modulus=_agg(mod_stats),
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
