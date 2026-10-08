"""Checkpoint selection and stable-event detection (val-only)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


PrimaryMetric = Literal["macro_operation_accuracy"]
TieBreaker = Literal["macro_operation_loss"]


@dataclass
class CheckpointSelector:
    """Select best checkpoint from validation metrics only.

    Never pass test metrics into ``observe``.
    """

    primary: PrimaryMetric = "macro_operation_accuracy"
    tie_breaker: TieBreaker = "macro_operation_loss"
    min_delta: float = 0.0
    best_step: int = 0
    best_primary: float = float("-inf")
    best_tie: float = float("inf")
    history: list[dict[str, Any]] = field(default_factory=list)

    def observe(
        self,
        step: int,
        *,
        macro_operation_accuracy: float,
        macro_operation_loss: float,
    ) -> bool:
        """Return True if this step becomes the new best."""
        acc = float(macro_operation_accuracy)
        loss = float(macro_operation_loss)
        self.history.append(
            {
                "step": int(step),
                "macro_operation_accuracy": acc,
                "macro_operation_loss": loss,
            }
        )
        improved_acc = acc > self.best_primary + self.min_delta
        tied_acc = abs(acc - self.best_primary) <= self.min_delta
        improved_tie = tied_acc and loss < self.best_tie
        if improved_acc or improved_tie or self.best_primary == float("-inf"):
            if improved_acc or self.best_primary == float("-inf"):
                self.best_primary = acc
                self.best_tie = loss
                self.best_step = int(step)
                return True
            if improved_tie:
                self.best_tie = loss
                self.best_step = int(step)
                return True
        return False


@dataclass
class StableEventDetector:
    """Require ``window`` consecutive evals above threshold."""

    name: str
    threshold: float
    window: int = 5
    maximize: bool = True
    _streak: int = 0
    triggered_step: int | None = None

    def observe(self, step: int, value: float) -> bool:
        if self.triggered_step is not None:
            return False
        ok = value >= self.threshold if self.maximize else value <= self.threshold
        if ok:
            self._streak += 1
        else:
            self._streak = 0
        if self._streak >= self.window:
            self.triggered_step = int(step)
            return True
        return False


def default_event_detectors() -> dict[str, StableEventDetector]:
    """Canonical Phase-1 event detectors (t_mem / t_gen / t_iid)."""
    return {
        "t_mem": StableEventDetector("t_mem", threshold=0.99, window=5),
        "t_gen": StableEventDetector("t_gen", threshold=0.90, window=5),
        "t_iid": StableEventDetector("t_iid", threshold=0.95, window=5),
    }


def task_scoped_detectors() -> dict[str, StableEventDetector]:
    """Phase-1 aliases plus ``A_*`` / ``B_*`` task events. Window and thresholds match."""
    detectors = default_event_detectors()
    for task in ("A", "B"):
        for name, threshold in (
            ("t_mem", 0.99),
            ("t_gen", 0.90),
            ("t_iid", 0.95),
        ):
            key = f"{task}_{name}"
            detectors[key] = StableEventDetector(key, threshold=threshold, window=5)
    return detectors


def task_event_values(loader_macros: dict[str, float]) -> tuple[dict[str, float], str | None]:
    """Map loader names to task-scoped event values.

    Lookup is by exact loader name, so the order of ``loader_macros`` does not
    matter. When both ``A_val`` and ``B_val`` are present, ``t_gen`` is omitted:
    the A/B average is not a definition of B having learned. A single-task run
    still fills ``t_mem`` / ``t_gen`` / ``t_iid`` and returns that task name as
    the compatibility annotation.
    """

    def _lookup(*names: str) -> float | None:
        for name in names:
            if name in loader_macros:
                return float(loader_macros[name])
        return None

    has_a = any(name.startswith("A_") for name in loader_macros)
    has_b = any(name.startswith("B_") for name in loader_macros)
    values: dict[str, float] = {}
    if has_a:
        for event, name in (
            ("A_t_mem", "A_train_eval"),
            ("A_t_gen", "A_val"),
            ("A_t_iid", "A_iid"),
        ):
            value = _lookup(name)
            if value is not None:
                values[event] = value
    if has_b:
        for event, name in (
            ("B_t_mem", "B_train_eval"),
            ("B_t_gen", "B_val"),
            ("B_t_iid", "B_iid"),
        ):
            value = _lookup(name)
            if value is not None:
                values[event] = value
    if has_a and has_b:
        return values, None
    if has_b and not has_a:
        for src, dst in (("B_t_mem", "t_mem"), ("B_t_gen", "t_gen"), ("B_t_iid", "t_iid")):
            if src in values:
                values[dst] = values[src]
        return values, "B"
    if has_a:
        for src, dst in (("A_t_mem", "t_mem"), ("A_t_gen", "t_gen"), ("A_t_iid", "t_iid")):
            if src in values:
                values[dst] = values[src]
        return values, "A"
    for event, name in (
        ("t_mem", "train_eval"),
        ("t_gen", "val"),
        ("t_iid", "iid"),
    ):
        value = _lookup(name)
        if value is not None:
            values[event] = value
    if not values:
        return values, None
    return values, "single"
