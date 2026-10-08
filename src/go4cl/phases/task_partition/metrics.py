"""Forgetting, retention, and the AB task-token gate."""

from __future__ import annotations

from typing import Any

import math


def _finite(value: float | None) -> bool:
    return value is not None and not math.isnan(float(value)) and not math.isinf(float(value))


def retention_block(
    before: dict[str, float],
    after: dict[str, float],
) -> dict[str, float]:
    """Forgetting is the drop in test accuracy from the shared AB checkpoint."""
    forgetting_a = float(before["A"]) - float(after["A"])
    forgetting_b = float(before["B"]) - float(after["B"])
    return {
        "forgetting_A": forgetting_a,
        "forgetting_B": forgetting_b,
        "mean_old_retention": (float(after["A"]) + float(after["B"])) / 2.0,
        "worst_old_retention": min(float(after["A"]), float(after["B"])),
        "acc_A": float(after["A"]),
        "acc_B": float(after["B"]),
        "acc_C": float(after["C"]),
    }


def gap_to_upper(
    final: dict[str, float],
    upper: dict[str, float],
) -> dict[str, float]:
    """Positive means the joint upper bound is more accurate than ``final``."""
    return {task: float(upper[task]) - float(final[task]) for task in ("A", "B", "C")}


def assess_partition(
    *,
    acc_a: float,
    acc_b: float,
    ab_matrix: list[list[float | None]],
    flip_rate: float | None,
    n_disagree: int,
    min_acc: float,
    diag_min: float,
    off_max: float,
    flip_min: float,
) -> dict[str, Any]:
    """Stop the C phase unless AB uses the task token on disagreeing items.

    ``ab_matrix[u][v]`` is P(prediction under TASK_u equals task v's label)
    on samples with ``y_A != y_B``.
    """
    reasons: list[str] = []
    if float(acc_a) < float(min_acc):
        reasons.append(f"A test accuracy {acc_a:.3f} < {min_acc:.2f}")
    if float(acc_b) < float(min_acc):
        reasons.append(f"B test accuracy {acc_b:.3f} < {min_acc:.2f}")
    if int(n_disagree) <= 0:
        reasons.append("no evaluation samples with y_A != y_B")
    else:
        diagonal = [ab_matrix[0][0], ab_matrix[1][1]]
        off = [ab_matrix[0][1], ab_matrix[1][0]]
        labels = (("TASK_A", "A"), ("TASK_B", "B"))
        for (token, target), value in zip(labels, diagonal, strict=True):
            if not _finite(value) or float(value) < float(diag_min):
                reasons.append(
                    f"{token} match to label {target} is {value} < {diag_min:.2f}"
                )
        crossed = (("TASK_A", "B"), ("TASK_B", "A"))
        for (token, target), value in zip(crossed, off, strict=True):
            if not _finite(value) or float(value) > float(off_max):
                reasons.append(
                    f"{token} match to label {target} is {value} > {off_max:.2f}"
                )
        if not _finite(flip_rate) or float(flip_rate) < float(flip_min):
            reasons.append(
                f"task-token flip rate {flip_rate} < {flip_min:.2f}; "
                "swapping TASK_A and TASK_B does not change the prediction"
            )
    return {
        "passed": not reasons,
        "reasons": reasons,
        "A_test_acc": float(acc_a),
        "B_test_acc": float(acc_b),
        "ab_matrix": ab_matrix,
        "task_token_flip_rate": None if flip_rate is None else float(flip_rate),
        "n_disagree_ab": int(n_disagree),
        "thresholds": {
            "min_acc": float(min_acc),
            "diag_min": float(diag_min),
            "off_max": float(off_max),
            "flip_min": float(flip_min),
        },
    }
