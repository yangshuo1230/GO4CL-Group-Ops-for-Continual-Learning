"""Validation selection and test confirmation for mechanism reports."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

EXPLORATORY_SAME_SPLIT = "exploratory_same_split"
VAL_SELECT_TEST_CONFIRM = "val_select_test_confirm"


def select_and_confirm(
    rows: Sequence[Mapping[str, Any]],
    *,
    name_key: str,
    val_key: str,
    test_key: str,
    higher_is_better: bool = True,
) -> dict[str, Any]:
    """Choose a component by its validation score and report the test score.

    The test score is not used to pick the component.
    """
    if not rows:
        raise ValueError("select_and_confirm requires at least one candidate")
    ordered = sorted(
        rows,
        key=lambda row: float(row[val_key]),
        reverse=higher_is_better,
    )
    chosen = ordered[0]
    return {
        "selected_component": chosen[name_key],
        "val_selection_score": float(chosen[val_key]),
        "test_confirmation_score": float(chosen[test_key]),
        "selection_protocol": VAL_SELECT_TEST_CONFIRM,
    }
