"""Phase-2 behavioral summaries: forgetting, jump, exposure AUC, grok order.

Fourier / probe-before-behavior is left to phase 3. These functions only use
the numeric eval history written during training.
"""

from __future__ import annotations

from typing import Any

from go4cl.defaults import PHASE2

# Protocols whose second phase (or mixture) should be compared to B-from-scratch.
TRANSFER_PROTOCOLS = (
    "sequential_ab",
    "sequential_ab_replay",
    "sequential_ba",
    "joint",
    "interleaved",
)
SWITCH_PROTOCOLS = (
    "sequential_ab",
    "sequential_ab_replay",
    "sequential_ba",
    "a_only_continued",
)

GROUP_KEYS = (
    "rho_slot",
    "rho_operand",
    "rho_mod",
    "task_seed",
    "model_seed",
    "direction",
    "d_model",
    "n_layers",
)


def _series(history: list[dict[str, Any]], key: str) -> list[tuple[float, float]]:
    pts: list[tuple[float, float]] = []
    for row in history:
        value = row.get(key)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            pts.append((float(row["step"]), float(value)))
    pts.sort()
    return pts


def _pick(history: list[dict[str, Any]], keys: tuple[str, ...]) -> str | None:
    for key in keys:
        if any(key in row for row in history):
            return key
    return None


def _auc(series: list[tuple[float, float]]) -> float | None:
    if len(series) < 2:
        return None
    area = 0.0
    for (s0, y0), (s1, y1) in zip(series, series[1:]):
        dt = s1 - s0
        if dt <= 0:
            continue
        area += dt * (y0 + y1) / 2.0
    span = series[-1][0] - series[0][0]
    if span <= 0:
        return None
    return area / span


def _first_reach(series: list[tuple[float, float]], threshold: float) -> float | None:
    for step, value in series:
        if value >= threshold:
            return step
    return None


def _steepest_slope(series: list[tuple[float, float]]) -> float:
    """Most negative per-step slope (0 if the curve never falls)."""
    best = 0.0
    for (s0, y0), (s1, y1) in zip(series, series[1:]):
        dt = s1 - s0
        if dt <= 0:
            continue
        slope = (y1 - y0) / dt
        if slope < best:
            best = slope
    return best


def _at_or_before(
    series: list[tuple[float, float]], step: float
) -> tuple[float, float] | None:
    chosen: tuple[float, float] | None = None
    for point in series:
        if point[0] <= step:
            chosen = point
        else:
            break
    return chosen


def _first_after(
    series: list[tuple[float, float]], step: float
) -> tuple[float, float] | None:
    for point in series:
        if point[0] > step:
            return point
    return None


def b_exposure(protocol: str, step: float, switch_step: float | None) -> float | None:
    """Map a global step onto B-example exposure, in full-batch step units.

    Joint and interleaved run 2S steps so each task expects S steps of data:
    exposure = step / 2. Sequential A→B counts only steps after the switch.
    Sequential A→B with replay scales post-switch steps by (1 - replay
    fraction) because that fraction of each B-phase batch is A.
    Sequential B→A counts only steps up to the switch.
    """
    if protocol == "b_only":
        return float(step)
    if protocol in {"sequential_ab", "sequential_ab_replay"}:
        if switch_step is None:
            return 0.0
        exposed = max(0.0, float(step) - float(switch_step))
        if protocol == "sequential_ab_replay":
            exposed *= 1.0 - float(PHASE2.sequential_ab_replay_ratio)
        return exposed
    if protocol == "sequential_ba":
        if switch_step is None:
            return float(step)
        return float(min(step, switch_step))
    if protocol in {"joint", "interleaved"}:
        return float(step) / 2.0
    return None


def _curve_stats(
    out: dict[str, Any],
    prefix: str,
    series: list[tuple[float, float]],
    loss_series: list[tuple[float, float]],
    boundary: float | None,
) -> None:
    if not series:
        return
    values = [y for _, y in series]
    out[f"max_{prefix}_acc"] = max(values)
    out[f"final_{prefix}_acc"] = series[-1][1]
    out[f"forgetting_{prefix}"] = max(values) - series[-1][1]
    if out[f"max_{prefix}_acc"] > 0:
        out[f"retention_{prefix}"] = series[-1][1] / out[f"max_{prefix}_acc"]
    if boundary is None:
        return
    before = _at_or_before(series, boundary)
    after = _first_after(series, boundary)
    if before is None or after is None:
        return
    out[f"acc_{prefix}_at_switch"] = before[1]
    out[f"acc_{prefix}_after_switch"] = after[1]
    out[f"jump_{prefix}"] = before[1] - after[1]
    post = [(s, y) for s, y in series if s >= before[0]]
    out[f"forget_rate_{prefix}"] = _steepest_slope(post)
    if loss_series:
        loss_before = _at_or_before(loss_series, boundary)
        loss_after = _first_after(loss_series, boundary)
        if loss_before is not None and loss_after is not None:
            out[f"jump_loss_{prefix}"] = loss_after[1] - loss_before[1]


def summarize_behavior(
    protocol: str,
    history: list[dict[str, Any]],
    *,
    switch_step: float | None = None,
    gen_threshold: float = 0.9,
) -> dict[str, Any]:
    """Summarize one run's eval history.

    Accuracy series prefer held-out test when it was logged, otherwise val.
    ``forgetting_*`` is max accuracy minus the final accuracy (negative means
    backward improvement). ``jump_*`` is the drop across the task switch.
    ``b_exposure_*`` puts B on a from-scratch-comparable x-axis.
    """
    hist = list(history)
    a_key = _pick(hist, ("A_test_acc", "A_val_acc"))
    b_key = _pick(hist, ("B_test_acc", "B_val_acc"))
    a_loss_key = _pick(hist, ("A_test_loss", "A_val_loss"))
    b_loss_key = _pick(hist, ("B_test_loss", "B_val_loss"))
    out: dict[str, Any] = {
        "protocol": protocol,
        "switch_step": switch_step,
        "n_eval": len(hist),
        "acc_key_A": a_key,
        "acc_key_B": b_key,
        "gen_threshold": gen_threshold,
    }
    a_series = _series(hist, a_key) if a_key else []
    b_series = _series(hist, b_key) if b_key else []
    a_loss = _series(hist, a_loss_key) if a_loss_key else []
    b_loss = _series(hist, b_loss_key) if b_loss_key else []
    boundary = float(switch_step) if switch_step is not None else None
    a_boundary = boundary if protocol in SWITCH_PROTOCOLS else None
    b_boundary = boundary if protocol == "sequential_ba" else None
    _curve_stats(out, "A", a_series, a_loss, a_boundary)
    _curve_stats(out, "B", b_series, b_loss, b_boundary)

    if b_series and protocol in {"b_only", *TRANSFER_PROTOCOLS}:
        exposed: list[tuple[float, float]] = []
        for step, value in b_series:
            if (
                protocol in {"sequential_ab", "sequential_ab_replay"}
                and boundary is not None
                and step <= boundary
            ):
                continue
            if (
                protocol == "sequential_ba"
                and boundary is not None
                and step > boundary
            ):
                continue
            x = b_exposure(protocol, step, boundary)
            if x is None:
                continue
            exposed.append((x, value))
        exposed.sort()
        out["b_exposure_auc"] = _auc(exposed)
        out["b_exposure_steps_to_gen"] = _first_reach(exposed, gen_threshold)

    out["A_steps_to_gen"] = _first_reach(a_series, gen_threshold)
    out["B_steps_to_gen"] = _first_reach(b_series, gen_threshold)
    a_hit = out["A_steps_to_gen"]
    b_hit = out["B_steps_to_gen"]
    if a_hit is None and b_hit is None:
        order = "neither"
    elif a_hit is None:
        order = "B_only"
    elif b_hit is None:
        order = "A_only"
    elif a_hit == b_hit:
        order = "simultaneous"
    elif a_hit < b_hit:
        order = "A_then_B"
    else:
        order = "B_then_A"
    out["grok_order"] = order
    return out


def _group_key(row: dict[str, Any]) -> tuple:
    return tuple(row.get(key) for key in GROUP_KEYS)


def forward_transfer_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Compare each mixed/sequential run to the matched B-only baseline.

    Positive ``delta_b_exposure_auc`` means a higher B curve than from-scratch.
    Positive ``delta_b_exposure_steps_to_gen`` means the threshold was reached
    in fewer B-exposure steps than from-scratch.
    """
    baselines: dict[tuple, dict[str, Any]] = {}
    for row in records:
        if row.get("protocol") == "b_only" and row.get("status") == "ok":
            baselines[_group_key(row)] = row
    out: list[dict[str, Any]] = []
    for row in records:
        protocol = row.get("protocol")
        if protocol not in TRANSFER_PROTOCOLS or row.get("status") != "ok":
            continue
        base = baselines.get(_group_key(row))
        if base is None:
            continue
        built: dict[str, Any] = {key: row.get(key) for key in GROUP_KEYS}
        built["protocol"] = protocol
        built["job_id"] = row.get("job_id")
        for field, higher_is_better in (
            ("b_exposure_auc", True),
            ("b_exposure_steps_to_gen", False),
        ):
            base_v = base.get(field)
            run_v = row.get(field)
            built[f"baseline_{field}"] = base_v
            built[field] = run_v
            if isinstance(base_v, (int, float)) and isinstance(run_v, (int, float)):
                built[f"delta_{field}"] = (
                    run_v - base_v if higher_is_better else base_v - run_v
                )
        out.append(built)
    return out
