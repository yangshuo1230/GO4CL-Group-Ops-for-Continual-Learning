"""Per-operation AUC, stable generalization, and fixed-A identity."""

from __future__ import annotations

import pytest

from go4cl.phases.transfer_mechanism.analysis import (
    assert_shared_task_a,
    component_transfer_rows,
    exposure_auc,
    first_reach_steps_to_gen,
    necessity_score,
    per_op_transfer_rows,
    stable_steps_to_gen,
    sufficiency_score,
    task_a_hash,
)
from go4cl.phases.transfer_mechanism.configs import MODULUS_CONDITIONS
from go4cl.tasks.relations import build_task_pair


def test_per_op_auc_and_stable_threshold() -> None:
    assert exposure_auc([(0.0, 0.0), (10.0, 1.0)]) == pytest.approx(0.5)
    dip = [
        (0.0, 0.5),
        (1.0, 0.91),
        (2.0, 0.92),
        (3.0, 0.93),
        (4.0, 0.94),
        (5.0, 0.2),
        (6.0, 0.95),
        (7.0, 0.96),
        (8.0, 0.97),
        (9.0, 0.98),
        (10.0, 0.99),
    ]
    assert first_reach_steps_to_gen(dip, 0.9) == 1.0
    assert stable_steps_to_gen(dip, 0.9, 5) == 10.0
    assert stable_steps_to_gen([(0.0, 1.0), (1.0, 1.0), (2.0, 1.0), (3.0, 1.0)], 0.9, 5) is None

    ops = [
        {
            "operation": "op0",
            "latent_id": 0,
            "slot": 1,
            "operand_pair": "0+1",
            "modulus": 23,
            "modulus_seen_in_A": True,
            "same_latent_modulus_as_A": True,
        },
        {
            "operation": "op1",
            "latent_id": 1,
            "slot": 0,
            "operand_pair": "2+3",
            "modulus": 29,
            "modulus_seen_in_A": False,
            "same_latent_modulus_as_A": False,
        },
    ]

    def _job(protocol: str, rows: list[tuple[float, float]]) -> dict:
        history = []
        for step, (op0, op1) in enumerate(rows):
            history.append(
                {
                    "curve": "B",
                    "pre_b": step == 0,
                    "step": float(step),
                    "b_exposure": float(step),
                    "B_test_acc": (op0 + op1) / 2.0,
                    "B_test_acc/op0": op0,
                    "B_test_acc/op1": op1,
                }
            )
        return {
            "status": "ok",
            "condition": "s0_o0_m1",
            "protocol": protocol,
            "task_seed": 0,
            "model_seed": 1,
            "gen_threshold": 0.9,
            "stable_window": 5,
            "operations": ops,
            "history": history,
        }

    baseline_rows = [(0.0, 0.0), (2.0, 0.0)]
    transfer_rows = [
        (0.2, 0.1),
        (0.95, 0.1),
        (0.96, 0.1),
        (0.97, 0.1),
        (0.98, 0.1),
        (0.99, 0.1),
    ]
    table = per_op_transfer_rows(
        [
            _job("b_only", baseline_rows),
            _job("sequential_ab", transfer_rows),
        ]
    )
    sequential = [row for row in table if row["protocol"] == "sequential_ab"]
    by_op = {row["operation"]: row for row in sequential}
    assert by_op["op0"]["modulus_seen_in_A"] is True
    assert by_op["op1"]["modulus_seen_in_A"] is False
    assert by_op["op0"]["same_latent_modulus_as_A"] is True
    assert by_op["op0"]["b_exposure_auc"] == pytest.approx(
        exposure_auc([(float(i), y) for i, (y, _) in enumerate(transfer_rows)])
    )
    assert by_op["op0"]["first_reach_steps_to_gen"] == 1.0
    assert by_op["op0"]["b_exposure_steps_to_gen"] == 1.0
    assert by_op["op0"]["stable_steps_to_gen"] == 5.0
    assert by_op["op0"]["delta_b_exposure_auc"] == pytest.approx(
        by_op["op0"]["b_exposure_auc"] - by_op["op0"]["baseline_b_exposure_auc"]
    )
    assert by_op["op0"]["stable_delta_steps_to_gen"] is None
    assert by_op["op1"]["stable_steps_to_gen"] is None

    assert necessity_score(0.8, 0.5, 0.2) == pytest.approx(0.5)
    assert necessity_score(0.8, 0.9, 0.2) < 0
    assert sufficiency_score(0.95, 0.8, 0.2) > 1
    assert sufficiency_score(0.4, 0.4, 0.4) is None
    component = component_transfer_rows(
        [
            {
                "status": "ok",
                "condition": "s0_o0_m1",
                "task_seed": 0,
                "model_seed": 0,
                "intervention": "full_A",
                "kind": "full_A",
                "groups_from_A": ["digit_embedding", "mlp"],
                "groups_from_init": [],
                "history": [
                    {"curve": "B", "b_exposure": 0, "pre_b": True, "B_test_acc": 0.1},
                    {"curve": "B", "b_exposure": 10, "B_test_acc": 0.9},
                ],
            },
            {
                "status": "ok",
                "condition": "s0_o0_m1",
                "task_seed": 0,
                "model_seed": 0,
                "intervention": "full_fresh",
                "kind": "full_fresh",
                "groups_from_A": [],
                "groups_from_init": ["digit_embedding"],
                "history": [
                    {"curve": "B", "b_exposure": 0, "pre_b": True, "B_test_acc": 0.1},
                    {"curve": "B", "b_exposure": 10, "B_test_acc": 0.3},
                ],
            },
            {
                "status": "ok",
                "condition": "s0_o0_m1",
                "task_seed": 0,
                "model_seed": 0,
                "intervention": "reset_mlp",
                "kind": "reset",
                "groups_from_A": ["digit_embedding"],
                "groups_from_init": ["mlp"],
                "history": [
                    {"curve": "B", "b_exposure": 0, "pre_b": True, "B_test_acc": 0.1},
                    {"curve": "B", "b_exposure": 10, "B_test_acc": 0.2},
                ],
            },
        ]
    )
    by_name = {row["intervention"]: row for row in component}
    # AUC of (0.1 -> 0.9) is 0.5; (0.1 -> 0.3) is 0.2; (0.1 -> 0.2) is 0.15.
    assert by_name["full_A"]["b_exposure_auc"] == pytest.approx(0.5)
    assert by_name["reset_mlp"]["necessity_score"] == pytest.approx((0.5 - 0.15) / (0.5 - 0.2))
    assert by_name["full_fresh"]["sufficiency_score"] == pytest.approx(0.0)
    assert by_name["full_A"]["sufficiency_score"] == pytest.approx(1.0)
    assert by_name["reset_mlp"]["pre_B_B_acc"] == pytest.approx(0.1)


def test_fixed_a_identical_across_modulus_conditions() -> None:
    digest = assert_shared_task_a(4, MODULUS_CONDITIONS)
    specs = []
    b_specs = []
    for cond in MODULUS_CONDITIONS:
        pair = build_task_pair(
            rho_slot=float(cond["rho_slot"]),
            rho_operand=float(cond["rho_operand"]),
            rho_mod=float(cond["rho_mod"]),
            task_seed=4,
            fixed_a=True,
        )
        specs.append(pair.task_a.to_dict())
        b_specs.append(pair.task_b.to_dict())
        assert task_a_hash(pair) == digest
        assert pair.pair_id.endswith("_fixedA")
    assert specs[0] == specs[1] == specs[2]
    assert len({str(spec) for spec in b_specs}) == 3
