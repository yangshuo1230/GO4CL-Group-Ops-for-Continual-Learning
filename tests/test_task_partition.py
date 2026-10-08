"""Task-partition specs, gate, and a tiny three-protocol run."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

import torch

from go4cl.constants import (
    PARTITION_VOCAB_SIZE,
    QUERY_TOKEN_IDS,
    TOKEN_Q0,
    TOKEN_TASK_C,
    VOCAB_SIZE,
)
from go4cl.phases.task_partition.config import smoke_config
from go4cl.phases.task_partition.data import (
    EqualPackedJointLoader,
    make_c_only_loader,
    peek_task_ids,
    prepare_dataset,
)
from go4cl.phases.task_partition.metrics import assess_partition, gap_to_upper, retention_block
from go4cl.phases.task_partition.run import run_task_partition
from go4cl.tasks.partition import PARTITION_MODULI, build_partition_tasks


def test_partition_tasks_match_the_operand_map() -> None:
    tasks = build_partition_tasks()
    expected = {
        "A": [(0, 1), (2, 3), (4, 5), (6, 7)],
        "B": [(0, 2), (1, 3), (4, 6), (5, 7)],
        "C": [(0, 4), (1, 5), (2, 6), (3, 7)],
    }
    assert TOKEN_Q0 == 66
    assert VOCAB_SIZE == 70
    assert TOKEN_TASK_C == 70
    assert PARTITION_VOCAB_SIZE == 71
    tokens = []
    for name, pairs in expected.items():
        task = tasks[name]
        tokens.append(task.task_token)
        assert task.moduli() == PARTITION_MODULI
        for slot, pair in enumerate(pairs):
            op = task.by_slot()[slot]
            assert (op.i, op.j) == pair
            assert op.modulus == PARTITION_MODULI[slot]
            assert QUERY_TOKEN_IDS[op.slot] == QUERY_TOKEN_IDS[slot]
    assert tokens == [64, 65, 70]
    digits = [0, 1, 2, 3, 4, 5, 6, 7]
    assert tasks["A"].evaluate_slot(digits, 0) == 1
    assert tasks["B"].evaluate_slot(digits, 0) == 2
    assert tasks["C"].evaluate_slot(digits, 0) == 4


def test_joint_and_c_only_mixes(tmp_path: Path) -> None:
    cfg = smoke_config(str(tmp_path))
    bundle = prepare_dataset(tmp_path / "data", cfg)
    tasks = bundle["tasks"]
    splits = bundle["splits"]
    ab = EqualPackedJointLoader(
        [tasks["A"], tasks["B"]], splits, batch_size=24, seed=0
    )
    ab_ids = peek_task_ids(ab)
    assert ab_ids.count(0) == 12
    assert ab_ids.count(1) == 12
    abc = EqualPackedJointLoader(
        [tasks["A"], tasks["B"], tasks["C"]], splits, batch_size=24, seed=1
    )
    abc_ids = peek_task_ids(abc)
    assert abc_ids.count(0) == abc_ids.count(1) == abc_ids.count(2) == 8
    c_only = make_c_only_loader(tasks["C"], splits, cfg)
    c_ids = peek_task_ids(c_only)
    assert set(c_ids) == {2}
    assert tasks["C"].task_token == TOKEN_TASK_C


def test_partition_gate_rejects_a_flat_task_token() -> None:
    failed = assess_partition(
        acc_a=0.95,
        acc_b=0.95,
        ab_matrix=[[0.95, 0.95], [0.95, 0.95]],
        flip_rate=0.02,
        n_disagree=100,
        min_acc=0.90,
        diag_min=0.80,
        off_max=0.20,
        flip_min=0.80,
    )
    assert failed["passed"] is False
    assert any("flip" in reason for reason in failed["reasons"])

    passed = assess_partition(
        acc_a=0.97,
        acc_b=0.96,
        ab_matrix=[[0.95, 0.04], [0.05, 0.94]],
        flip_rate=0.93,
        n_disagree=80,
        min_acc=0.90,
        diag_min=0.80,
        off_max=0.20,
        flip_min=0.80,
    )
    assert passed["passed"] is True


def test_retention_and_gap() -> None:
    block = retention_block({"A": 0.9, "B": 0.8, "C": 0.1}, {"A": 0.4, "B": 0.6, "C": 0.7})
    assert block["forgetting_A"] == pytest.approx(0.5)
    assert block["forgetting_B"] == pytest.approx(0.2)
    assert block["mean_old_retention"] == pytest.approx(0.5)
    assert block["worst_old_retention"] == pytest.approx(0.4)
    gap = gap_to_upper(
        {"A": 0.4, "B": 0.6, "C": 0.7},
        {"A": 0.9, "B": 0.9, "C": 0.9},
    )
    assert gap["A"] == pytest.approx(0.5)
    assert gap["B"] == pytest.approx(0.3)
    assert gap["C"] == pytest.approx(0.2)


def test_unpartitioned_ab_does_not_start_c(tmp_path: Path) -> None:
    cfg = smoke_config(str(tmp_path))
    cfg.require_partition = True
    cfg.device = "cpu"
    result = run_task_partition(cfg)
    assert result["status"] == "stopped"
    assert not (tmp_path / "c_only").exists()
    assert (tmp_path / "PARTITION_DIAGNOSTIC.md").is_file()
    assert (tmp_path / "AB_joint_checkpoint.pt").is_file()


def test_smoke_runs_three_protocols_from_one_ab_checkpoint(tmp_path: Path) -> None:
    cfg = smoke_config(str(tmp_path))
    cfg.device = "cpu"
    result = run_task_partition(cfg)
    assert result["status"] == "ok"
    canonical = tmp_path / "AB_joint_checkpoint.pt"
    assert canonical.is_file()
    c_phase = json.loads((tmp_path / "c_only" / "phase.json").read_text())
    ab_phase = json.loads((tmp_path / "ab_continued" / "phase.json").read_text())
    abc_phase = json.loads((tmp_path / "abc_joint" / "phase.json").read_text())
    assert c_phase["parent_checkpoint"] == str(canonical)
    assert ab_phase["parent_checkpoint"] == str(canonical)
    assert c_phase["parent_sha256"] == ab_phase["parent_sha256"]
    assert c_phase["mixes_other_tasks"] is False
    assert abc_phase["init_checkpoint"] == str(tmp_path / "init_checkpoint.pt")
    assert (tmp_path / "c_only" / "ckpts" / "c_step0.pt").is_file()
    assert (tmp_path / "c_only" / "ckpts" / "c_step2.pt").is_file()
    assert (tmp_path / "c_only" / "ckpts" / "c_first_stable.pt").is_file()
    for name in (
        "accuracy_curves.png",
        "final_accuracy_comparison.png",
        "task_token_counterfactual.png",
        "routing_dynamics.png",
    ):
        figure = tmp_path / "figures" / name
        assert figure.is_file() and figure.stat().st_size > 0
    comparison = result["comparison"]["ab_then_c"]
    assert "forgetting_A" in comparison
    assert "mean_old_retention" in comparison
    assert "worst_old_retention" in comparison
    assert result["per_query_test"]["c_final"]["A"][0]["query"] == "Q0"
    package = Path(__file__).resolve().parents[1] / "src" / "go4cl" / "phases" / "task_partition"
    text = "\n".join(path.read_text(encoding="utf-8") for path in package.glob("*.py"))
    assert "MixedPackedReplayLoader" not in text
    assert "from go4cl.data.null_task" not in text
    assert "sequential_ab_replay" not in text
    model = torch.load(canonical, map_location="cpu", weights_only=False)
    assert model["model_config"]["vocab_size"] == PARTITION_VOCAB_SIZE
    assert "optimizer_state" in model
