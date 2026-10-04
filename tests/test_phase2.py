"""Phase 2 grids, packed joint batches, and behavioral summaries."""

from __future__ import annotations

from pathlib import Path

import pytest

from go4cl.metrics.continual import forward_transfer_rows, summarize_behavior
from go4cl.phases.phase2.grid import (
    CAPACITY_CONDITIONS,
    rho_grid,
)


def test_rho_grids_match_the_plan() -> None:
    extreme = rho_grid("extreme")
    full = rho_grid("full")
    assert len(extreme) == 8
    assert len(full) == 27
    assert {(c.rho_slot, c.rho_operand, c.rho_mod) for c in extreme} <= {
        (c.rho_slot, c.rho_operand, c.rho_mod) for c in full
    }
    assert len(CAPACITY_CONDITIONS) == 6
    assert {c.name for c in CAPACITY_CONDITIONS} >= {
        "identical",
        "slot_diff",
        "operand_diff",
        "mod_diff",
        "disjoint",
        "partial",
    }


def test_summarize_sequential_forgetting_and_exposure() -> None:
    history = [
        {
            "step": 100,
            "segment": "phase_a",
            "A_val_acc": 0.2,
            "A_val_loss": 2.0,
            "B_val_acc": 0.0,
        },
        {
            "step": 200,
            "segment": "phase_a",
            "A_val_acc": 0.95,
            "A_val_loss": 0.2,
            "B_val_acc": 0.1,
        },
        {
            "step": 300,
            "segment": "phase_b",
            "A_val_acc": 0.5,
            "A_val_loss": 1.0,
            "B_val_acc": 0.2,
        },
        {
            "step": 400,
            "segment": "phase_b",
            "A_val_acc": 0.4,
            "A_val_loss": 1.2,
            "B_val_acc": 0.8,
        },
    ]
    summary = summarize_behavior("sequential_ab", history, switch_step=200)
    assert summary["forgetting_A"] == pytest.approx(0.55)
    assert summary["jump_A"] == pytest.approx(0.45)
    assert summary["jump_loss_A"] == pytest.approx(0.8)
    assert summary["forget_rate_A"] == pytest.approx(-0.0045)
    assert summary["retention_A"] == pytest.approx(0.4 / 0.95)
    assert summary["b_exposure_auc"] == pytest.approx(0.5)
    assert summary["grok_order"] == "A_only"
    assert summary["b_exposure_steps_to_gen"] is None


def test_forward_transfer_signs() -> None:
    keys = {
        "rho_slot": 1.0,
        "rho_operand": 0.0,
        "rho_mod": 1.0,
        "task_seed": 0,
        "model_seed": 1,
        "direction": "forward",
        "d_model": 64,
        "n_layers": 3,
        "status": "ok",
    }
    rows = forward_transfer_rows(
        [
            {**keys, "protocol": "b_only", "b_exposure_auc": 0.4, "b_exposure_steps_to_gen": 100},
            {
                **keys,
                "protocol": "sequential_ab",
                "b_exposure_auc": 0.55,
                "b_exposure_steps_to_gen": 80,
            },
        ]
    )
    assert len(rows) == 1
    assert rows[0]["delta_b_exposure_auc"] == pytest.approx(0.15)
    assert rows[0]["delta_b_exposure_steps_to_gen"] == pytest.approx(20)


def test_balanced_packed_joint_is_half_half() -> None:
    import numpy as np

    from go4cl.data.generate import build_shared_residue_splits
    from go4cl.data.packed import BalancedPackedJointLoader
    from go4cl.tasks.relations import build_task_pair

    pair = build_task_pair(rho_slot=1.0, rho_operand=1.0, rho_mod=1.0, task_seed=0)
    splits = build_shared_residue_splits(
        [pair.task_a, pair.task_b], data_seed=0, ratios=(0.6, 0.2, 0.2)
    )
    loader = BalancedPackedJointLoader(
        pair.task_a,
        pair.task_b,
        splits,
        batch_size=8,
        seed=0,
    )
    stream = iter(loader)
    batch = next(stream)
    assert batch["tokens"].shape == (8, 10)
    task_ids = batch["task_ids"].tolist()
    assert task_ids.count(0) == 4
    assert task_ids.count(1) == 4
    # Labels follow the queried operand pair.
    ops = {0: pair.task_a.by_slot(), 1: pair.task_b.by_slot()}
    tokens = batch["tokens"].numpy()
    labels = batch["labels"].numpy()
    slots = batch["slots"].numpy()
    for i in range(8):
        op = ops[task_ids[i]][int(slots[i])]
        assert int(labels[i]) == (int(tokens[i, op.i]) + int(tokens[i, op.j])) % op.modulus
    # A second draw still balances; the stream is not a single frozen batch.
    batch2 = next(stream)
    assert batch2["task_ids"].tolist().count(0) == 4
    assert not np.array_equal(batch["tokens"].numpy(), batch2["tokens"].numpy())


def test_prepare_phase2_dataset_packed(tmp_path: Path) -> None:
    from go4cl.phases.phase2.data import prepare_phase2_dataset

    meta = prepare_phase2_dataset(
        tmp_path,
        rho_slot=0.0,
        rho_operand=1.0,
        rho_mod=0.0,
        task_seed=0,
        data_seed=0,
        n_aliases=1,
        train_frac=0.6,
        direction="forward",
    )
    assert meta["train_mode"] == "packed_online"
    assert meta["n_train_a"] == 0
    assert meta["n_val_a"] > 0
    assert meta["n_val_b"] > 0
    assert (tmp_path / "data" / meta["tag"] / "manifest.json").exists()
    swapped = prepare_phase2_dataset(
        tmp_path,
        rho_slot=0.0,
        rho_operand=1.0,
        rho_mod=0.0,
        task_seed=0,
        data_seed=0,
        n_aliases=1,
        train_frac=0.6,
        direction="swap",
    )
    assert swapped["pair_id"].endswith("_swap")
    assert swapped["dataset_hash"] != meta["dataset_hash"]


def test_protocols_dry_run_lists_every_protocol(tmp_path: Path) -> None:
    from go4cl.phases.phase2.protocols import run_protocols
    import argparse

    args = argparse.Namespace(
        out=str(tmp_path),
        steps=4,
        train_frac=0.6,
        weight_decay=0.5,
        batch_size=8,
        n_aliases=1,
        lr=1e-3,
        d_model=32,
        n_layers=1,
        n_heads=4,
        task_seeds=[0],
        model_seeds=[0],
        data_seed=0,
        directions=["forward"],
        switch_on="fixed",
        gpus="0",
        workers_per_gpu=1,
        wandb_project="go4cl",
        wandb_group="test",
        wandb_mode="disabled",
        dry_run=True,
        protocols=[
            "a_only",
            "b_only",
            "joint",
            "interleaved",
            "sequential_ab",
            "sequential_ba",
            "a_only_continued",
        ],
        rho_slot=1.0,
        rho_operand=1.0,
        rho_mod=1.0,
    )
    run_protocols(args)
    payload = __import__("json").loads((tmp_path / "jobs.json").read_text())
    assert payload["n_jobs"] == 7
    assert {job["protocol"] for job in payload["jobs"]} == set(args.protocols)


def test_short_sequential_and_joint_on_packed_data(tmp_path: Path) -> None:
    from go4cl.model.transformer import ModelConfig
    from go4cl.phases.phase2.data import prepare_phase2_dataset
    from go4cl.train.loop import TrainConfig
    from go4cl.train.protocols import run_protocol

    meta = prepare_phase2_dataset(
        tmp_path,
        rho_slot=1.0,
        rho_operand=1.0,
        rho_mod=1.0,
        task_seed=1,
        data_seed=0,
        n_aliases=1,
        train_frac=0.6,
    )
    model_cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_mlp=64)
    train_cfg = TrainConfig(
        lr=1e-3,
        weight_decay=0.0,
        batch_size=8,
        max_steps=2,
        eval_every=1,
        ckpt_every=10,
        device="cpu",
    )
    seq = run_protocol(
        "sequential_ab",
        meta["data_dir"],
        tmp_path / "seq",
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_seed=0,
        phase_steps=2,
        wandb_enabled=False,
        include_test=True,
        switch_on="fixed",
        eval_n_per_operation=2,
    )
    assert seq.metrics["switch_step"] == 2
    assert "forgetting_A_from_switch" in seq.metrics
    assert seq.metrics["behavior"]["n_eval"] >= 1
    assert (tmp_path / "seq" / "eval_history.jsonl").is_file()
    assert (tmp_path / "seq" / "ckpts" / "theta_A.pt").is_file()

    joint = run_protocol(
        "joint",
        meta["data_dir"],
        tmp_path / "joint",
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_seed=0,
        phase_steps=1,
        wandb_enabled=False,
        include_test=True,
        eval_n_per_operation=2,
    )
    # Joint matches exposure by training 2S steps.
    assert joint.metrics["final_step"] == 2
    assert "A_test_acc" in joint.metrics
    assert "B_test_acc" in joint.metrics
    assert joint.metrics["behavior"]["grok_order"] in {
        "neither",
        "A_only",
        "B_only",
        "A_then_B",
        "B_then_A",
        "simultaneous",
    }
