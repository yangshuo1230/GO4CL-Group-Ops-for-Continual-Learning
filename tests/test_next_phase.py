"""CPU tests for stable t90, modulus summaries, and the three dry-run grids."""

from __future__ import annotations

from pathlib import Path

import torch

from go4cl.data.generate import build_shared_residue_splits
from go4cl.data.packed import CoverageReplayLoader
from go4cl.metrics.continual import stable_time_to_threshold
from go4cl.phases.phase2.causal_modulus import (
    N_MAIN_RUNS,
    build_causal_tasks,
    plan_causal_jobs,
    task_hash,
    write_dry_run,
)
from go4cl.phases.phase2.modulus_analysis import summarize_groups
from go4cl.phases.phase2.param_patch import apply_patch, assert_patch_sanity, change_norm
from go4cl.phases.common import TrainJob
from go4cl.phases.phase2.next_launch import pin_jobs
from go4cl.phases.phase2.replay_coverage import (
    N_MAIN_RUNS as COVERAGE_N,
    job_count_report,
    plan_coverage_jobs,
)
from go4cl.phases.transfer_mechanism.checkpoint_mix import clone_state_dict
from go4cl.phases.transfer_mechanism.parameter_groups import COARSE_GROUPS, build_registry
from go4cl.train.protocol_impl import a_ops_mastered


def test_stable_time_matches_five_consecutive_hits() -> None:
    series = [(float(i), 0.95) for i in range(1, 6)]
    assert stable_time_to_threshold(series, threshold=0.9, window=5) == 5.0
    short = [(float(i), 0.95) for i in range(1, 5)]
    assert stable_time_to_threshold(short, threshold=0.9, window=5) is None
    broken = [(1.0, 0.95), (2.0, 0.2), (3.0, 0.95), (4.0, 0.95), (5.0, 0.95), (6.0, 0.95), (7.0, 0.95)]
    assert stable_time_to_threshold(broken, threshold=0.9, window=5) == 7.0


def test_censored_t90_stays_in_the_group() -> None:
    rows = [
        {
            "overlap_group": "rho_mod=0",
            "modulus": 23,
            "delta_auc": 0.1,
            "delta_stable_t90": None,
            "censored_stable_t90": True,
            "censored_first_t90": True,
        },
        {
            "overlap_group": "rho_mod=0",
            "modulus": 29,
            "delta_auc": -0.2,
            "delta_stable_t90": 1000.0,
            "censored_stable_t90": False,
            "censored_first_t90": False,
        },
    ]
    summary = {row["group"]: row for row in summarize_groups(rows)}
    assert summary["rho_mod=0"]["n"] == 2
    assert summary["rho_mod=0"]["n_censored_stable_t90"] == 1
    assert summary["rho_mod=0"]["delta_stable_t90_n_finite"] == 1
    assert summary["p23"]["n"] == 1


def test_causal_grid_holds_b_fixed_and_dedups() -> None:
    built = build_causal_tasks(23, 0)
    assert task_hash(built.same.task_b) == task_hash(built.different.task_b)
    same_ops = built.same.task_a.by_latent()
    diff_ops = built.different.task_a.by_latent()
    b_ops = built.task_b.by_latent()
    assert same_ops[0].modulus == 23
    assert same_ops[0].operand_pair != b_ops[0].operand_pair
    assert same_ops[0].slot != b_ops[0].slot
    assert diff_ops[0].modulus != 23
    assert diff_ops[0].modulus not in {op.modulus for op in built.task_b.operations}
    for latent in (1, 2, 3):
        assert same_ops[latent].to_dict() == diff_ops[latent].to_dict()
    jobs = plan_causal_jobs()
    assert len(jobs) == N_MAIN_RUNS == 36
    assert len([job for job in jobs if job.protocol == "b_only"]) == 12
    keys = [job.b_task_hash for job in jobs if job.target_modulus == 23 and job.construction == 0]
    assert len(set(keys)) == 1


def test_a_mastery_requires_every_operation() -> None:
    history = [{"A_val_acc/task0/lat0/slot0": 0.95, "A_val_acc/task0/lat1/slot1": 0.5}]
    keys = ["A_val_acc/task0/lat0/slot0", "A_val_acc/task0/lat1/slot1"]
    assert a_ops_mastered(history, keys, 0.9) is False
    history.append({"A_val_acc/task0/lat0/slot0": 0.95, "A_val_acc/task0/lat1/slot1": 0.91})
    assert a_ops_mastered(history, keys, 0.9) is True


def test_coverage_grid_shares_theta_a_and_replay_count(tmp_path: Path) -> None:
    jobs = plan_coverage_jobs(batch_size=8192)
    report = job_count_report(jobs)
    assert report["n_main_runs"] == COVERAGE_N == 36
    assert report["n_a_sources"] == 3
    assert report["shared_theta_a"]
    assert report["a_replay_examples_per_step"] == [820]
    low = [job for job in jobs if job.coverage == 0.01 and job.model_seed == 0]
    high = [job for job in jobs if job.coverage == 1.0 and job.model_seed == 1 and job.condition == low[0].condition]
    assert low[0].buffer["unique_pairs"] < high[0].buffer["unique_pairs"]
    for row in low[0].buffer["per_operation"]:
        assert row["n_selected"] >= 1
    coverages = [row["actual_coverage"] for row in low[0].buffer["per_operation"]]
    assert max(coverages) - min(coverages) < 0.05
    same_seed = [
        job.buffer["train_pairs_by_latent"]
        for job in jobs
        if job.condition == low[0].condition and job.coverage == 0.01
    ]
    assert same_seed[0] == same_seed[1] == same_seed[2]
    from go4cl.phases.phase2.replay_coverage import write_dry_run

    payload = write_dry_run(tmp_path, jobs)
    assert payload["formal_gpu_started"] is False
    assert len(payload["a_pretrain"]) == 3


def test_coverage_loader_keeps_replay_fraction() -> None:
    from go4cl.tasks.relations import build_task_pair

    pair = build_task_pair(rho_slot=0, rho_operand=0, rho_mod=0, task_seed=0, fixed_a=True)
    splits = build_shared_residue_splits([pair.task_a, pair.task_b], data_seed=0, ratios=(0.8, 0.1, 0.1))
    from go4cl.phases.phase2.replay_coverage import select_coverage_pairs

    low = select_coverage_pairs(pair.task_a, splits, 0.05, buffer_seed=0)
    full = select_coverage_pairs(pair.task_a, splits, 1.0, buffer_seed=0)
    loader_low = CoverageReplayLoader(
        pair.task_a,
        pair.task_b,
        splits,
        batch_size=40,
        frac_a=0.1,
        train_pairs_by_latent=low["train_pairs_by_latent"],
    )
    loader_full = CoverageReplayLoader(
        pair.task_a,
        pair.task_b,
        splits,
        batch_size=40,
        frac_a=0.1,
        train_pairs_by_latent=full["train_pairs_by_latent"],
    )
    assert loader_low.n_packs_a == loader_full.n_packs_a
    batch = next(iter(loader_low))
    assert batch["tokens"].shape[0] == 40


def test_param_patch_sanity_and_isolation() -> None:
    from go4cl.model.transformer import ModelConfig, ModularTransformer

    torch.manual_seed(0)
    model = ModularTransformer(ModelConfig(n_layers=2, d_model=32, n_heads=4, d_mlp=64))
    theta_a = clone_state_dict(model.state_dict())
    with torch.no_grad():
        for value in model.parameters():
            value.add_(0.2)
    theta_ab = clone_state_dict(model.state_dict())
    report = assert_patch_sanity(model, theta_a, theta_ab)
    assert report["unclassified_trainable"] == []
    registry = build_registry(model)
    empty = apply_patch(theta_ab, theta_a, [], registry)
    full = apply_patch(theta_ab, theta_a, list(COARSE_GROUPS), registry)
    assert change_norm(empty, theta_ab) == 0.0
    assert change_norm(full, theta_a) == 0.0
    full["tok_emb.weight"].fill_(0)
    again = apply_patch(theta_ab, theta_a, list(COARSE_GROUPS), registry)
    assert not torch.equal(again["tok_emb.weight"], full["tok_emb.weight"])


def test_pin_jobs_does_not_leave_every_job_on_gpu_zero() -> None:
    jobs = [
        TrainJob(
            job_id=f"j{i}",
            data_dir="data",
            out_dir=f"out/{i}",
            gpu=0,
            steps=1,
            lr=1e-3,
            weight_decay=0.3,
            d_model=32,
            n_layers=1,
            model_seed=i,
            wandb_project="go4cl",
            wandb_group="g",
            wandb_mode="disabled",
            wandb_tags=("phase2",),
            wandb_config={},
        )
        for i in range(6)
    ]
    pinned = pin_jobs(jobs, [4, 5, 6, 7], 1)
    assert [job.gpu for job in pinned] == [4, 5, 6, 7, 4, 5]


def test_causal_dry_run_writes_hashes(tmp_path: Path) -> None:
    payload = write_dry_run(tmp_path, plan_causal_jobs())
    assert payload["counts"]["n_main_runs"] == 36
    assert payload["counts"]["b_only_deduped"] is True
    assert payload["formal_gpu_started"] is False
    assert (tmp_path / "jobs.json").is_file()
