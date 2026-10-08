"""CPU checks for the experiment-validity fixes. No training grid is launched."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from go4cl.analysis.context import resolve_checkpoint
from go4cl.analysis.probes import fit_linear_probe
from go4cl.analysis.selection import select_and_confirm
from go4cl.data.manifest import DataManifest, hash_payload
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.phase2.data import prepare_phase2_dataset
from go4cl.phases.phase2.launch import assert_share_a_compatible
from go4cl.phases.task_partition.config import smoke_config
from go4cl.phases.task_partition.data import prepare_dataset
from go4cl.phases.task_partition.mechanism import (
    build_mechanism_analysis_bundle,
    shared_analysis_seed,
)
from go4cl.tasks.relations import build_task_pair, swap_ab
from go4cl.train.checkpoint_selection import task_event_values
from go4cl.train.loop import TrainConfig, TrainState, _append_eval_history, build_optimizer
from go4cl.train.protocol_impl import canonical_optimizer_transition, phase_b_optimizer
from go4cl.train.protocols import assert_null_task_protocol
from go4cl.utils.checkpoint import load_checkpoint, save_checkpoint
from go4cl.utils.wandb_log import operation_acc_metrics


def test_share_a_with_swap_is_rejected() -> None:
    with pytest.raises(ValueError, match="no longer|not a fixed source|changes with the overlap"):
        assert_share_a_compatible(share_a=True, directions=["forward", "swap"])
    assert_share_a_compatible(share_a=False, directions=["swap"])
    assert_share_a_compatible(share_a=True, directions=["forward"])


def test_swap_source_hash_is_not_the_canonical_a(tmp_path: Path) -> None:
    canonical = build_task_pair(
        rho_slot=0.0, rho_operand=0.0, rho_mod=0.5, task_seed=1, fixed_a=True
    )
    swapped = swap_ab(canonical)
    assert hash_payload(swapped.task_a.to_dict()) != hash_payload(canonical.task_a.to_dict())
    meta = prepare_phase2_dataset(
        tmp_path,
        rho_slot=0.0,
        rho_operand=0.0,
        rho_mod=0.5,
        task_seed=1,
        data_seed=0,
        n_aliases=1,
        train_frac=0.6,
        direction="swap",
        fixed_a=True,
    )
    manifest = DataManifest.load(Path(meta["data_dir"]) / "manifest.json")
    assert manifest.fixed_a is True
    assert manifest.task_a_hash == hash_payload(swapped.task_a.to_dict())
    assert manifest.task_a_hash != hash_payload(canonical.task_a.to_dict())
    again = prepare_phase2_dataset(
        tmp_path,
        rho_slot=0.0,
        rho_operand=0.0,
        rho_mod=0.5,
        task_seed=1,
        data_seed=0,
        n_aliases=1,
        train_frac=0.6,
        direction="swap",
        fixed_a=True,
    )
    assert again["task_a_hash"] == meta["task_a_hash"]
    manifest_path = Path(meta["data_dir"]) / "manifest.json"
    raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    raw.pop("fixed_a")
    raw.pop("task_a_hash")
    manifest_path.write_text(json.dumps(raw), encoding="utf-8")
    legacy = prepare_phase2_dataset(
        tmp_path,
        rho_slot=0.0,
        rho_operand=0.0,
        rho_mod=0.5,
        task_seed=1,
        data_seed=0,
        n_aliases=1,
        train_frac=0.6,
        direction="swap",
        fixed_a=True,
    )
    assert legacy["task_pair_hash"] == meta["task_pair_hash"]
    stamped = DataManifest.load(manifest_path)
    stamped.fixed_a = False
    stamped.save(manifest_path)
    with pytest.raises(ValueError, match="fixed_a"):
        prepare_phase2_dataset(
            tmp_path,
            rho_slot=0.0,
            rho_operand=0.0,
            rho_mod=0.5,
            task_seed=1,
            data_seed=0,
            n_aliases=1,
            train_frac=0.6,
            direction="swap",
            fixed_a=True,
        )


def _step_once(model: ModularTransformer, opt: torch.optim.Optimizer) -> None:
    tokens = torch.randint(0, 10, (4, 10))
    labels = torch.randint(0, 5, (4,))
    loss = model(tokens, labels)["loss"]
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def test_fresh_and_preserve_share_weights_and_differ_in_adam(tmp_path: Path) -> None:
    torch.manual_seed(0)
    cfg = ModelConfig(d_model=16, n_layers=1, n_heads=2, d_mlp=32)
    source = ModularTransformer(cfg)
    train_cfg = TrainConfig(lr=1e-2, weight_decay=0.0, device="cpu", batch_size=4)
    opt = build_optimizer(source, train_cfg)
    _step_once(source, opt)
    path = tmp_path / "theta_A.pt"
    save_checkpoint(path, source, optimizer=opt, step=7, meta={"role": "theta_A"})
    moment = next(iter(opt.state.values()))["exp_avg"].detach().clone()
    assert float(moment.abs().sum()) > 0

    fresh_model, payload = load_checkpoint(path, map_location="cpu")
    fresh_opt, mode = phase_b_optimizer(
        fresh_model, train_cfg, transition="fresh", payload=payload, checkpoint_path=str(path)
    )
    assert mode == "fresh"
    assert canonical_optimizer_transition("reset") == "fresh"
    assert fresh_opt.state == {}

    preserve_model, payload = load_checkpoint(path, map_location="cpu")
    preserve_opt, mode = phase_b_optimizer(
        preserve_model,
        train_cfg,
        transition="preserve",
        payload=payload,
        checkpoint_path=str(path),
    )
    assert mode == "preserve"
    restored = next(iter(preserve_opt.state.values()))
    assert int(restored["step"]) == 1
    assert torch.equal(restored["exp_avg"], moment)
    for left, right in zip(fresh_model.parameters(), preserve_model.parameters()):
        assert torch.equal(left, right)


def test_b_events_ignore_loader_order_and_do_not_average() -> None:
    first = {
        "A_val": 0.1,
        "B_val": 0.95,
        "A_train_eval": 0.2,
        "B_train_eval": 0.99,
        "A_iid": 0.3,
        "B_iid": 0.96,
    }
    reversed_loaders = {key: first[key] for key in reversed(list(first))}
    left, compat_left = task_event_values(first)
    right, compat_right = task_event_values(reversed_loaders)
    assert left == right
    assert compat_left is None and compat_right is None
    assert left["B_t_gen"] == pytest.approx(0.95)
    assert left["A_t_gen"] == pytest.approx(0.1)
    assert "t_gen" not in left
    only_a, compat = task_event_values({"A_val": 0.91, "A_train_eval": 1.0, "A_iid": 0.96})
    assert compat == "A"
    assert only_a["t_gen"] == pytest.approx(0.91)
    assert only_a["A_t_gen"] == pytest.approx(0.91)


def test_sequential_analysis_requires_an_explicit_role(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    (ckpt / "best.pt").write_bytes(b"legacy")
    (ckpt / "theta_A.pt").write_bytes(b"a")
    (ckpt / "B_best_val.pt").write_bytes(b"b")
    (ckpt / "AB_tradeoff_best.pt").write_bytes(b"trade")
    (tmp_path / "config_resolved.json").write_text(
        json.dumps({"protocol": "sequential_ab"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="explicit checkpoint role"):
        resolve_checkpoint(tmp_path, "best")
    assert resolve_checkpoint(tmp_path, "theta_A").name == "theta_A.pt"
    assert resolve_checkpoint(tmp_path, "B_best_val").name == "B_best_val.pt"
    assert resolve_checkpoint(tmp_path, "AB_tradeoff_best").name == "AB_tradeoff_best.pt"


def test_analysis_bundle_is_identical_for_two_checkpoints(tmp_path: Path) -> None:
    cfg = smoke_config(str(tmp_path))
    prepared = prepare_dataset(tmp_path / "data", cfg)
    kwargs = dict(
        n_contexts=int(cfg.mech_n_contexts),
        n_per_operation=int(cfg.mech_n_per_operation),
        seed=shared_analysis_seed(int(cfg.eval_seed)),
    )
    assert shared_analysis_seed(0) == shared_analysis_seed(0)
    first = build_mechanism_analysis_bundle(prepared["tasks"], prepared["splits"], **kwargs)
    second = build_mechanism_analysis_bundle(prepared["tasks"], prepared["splits"], **kwargs)
    assert first["hash"] == second["hash"]
    assert first["counterfactual_digits"] == second["counterfactual_digits"]
    assert first["routing_tokens"] == second["routing_tokens"]
    assert first["probe_tokens"] == second["probe_tokens"]
    assert first["probe_train_idx"] == second["probe_train_idx"]
    assert first["probe_held_idx"] == second["probe_held_idx"]
    assert first["probe_train_idx"] != first["probe_held_idx"]


def test_linear_probe_seed_is_local_and_reproducible() -> None:
    torch.manual_seed(123)
    x = torch.randn(8, 4)
    y = torch.randint(0, 3, (8,))
    before = torch.get_rng_state().clone()
    first = fit_linear_probe(x, y, {"held": x[:2]}, {"held": y[:2]}, n_classes=3, steps=3, seed=7)
    mid = torch.get_rng_state().clone()
    second = fit_linear_probe(x, y, {"held": x[:2]}, {"held": y[:2]}, n_classes=3, steps=3, seed=7)
    after = torch.get_rng_state().clone()
    other = fit_linear_probe(x, y, {"held": x[:2]}, {"held": y[:2]}, n_classes=3, steps=3, seed=8)
    assert torch.equal(before, mid)
    assert torch.equal(mid, after)
    assert torch.equal(first["weight"], second["weight"])
    assert first["seed"] == 7
    assert not torch.equal(first["weight"], other["weight"])


def test_operation_history_keeps_task_latent_and_slot() -> None:
    metrics = operation_acc_metrics(
        {
            "task0/lat0/slot0": {"accuracy": 0.2},
            "task0/lat1/slot1": {"accuracy": 0.4},
            "task1/lat0/slot2": {"accuracy": 0.9},
        },
        prefix="A_val_acc",
    )
    assert metrics["A_val_acc/task0/lat0/slot0"] == pytest.approx(0.2)
    assert metrics["A_val_acc/task1/lat0/slot2"] == pytest.approx(0.9)
    assert "A_val_acc/op0" not in metrics
    single = operation_acc_metrics(
        {"task0/lat0/slot0": {"accuracy": 0.5}, "task0/lat1/slot1": {"accuracy": 0.7}},
        prefix="B_val_acc",
    )
    assert single["B_val_acc/op0"] == pytest.approx(0.5)
    assert single["B_val_acc/op1"] == pytest.approx(0.7)
    state = TrainState(step=1)
    _append_eval_history(state, {"step": 1, **single}, segment="phase_b")
    assert state.eval_history[0]["B_val_acc/op0"] == pytest.approx(0.5)
    shared = [(0.0, 0.1), (10.0, 0.4)]
    novel = [(0.0, 0.1), (10.0, 0.9)]
    assert shared != novel


def test_null_task_replay_and_interleaved_are_rejected() -> None:
    with pytest.raises(ValueError, match="undefined"):
        assert_null_task_protocol("sequential_ab_replay", True, replay_ratio=0.1)
    with pytest.raises(ValueError, match="undefined"):
        assert_null_task_protocol("interleaved", True)
    with pytest.raises(ValueError, match="undefined"):
        assert_null_task_protocol("sequential_ab", True, replay_ratio=0.1)
    assert_null_task_protocol("sequential_ab", True, replay_ratio=0.0)
    assert_null_task_protocol("joint", True)
    assert_null_task_protocol("interleaved", False)


def test_val_selection_does_not_use_the_test_score() -> None:
    chosen = select_and_confirm(
        [
            {"name": "layer0", "val": 0.2, "test": 0.99},
            {"name": "layer1", "val": 0.8, "test": 0.1},
        ],
        name_key="name",
        val_key="val",
        test_key="test",
    )
    assert chosen["selected_component"] == "layer1"
    assert chosen["val_selection_score"] == pytest.approx(0.8)
    assert chosen["test_confirmation_score"] == pytest.approx(0.1)
    assert chosen["selection_protocol"] == "val_select_test_confirm"


def test_numpy_rng_is_unchanged_by_probe_import() -> None:
    state = np.random.get_state()
    _ = shared_analysis_seed(3)
    assert np.array_equal(np.random.get_state()[1], state[1])
