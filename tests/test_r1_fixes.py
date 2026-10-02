"""R1 fixes: OperationKey, contexts, macro selector, analysis data, seeds."""

from __future__ import annotations

import numpy as np
import torch

from go4cl.data.context import ContextBuilder, ContextRequest, build_analysis_dataset
from go4cl.data.dataset import ModularAdditionDataset
from go4cl.data.packed import sample_packed_examples
from go4cl.data.residue_pairs import stratified_residue_pair_split
from go4cl.metrics.behavioral import evaluate
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.phase1.mechanisms import filter_by_operation
from go4cl.tasks.spec import Operation, OperationKey, TaskSpec
from go4cl.train.checkpoint_selection import CheckpointSelector, StableEventDetector
from go4cl.analysis.causal import estimate_class_means, steer_at_layer


def _tiny_four_diff_same_mod() -> tuple[TaskSpec, dict[int, object]]:
    """Two ops share modulus 29; two others differ — pair_same-like identity test."""
    ops = (
        Operation(latent_id=0, i=0, j=1, modulus=29, slot=0),
        Operation(latent_id=1, i=2, j=3, modulus=29, slot=1),
        Operation(latent_id=2, i=4, j=5, modulus=23, slot=2),
        Operation(latent_id=3, i=6, j=7, modulus=31, slot=3),
    )
    task = TaskSpec(name="A", task_id=0, operations=ops)
    splits = {
        29: stratified_residue_pair_split(29, ratios=(0.6, 0.2, 0.2), data_seed=0),
        23: stratified_residue_pair_split(23, ratios=(0.6, 0.2, 0.2), data_seed=0),
        31: stratified_residue_pair_split(31, ratios=(0.6, 0.2, 0.2), data_seed=0),
    }
    return task, splits


def test_operation_key_is_report_primary() -> None:
    op = Operation(latent_id=1, i=0, j=1, modulus=29, slot=2)
    key = OperationKey.from_operation(op, task_id=0)
    assert key.report_key() == "task0/lat1/slot2"
    assert key.latent_id == 1 and key.slot == 2


def test_packed_queries_share_digits_and_carry_latent_id() -> None:
    task, splits = _tiny_four_diff_same_mod()
    rng = np.random.default_rng(0)
    exs = sample_packed_examples(rng, task, splits, split_name="train")
    assert len(exs) == 4
    digits0 = exs[0].tokens[:8]
    assert all(e.tokens[:8] == digits0 for e in exs)
    assert {e.latent_id for e in exs} == {0, 1, 2, 3}
    for e, op in zip(exs, task.operations, strict=True):
        assert e.label == (e.tokens[op.i] + e.tokens[op.j]) % op.modulus


def test_packed_id_vs_nuisance_target_operands() -> None:
    task, splits = _tiny_four_diff_same_mod()
    builder = ContextBuilder(task, splits)
    op = task.operations[0]
    key = OperationKey.from_operation(op, task_id=0)
    rng = np.random.default_rng(1)
    packed = builder.sample_eval_for_target(
        rng,
        ContextRequest(
            target_operation=key,
            target_split="val",
            distractor_split="train",
            context_mode="packed_id",
        ),
    )
    nuisance = builder.sample_eval_for_target(
        rng,
        ContextRequest(
            target_operation=key,
            target_split="val",
            context_mode="nuisance_random",
        ),
    )
    assert packed.latent_id == op.latent_id
    assert nuisance.latent_id == op.latent_id
    # Target label matches operands in both modes
    for e in (packed, nuisance):
        assert e.label == (e.tokens[op.i] + e.tokens[op.j]) % op.modulus


def test_filter_by_operation_separates_same_modulus() -> None:
    task, splits = _tiny_four_diff_same_mod()
    builder = ContextBuilder(task, splits)
    rng = np.random.default_rng(2)
    examples = []
    for _ in range(40):
        examples.extend(builder.sample_train_pack(rng))
    ds = ModularAdditionDataset.from_examples(examples, task_id=0)
    a = filter_by_operation(ds, latent_id=0, slot=0)
    b = filter_by_operation(ds, latent_id=1, slot=1)
    assert len(a) > 0 and len(b) > 0
    assert set(a.latent_ids.tolist()) == {0}
    assert set(b.latent_ids.tolist()) == {1}
    assert set(a.moduli.tolist()) == {29}
    assert set(b.moduli.tolist()) == {29}


def test_macro_accuracy_not_dominated_by_sample_count() -> None:
    # Two ops: many easy examples for lat0, few for lat1 → micro high, macro mid
    tokens = np.zeros((10, 10), dtype=np.int64)
    labels = np.zeros(10, dtype=np.int64)
    slots = np.array([0] * 8 + [1] * 2, dtype=np.int64)
    moduli = np.array([29] * 8 + [23] * 2, dtype=np.int64)
    latent_ids = np.array([0] * 8 + [1] * 2, dtype=np.int64)
    task_ids = np.zeros(10, dtype=np.int64)
    # Craft a model that always predicts 0 → lat0 all correct, lat1 all wrong if labels=1
    labels[8:] = 1
    ds = ModularAdditionDataset(tokens, labels, slots, moduli, task_ids, latent_ids)
    loader = torch.utils.data.DataLoader(ds, batch_size=10)
    model = ModularTransformer(ModelConfig(d_model=32, n_layers=1, n_heads=4, d_mlp=64))
    # Force head bias toward class 0
    with torch.no_grad():
        model.head.bias.zero_()
        model.head.bias[0] = 10.0
    device = torch.device("cpu")
    result = evaluate(model, loader, device)
    assert abs(result.accuracy - 0.8) < 1e-6
    # macro = mean(1.0, 0.0) = 0.5
    assert abs(result.macro_operation_accuracy - 0.5) < 1e-6
    assert len(result.by_operation) == 2


def test_checkpoint_selector_tie_breaks_on_loss() -> None:
    sel = CheckpointSelector()
    assert sel.observe(10, macro_operation_accuracy=0.9, macro_operation_loss=1.0)
    assert sel.best_step == 10
    # Same accuracy, worse loss → no update
    assert not sel.observe(20, macro_operation_accuracy=0.9, macro_operation_loss=1.5)
    assert sel.best_step == 10
    # Same accuracy, better loss → update
    assert sel.observe(30, macro_operation_accuracy=0.9, macro_operation_loss=0.5)
    assert sel.best_step == 30
    # Higher accuracy → update even if loss worse
    assert sel.observe(40, macro_operation_accuracy=0.95, macro_operation_loss=2.0)
    assert sel.best_step == 40


def test_stable_event_needs_consecutive_window() -> None:
    det = StableEventDetector("t_gen", threshold=0.9, window=3)
    assert not det.observe(1, 0.95)
    assert not det.observe(2, 0.95)
    assert det.observe(3, 0.95)
    assert det.triggered_step == 3
    assert not det.observe(4, 0.99)  # already triggered


def test_sampler_seed_changes_stream_model_seed_does_not() -> None:
    task, splits = _tiny_four_diff_same_mod()
    builder = ContextBuilder(task, splits)
    a = builder.sample_train_pack(np.random.default_rng(7))
    b = builder.sample_train_pack(np.random.default_rng(7))
    c = builder.sample_train_pack(np.random.default_rng(8))
    assert [e.tokens for e in a] == [e.tokens for e in b]
    assert [e.tokens for e in a] != [e.tokens for e in c]


def test_analysis_dataset_nonempty_and_deterministic() -> None:
    task, splits = _tiny_four_diff_same_mod()
    a = build_analysis_dataset(
        task,
        splits,
        split="train",
        context_mode="packed_id",
        analysis_seed=11,
        aliases_per_pair=1,
        contexts_per_pair=1,
        target_latent_ids=[0],
    )
    b = build_analysis_dataset(
        task,
        splits,
        split="train",
        context_mode="packed_id",
        analysis_seed=11,
        aliases_per_pair=1,
        contexts_per_pair=1,
        target_latent_ids=[0],
    )
    assert len(a) > 0
    assert [e.tokens for e in a] == [e.tokens for e in b]
    assert all(e.latent_id == 0 for e in a)


def test_primary_val_loader_names() -> None:
    from go4cl.train.loop import is_primary_val_loader

    assert is_primary_val_loader("A_val")
    assert is_primary_val_loader("B_val")
    assert is_primary_val_loader("val")
    assert not is_primary_val_loader("A_val_nuisance")
    assert not is_primary_val_loader("A_train_eval")
    assert not is_primary_val_loader("A_iid")


def test_train_segment_preserves_optimizer() -> None:
    from go4cl.train.loop import TrainConfig, build_optimizer, train_segment

    cfg = ModelConfig(d_model=32, n_layers=1, n_heads=4, d_mlp=64)
    model = ModularTransformer(cfg)
    # tiny fixed loader
    tokens = torch.randint(0, 64, (32, 10))
    labels = torch.randint(0, 7, (32,))
    ds = ModularAdditionDataset(
        tokens.numpy(),
        labels.numpy(),
        slots=np.zeros(32, dtype=np.int64),
        moduli=np.full(32, 7, dtype=np.int64),
        task_ids=np.zeros(32, dtype=np.int64),
        latent_ids=np.zeros(32, dtype=np.int64),
    )
    loader = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
    opt = build_optimizer(model, TrainConfig(lr=1e-3, max_steps=2, device="cpu"))
    id0 = id(opt)
    # touch optimizer so state is non-empty
    batch = next(iter(loader))
    loss = model(batch["tokens"], batch["labels"])["loss"]
    loss.backward()
    opt.step()
    state_keys_before = set(opt.state_dict()["state"].keys())

    seg = train_segment(
        model,
        loader,
        cfg=TrainConfig(
            lr=1e-3,
            max_steps=3,
            eval_every=1000,
            ckpt_every=1000,
            device="cpu",
            batch_size=8,
        ),
        optimizer=opt,
        start_step=0,
        optimizer_transition="preserve",
        track_events=False,
    )
    assert id(seg.optimizer) == id0
    assert set(seg.optimizer.state_dict()["state"].keys()) >= state_keys_before

    seg_reset = train_segment(
        model,
        loader,
        cfg=TrainConfig(
            lr=1e-3,
            max_steps=2,
            eval_every=1000,
            ckpt_every=1000,
            device="cpu",
        ),
        optimizer=opt,
        start_step=seg.state.step,
        optimizer_transition="reset",
        track_events=False,
    )
    assert id(seg_reset.optimizer) != id0


def test_steering_uses_reference_not_eval_means() -> None:
    cfg = ModelConfig(d_model=32, n_layers=2, n_heads=4, d_mlp=64)
    model = ModularTransformer(cfg)
    b, t, d, p = 16, 10, 32, 7
    # Fake resid_post and labels
    resid_train = torch.randn(b, t, d)
    resid_test = torch.randn(b, t, d)
    y_train = torch.randint(0, p, (b,))
    y_test = torch.randint(0, p, (b,))
    means = estimate_class_means(resid_train[:, -1, :], y_train, n_classes=p)
    out = steer_at_layer(
        model,
        resid_test,
        y_test,
        layer_idx=0,
        modulus=p,
        class_means=means,
    )
    assert out["direction_source"] == "external_class_means"
    legacy = steer_at_layer(
        model,
        resid_test,
        y_test,
        layer_idx=0,
        modulus=p,
    )
    assert legacy["direction_source"] == "eval_set_legacy"
