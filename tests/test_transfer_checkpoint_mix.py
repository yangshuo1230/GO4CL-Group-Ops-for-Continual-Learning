"""Reset/keep hybrids, fresh optimizer, and the replay refusal."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.transfer_mechanism.checkpoint_mix import (
    build_b_optimizer,
    build_hybrid,
    clone_state_dict,
    load_theta_checkpoint,
    optimizer_has_adam_moments,
    reject_nonzero_replay,
    save_hybrid_checkpoint,
    save_theta_checkpoint,
    state_dict_from_model,
)
from go4cl.phases.transfer_mechanism.configs import TransferConfig
from go4cl.phases.transfer_mechanism.launch import component_jobs, modulus_jobs
from go4cl.phases.transfer_mechanism.parameter_groups import build_registry
from go4cl.train.loop import TrainConfig
from go4cl.utils.checkpoint import load_checkpoint
from go4cl.utils.config import load_config

_ROOT = Path(__file__).resolve().parents[1]
_MODULUS = _ROOT / "configs/transfer_mechanism/modulus_specificity.yaml"
_COMPONENT = _ROOT / "configs/transfer_mechanism/component_reset.yaml"


def _states():
    cfg = ModelConfig(d_model=8, n_layers=2, n_heads=2, d_mlp=16, dropout=0.0)
    torch.manual_seed(0)
    init = ModularTransformer(cfg)
    torch.manual_seed(1)
    trained = ModularTransformer(cfg)
    theta_0 = state_dict_from_model(init)
    theta_a = state_dict_from_model(trained)
    registry = build_registry(init)
    return init, theta_0, theta_a, registry


def _mix(name: str):
    model, theta_0, theta_a, registry = _states()
    hybrid, manifest = build_hybrid(
        theta_0,
        theta_a,
        name,
        registry,
        n_layers=2,
        task_seed=3,
        model_seed=4,
        dataset_hash="hash",
    )
    return model, theta_0, theta_a, hybrid, manifest


def test_reset_one_target_comes_from_init() -> None:
    _model, theta_0, _theta_a, hybrid, manifest = _mix("reset_digit_embedding")
    assert torch.equal(hybrid["tok_emb.weight"][:64], theta_0["tok_emb.weight"][:64])
    assert manifest["reset_groups"] == ["digit_embedding"]
    assert "digit_embedding" not in manifest["kept_groups"]
    _model, theta_0, theta_a, hybrid_l, manifest_l = _mix("reset_attention_layer_0")
    assert torch.equal(
        hybrid_l["blocks.0.attn.qkv.weight"], theta_0["blocks.0.attn.qkv.weight"]
    )
    assert torch.equal(hybrid_l["blocks.0.ln1.weight"], theta_0["blocks.0.ln1.weight"])
    assert torch.equal(
        hybrid_l["blocks.1.attn.qkv.weight"], theta_a["blocks.1.attn.qkv.weight"]
    )
    assert torch.equal(hybrid_l["blocks.1.ln1.weight"], theta_a["blocks.1.ln1.weight"])
    assert "attention_layer_0" in manifest_l["reset_groups"]
    assert "attention_layer_1" in manifest_l["kept_groups"]


def test_reset_one_other_groups_stay_on_theta_a() -> None:
    _model, theta_0, theta_a, hybrid, _manifest = _mix("reset_digit_embedding")
    assert torch.equal(hybrid["tok_emb.weight"][64:], theta_a["tok_emb.weight"][64:])
    assert torch.equal(hybrid["pos_emb.weight"], theta_a["pos_emb.weight"])
    assert torch.equal(
        hybrid["blocks.0.attn.qkv.weight"], theta_a["blocks.0.attn.qkv.weight"]
    )
    assert torch.equal(
        hybrid["blocks.0.attn.out.bias"], theta_a["blocks.0.attn.out.bias"]
    )
    assert torch.equal(hybrid["blocks.0.ln1.bias"], theta_a["blocks.0.ln1.bias"])
    assert torch.equal(hybrid["blocks.1.mlp.fc1.weight"], theta_a["blocks.1.mlp.fc1.weight"])
    assert torch.equal(hybrid["blocks.1.ln2.weight"], theta_a["blocks.1.ln2.weight"])
    assert torch.equal(hybrid["ln_f.weight"], theta_a["ln_f.weight"])
    assert torch.equal(hybrid["head.weight"], theta_a["head.weight"])
    assert torch.equal(hybrid["head.bias"], theta_a["head.bias"])
    assert not torch.equal(hybrid["tok_emb.weight"][:64], theta_a["tok_emb.weight"][:64])
    del theta_0


def test_keep_only_and_keep_multiple() -> None:
    _model, theta_0, theta_a, hybrid, manifest = _mix("keep_mlp")
    assert torch.equal(hybrid["blocks.0.mlp.fc2.weight"], theta_a["blocks.0.mlp.fc2.weight"])
    assert torch.equal(hybrid["blocks.1.ln2.bias"], theta_a["blocks.1.ln2.bias"])
    assert torch.equal(hybrid["blocks.0.attn.qkv.weight"], theta_0["blocks.0.attn.qkv.weight"])
    assert torch.equal(hybrid["tok_emb.weight"], theta_0["tok_emb.weight"])
    assert torch.equal(hybrid["head.weight"], theta_0["head.weight"])
    assert manifest["kept_groups"] == ["mlp"]
    _model, theta_0, theta_a, hybrid, manifest = _mix("keep_digit_embedding+mlp")
    assert torch.equal(hybrid["tok_emb.weight"][:64], theta_a["tok_emb.weight"][:64])
    assert torch.equal(hybrid["tok_emb.weight"][64:], theta_0["tok_emb.weight"][64:])
    assert torch.equal(hybrid["pos_emb.weight"], theta_0["pos_emb.weight"])
    assert torch.equal(hybrid["blocks.0.mlp.fc1.bias"], theta_a["blocks.0.mlp.fc1.bias"])
    assert torch.equal(hybrid["blocks.0.attn.out.weight"], theta_0["blocks.0.attn.out.weight"])
    assert torch.equal(hybrid["ln_f.bias"], theta_0["ln_f.bias"])
    assert "digit_embedding" in manifest["kept_groups"]
    assert "mlp" in manifest["kept_groups"]
    assert "attention" in manifest["reset_groups"]


def test_inputs_and_files_are_not_mutated(tmp_path: Path) -> None:
    model, theta_0, theta_a, registry = _states()
    snap_0 = clone_state_dict(theta_0)
    snap_a = clone_state_dict(theta_a)
    save_theta_checkpoint(
        tmp_path / "theta_0.pt",
        model,
        task_pair={"name": "pair"},
        dataset_hash="hash",
        task_seed=1,
        model_seed=2,
        step=0,
    )
    torch.manual_seed(1)
    trained = ModularTransformer(model.cfg)
    save_theta_checkpoint(
        tmp_path / "theta_A.pt",
        trained,
        task_pair={"name": "pair"},
        dataset_hash="hash",
        task_seed=1,
        model_seed=2,
        step=4,
    )
    digest_0 = (tmp_path / "theta_0.pt").read_bytes()
    digest_a = (tmp_path / "theta_A.pt").read_bytes()
    payload, _state = load_theta_checkpoint(tmp_path / "theta_0.pt")
    for key in (
        "model_state_dict",
        "model_config",
        "task_pair",
        "dataset_hash",
        "task_seed",
        "model_seed",
        "parameter_group_schema_version",
    ):
        assert key in payload
    assert "optimizer_state" not in payload
    build_hybrid(
        theta_0,
        theta_a,
        "reset_output",
        registry,
        n_layers=2,
        task_seed=1,
        model_seed=2,
        dataset_hash="hash",
    )
    for key in theta_0:
        assert torch.equal(theta_0[key], snap_0[key])
        assert torch.equal(theta_a[key], snap_a[key])
    assert (tmp_path / "theta_0.pt").read_bytes() == digest_0
    assert (tmp_path / "theta_A.pt").read_bytes() == digest_a


def test_hybrid_checkpoint_loads(tmp_path: Path) -> None:
    model, _theta_0, _theta_a, hybrid, manifest = _mix("reset_mlp")
    path = tmp_path / "hybrid.pt"
    save_hybrid_checkpoint(
        path, hybrid, model_config=model.cfg.to_dict(), manifest=manifest
    )
    loaded, payload = load_checkpoint(path, map_location="cpu")
    assert torch.equal(loaded.blocks[0].mlp.fc1.weight, hybrid["blocks.0.mlp.fc1.weight"])
    assert payload["model_config"]["d_model"] == model.cfg.d_model
    loaded.load_state_dict(payload["model_state_dict"])
    assert torch.equal(loaded.head.weight, hybrid["head.weight"])


def test_fresh_optimizer_has_no_adam_moments() -> None:
    torch.manual_seed(0)
    model = ModularTransformer(ModelConfig(d_model=8, n_layers=1, n_heads=2, d_mlp=16))
    cfg = TrainConfig(
        lr=1e-3,
        weight_decay=0.0,
        max_steps=1,
        device="cpu",
        compile_model=False,
        log_to_wandb=False,
    )
    opt = build_b_optimizer(model, cfg, carry_optimizer_state=False, payload=None)
    tokens = torch.randint(0, model.cfg.vocab_size, (4, model.cfg.context_length))
    labels = torch.randint(0, model.cfg.n_classes, (4,))
    loss = model(tokens, labels)["loss"]
    loss.backward()
    opt.step()
    assert optimizer_has_adam_moments(opt)
    payload = {"optimizer_state": opt.state_dict()}
    fresh = build_b_optimizer(
        model, cfg, carry_optimizer_state=False, payload=payload
    )
    assert not optimizer_has_adam_moments(fresh)
    carried = build_b_optimizer(
        model, cfg, carry_optimizer_state=True, payload=payload
    )
    assert optimizer_has_adam_moments(carried)
    param = next(model.parameters())
    assert torch.equal(carried.state[param]["exp_avg"], opt.state[param]["exp_avg"])


def test_nonzero_replay_is_rejected() -> None:
    with pytest.raises(ValueError, match="without replay"):
        reject_nonzero_replay(0.1)
    raw = load_config(_COMPONENT)
    raw["replay_ratio"] = 0.1
    with pytest.raises(ValueError, match="without replay"):
        TransferConfig.from_dict(raw).validate()
    modulus = TransferConfig.from_yaml(_MODULUS)
    component = TransferConfig.from_yaml(_COMPONENT)
    assert modulus.replay_ratio == 0.0
    assert component.replay_ratio == 0.0
    assert modulus.carry_optimizer_state is False
    assert component.carry_optimizer_state is False
    assert modulus.fixed_a is True
    assert len(modulus_jobs(modulus, "final")) == 90
    assert len(modulus_jobs(modulus, "pilot")) == 54
    final = component_jobs(component, "final")
    assert sum(job["kind"] == "source" for job in final) == 15
    assert sum(job["kind"] == "intervention" for job in final) == 180
    assert len(final) == 195
    with pytest.raises(ValueError, match="not allowed"):
        bad = TransferConfig.from_yaml(_MODULUS)
        bad.protocols = ["b_only", "sequential_ab_replay"]
        bad.validate()
