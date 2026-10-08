"""Parameter-group coverage for checkpoint mixing."""

from __future__ import annotations

import torch

from go4cl.constants import NUM_DIGITS
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.transfer_mechanism.parameter_groups import (
    COARSE_GROUPS,
    attention_head_group,
    build_registry,
    head_parameter_slices,
    parameter_group_manifest,
)


def _model() -> ModularTransformer:
    torch.manual_seed(0)
    return ModularTransformer(ModelConfig())


def test_groups_cover_every_parameter_once() -> None:
    model = _model()
    slices = build_registry(model)
    manifest = parameter_group_manifest(model)
    total = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    assert manifest["total_numel"] == total
    assert sum(group["numel"] for group in manifest["groups"].values()) == total
    assigned: dict[str, int] = {name: 0 for name in COARSE_GROUPS}
    for sl in slices:
        assert sl.group in COARSE_GROUPS
        assigned[sl.group] += sl.numel
    assert all(n > 0 for n in assigned.values())
    names = {name for name, _ in model.named_parameters()}
    assert {sl.name for sl in slices} == names
    qkv = [sl for sl in slices if sl.name.endswith("attn.qkv.weight")]
    assert qkv and all(sl.group == "attention" and sl.subgroup.startswith("attention_layer_") for sl in qkv)
    assert any(sl.name.endswith("ln1.weight") and sl.group == "attention" for sl in slices)
    assert any(sl.name.endswith("attn.out.bias") and sl.group == "attention" for sl in slices)
    assert any(sl.name.endswith("ln2.weight") and sl.group == "mlp" for sl in slices)
    assert any(sl.name == "ln_f.weight" and sl.group == "output" for sl in slices)
    assert any(sl.name == "head.bias" and sl.group == "output" for sl in slices)
    try:
        head_parameter_slices(model, 0, 0)
    except NotImplementedError as exc:
        assert attention_head_group(0, 0) in str(exc)
    else:
        raise AssertionError("single-head mixing should stay unimplemented")


def test_digit_and_control_rows_are_split() -> None:
    model = _model()
    slices = build_registry(model)
    digit = [sl for sl in slices if sl.group == "digit_embedding"]
    assert len(digit) == 1
    assert digit[0].name == "tok_emb.weight"
    assert digit[0].rows == (0, NUM_DIGITS)
    control_rows = [
        sl for sl in slices if sl.name == "tok_emb.weight" and sl.group == "control_embedding"
    ]
    assert len(control_rows) == 1
    assert control_rows[0].rows == (NUM_DIGITS, model.cfg.vocab_size)
    assert control_rows[0].rows[0] >= NUM_DIGITS
    pos = [sl for sl in slices if sl.name == "pos_emb.weight"]
    assert len(pos) == 1 and pos[0].group == "control_embedding" and pos[0].rows is None
    for sl in slices:
        if sl.group == "control_embedding" and sl.rows is not None:
            assert sl.rows[0] >= NUM_DIGITS
        if sl.group == "digit_embedding":
            assert sl.rows == (0, NUM_DIGITS)


def test_layer_subgroups_stay_inside_coarse_groups() -> None:
    slices = build_registry(_model())
    for sl in slices:
        if sl.subgroup and sl.subgroup.startswith("attention_layer_"):
            assert sl.group == "attention"
        if sl.subgroup and sl.subgroup.startswith("mlp_layer_"):
            assert sl.group == "mlp"


def test_manifest_is_determined_by_architecture() -> None:
    first = parameter_group_manifest(_model())
    second = parameter_group_manifest(
        ModularTransformer(ModelConfig())
    )
    assert first == second


def test_unknown_parameter_fails_loudly() -> None:
    import torch.nn as nn

    model = _model()
    model.extra_linear = nn.Linear(4, 4)
    try:
        build_registry(model)
    except RuntimeError as exc:
        assert "unassigned trainable parameter" in str(exc)
        assert "extra_linear" in str(exc)
    else:
        raise AssertionError("an unrecognized parameter must not be mixed silently")


def test_single_head_mixing_error_names_the_fused_qkv() -> None:
    try:
        head_parameter_slices(_model(), layer=1, head=2)
    except NotImplementedError as exc:
        text = str(exc)
    else:
        raise AssertionError("single-head mixing must stay unimplemented")
    assert "not implemented" in text
    assert "fused" in text
    assert "qkv" in text
