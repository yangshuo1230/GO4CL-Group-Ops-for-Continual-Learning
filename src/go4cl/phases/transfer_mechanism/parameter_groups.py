"""Deterministic parameter-group registry for checkpoint mixing.

Coarse groups partition every trainable element exactly once. Embedding rows
may belong to different groups. Layer subgroups are subsets of attention / MLP
and exist so later interventions can name ``attention_layer_0`` without a
second partitioning scheme. Single-head mixing is named but not implemented:
Q, K, and V are one fused Linear.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from go4cl.constants import NUM_DIGITS
from go4cl.model.transformer import ModularTransformer
from go4cl.phases.transfer_mechanism import PARAMETER_GROUP_SCHEMA_VERSION

COARSE_GROUPS: tuple[str, ...] = (
    "digit_embedding",
    "control_embedding",
    "attention",
    "mlp",
    "output",
)

_ATTENTION = re.compile(r"^blocks\.(\d+)\.(?:ln1|attn)(?:\.|$)")
_MLP = re.compile(r"^blocks\.(\d+)\.(?:ln2|mlp)(?:\.|$)")


@dataclass(frozen=True)
class ParamSlice:
    """One contiguous piece of a trainable parameter."""

    name: str
    group: str
    subgroup: str | None
    shape: tuple[int, ...]
    parameter_shape: tuple[int, ...]
    numel: int
    rows: tuple[int, int] | None
    layer: int | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "group": self.group,
            "subgroup": self.subgroup,
            "shape": list(self.shape),
            "parameter_shape": list(self.parameter_shape),
            "numel": int(self.numel),
            "rows": list(self.rows) if self.rows is not None else None,
            "layer": self.layer,
        }


def attention_layer_group(layer: int) -> str:
    return f"attention_layer_{int(layer)}"


def mlp_layer_group(layer: int) -> str:
    return f"mlp_layer_{int(layer)}"


def attention_head_group(layer: int, head: int) -> str:
    """Reserved name. Per-head slices are not implemented."""
    return f"attention_layer_{int(layer)}_head_{int(head)}"


def known_group_names(n_layers: int) -> set[str]:
    names = set(COARSE_GROUPS)
    for layer in range(int(n_layers)):
        names.add(attention_layer_group(layer))
        names.add(mlp_layer_group(layer))
    return names


def head_parameter_slices(
    model: ModularTransformer, layer: int, head: int
) -> list[ParamSlice]:
    """Extension point for a later per-head mix. Not available this round."""
    raise NotImplementedError(
        "Single-head mixing is reserved and not implemented. "
        f"Requested {attention_head_group(layer, head)} on "
        f"blocks.{int(layer)}.attn. Q, K, and V are one fused Linear "
        "(qkv.weight / qkv.bias); this round does not slice heads."
    )


def slice_selected(sl: ParamSlice, groups: set[str]) -> bool:
    if sl.group in groups:
        return True
    return bool(sl.subgroup and sl.subgroup in groups)


def build_registry(model: nn.Module) -> list[ParamSlice]:
    """Classify every trainable parameter. Raises if any element is left out."""
    n_layers = _n_layers(model)
    vocab = _vocab_size(model)
    slices: list[ParamSlice] = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        slices.extend(_classify(name, param, n_layers=n_layers, vocab_size=vocab))
    assert_partition(model, slices)
    return slices


def parameter_group_manifest(model: nn.Module) -> dict[str, Any]:
    slices = build_registry(model)
    by_group: dict[str, dict[str, Any]] = {}
    for group in COARSE_GROUPS:
        members = [sl for sl in slices if sl.group == group]
        by_group[group] = {
            "numel": int(sum(sl.numel for sl in members)),
            "n_slices": len(members),
        }
    return {
        "schema_version": PARAMETER_GROUP_SCHEMA_VERSION,
        "coarse_groups": list(COARSE_GROUPS),
        "groups": by_group,
        "total_numel": int(sum(sl.numel for sl in slices)),
        "slices": [sl.to_dict() for sl in slices],
        "notes": [
            "Digit rows are tok_emb[0:NUM_DIGITS]. Other token rows and pos_emb "
            "are control_embedding.",
            "Q, K, and V are the fused blocks.*.attn.qkv Linear, including bias.",
            "Pre-attention LayerNorm is blocks.*.ln1. Pre-MLP LayerNorm is blocks.*.ln2.",
            "Final LayerNorm and the unembedding head are the output group.",
            "buffers such as attention masks are not trainable and are not mixed.",
            "attention_layer_k and mlp_layer_k are subsets for later interventions.",
            "Single-head mixing is not implemented.",
        ],
    }


def logical_units(selected: set[str], n_layers: int) -> list[str]:
    """Labels used in manifests. Expand a family only when a layer subset is named."""
    units = ["digit_embedding", "control_embedding"]
    attn = [attention_layer_group(i) for i in range(n_layers)]
    mlp = [mlp_layer_group(i) for i in range(n_layers)]
    if any(name in selected for name in attn) and "attention" not in selected:
        units.extend(attn)
    else:
        units.append("attention")
    if any(name in selected for name in mlp) and "mlp" not in selected:
        units.extend(mlp)
    else:
        units.append("mlp")
    units.append("output")
    return units


def unit_selected(unit: str, selected: set[str]) -> bool:
    if unit in selected:
        return True
    if unit.startswith("attention_layer_") and "attention" in selected:
        return True
    if unit.startswith("mlp_layer_") and "mlp" in selected:
        return True
    return False


def assert_partition(model: nn.Module, slices: list[ParamSlice]) -> None:
    by_name: dict[str, list[ParamSlice]] = {}
    for sl in slices:
        by_name.setdefault(sl.name, []).append(sl)
    assigned = 0
    seen_names: set[str] = set()
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        seen_names.add(name)
        members = by_name.get(name)
        if not members:
            raise RuntimeError(f"parameter {name} is not assigned to a group")
        if any(sl.rows is None for sl in members):
            if len(members) != 1 or members[0].rows is not None:
                raise RuntimeError(f"parameter {name} is assigned more than once")
            if members[0].numel != param.numel():
                raise RuntimeError(
                    f"parameter {name} numel {param.numel()} != slice {members[0].numel}"
                )
            assigned += members[0].numel
            continue
        if param.ndim < 1:
            raise RuntimeError(f"row mix on scalar parameter {name}")
        cover = torch.zeros(param.shape[0], dtype=torch.bool)
        for sl in members:
            assert sl.rows is not None
            start, stop = sl.rows
            if start < 0 or stop > param.shape[0] or start >= stop:
                raise RuntimeError(f"bad row span {sl.rows} for {name}")
            if bool(cover[start:stop].any()):
                raise RuntimeError(f"overlapping rows on {name}: {sl.rows}")
            cover[start:stop] = True
            expected = int(stop - start)
            for dim in param.shape[1:]:
                expected *= int(dim)
            if sl.numel != expected:
                raise RuntimeError(f"numel mismatch on {name} rows {sl.rows}")
            assigned += sl.numel
        if not bool(cover.all()):
            missing = (~cover).nonzero(as_tuple=False).flatten().tolist()
            raise RuntimeError(f"unassigned rows on {name}: {missing[:8]}")
    extra = sorted(set(by_name) - seen_names)
    if extra:
        raise RuntimeError(f"slices name unknown parameters: {extra}")
    total = sum(int(p.numel()) for p in model.parameters() if p.requires_grad)
    if assigned != total:
        raise RuntimeError(f"assigned numel {assigned} != parameter numel {total}")


def _n_layers(model: nn.Module) -> int:
    blocks = getattr(model, "blocks", None)
    if isinstance(blocks, nn.ModuleList):
        return len(blocks)
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "n_layers"):
        return int(cfg.n_layers)
    raise RuntimeError("model has no blocks / n_layers; cannot build parameter groups")


def _vocab_size(model: nn.Module) -> int:
    tok = getattr(model, "tok_emb", None)
    if isinstance(tok, nn.Embedding):
        return int(tok.num_embeddings)
    cfg = getattr(model, "cfg", None)
    if cfg is not None and hasattr(cfg, "vocab_size"):
        return int(cfg.vocab_size)
    raise RuntimeError("model has no token embedding; cannot split digit rows")


def _classify(
    name: str,
    param: torch.Tensor,
    *,
    n_layers: int,
    vocab_size: int,
) -> list[ParamSlice]:
    full_shape = tuple(int(d) for d in param.shape)
    if name == "tok_emb.weight":
        return _token_rows(name, param, vocab_size=vocab_size)
    if name == "pos_emb.weight":
        return [_full(name, param, group="control_embedding", layer=None, subgroup=None)]
    attn = _ATTENTION.match(name)
    if attn:
        layer = int(attn.group(1))
        _check_layer(name, layer, n_layers)
        return [
            _full(
                name,
                param,
                group="attention",
                layer=layer,
                subgroup=attention_layer_group(layer),
            )
        ]
    mlp = _MLP.match(name)
    if mlp:
        layer = int(mlp.group(1))
        _check_layer(name, layer, n_layers)
        return [
            _full(
                name,
                param,
                group="mlp",
                layer=layer,
                subgroup=mlp_layer_group(layer),
            )
        ]
    if name.startswith("ln_f.") or name.startswith("head."):
        return [_full(name, param, group="output", layer=None, subgroup=None)]
    raise RuntimeError(
        f"unassigned trainable parameter {name} shape {full_shape}. "
        "Add it to exactly one transfer group before mixing checkpoints."
    )


def _check_layer(name: str, layer: int, n_layers: int) -> None:
    if not 0 <= layer < n_layers:
        raise RuntimeError(f"{name} layer {layer} outside 0..{n_layers - 1}")


def _full(
    name: str,
    param: torch.Tensor,
    *,
    group: str,
    layer: int | None,
    subgroup: str | None,
) -> ParamSlice:
    shape = tuple(int(d) for d in param.shape)
    return ParamSlice(
        name=name,
        group=group,
        subgroup=subgroup,
        shape=shape,
        parameter_shape=shape,
        numel=int(param.numel()),
        rows=None,
        layer=layer,
    )


def _token_rows(name: str, param: torch.Tensor, *, vocab_size: int) -> list[ParamSlice]:
    if param.ndim != 2:
        raise RuntimeError(f"{name} expected rank 2, got {tuple(param.shape)}")
    if int(param.shape[0]) != int(vocab_size):
        raise RuntimeError(
            f"{name} rows {param.shape[0]} != vocab_size {vocab_size}"
        )
    if vocab_size < NUM_DIGITS:
        raise RuntimeError(
            f"vocab_size {vocab_size} < NUM_DIGITS {NUM_DIGITS}; cannot split digit rows"
        )
    width = int(param.shape[1])
    digit_rows = (0, NUM_DIGITS)
    control_rows = (NUM_DIGITS, vocab_size)
    out = [
        ParamSlice(
            name=name,
            group="digit_embedding",
            subgroup=None,
            shape=(digit_rows[1] - digit_rows[0], width),
            parameter_shape=tuple(int(d) for d in param.shape),
            numel=(digit_rows[1] - digit_rows[0]) * width,
            rows=digit_rows,
            layer=None,
        )
    ]
    if control_rows[1] > control_rows[0]:
        out.append(
            ParamSlice(
                name=name,
                group="control_embedding",
                subgroup=None,
                shape=(control_rows[1] - control_rows[0], width),
                parameter_shape=tuple(int(d) for d in param.shape),
                numel=(control_rows[1] - control_rows[0]) * width,
                rows=control_rows,
                layer=None,
            )
        )
    return out
