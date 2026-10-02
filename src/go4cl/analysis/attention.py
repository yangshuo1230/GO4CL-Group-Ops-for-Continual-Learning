"""Attention routing diagnostics: does query attend to operand positions?"""

from __future__ import annotations

from typing import Any

import torch

from go4cl.constants import SEQ_LEN_OPERANDS


def query_attention_to_operands(
    attn: list[torch.Tensor],
    *,
    operand_i: int,
    operand_j: int,
) -> dict[str, Any]:
    """Summarize attention mass from the query token onto operand digits.

    ``attn[layer]``: [B, H, T, T]. Query index is T-1.
    Digit positions are 0..SEQ_LEN_OPERANDS-1.
    """
    if not attn:
        return {"layers": [], "mean_operand_mass": 0.0}

    layers: list[dict[str, Any]] = []
    operand_masses: list[float] = []
    for li, a in enumerate(attn):
        # a: [B, H, T, T] — take query row
        q = a[:, :, -1, :]  # [B, H, T]
        # mean over batch and heads
        mass = q.mean(dim=(0, 1))  # [T]
        digit_mass = mass[:SEQ_LEN_OPERANDS]
        op_mass = float((mass[operand_i] + mass[operand_j]).item())
        other_digit = float(
            digit_mass.sum().item() - mass[operand_i].item() - mass[operand_j].item()
        )
        per_head = q.mean(dim=0)  # [H, T]
        head_op = (per_head[:, operand_i] + per_head[:, operand_j]).tolist()
        layers.append(
            {
                "layer": li,
                "operand_mass": op_mass,
                "other_digit_mass": other_digit,
                "task_mass": float(mass[SEQ_LEN_OPERANDS].item())
                if mass.numel() > SEQ_LEN_OPERANDS
                else 0.0,
                "query_self_mass": float(mass[-1].item()),
                "mass_by_pos": [float(x) for x in mass.tolist()],
                "operand_mass_by_head": [float(x) for x in head_op],
            }
        )
        operand_masses.append(op_mass)

    return {
        "operand_i": operand_i,
        "operand_j": operand_j,
        "layers": layers,
        "mean_operand_mass": float(sum(operand_masses) / max(len(operand_masses), 1)),
        "last_layer_operand_mass": float(operand_masses[-1]) if operand_masses else 0.0,
    }
