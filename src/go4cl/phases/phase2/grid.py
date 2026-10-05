"""Condition grids for phase-2 protocols, relation matrix, and capacity."""

from __future__ import annotations

from dataclasses import dataclass

ALL_PROTOCOLS: tuple[str, ...] = (
    "a_only",
    "b_only",
    "joint",
    "interleaved",
    "sequential_ab",
    "sequential_ba",
    "a_only_continued",
)

# 2B compares joint vs both orders on one manifest, plus single-task baselines.
RELATION_PROTOCOLS: tuple[str, ...] = (
    "a_only",
    "b_only",
    "joint",
    "sequential_ab",
    "sequential_ba",
)

# 2D asks whether a compatible solution exists and whether B is learnable.
CAPACITY_PROTOCOLS: tuple[str, ...] = (
    "a_only",
    "b_only",
    "joint",
    "sequential_ab",
)


@dataclass(frozen=True)
class Condition:
    name: str
    rho_slot: float
    rho_operand: float
    rho_mod: float


# Representative overlaps from the plan: identical, one-factor flips, disjoint, partial.
CAPACITY_CONDITIONS: tuple[Condition, ...] = (
    Condition("identical", 1.0, 1.0, 1.0),
    Condition("slot_diff", 0.0, 1.0, 1.0),
    Condition("operand_diff", 1.0, 0.0, 1.0),
    Condition("mod_diff", 1.0, 1.0, 0.0),
    Condition("disjoint", 0.0, 0.0, 0.0),
    Condition("partial", 0.5, 0.5, 0.5),
)


def rho_grid(which: str) -> list[Condition]:
    """``extreme`` is {0,1}^3 (8 cells). ``full`` is {0,0.5,1}^3 (27 cells)."""
    if which == "extreme":
        levels = (0.0, 1.0)
    elif which == "full":
        levels = (0.0, 0.5, 1.0)
    else:
        raise ValueError(f"grid must be extreme|full, got {which}")
    cells: list[Condition] = []
    for rho_slot in levels:
        for rho_operand in levels:
            for rho_mod in levels:
                cells.append(
                    Condition(
                        f"s{rho_slot:g}_o{rho_operand:g}_m{rho_mod:g}",
                        rho_slot,
                        rho_operand,
                        rho_mod,
                    )
                )
    return cells


def conditions_by_name(names: list[str]) -> list[Condition]:
    """Select named cells from the full 27-grid (e.g. ``s0.5_o0.5_m1``)."""
    lookup = {cell.name: cell for cell in rho_grid("full")}
    missing = [name for name in names if name not in lookup]
    if missing:
        raise SystemExit(
            f"unknown condition(s) {missing}; expected names like s0.5_o0.5_m1"
        )
    return [lookup[name] for name in names]
