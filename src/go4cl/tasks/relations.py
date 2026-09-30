"""Orthogonal A/B task-relation construction (slot / operand / modulus overlap)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from go4cl.constants import LATENT_OPS, MODULUS_PAIRS, NUM_LATENT_OPS
from go4cl.tasks.spec import Operation, TaskSpec, perfect_matchings


VALID_OVERLAPS = (0.0, 0.5, 1.0)


@dataclass(frozen=True)
class TaskPairSpec:
    """A paired A/B task with measured overlap factors."""

    task_a: TaskSpec
    task_b: TaskSpec
    rho_slot: float
    rho_operand: float
    rho_mod: float
    task_seed: int
    pair_id: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_id": self.pair_id,
            "task_seed": self.task_seed,
            "rho_slot": self.rho_slot,
            "rho_operand": self.rho_operand,
            "rho_mod": self.rho_mod,
            "task_a": self.task_a.to_dict(),
            "task_b": self.task_b.to_dict(),
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskPairSpec:
        return cls(
            task_a=TaskSpec.from_dict(d["task_a"]),
            task_b=TaskSpec.from_dict(d["task_b"]),
            rho_slot=float(d["rho_slot"]),
            rho_operand=float(d["rho_operand"]),
            rho_mod=float(d["rho_mod"]),
            task_seed=int(d["task_seed"]),
            pair_id=str(d["pair_id"]),
        )


def _validate_rho(name: str, rho: float) -> None:
    if rho not in VALID_OVERLAPS:
        raise ValueError(f"{name} must be in {VALID_OVERLAPS}, got {rho}")


def _choose_fixed_points(rng: np.random.Generator, rho: float) -> set[int]:
    """Choose latent ids that keep the property (rho in {0, 0.5, 1})."""
    n_fixed = int(round(rho * NUM_LATENT_OPS))
    if n_fixed == 0:
        return set()
    if n_fixed == NUM_LATENT_OPS:
        return set(LATENT_OPS)
    choices = list(LATENT_OPS)
    rng.shuffle(choices)
    return set(choices[:n_fixed])


def _sample_two_matchings(
    rng: np.random.Generator, rho_operand: float
) -> tuple[tuple[frozenset[int], ...], tuple[frozenset[int], ...]]:
    """Sample two perfect matchings with the desired edge-overlap fraction."""
    all_m = perfect_matchings()
    n_shared = int(round(rho_operand * NUM_LATENT_OPS))
    rng.shuffle(all_m)
    for m_a in all_m:
        for m_b in all_m:
            shared = len(set(m_a) & set(m_b))
            if shared == n_shared:
                # Forbid accidental duplicate edges beyond the intended share:
                # already enforced by exact shared count.
                return m_a, m_b
    raise RuntimeError(f"failed to find matchings with rho_operand={rho_operand}")


def _assign_moduli(
    rng: np.random.Generator, rho_mod: float
) -> tuple[dict[int, int], dict[int, int]]:
    """Assign moduli to latent ops for A and B given rho_mod."""
    pairs = list(MODULUS_PAIRS)
    rng.shuffle(pairs)
    # Each latent gets one modulus pair; A takes one member, B keeps or flips
    fixed = _choose_fixed_points(rng, rho_mod)
    mods_a: dict[int, int] = {}
    mods_b: dict[int, int] = {}
    for z, (p, q) in zip(LATENT_OPS, pairs):
        if rng.random() < 0.5:
            a_mod, b_alt = p, q
        else:
            a_mod, b_alt = q, p
        mods_a[z] = a_mod
        mods_b[z] = a_mod if z in fixed else b_alt
    return mods_a, mods_b


def compute_overlaps(task_a: TaskSpec, task_b: TaskSpec) -> tuple[float, float, float]:
    """Compute (rho_slot, rho_operand, rho_mod) over latent identities."""
    a = task_a.by_latent()
    b = task_b.by_latent()
    slot = sum(1 for z in LATENT_OPS if a[z].slot == b[z].slot) / NUM_LATENT_OPS
    operand = (
        sum(1 for z in LATENT_OPS if a[z].operand_pair == b[z].operand_pair) / NUM_LATENT_OPS
    )
    mod = sum(1 for z in LATENT_OPS if a[z].modulus == b[z].modulus) / NUM_LATENT_OPS
    return slot, operand, mod


def build_task_pair(
    *,
    rho_slot: float = 1.0,
    rho_operand: float = 1.0,
    rho_mod: float = 1.0,
    task_seed: int = 0,
    pair_id: str | None = None,
) -> TaskPairSpec:
    """
    Construct A/B tasks with controlled overlaps on latent operations.

    Construction:
    - Assign each latent a modulus pair; B keeps or flips per rho_mod.
    - Assign each latent an operand edge from two matchings with given overlap.
    - Assign each latent a query slot via permutations with given fixed-point rate.
    """
    _validate_rho("rho_slot", rho_slot)
    _validate_rho("rho_operand", rho_operand)
    _validate_rho("rho_mod", rho_mod)

    rng = np.random.default_rng(task_seed)
    mods_a, mods_b = _assign_moduli(rng, rho_mod)
    m_a, m_b = _sample_two_matchings(rng, rho_operand)

    # Bind edges to latents: shared edges keep the same latent; others are matched arbitrarily
    shared_edges = list(set(m_a) & set(m_b))
    only_a = list(set(m_a) - set(m_b))
    only_b = list(set(m_b) - set(m_a))
    rng.shuffle(shared_edges)
    rng.shuffle(only_a)
    rng.shuffle(only_b)

    latents = list(LATENT_OPS)
    rng.shuffle(latents)
    edge_a: dict[int, frozenset[int]] = {}
    edge_b: dict[int, frozenset[int]] = {}
    for z, e in zip(latents[: len(shared_edges)], shared_edges):
        edge_a[z] = e
        edge_b[z] = e
    rem = latents[len(shared_edges) :]
    for z, ea, eb in zip(rem, only_a, only_b):
        edge_a[z] = ea
        edge_b[z] = eb

    # Slot assignment: start from identity for A, permute for B relative to A
    slots_a = {z: z for z in LATENT_OPS}  # temporary; then random relabel
    slot_relabel = list(range(NUM_LATENT_OPS))
    rng.shuffle(slot_relabel)
    slots_a = {z: slot_relabel[z] for z in LATENT_OPS}

    fixed_slots = _choose_fixed_points(rng, rho_slot)
    # Build B slots: keep fixed latents' slots, derange the rest among remaining slots
    kept = {z: slots_a[z] for z in fixed_slots}
    free_latents = [z for z in LATENT_OPS if z not in fixed_slots]
    free_slots = [slots_a[z] for z in free_latents]
    if free_latents:
        for _ in range(1000):
            rng.shuffle(free_slots)
            # Prefer a derangement relative to A's slot mapping where possible
            if len(free_latents) == 1 or all(
                free_slots[i] != slots_a[free_latents[i]] for i in range(len(free_latents))
            ):
                break
        slots_b = dict(kept)
        for z, s in zip(free_latents, free_slots):
            slots_b[z] = s
    else:
        slots_b = dict(kept)

    def _ops(
        edges: dict[int, frozenset[int]],
        mods: dict[int, int],
        slots: dict[int, int],
    ) -> tuple[Operation, ...]:
        ops: list[Operation] = []
        for z in LATENT_OPS:
            i, j = sorted(edges[z])
            if rng.random() < 0.5:
                i, j = j, i
            ops.append(
                Operation(latent_id=z, i=i, j=j, modulus=mods[z], slot=slots[z])
            )
        return tuple(ops)

    task_a = TaskSpec(name="A", task_id=0, operations=_ops(edge_a, mods_a, slots_a))
    task_b = TaskSpec(name="B", task_id=1, operations=_ops(edge_b, mods_b, slots_b))
    got_slot, got_op, got_mod = compute_overlaps(task_a, task_b)

    # Numerical tolerance for float representation of 0/0.5/1
    def _close(a: float, b: float) -> bool:
        return abs(a - b) < 1e-9

    if not (_close(got_slot, rho_slot) and _close(got_op, rho_operand) and _close(got_mod, rho_mod)):
        raise RuntimeError(
            "constructed overlaps mismatch request: "
            f"got {(got_slot, got_op, got_mod)} vs {(rho_slot, rho_operand, rho_mod)}"
        )

    if pair_id is None:
        pair_id = (
            f"s{rho_slot:g}_o{rho_operand:g}_m{rho_mod:g}_seed{task_seed}"
        )
    return TaskPairSpec(
        task_a=task_a,
        task_b=task_b,
        rho_slot=got_slot,
        rho_operand=got_op,
        rho_mod=got_mod,
        task_seed=task_seed,
        pair_id=pair_id,
    )


def swap_ab(pair: TaskPairSpec) -> TaskPairSpec:
    """Return B→A counterpart with swapped task tokens / names."""
    a = pair.task_a
    b = pair.task_b
    new_a = TaskSpec(name="A", task_id=0, operations=b.operations)
    new_b = TaskSpec(name="B", task_id=1, operations=a.operations)
    return TaskPairSpec(
        task_a=new_a,
        task_b=new_b,
        rho_slot=pair.rho_slot,
        rho_operand=pair.rho_operand,
        rho_mod=pair.rho_mod,
        task_seed=pair.task_seed,
        pair_id=pair.pair_id + "_swap",
    )
