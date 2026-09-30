"""Multi-operation task construction for Phase 1B."""

from __future__ import annotations

from typing import Literal, Sequence

import numpy as np

from go4cl.constants import MODULUS_PAIRS, NUM_LATENT_OPS, PRIMES, SEQ_LEN_OPERANDS
from go4cl.tasks.relations import TaskPairSpec
from go4cl.tasks.spec import Operation, TaskSpec, perfect_matchings

MultiOpVariant = Literal["one", "four_diff", "pair_same", "all_same"]
VARIANTS: tuple[MultiOpVariant, ...] = ("one", "four_diff", "pair_same", "all_same")


def choose_base_moduli(task_seed: int) -> tuple[int, int, int, int]:
    """
    Pick (p, q, p2, p3) for a task seed.

    ``(p, q)`` is a nearby pair from ``MODULUS_PAIRS`` (used by four_diff on the
    first two latents). ``p2, p3`` are drawn from the remaining primes.
    """
    rng = np.random.default_rng(10_000 + int(task_seed))
    pairs = list(MODULUS_PAIRS)
    rng.shuffle(pairs)
    p, q = pairs[0]
    if rng.random() < 0.5:
        p, q = q, p
    rest = [x for pair in pairs[1:] for x in pair]
    rng.shuffle(rest)
    p2, p3 = int(rest[0]), int(rest[1])
    return int(p), int(q), p2, p3


def _sample_matching_and_slots(
    task_seed: int,
) -> tuple[list[tuple[int, int]], list[int]]:
    """Deterministic perfect matching + slot assignment for ``task_seed``."""
    rng = np.random.default_rng(int(task_seed))
    all_m = perfect_matchings(SEQ_LEN_OPERANDS)
    matching = all_m[int(rng.integers(0, len(all_m)))]
    edges = list(matching)
    rng.shuffle(edges)
    ordered: list[tuple[int, int]] = []
    for edge in edges:
        i, j = sorted(edge)
        if rng.random() < 0.5:
            i, j = j, i
        ordered.append((int(i), int(j)))
    slots = list(range(NUM_LATENT_OPS))
    rng.shuffle(slots)
    return ordered, slots


def moduli_for_variant(
    variant: MultiOpVariant, base: tuple[int, int, int, int]
) -> tuple[int, ...]:
    p, q, p2, p3 = base
    if variant == "one":
        return (p,)
    if variant == "four_diff":
        return (p, q, p2, p3)
    if variant == "pair_same":
        return (p, p, p2, p3)
    if variant == "all_same":
        return (p, p, p, p)
    raise ValueError(f"unknown variant: {variant}")


def build_multi_op_task(
    variant: MultiOpVariant,
    *,
    task_seed: int = 0,
    name: str = "A",
    task_id: int = 0,
    base_moduli: tuple[int, int, int, int] | None = None,
) -> TaskSpec:
    """
    Build a Phase 1B task.

    Across variants with the same ``task_seed``, operand matching and slots are
    identical; only the modulus assignment changes (except ``one``, which keeps
    the first latent's edge/slot only).
    """
    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant}")
    base = base_moduli if base_moduli is not None else choose_base_moduli(task_seed)
    for m in base:
        if m not in PRIMES:
            raise ValueError(f"modulus {m} not in PRIMES={PRIMES}")

    ordered, slots = _sample_matching_and_slots(task_seed)
    mods = moduli_for_variant(variant, base)

    if variant == "one":
        i, j = ordered[0]
        return TaskSpec(
            name=name,
            task_id=task_id,
            operations=(
                Operation(
                    latent_id=0,
                    i=i,
                    j=j,
                    modulus=mods[0],
                    slot=slots[0],
                ),
            ),
        )

    ops = []
    for z in range(NUM_LATENT_OPS):
        i, j = ordered[z]
        ops.append(
            Operation(
                latent_id=z,
                i=i,
                j=j,
                modulus=mods[z],
                slot=slots[z],
            )
        )
    return TaskSpec(name=name, task_id=task_id, operations=tuple(ops))


def build_multi_op_pair(
    variant: MultiOpVariant,
    *,
    task_seed: int = 0,
    pair_id: str | None = None,
    base_moduli: tuple[int, int, int, int] | None = None,
) -> TaskPairSpec:
    """A/B pair for Phase 1B; B is an identical clone (``a_only`` training)."""
    base = base_moduli if base_moduli is not None else choose_base_moduli(task_seed)
    task_a = build_multi_op_task(
        variant, task_seed=task_seed, name="A", task_id=0, base_moduli=base
    )
    task_b = build_multi_op_task(
        variant, task_seed=task_seed, name="B", task_id=1, base_moduli=base
    )
    mods = moduli_for_variant(variant, base)
    if pair_id is None:
        mod_tag = "-".join(str(m) for m in mods)
        pair_id = f"multi_{variant}_m{mod_tag}_seed{task_seed}"
    return TaskPairSpec(
        task_a=task_a,
        task_b=task_b,
        rho_slot=1.0,
        rho_operand=1.0,
        rho_mod=1.0,
        task_seed=task_seed,
        pair_id=pair_id,
    )


def describe_variant(variant: MultiOpVariant, base: Sequence[int]) -> str:
    mods = moduli_for_variant(variant, tuple(base))  # type: ignore[arg-type]
    return f"{variant}: moduli={list(mods)}"
