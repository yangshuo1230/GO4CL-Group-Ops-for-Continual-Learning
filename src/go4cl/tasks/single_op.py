"""Single-operation task construction for Phase 1A."""

from __future__ import annotations

import numpy as np

from go4cl.constants import PRIMES, SEQ_LEN_OPERANDS
from go4cl.tasks.relations import TaskPairSpec
from go4cl.tasks.spec import Operation, TaskSpec


def build_single_op_task(
    modulus: int,
    *,
    name: str = "A",
    task_id: int = 0,
    slot: int = 0,
    i: int = 0,
    j: int = 1,
    latent_id: int = 0,
) -> TaskSpec:
    """Build a task with exactly one valid query / modular-addition op."""
    if modulus not in PRIMES and modulus < 2:
        raise ValueError(f"unexpected modulus {modulus}")
    return TaskSpec(
        name=name,
        task_id=task_id,
        operations=(
            Operation(
                latent_id=latent_id,
                i=i,
                j=j,
                modulus=modulus,
                slot=slot,
            ),
        ),
    )


def build_single_op_pair(
    modulus: int,
    *,
    task_seed: int = 0,
    pair_id: str | None = None,
) -> TaskPairSpec:
    """
    A/B pair for single-op experiments.

    Phase 1 only trains A (``a_only``). B is an identical clone with the TASK_B
    token so existing loaders/protocols keep working without special cases.
    """
    rng = np.random.default_rng(task_seed)
    positions = list(range(SEQ_LEN_OPERANDS))
    rng.shuffle(positions)
    i, j = int(positions[0]), int(positions[1])
    if rng.random() < 0.5:
        i, j = j, i
    slot = int(rng.integers(0, 4))

    task_a = build_single_op_task(
        modulus, name="A", task_id=0, slot=slot, i=i, j=j, latent_id=0
    )
    task_b = build_single_op_task(
        modulus, name="B", task_id=1, slot=slot, i=i, j=j, latent_id=0
    )
    if pair_id is None:
        pair_id = f"single_p{modulus}_seed{task_seed}"
    return TaskPairSpec(
        task_a=task_a,
        task_b=task_b,
        rho_slot=1.0,
        rho_operand=1.0,
        rho_mod=1.0,
        task_seed=task_seed,
        pair_id=pair_id,
    )
