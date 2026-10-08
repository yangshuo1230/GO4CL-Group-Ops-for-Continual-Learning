"""Fixed A/B/C task-partition specs.

Three tasks share query tokens Q0–Q3 and moduli (23, 37, 41, 53).
They differ only in which operand positions each query reads, so the
model has to use the task token to choose the pair.
"""

from __future__ import annotations

from go4cl.constants import QUERY_TOKEN_IDS
from go4cl.tasks.spec import Operation, TaskSpec

PARTITION_MODULI: tuple[int, ...] = (23, 37, 41, 53)

# Query slot -> operand positions. Each task is a perfect matching on 0..7.
PARTITION_PAIRS: dict[str, tuple[tuple[int, int], ...]] = {
    "A": ((0, 1), (2, 3), (4, 5), (6, 7)),
    "B": ((0, 2), (1, 3), (4, 6), (5, 7)),
    "C": ((0, 4), (1, 5), (2, 6), (3, 7)),
}

PARTITION_TASK_IDS: dict[str, int] = {"A": 0, "B": 1, "C": 2}
PARTITION_ORDER: tuple[str, ...] = ("A", "B", "C")


def build_partition_tasks() -> dict[str, TaskSpec]:
    """Return Task A, B, and C in that order."""
    tasks: dict[str, TaskSpec] = {}
    for name in PARTITION_ORDER:
        pairs = PARTITION_PAIRS[name]
        if len(pairs) != len(PARTITION_MODULI):
            raise RuntimeError(f"{name} pair count does not match moduli")
        ops = tuple(
            Operation(
                latent_id=slot,
                i=int(pair[0]),
                j=int(pair[1]),
                modulus=int(PARTITION_MODULI[slot]),
                slot=slot,
            )
            for slot, pair in enumerate(pairs)
        )
        tasks[name] = TaskSpec(
            name=name,
            task_id=PARTITION_TASK_IDS[name],
            operations=ops,
        )
    moduli = {name: task.moduli() for name, task in tasks.items()}
    if len(set(moduli.values())) != 1:
        raise RuntimeError(f"tasks must share per-query moduli, got {moduli}")
    queries = {
        name: tuple(QUERY_TOKEN_IDS[op.slot] for op in task.operations)
        for name, task in tasks.items()
    }
    if len(set(queries.values())) != 1:
        raise RuntimeError(f"tasks must share query tokens, got {queries}")
    tokens = [task.task_token for task in tasks.values()]
    if len(set(tokens)) != len(tokens):
        raise RuntimeError(f"task tokens must be distinct, got {tokens}")
    return tasks


def task_descriptions(tasks: dict[str, TaskSpec] | None = None) -> list[dict[str, object]]:
    """Human-readable operand map for the manifest."""
    tasks = tasks or build_partition_tasks()
    rows: list[dict[str, object]] = []
    for name in PARTITION_ORDER:
        task = tasks[name]
        for op in sorted(task.operations, key=lambda item: item.slot):
            rows.append(
                {
                    "task": name,
                    "task_token_id": int(task.task_token),
                    "query": f"Q{op.slot}",
                    "slot": int(op.slot),
                    "operands": [int(op.i), int(op.j)],
                    "modulus": int(op.modulus),
                    "latent_id": int(op.latent_id),
                }
            )
    return rows
