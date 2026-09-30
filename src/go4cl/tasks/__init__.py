"""Task package exports."""

from go4cl.tasks.relations import TaskPairSpec, build_task_pair, compute_overlaps, swap_ab
from go4cl.tasks.spec import Operation, TaskSpec

__all__ = [
    "Operation",
    "TaskSpec",
    "TaskPairSpec",
    "build_task_pair",
    "compute_overlaps",
    "swap_ab",
]
