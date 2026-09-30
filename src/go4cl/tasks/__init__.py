"""Task package exports."""

from go4cl.tasks.multi_op import build_multi_op_pair, build_multi_op_task
from go4cl.tasks.relations import TaskPairSpec, build_task_pair, compute_overlaps, swap_ab
from go4cl.tasks.single_op import build_single_op_pair, build_single_op_task
from go4cl.tasks.spec import Operation, TaskSpec

__all__ = [
    "Operation",
    "TaskSpec",
    "TaskPairSpec",
    "build_task_pair",
    "build_single_op_task",
    "build_single_op_pair",
    "build_multi_op_task",
    "build_multi_op_pair",
    "compute_overlaps",
    "swap_ab",
]
