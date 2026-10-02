"""Task specifications for modular-addition multi-query tasks."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Sequence

from go4cl.constants import (
    LATENT_OPS,
    NUM_LATENT_OPS,
    NUM_QUERIES,
    SEQ_LEN_OPERANDS,
    TASK_TOKEN_IDS,
)


@dataclass(frozen=True)
class OperationKey:
    """Stable identity for one latent op inside a task.

    Report key is always OperationKey (not modulus alone). ``slot`` and
    ``modulus`` are attributes of the underlying Operation; slot alone is not
    a global key across task_seed reassignments.
    """

    task_id: int
    latent_id: int
    slot: int

    def report_key(self) -> str:
        return f"task{self.task_id}/lat{self.latent_id}/slot{self.slot}"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> OperationKey:
        return cls(
            task_id=int(d["task_id"]),
            latent_id=int(d["latent_id"]),
            slot=int(d["slot"]),
        )

    @classmethod
    def from_operation(cls, op: Operation, *, task_id: int) -> OperationKey:
        return cls(task_id=int(task_id), latent_id=int(op.latent_id), slot=int(op.slot))


@dataclass(frozen=True)
class Operation:
    """One modular-addition op: (x_i + x_j) mod p, exposed via a query slot."""

    latent_id: int
    i: int
    j: int
    modulus: int
    slot: int

    def __post_init__(self) -> None:
        if not (0 <= self.i < SEQ_LEN_OPERANDS and 0 <= self.j < SEQ_LEN_OPERANDS):
            raise ValueError(f"operand indices out of range: {(self.i, self.j)}")
        if self.i == self.j:
            raise ValueError("operand indices must be distinct")
        if not (0 <= self.slot < NUM_QUERIES):
            raise ValueError(f"slot out of range: {self.slot}")
        if self.modulus < 2:
            raise ValueError(f"modulus must be >= 2, got {self.modulus}")

    @property
    def operand_pair(self) -> frozenset[int]:
        return frozenset({self.i, self.j})

    def evaluate(self, x: Sequence[int]) -> int:
        return (int(x[self.i]) + int(x[self.j])) % self.modulus

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Operation:
        return cls(
            latent_id=int(d["latent_id"]),
            i=int(d["i"]),
            j=int(d["j"]),
            modulus=int(d["modulus"]),
            slot=int(d["slot"]),
        )


@dataclass(frozen=True)
class TaskSpec:
    """One-to-four operations. Full tasks (4 ops) must be a perfect matching."""

    name: str
    task_id: int  # 0 -> TASK_A token, 1 -> TASK_B token
    operations: tuple[Operation, ...]

    def __post_init__(self) -> None:
        n = len(self.operations)
        if not (1 <= n <= NUM_LATENT_OPS):
            raise ValueError(f"expected 1..{NUM_LATENT_OPS} ops, got {n}")

        slots = [op.slot for op in self.operations]
        if len(set(slots)) != n:
            raise ValueError(f"slots must be unique, got {slots}")
        if any(s < 0 or s >= NUM_QUERIES for s in slots):
            raise ValueError(f"slots out of range: {slots}")

        latents = [op.latent_id for op in self.operations]
        if len(set(latents)) != n:
            raise ValueError(f"latent ids must be unique, got {latents}")

        used: list[int] = []
        for op in self.operations:
            used.extend([op.i, op.j])
        if len(used) != len(set(used)):
            raise ValueError(
                "operand positions must not overlap across ops; "
                f"got positions {sorted(used)}"
            )

        # Full 4-op scientific tasks keep the original hard constraints.
        if n == NUM_LATENT_OPS:
            if sorted(slots) != list(range(NUM_QUERIES)):
                raise ValueError(f"slots must be a permutation of 0..3, got {slots}")
            if sorted(latents) != list(LATENT_OPS):
                raise ValueError(f"latent ids must be a permutation of 0..3, got {latents}")
            if sorted(used) != list(range(SEQ_LEN_OPERANDS)):
                raise ValueError(
                    "operations must form a perfect matching on positions 0..7; "
                    f"got positions {sorted(used)}"
                )

    @property
    def n_ops(self) -> int:
        return len(self.operations)

    @property
    def is_single_op(self) -> bool:
        return self.n_ops == 1

    @property
    def task_token(self) -> int:
        return TASK_TOKEN_IDS[self.task_id]

    def by_slot(self) -> dict[int, Operation]:
        return {op.slot: op for op in self.operations}

    def by_latent(self) -> dict[int, Operation]:
        return {op.latent_id: op for op in self.operations}

    def operation_keys(self) -> tuple[OperationKey, ...]:
        return tuple(
            OperationKey.from_operation(op, task_id=self.task_id)
            for op in self.operations
        )

    def moduli(self) -> tuple[int, ...]:
        return tuple(op.modulus for op in sorted(self.operations, key=lambda o: o.slot))

    def evaluate_slot(self, x: Sequence[int], slot: int) -> int:
        return self.by_slot()[slot].evaluate(x)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "task_id": self.task_id,
            "operations": [op.to_dict() for op in self.operations],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> TaskSpec:
        ops = tuple(Operation.from_dict(o) for o in d["operations"])
        return cls(name=str(d["name"]), task_id=int(d["task_id"]), operations=ops)


def perfect_matchings(n: int = SEQ_LEN_OPERANDS) -> list[tuple[frozenset[int], ...]]:
    """Enumerate perfect matchings on n labeled points (n even)."""
    if n % 2 != 0:
        raise ValueError("n must be even")
    points = list(range(n))

    def _rec(remaining: list[int]) -> list[list[frozenset[int]]]:
        if not remaining:
            return [[]]
        a = remaining[0]
        out: list[list[frozenset[int]]] = []
        for idx in range(1, len(remaining)):
            b = remaining[idx]
            edge = frozenset({a, b})
            rest = remaining[1:idx] + remaining[idx + 1 :]
            for matching in _rec(rest):
                out.append([edge] + matching)
        return out

    return [tuple(m) for m in _rec(points)]


def matching_to_ordered_pairs(
    matching: Sequence[frozenset[int]],
    rng_order: Iterable[tuple[int, int]] | None = None,
) -> list[tuple[int, int]]:
    """Convert unordered edges to ordered (i, j) with i < j by default."""
    pairs: list[tuple[int, int]] = []
    if rng_order is None:
        for edge in matching:
            i, j = sorted(edge)
            pairs.append((i, j))
        return pairs
    for edge, (i, j) in zip(matching, rng_order, strict=True):
        if frozenset({i, j}) != edge:
            raise ValueError("ordered pair does not match edge")
        pairs.append((i, j))
    return pairs
