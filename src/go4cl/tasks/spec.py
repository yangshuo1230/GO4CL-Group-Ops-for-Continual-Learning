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
    """Four operations forming a perfect matching on the eight input positions."""

    name: str
    task_id: int  # 0 -> TASK_A token, 1 -> TASK_B token
    operations: tuple[Operation, ...]

    def __post_init__(self) -> None:
        if len(self.operations) != NUM_LATENT_OPS:
            raise ValueError(f"expected {NUM_LATENT_OPS} ops, got {len(self.operations)}")
        slots = [op.slot for op in self.operations]
        if sorted(slots) != list(range(NUM_QUERIES)):
            raise ValueError(f"slots must be a permutation of 0..3, got {slots}")
        latents = [op.latent_id for op in self.operations]
        if sorted(latents) != list(LATENT_OPS):
            raise ValueError(f"latent ids must be a permutation of 0..3, got {latents}")
        used: list[int] = []
        for op in self.operations:
            used.extend([op.i, op.j])
        if sorted(used) != list(range(SEQ_LEN_OPERANDS)):
            raise ValueError(
                "operations must form a perfect matching on positions 0..7; "
                f"got positions {sorted(used)}"
            )

    @property
    def task_token(self) -> int:
        return TASK_TOKEN_IDS[self.task_id]

    def by_slot(self) -> dict[int, Operation]:
        return {op.slot: op for op in self.operations}

    def by_latent(self) -> dict[int, Operation]:
        return {op.latent_id: op for op in self.operations}

    def moduli(self) -> tuple[int, ...]:
        return tuple(self.by_slot()[s].modulus for s in range(NUM_QUERIES))

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
