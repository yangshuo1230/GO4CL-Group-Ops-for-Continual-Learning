"""Unified context construction for packed / nuisance evaluation and analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from go4cl.constants import NUM_DIGITS, QUERY_TOKEN_IDS, SEQ_LEN_OPERANDS
from go4cl.data.generate import Example, sample_raw_operands
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.tasks.spec import Operation, OperationKey, TaskSpec

ContextMode = Literal["packed_id", "nuisance_random"]
SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class ContextRequest:
    target_operation: OperationKey
    target_split: SplitName
    distractor_split: SplitName | None = "train"
    context_mode: ContextMode = "packed_id"


@dataclass(frozen=True)
class ContextRecord:
    digits: tuple[int, ...]
    pairs_by_latent: dict[int, tuple[int, int]]


class ContextBuilder:
    """Single source of truth for 8-digit contexts + per-op query examples."""

    def __init__(
        self,
        task: TaskSpec,
        splits: dict[int, ResiduePairSplit],
    ) -> None:
        self.task = task
        self.splits = splits
        self._pool_arrays: dict[tuple[int, str], np.ndarray] = {}
        self._alias_tables: dict[int, tuple[np.ndarray, np.ndarray]] = {}

    def _pool(self, modulus: int, split_name: SplitName) -> list[tuple[int, int]]:
        pool = self.splits[modulus].get(split_name)
        if not pool:
            raise ValueError(
                f"empty {split_name} residue-pair pool for modulus {modulus}"
            )
        return pool

    def _draw_pair(
        self,
        rng: np.random.Generator,
        op: Operation,
        split_name: SplitName,
    ) -> tuple[tuple[int, int], int, int]:
        pool = self._pool(op.modulus, split_name)
        pair = pool[int(rng.integers(0, len(pool)))]
        swap = bool(rng.integers(0, 2))
        raw_i, raw_j = sample_raw_operands(rng, pair, op.modulus, swap=swap)
        return pair, raw_i, raw_j

    def sample_packed_digits(
        self,
        rng: np.random.Generator,
        *,
        split_by_latent: dict[int, SplitName] | None = None,
        default_split: SplitName = "train",
    ) -> ContextRecord:
        """Fill all ops into one shared digit context."""
        if self.task.n_ops == 1:
            digits = [int(rng.integers(0, NUM_DIGITS)) for _ in range(SEQ_LEN_OPERANDS)]
        else:
            digits = [0] * SEQ_LEN_OPERANDS

        pairs_by_latent: dict[int, tuple[int, int]] = {}
        for op in self.task.operations:
            split_name = (
                split_by_latent.get(op.latent_id, default_split)
                if split_by_latent
                else default_split
            )
            pair, raw_i, raw_j = self._draw_pair(rng, op, split_name)
            digits[op.i] = raw_i
            digits[op.j] = raw_j
            pairs_by_latent[op.latent_id] = pair
        return ContextRecord(digits=tuple(digits), pairs_by_latent=pairs_by_latent)

    def sample_nuisance_digits(
        self,
        rng: np.random.Generator,
        op: Operation,
        *,
        split_name: SplitName,
    ) -> ContextRecord:
        """Fix only ``op`` operands; other six positions ~U{0..63}."""
        digits = [int(rng.integers(0, NUM_DIGITS)) for _ in range(SEQ_LEN_OPERANDS)]
        pair, raw_i, raw_j = self._draw_pair(rng, op, split_name)
        digits[op.i] = raw_i
        digits[op.j] = raw_j
        return ContextRecord(
            digits=tuple(digits),
            pairs_by_latent={op.latent_id: pair},
        )

    def emit_query_example(
        self,
        digits: tuple[int, ...] | list[int],
        op: Operation,
        *,
        residue_pair: tuple[int, int],
        split: str,
    ) -> Example:
        tokens = tuple(list(digits) + [self.task.task_token, QUERY_TOKEN_IDS[op.slot]])
        label = (int(digits[op.i]) + int(digits[op.j])) % op.modulus
        return Example(
            tokens=tokens,
            label=label,
            task_name=self.task.name,
            slot=op.slot,
            modulus=op.modulus,
            residue_pair=residue_pair,
            split=split,
            latent_id=op.latent_id,
            task_id=self.task.task_id,
        )

    def emit_all_queries(
        self,
        record: ContextRecord,
        *,
        split: str,
    ) -> list[Example]:
        return [
            self.emit_query_example(
                record.digits,
                op,
                residue_pair=record.pairs_by_latent[op.latent_id],
                split=split,
            )
            for op in self.task.operations
        ]

    def sample_train_pack(self, rng: np.random.Generator) -> list[Example]:
        record = self.sample_packed_digits(rng, default_split="train")
        return self.emit_all_queries(record, split="train")

    def _pool_array(self, modulus: int, split_name: SplitName) -> np.ndarray:
        key = (int(modulus), str(split_name))
        cached = self._pool_arrays.get(key)
        if cached is None:
            cached = np.asarray(self._pool(modulus, split_name), dtype=np.int64)
            if cached.ndim != 2 or cached.shape[1] != 2:
                raise ValueError(f"residue pool for p={modulus} must be [N, 2]")
            self._pool_arrays[key] = cached
        return cached

    def _alias_table(self, modulus: int) -> tuple[np.ndarray, np.ndarray]:
        """Padded raw-token table ``[p, max_alias]`` and per-residue counts."""
        cached = self._alias_tables.get(int(modulus))
        if cached is not None:
            return cached
        rows = [
            [x for x in range(NUM_DIGITS) if x % modulus == residue]
            for residue in range(modulus)
        ]
        width = max(len(row) for row in rows)
        table = np.zeros((modulus, width), dtype=np.int64)
        counts = np.empty(modulus, dtype=np.int64)
        for residue, row in enumerate(rows):
            table[residue, : len(row)] = row
            counts[residue] = len(row)
        cached = (table, counts)
        self._alias_tables[int(modulus)] = cached
        return cached

    def _sample_aliases(
        self, rng: np.random.Generator, residues: np.ndarray, modulus: int
    ) -> np.ndarray:
        table, counts = self._alias_table(modulus)
        choice = rng.integers(0, counts[residues])
        return table[residues, choice]

    def sample_train_batch(
        self, rng: np.random.Generator, n_packs: int
    ) -> dict[str, np.ndarray]:
        """Draw ``n_packs`` train contexts as arrays, one row per query.

        Same sampling rule as ``sample_train_pack`` repeated ``n_packs`` times
        (uniform train residue pair, fair swap, uniform raw alias). Random
        numbers are drawn in batch order, so a seed does not replay the scalar
        stream. Query order inside a pack follows ``task.operations``.
        """
        n_packs = int(n_packs)
        if n_packs <= 0:
            raise ValueError(f"n_packs must be positive, got {n_packs}")
        ops = self.task.operations
        n_ops = len(ops)
        digits = np.zeros((n_packs, SEQ_LEN_OPERANDS), dtype=np.int64)
        if n_ops == 1:
            digits[:] = rng.integers(0, NUM_DIGITS, size=digits.shape)

        labels = np.empty((n_packs, n_ops), dtype=np.int64)
        slots = np.empty((n_packs, n_ops), dtype=np.int64)
        moduli = np.empty((n_packs, n_ops), dtype=np.int64)
        latent_ids = np.empty((n_packs, n_ops), dtype=np.int64)
        query_ids = np.empty(n_ops, dtype=np.int64)
        for k, op in enumerate(ops):
            pool = self._pool_array(op.modulus, "train")
            picked = pool[rng.integers(0, pool.shape[0], size=n_packs)]
            swap = rng.integers(0, 2, size=n_packs).astype(bool)
            r_i = np.where(swap, picked[:, 1], picked[:, 0])
            r_j = np.where(swap, picked[:, 0], picked[:, 1])
            raw_i = self._sample_aliases(rng, r_i, op.modulus)
            raw_j = self._sample_aliases(rng, r_j, op.modulus)
            digits[:, op.i] = raw_i
            digits[:, op.j] = raw_j
            labels[:, k] = (raw_i + raw_j) % op.modulus
            slots[:, k] = op.slot
            moduli[:, k] = op.modulus
            latent_ids[:, k] = op.latent_id
            query_ids[k] = QUERY_TOKEN_IDS[op.slot]

        tokens = np.empty((n_packs, n_ops, SEQ_LEN_OPERANDS + 2), dtype=np.int64)
        tokens[:, :, :SEQ_LEN_OPERANDS] = digits[:, None, :]
        tokens[:, :, SEQ_LEN_OPERANDS] = self.task.task_token
        tokens[:, :, SEQ_LEN_OPERANDS + 1] = query_ids[None, :]
        n = n_packs * n_ops
        return {
            "tokens": tokens.reshape(n, SEQ_LEN_OPERANDS + 2),
            "labels": labels.reshape(n),
            "slots": slots.reshape(n),
            "moduli": moduli.reshape(n),
            "latent_ids": latent_ids.reshape(n),
            "task_ids": np.full(n, self.task.task_id, dtype=np.int64),
        }

    def sample_eval_for_target(
        self,
        rng: np.random.Generator,
        request: ContextRequest,
    ) -> Example:
        """One example for ``request.target_operation`` under the requested mode."""
        op = self.task.by_latent()[request.target_operation.latent_id]
        if (
            request.target_operation.task_id != self.task.task_id
            or request.target_operation.slot != op.slot
        ):
            raise ValueError(
                f"OperationKey {request.target_operation} does not match task "
                f"{self.task.name} op latent={op.latent_id} slot={op.slot}"
            )

        if request.context_mode == "nuisance_random":
            record = self.sample_nuisance_digits(
                rng, op, split_name=request.target_split
            )
            return self.emit_query_example(
                record.digits,
                op,
                residue_pair=record.pairs_by_latent[op.latent_id],
                split=request.target_split,
            )

        # packed_id: target uses target_split; distractors use distractor_split
        distractor = request.distractor_split or "train"
        split_by_latent = {
            other.latent_id: (
                request.target_split
                if other.latent_id == op.latent_id
                else distractor
            )
            for other in self.task.operations
        }
        record = self.sample_packed_digits(rng, split_by_latent=split_by_latent)
        return self.emit_query_example(
            record.digits,
            op,
            residue_pair=record.pairs_by_latent[op.latent_id],
            split=request.target_split,
        )


def build_analysis_dataset(
    task: TaskSpec,
    splits: dict[int, ResiduePairSplit],
    *,
    split: SplitName = "train",
    context_mode: ContextMode = "packed_id",
    analysis_seed: int = 0,
    aliases_per_pair: int = 4,
    contexts_per_pair: int = 1,
    distractor_split: SplitName = "train",
    target_latent_ids: list[int] | None = None,
) -> list[Example]:
    """Deterministic analysis examples from residue pools (does not mutate manifest).

    For ``packed_id`` + non-train ``split``, only the target op draws from
    ``split``; other ops use ``distractor_split`` (default train).
    """
    builder = ContextBuilder(task, splits)
    rng = np.random.default_rng(int(analysis_seed))
    targets = (
        [task.by_latent()[i] for i in target_latent_ids]
        if target_latent_ids is not None
        else list(task.operations)
    )
    examples: list[Example] = []
    for op in targets:
        pool = builder._pool(op.modulus, split)
        for pair in pool:
            for _alias in range(int(aliases_per_pair)):
                for _ctx in range(int(contexts_per_pair)):
                    # Re-seeded path via rng advances only; pair fixed for target.
                    if context_mode == "nuisance_random":
                        digits = [
                            int(rng.integers(0, NUM_DIGITS))
                            for _ in range(SEQ_LEN_OPERANDS)
                        ]
                        swap = bool(rng.integers(0, 2))
                        raw_i, raw_j = sample_raw_operands(
                            rng, pair, op.modulus, swap=swap
                        )
                        digits[op.i] = raw_i
                        digits[op.j] = raw_j
                        examples.append(
                            builder.emit_query_example(
                                tuple(digits),
                                op,
                                residue_pair=pair,
                                split=split,
                            )
                        )
                    else:
                        split_by_latent = {
                            other.latent_id: (
                                split
                                if other.latent_id == op.latent_id
                                else distractor_split
                            )
                            for other in task.operations
                        }
                        # Force target pair: sample distractors then overwrite target.
                        record = builder.sample_packed_digits(
                            rng, split_by_latent=split_by_latent
                        )
                        digits = list(record.digits)
                        swap = bool(rng.integers(0, 2))
                        raw_i, raw_j = sample_raw_operands(
                            rng, pair, op.modulus, swap=swap
                        )
                        digits[op.i] = raw_i
                        digits[op.j] = raw_j
                        pairs = dict(record.pairs_by_latent)
                        pairs[op.latent_id] = pair
                        examples.append(
                            builder.emit_query_example(
                                tuple(digits),
                                op,
                                residue_pair=pair,
                                split=split,
                            )
                        )
    return examples
