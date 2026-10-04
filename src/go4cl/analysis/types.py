"""Stable identifiers for mechanistic analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class OperationRef:
    """One queried modular-addition operation inside a task.

    ``latent_id`` is the TaskSpec identity and is the preferred filter key.
    ``slot`` is the query-token index (0..3) and is only a fallback for
    legacy datasets that stored no ``latent_ids``.
    ``modulus`` is derived (the p in (x_i + x_j) mod p) and is not unique
    when two ops share a modulus (``pair_same`` / ``all_same``).
    """

    latent_id: int
    modulus: int
    operand_i: int
    operand_j: int
    slot: int

    def report_key(self) -> str:
        return f"lat{self.latent_id}/slot{self.slot}/p{self.modulus}"


@dataclass(frozen=True)
class AnalysisTarget:
    """A (job, checkpoint, data) triple ready for analysis."""

    job_dir: Path
    data_dir: Path
    ckpt_path: Path
    ckpt_kind: str
