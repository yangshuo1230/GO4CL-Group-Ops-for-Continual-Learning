"""Optional TASK-slot negatives with a fixed reject label (default 0).

Does not change eval loaders or datasets. Real A/B examples keep their
labels; extra rows only overwrite the TASK position and set ``labels``.

Token pools (TASK slot, no vocab expansion):
- phase A: ``TASK_B`` plus digit ids ``0..63`` (non-A, including B)
- phase B / joint: digit ids only (neither A nor B; no A replay)

Label ``0`` collides with a real residue class; that is intentional for this
probe, not a reserved reject class.
"""

from __future__ import annotations

from typing import Iterator, Literal, Sequence

import numpy as np
import torch

from go4cl.constants import NUM_DIGITS, SEQ_LEN_OPERANDS, TOKEN_TASK_B

NullTaskPhase = Literal["a", "b", "joint"]


def foreign_task_token_ids(phase: NullTaskPhase) -> tuple[int, ...]:
    """TASK-slot ids treated as 'not the current real task(s)'."""
    digits = tuple(range(NUM_DIGITS))
    if phase == "a":
        return (TOKEN_TASK_B, *digits)
    if phase in {"b", "joint"}:
        return digits
    raise ValueError(f"phase must be a|b|joint, got {phase}")


def mix_null_task_batch(
    batch: dict[str, torch.Tensor],
    rng: np.random.Generator,
    *,
    token_ids: Sequence[int],
    ratio: float,
    label: int,
) -> dict[str, torch.Tensor]:
    """Append copies of random rows with TASK token replaced and label fixed."""
    if ratio <= 0:
        return batch
    tokens = batch["tokens"]
    n = int(tokens.shape[0])
    n_extra = max(int(round(n * float(ratio))), 1)
    ids = np.asarray(list(token_ids), dtype=np.int64)
    if ids.size == 0:
        raise ValueError("token_ids must be non-empty")
    idx = rng.integers(0, n, size=n_extra)
    extra: dict[str, torch.Tensor] = {}
    for key, value in batch.items():
        extra[key] = value[idx].clone()
    extra["tokens"][:, SEQ_LEN_OPERANDS] = torch.as_tensor(
        rng.choice(ids, size=n_extra), dtype=extra["tokens"].dtype
    )
    extra["labels"] = torch.full(
        (n_extra,), int(label), dtype=batch["labels"].dtype
    )
    return {key: torch.cat([batch[key], extra[key]], dim=0) for key in batch}


class NullTaskTokenMixer:
    """Wrap an infinite packed train loader; eval loaders must stay unwrapped."""

    def __init__(
        self,
        base,
        *,
        phase: NullTaskPhase,
        ratio: float,
        label: int = 0,
        seed: int = 0,
    ) -> None:
        if ratio <= 0:
            raise ValueError(f"null-task ratio must be positive, got {ratio}")
        self.base = base
        self.phase = phase
        self.token_ids = foreign_task_token_ids(phase)
        self.ratio = float(ratio)
        self.label = int(label)
        self.seed = int(seed)
        self.batch_size = getattr(base, "batch_size", None)
        self.n_packs = getattr(base, "n_packs", None)
        self.n_ops = getattr(base, "n_ops", None)
        self.dataset = getattr(base, "dataset", None)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rng = np.random.default_rng(self.seed)
        for batch in self.base:
            yield mix_null_task_batch(
                batch,
                rng,
                token_ids=self.token_ids,
                ratio=self.ratio,
                label=self.label,
            )


def wrap_null_task_loader(
    loader,
    *,
    phase: NullTaskPhase,
    ratio: float,
    label: int = 0,
    seed: int = 0,
    enabled: bool = True,
):
    if not enabled or ratio <= 0:
        return loader
    return NullTaskTokenMixer(
        loader, phase=phase, ratio=ratio, label=label, seed=seed
    )
