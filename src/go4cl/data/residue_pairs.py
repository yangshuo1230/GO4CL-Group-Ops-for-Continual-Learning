"""Unordered residue-pair splits stratified by output class."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

SplitName = Literal["train", "val", "test"]


def unordered_residue_pair(r1: int, r2: int) -> tuple[int, int]:
    return (min(r1, r2), max(r1, r2))


def all_unordered_pairs(modulus: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for r1 in range(modulus):
        for r2 in range(r1, modulus):
            pairs.append((r1, r2))
    return pairs


def pair_label(pair: tuple[int, int], modulus: int) -> int:
    return (pair[0] + pair[1]) % modulus


@dataclass(frozen=True)
class ResiduePairSplit:
    """Train/val/test partition of unordered residue pairs for one modulus."""

    modulus: int
    train: tuple[tuple[int, int], ...]
    val: tuple[tuple[int, int], ...]
    test: tuple[tuple[int, int], ...]
    ratios: tuple[float, float, float]
    data_seed: int

    def get(self, split: SplitName) -> tuple[tuple[int, int], ...]:
        return {"train": self.train, "val": self.val, "test": self.test}[split]

    def pair_to_split(self) -> dict[tuple[int, int], SplitName]:
        mapping: dict[tuple[int, int], SplitName] = {}
        for name in ("train", "val", "test"):
            for p in self.get(name):  # type: ignore[arg-type]
                mapping[p] = name  # type: ignore[assignment]
        return mapping

    def to_dict(self) -> dict[str, Any]:
        return {
            "modulus": self.modulus,
            "train": [list(p) for p in self.train],
            "val": [list(p) for p in self.val],
            "test": [list(p) for p in self.test],
            "ratios": list(self.ratios),
            "data_seed": self.data_seed,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResiduePairSplit:
        def _pairs(key: str) -> tuple[tuple[int, int], ...]:
            return tuple((int(a), int(b)) for a, b in d[key])

        return cls(
            modulus=int(d["modulus"]),
            train=_pairs("train"),
            val=_pairs("val"),
            test=_pairs("test"),
            ratios=(float(d["ratios"][0]), float(d["ratios"][1]), float(d["ratios"][2])),
            data_seed=int(d["data_seed"]),
        )


def stratified_residue_pair_split(
    modulus: int,
    *,
    ratios: tuple[float, float, float] = (0.6, 0.2, 0.2),
    data_seed: int = 0,
) -> ResiduePairSplit:
    """
    Stratify unordered residue pairs by output class y=(r1+r2) mod p.

    Priority for small moduli: each output class should have at least one
    unordered pair in val and test when the class has enough pairs.
    """
    if abs(sum(ratios) - 1.0) > 1e-6:
        raise ValueError(f"ratios must sum to 1, got {ratios}")

    rng = np.random.default_rng(data_seed)
    by_label: dict[int, list[tuple[int, int]]] = {y: [] for y in range(modulus)}
    for pair in all_unordered_pairs(modulus):
        by_label[pair_label(pair, modulus)].append(pair)

    train: list[tuple[int, int]] = []
    val: list[tuple[int, int]] = []
    test: list[tuple[int, int]] = []

    for y in range(modulus):
        pairs = by_label[y]
        rng.shuffle(pairs)
        n = len(pairs)
        if n == 0:
            continue
        if n == 1:
            # Prefer train coverage for singleton classes
            train.append(pairs[0])
            continue
        if n == 2:
            train.append(pairs[0])
            # Split the remaining between val/test deterministically
            if rng.random() < 0.5:
                val.append(pairs[1])
            else:
                test.append(pairs[1])
            continue

        # Ensure at least one in val and test when possible
        test.append(pairs[0])
        val.append(pairs[1])
        remaining = pairs[2:]
        n_rem = len(remaining)
        n_train = int(round(ratios[0] * n))
        # Already assigned 2; allocate remaining preferentially to train
        # so overall ratios approximate the target.
        already_train = 0
        target_train = max(0, n_train - already_train)
        # Cap so val/test keep their single guaranteed items
        target_train = min(target_train, n_rem)
        # Recompute desired val/test among leftover after train allocation
        leftover_after_train = n_rem - target_train
        # Of the original n, we want ~ratios[1] in val and ~ratios[2] in test,
        # and we already placed 1 each.
        target_val_extra = max(0, int(round(ratios[1] * n)) - 1)
        target_test_extra = max(0, leftover_after_train - target_val_extra)
        # Adjust if rounding overflow
        while target_train + target_val_extra + target_test_extra > n_rem:
            if target_test_extra > 0:
                target_test_extra -= 1
            elif target_val_extra > 0:
                target_val_extra -= 1
            else:
                target_train -= 1
        while target_train + target_val_extra + target_test_extra < n_rem:
            target_train += 1

        train.extend(remaining[:target_train])
        val.extend(remaining[target_train : target_train + target_val_extra])
        test.extend(remaining[target_train + target_val_extra :])

    return ResiduePairSplit(
        modulus=modulus,
        train=tuple(train),
        val=tuple(val),
        test=tuple(test),
        ratios=ratios,
        data_seed=data_seed,
    )


def assert_disjoint(split: ResiduePairSplit) -> None:
    sets = [set(split.train), set(split.val), set(split.test)]
    for i in range(3):
        for j in range(i + 1, 3):
            inter = sets[i] & sets[j]
            if inter:
                raise AssertionError(f"split overlap between {i} and {j}: {inter}")
    all_pairs = set(all_unordered_pairs(split.modulus))
    covered = sets[0] | sets[1] | sets[2]
    if covered != all_pairs:
        raise AssertionError(
            f"incomplete coverage for p={split.modulus}: "
            f"missing {all_pairs - covered}, extra {covered - all_pairs}"
        )
