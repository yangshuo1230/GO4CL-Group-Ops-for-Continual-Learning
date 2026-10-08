"""Locked experiment defaults — single source for CLI argparse defaults.

CLI parsers must import from here rather than repeating literals. Changing a
value here changes every command that uses that dataclass. Per-run
``config_resolved.json`` remains the authoritative record of what actually ran.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelDefaults:
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    lr: float = 1e-3
    n_aliases: int = 16


@dataclass(frozen=True)
class SingleOpDefaults:
    """Phase 1A single-op training (calibrate / scan-moduli)."""

    train_frac: float = 0.8
    train_fracs: tuple[float, ...] = (0.4, 0.6, 0.8)
    weight_decays: tuple[float, ...] = (0.1, 0.3, 1.0)
    scan_weight_decay: float = 0.3
    steps: int = 100_000
    steps: int = 100_000
    batch_size: int = 2048  # query examples per step, sampled with replacement
    calib_modulus: int = 31
    n_train_pairs: int = 150


@dataclass(frozen=True)
class MultiOpDefaults:
    """Phase 1B packed multi-op training."""

    train_frac: float = 0.8
    weight_decay: float = 0.3
    steps: int = 100_000
    batch_size: int = 8192  # query examples per step, not packed contexts
    variants: tuple[str, ...] = ("all_same", "four_diff", "pair_same")


@dataclass(frozen=True)
class Phase2Defaults:
    """Phase 2 dual-task protocols / relation matrix / capacity."""

    train_frac: float = 0.8
    weight_decay: float = 0.3
    steps: int = 100_000
    batch_size: int = 8192
    n_aliases: int = 16
    # sequential_ab_replay only: fraction of each phase-B packed batch that is A.
    sequential_ab_replay_ratio: float = 0.1


MODEL = ModelDefaults()
SINGLE_OP = SingleOpDefaults()
MULTI_OP = MultiOpDefaults()
PHASE2 = Phase2Defaults()
