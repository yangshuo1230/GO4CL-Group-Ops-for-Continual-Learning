"""Resolved configuration for the three-task partition experiment."""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

# Nearest batch to the phase-2 8192 that is divisible by both 8 (A/B, 4 ops)
# and 12 (A/B/C, 4 ops). 8184 = 24 * 341.
FORMAL_BATCH_SIZE = 8184
BATCH_UNIT = 24

FORMAL_LAUNCH = "bash scripts/task_partition/run_formal.sh"


@dataclass
class PartitionConfig:
    """Formal-run defaults. Smoke overrides the step counts in place."""

    out_dir: str = "runs/task_partition/formal"
    model_seed: int = 0
    data_seed: int = 0
    sampler_seed: int = 0
    eval_seed: int = 0
    train_frac: float = 0.8
    lr: float = 1e-3
    weight_decay: float = 0.3
    batch_size: int = FORMAL_BATCH_SIZE
    d_model: int = 64
    n_layers: int = 3
    n_heads: int = 4
    d_mlp: int = 256
    ab_steps: int = 200_000
    continuation_steps: int = 100_000
    abc_steps: int = 300_000
    eval_every: int = 1000
    ckpt_every: int = 10_000
    eval_n_per_operation: int = 256
    c_milestones: tuple[int, ...] = (1000, 5000, 10_000, 20_000)
    stable_threshold: float = 0.90
    stable_window: int = 5
    partition_min_acc: float = 0.90
    counterfactual_diag_min: float = 0.80
    counterfactual_off_max: float = 0.20
    counterfactual_flip_min: float = 0.80
    require_partition: bool = True
    mech_n_contexts: int = 256
    mech_n_per_operation: int = 64
    probe_steps: int = 200
    compile_model: bool = True
    device: str = "cuda"
    smoke: bool = False

    def __post_init__(self) -> None:
        self.out_dir = str(self.out_dir)
        self.c_milestones = tuple(int(step) for step in self.c_milestones)
        self.validate()

    def validate(self) -> None:
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.batch_size <= 0 or self.batch_size % BATCH_UNIT != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be a positive multiple of "
                f"{BATCH_UNIT} so A/B steps are 1:1 and A/B/C steps are 1:1:1. "
                f"Use {FORMAL_BATCH_SIZE} instead of 8192."
            )
        for name in ("ab_steps", "continuation_steps", "abc_steps", "eval_every"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")
        for name in ("ab_steps", "continuation_steps", "abc_steps"):
            steps = int(getattr(self, name))
            if steps % int(self.eval_every) != 0:
                raise ValueError(
                    f"{name}={steps} must be divisible by eval_every={self.eval_every} "
                    "so the phase boundary is evaluated"
                )
        for step in self.c_milestones:
            if step <= 0 or step > int(self.continuation_steps):
                raise ValueError(
                    f"C milestone {step} is outside 1..{self.continuation_steps}"
                )
            if step % int(self.eval_every) != 0:
                raise ValueError(
                    f"C milestone {step} must fall on an eval step "
                    f"(eval_every={self.eval_every})"
                )
        if self.stable_window < 1:
            raise ValueError("stable_window must be >= 1")
        if not 0.0 <= float(self.stable_threshold) <= 1.0:
            raise ValueError("stable_threshold must be in [0, 1]")
        if not 0.0 < float(self.train_frac) < 1.0:
            raise ValueError("train_frac must be in (0, 1)")

    def phase_sampler_seed(self, phase: str) -> int:
        """Deterministic stream offset per phase. C never reads the A/B stream."""
        offsets = {
            "ab_joint": 0,
            "c_only": 50_000,
            "ab_continued": 29,
            "abc_joint": 17,
        }
        if phase not in offsets:
            raise KeyError(phase)
        return int(self.sampler_seed) + offsets[phase]

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["c_milestones"] = list(self.c_milestones)
        payload["phase_sampler_seeds"] = {
            phase: self.phase_sampler_seed(phase)
            for phase in ("ab_joint", "c_only", "ab_continued", "abc_joint")
        }
        return payload


def smoke_config(out_dir: str) -> PartitionConfig:
    """Tiny run that exercises all three protocols, the gate, and the figures."""
    return PartitionConfig(
        out_dir=out_dir,
        ab_steps=4,
        continuation_steps=4,
        abc_steps=6,
        batch_size=24,
        eval_every=2,
        ckpt_every=2,
        eval_n_per_operation=2,
        c_milestones=(2,),
        stable_threshold=0.0,
        stable_window=1,
        require_partition=False,
        mech_n_contexts=4,
        mech_n_per_operation=2,
        probe_steps=20,
        compile_model=False,
        smoke=True,
    )


def load_config(
    path: Path | str,
    *,
    out_dir: str | None = None,
    device: str | None = None,
    model_seed: int | None = None,
) -> PartitionConfig:
    raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(raw, dict):
        raise ValueError(f"config must be a mapping, got {type(raw).__name__}")
    if out_dir is not None:
        raw["out_dir"] = out_dir
    if device is not None:
        raw["device"] = device
    if model_seed is not None:
        raw["model_seed"] = int(model_seed)
    known = {item.name for item in fields(PartitionConfig)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise ValueError(f"unknown config keys: {unknown}")
    return PartitionConfig(**raw)
