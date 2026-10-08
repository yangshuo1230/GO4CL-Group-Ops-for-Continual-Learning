"""YAML configs for the forward-transfer mechanism grid.

Both experiments require ``fixed_a`` and ``replay_ratio=0``.
``carry_optimizer_state`` exists so a later control can keep Adam moments;
the configs shipped with this round set it false.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from go4cl.phases.transfer_mechanism.checkpoint_mix import (
    DEFAULT_INTERVENTIONS,
    reject_nonzero_replay,
    reserved_interventions,
)
from go4cl.utils.config import load_config

ALLOWED_PROTOCOLS = ("b_only", "sequential_ab")

MODULUS_CONDITIONS: tuple[dict[str, Any], ...] = (
    {"name": "s0_o0_m0", "rho_slot": 0.0, "rho_operand": 0.0, "rho_mod": 0.0},
    {"name": "s0_o0_m0.5", "rho_slot": 0.0, "rho_operand": 0.0, "rho_mod": 0.5},
    {"name": "s0_o0_m1", "rho_slot": 0.0, "rho_operand": 0.0, "rho_mod": 1.0},
)
COMPONENT_CONDITION: dict[str, Any] = {
    "name": "s0_o0_m1",
    "rho_slot": 0.0,
    "rho_operand": 0.0,
    "rho_mod": 1.0,
}


@dataclass
class TransferConfig:
    experiment: str
    fixed_a: bool
    replay_ratio: float
    carry_optimizer_state: bool
    protocols: list[str]
    conditions: list[dict[str, Any]]
    interventions: list[str]
    reserved_interventions: list[str]
    task_seeds: list[int]
    pilot_model_seeds: list[int]
    final_model_seeds: list[int]
    seed_set: str
    data_seed: int
    b_sampler_seed: int
    eval_seed: int
    train_frac: float
    n_aliases: int
    weight_decay: float
    batch_size: int
    steps: int
    lr: float
    eval_every: int
    gen_threshold: float
    stable_window: int
    d_model: int
    n_layers: int
    n_heads: int
    d_mlp: int
    eval_n_per_operation: int
    out_dir: str
    device: str
    compile_model: bool
    include_reserved: bool = False
    config_path: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    def model_seeds(self, seed_set: str | None = None) -> list[int]:
        which = seed_set or self.seed_set
        if which == "pilot":
            return [int(s) for s in self.pilot_model_seeds]
        if which == "final":
            return [int(s) for s in self.final_model_seeds]
        raise ValueError(f"seed_set must be pilot|final, got {which}")

    def active_interventions(self) -> list[str]:
        names = list(self.interventions)
        if self.include_reserved:
            for name in self.reserved_interventions:
                if name not in names:
                    names.append(name)
        return names

    def validate(self) -> None:
        reject_nonzero_replay(self.replay_ratio)
        if not self.fixed_a:
            raise ValueError("transfer-mechanism experiments require fixed_a=true")
        if self.seed_set not in {"pilot", "final"}:
            raise ValueError(f"seed_set must be pilot|final, got {self.seed_set}")
        if self.stable_window < 1:
            raise ValueError("stable_window must be >= 1")
        if self.batch_size % 4 != 0:
            raise ValueError(
                f"batch_size={self.batch_size} must be a multiple of 4 operations"
            )
        unknown = [p for p in self.protocols if p not in ALLOWED_PROTOCOLS]
        if unknown:
            raise ValueError(
                f"protocols {unknown} are not allowed. "
                f"Only {list(ALLOWED_PROTOCOLS)} are supported, and replay is refused."
            )
        if self.experiment == "modulus_specificity":
            _expect_conditions(self.conditions, MODULUS_CONDITIONS)
            if set(self.protocols) != set(ALLOWED_PROTOCOLS):
                raise ValueError(
                    "modulus specificity protocols must be exactly "
                    f"{list(ALLOWED_PROTOCOLS)}, got {self.protocols}"
                )
        elif self.experiment == "component_reset":
            _expect_conditions(self.conditions, (COMPONENT_CONDITION,))
            if list(self.interventions) != list(DEFAULT_INTERVENTIONS):
                raise ValueError(
                    "component_reset interventions must be the 12 reset/keep "
                    f"conditions, got {self.interventions}"
                )
            expected_reserved = list(reserved_interventions(3))
            if list(self.reserved_interventions) != expected_reserved:
                raise ValueError(
                    "reserved_interventions drifted from the layer / keep_multiple list"
                )
        else:
            raise ValueError(f"unknown experiment {self.experiment}")
        for name in self.active_interventions():
            from go4cl.phases.transfer_mechanism.checkpoint_mix import parse_intervention

            parse_intervention(name, n_layers=int(self.n_layers))

    @classmethod
    def from_yaml(cls, path: Path | str) -> TransferConfig:
        path = Path(path)
        cfg = cls.from_dict(load_config(path))
        cfg.config_path = str(path)
        cfg.validate()
        return cfg

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TransferConfig:
        data = dict(raw)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        extra = {key: value for key, value in data.items() if key not in known}
        kwargs = {key: data[key] for key in known if key in data and key != "extra"}
        kwargs["extra"] = extra
        cfg = cls(**kwargs)
        cfg.task_seeds = [int(s) for s in cfg.task_seeds]
        cfg.pilot_model_seeds = [int(s) for s in cfg.pilot_model_seeds]
        cfg.final_model_seeds = [int(s) for s in cfg.final_model_seeds]
        cfg.protocols = [str(p) for p in cfg.protocols]
        cfg.interventions = [str(p) for p in cfg.interventions]
        cfg.reserved_interventions = [str(p) for p in cfg.reserved_interventions]
        cfg.replay_ratio = float(cfg.replay_ratio)
        cfg.fixed_a = bool(cfg.fixed_a)
        cfg.carry_optimizer_state = bool(cfg.carry_optimizer_state)
        return cfg


def _expect_conditions(
    got: list[dict[str, Any]], expected: tuple[dict[str, Any], ...] | tuple[dict[str, Any]]
) -> None:
    def _key(row: dict[str, Any]) -> tuple:
        return (
            str(row["name"]),
            float(row["rho_slot"]),
            float(row["rho_operand"]),
            float(row["rho_mod"]),
        )

    if [_key(row) for row in got] != [_key(row) for row in expected]:
        raise ValueError(f"conditions {got} != required {list(expected)}")
