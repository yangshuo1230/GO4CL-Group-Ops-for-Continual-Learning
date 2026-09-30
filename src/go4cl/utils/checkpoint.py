"""Checkpoint save/load helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch

from go4cl.model.transformer import ModelConfig, ModularTransformer


def save_checkpoint(
    path: Path | str,
    model: ModularTransformer,
    *,
    optimizer: torch.optim.Optimizer | None = None,
    step: int = 0,
    meta: dict[str, Any] | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state": model.state_dict(),
        "model_config": model.cfg.to_dict(),
        "step": step,
        "meta": meta or {},
    }
    if optimizer is not None:
        payload["optimizer_state"] = optimizer.state_dict()
    torch.save(payload, path)


def load_checkpoint(
    path: Path | str,
    *,
    map_location: str | torch.device = "cpu",
    model: ModularTransformer | None = None,
) -> tuple[ModularTransformer, dict[str, Any]]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    cfg = ModelConfig.from_dict(payload["model_config"])
    if model is None:
        model = ModularTransformer(cfg)
    model.load_state_dict(payload["model_state"])
    return model, payload


def write_json(path: Path | str, obj: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")
