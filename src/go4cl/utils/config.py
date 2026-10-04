"""YAML/JSON config loading.

Deprecated as a configuration system: CLI defaults live in ``go4cl.defaults``
and each run writes ``config_resolved.json``. Keep these helpers only for
ad-hoc YAML/JSON files if you pass an explicit path in a script.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: Path | str) -> dict[str, Any]:
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix in {".yaml", ".yml"}:
        data = yaml.safe_load(text)
    elif path.suffix == ".json":
        import json

        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config suffix: {path.suffix}")
    if not isinstance(data, dict):
        raise ValueError("config root must be a mapping")
    return data


def deep_update(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_update(out[k], v)
        else:
            out[k] = v
    return out
