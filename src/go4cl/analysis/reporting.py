"""CSV / JSON / Markdown writers for analysis reports.

Keep existing column names. Do not invent a new results schema here.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from go4cl.analysis.cache import to_jsonable


def strip_private(obj: Any) -> Any:
    """Drop keys starting with ``_`` (loaders, cached tensors) before serialize."""
    if isinstance(obj, dict):
        return {
            k: strip_private(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, list):
        return [strip_private(v) for v in obj]
    return obj


def write_csv_rows(
    path: Path | str,
    rows: Sequence[Mapping[str, Any]],
    fields: Iterable[str],
    *,
    extrasaction: str = "ignore",
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(fields)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fieldnames, extrasaction=extrasaction
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))
    return path


def write_json_report(path: Path | str, payload: Any, *, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_jsonable(payload), indent=indent, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return path


def write_markdown(path: Path | str, lines: Sequence[str] | str) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = lines if isinstance(lines, str) else "\n".join(lines)
    if not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path
