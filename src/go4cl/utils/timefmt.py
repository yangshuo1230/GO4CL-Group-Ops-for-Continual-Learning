"""UTC timestamps and local run stamps (shared by phases and analysis)."""

from __future__ import annotations

from datetime import datetime, timezone


def utc_now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def stamp() -> str:
    """Local wall-clock stamp used in run directory names (`YYYYMMDD_HHMMSS`)."""
    return datetime.now().strftime("%Y%m%d_%H%M%S")
