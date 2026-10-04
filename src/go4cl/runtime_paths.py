"""Scratch / cache paths on the NFS workspace (avoid filling system ``/tmp``)."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

# Repo root: .../GO4CL-Group-Ops-for-Continual-Learning
_REPO_ROOT = Path(__file__).resolve().parents[2]
# Sibling workspace cache (shared across projects on this machine)
_WORKSPACE = _REPO_ROOT.parent

DEFAULT_TMP = _REPO_ROOT / ".tmp"
DEFAULT_UV_CACHE = _WORKSPACE / ".cache" / "uv"
DEFAULT_TORCH_TMP = _REPO_ROOT / ".tmp" / "torch"


def configure_scratch_dirs(
    *,
    tmp: Path | str | None = None,
    uv_cache: Path | str | None = None,
) -> Path:
    """Point tempfile / common caches at NFS-backed dirs; return TMPDIR path.

    Idempotent. Safe to call in CLI main and in ProcessPool workers.
    Does not override variables the user already set explicitly, except when
    the current TMPDIR is missing or still the tiny system ``/tmp``.
    """
    tmp_path = Path(tmp) if tmp is not None else DEFAULT_TMP
    uv_path = Path(uv_cache) if uv_cache is not None else DEFAULT_UV_CACHE
    torch_tmp = DEFAULT_TORCH_TMP

    tmp_path.mkdir(parents=True, exist_ok=True)
    uv_path.mkdir(parents=True, exist_ok=True)
    torch_tmp.mkdir(parents=True, exist_ok=True)

    current = os.environ.get("TMPDIR") or os.environ.get("TMP") or os.environ.get("TEMP")
    force = current is None or current in {"/tmp", "/var/tmp"} or not Path(current).exists()
    if force or tmp is not None:
        os.environ["TMPDIR"] = str(tmp_path)
        os.environ["TMP"] = str(tmp_path)
        os.environ["TEMP"] = str(tmp_path)
        tempfile.tempdir = str(tmp_path)

    if "UV_CACHE_DIR" not in os.environ or uv_cache is not None:
        os.environ["UV_CACHE_DIR"] = str(uv_path)

    # Keep torch / cuda scratch off the root overlay when possible
    os.environ.setdefault("TORCHINDUCTOR_CACHE_DIR", str(torch_tmp / "inductor"))
    os.environ.setdefault("CUDA_CACHE_PATH", str(torch_tmp / "cuda"))
    Path(os.environ["TORCHINDUCTOR_CACHE_DIR"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["CUDA_CACHE_PATH"]).mkdir(parents=True, exist_ok=True)

    return Path(os.environ["TMPDIR"])
