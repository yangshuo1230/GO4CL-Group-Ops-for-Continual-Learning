"""Discover scan-moduli jobs / checkpoints / paired datasets for 1A-mech."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from go4cl.analysis.context import try_resolve_checkpoint
from go4cl.data.manifest import DataManifest

_JOB_RE = re.compile(
    r"^single_p(?P<p>\d+)_.*__wd(?P<wd>[^_]+)_steps(?P<steps>\d+)__ms(?P<ms>\d+)$"
)


@dataclass(frozen=True)
class MechTarget:
    modulus: int
    job_dir: Path
    data_dir: Path
    ckpt_path: Path
    ckpt_kind: str
    job_id: str
    experiment_id: str
    operand_i: int
    operand_j: int
    slot: int


def _parse_job_id(name: str) -> dict[str, str] | None:
    m = _JOB_RE.match(name)
    return m.groupdict() if m else None


def _experiment_id_from_job(job_id: str) -> str:
    # single_p31_..._p1a__wd0.5_steps20000__ms0 -> single_p31_..._p1a
    if "__" in job_id:
        return job_id.split("__", 1)[0]
    return job_id


def discover_targets(
    ckpt_root: Path | str,
    *,
    moduli: list[int] | None = None,
    ckpt_kinds: list[str] | None = None,
) -> list[MechTarget]:
    """Find (modulus, ckpt) pairs under a scan-moduli stamp directory."""
    root = Path(ckpt_root)
    runs_dir = root / "runs" if (root / "runs").is_dir() else root
    data_root = root / "data" if (root / "data").is_dir() else root.parent / "data"
    kinds = list(ckpt_kinds or ["final", "best"])
    want = set(moduli) if moduli else None

    targets: list[MechTarget] = []
    if not runs_dir.is_dir():
        return targets

    for job_dir in sorted(runs_dir.iterdir()):
        if not job_dir.is_dir():
            continue
        parsed = _parse_job_id(job_dir.name)
        if parsed is None:
            continue
        p = int(parsed["p"])
        if want is not None and p not in want:
            continue
        exp_id = _experiment_id_from_job(job_dir.name)
        data_dir = data_root / exp_id
        if not (data_dir / "manifest.json").is_file():
            # fall back: search by modulus tag
            matches = sorted(data_root.glob(f"single_p{p}_*")) if data_root.is_dir() else []
            if not matches:
                continue
            data_dir = matches[0]
            exp_id = data_dir.name
        manifest = DataManifest.load(data_dir / "manifest.json")
        op = manifest.task_pair.task_a.operations[0]
        for kind in kinds:
            ckpt = try_resolve_checkpoint(job_dir, kind)
            if ckpt is None:
                continue
            targets.append(
                MechTarget(
                    modulus=p,
                    job_dir=job_dir,
                    data_dir=data_dir,
                    ckpt_path=ckpt,
                    ckpt_kind=kind,
                    job_id=job_dir.name,
                    experiment_id=exp_id,
                    operand_i=int(op.i),
                    operand_j=int(op.j),
                    slot=int(op.slot),
                )
            )
    return targets
