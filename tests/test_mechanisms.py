"""Tests for 1C multi-op mechanism helpers."""

from __future__ import annotations

from pathlib import Path

from go4cl.data.dataset import ModularAdditionDataset
from go4cl.data.manifest import DataManifest
from go4cl.phases.phase1.mechanisms import (
    DEFAULT_JOB,
    _ops_from_manifest,
    _resolve_ckpt,
    _resolve_data_dir,
    filter_by_modulus,
)


def test_resolve_default_four_diff_job() -> None:
    job = Path(DEFAULT_JOB)
    if not job.is_dir():
        return
    ckpt = _resolve_ckpt(job, "best")
    data = _resolve_data_dir(job)
    assert ckpt.is_file()
    assert (data / "manifest.json").is_file()
    ops = _ops_from_manifest(DataManifest.load(data / "manifest.json"))
    assert [o.modulus for o in ops] == [47, 43, 37, 23]


def test_filter_by_modulus() -> None:
    data = Path(DEFAULT_JOB).parent.parent / "data"
    # find four_diff data
    matches = sorted(data.glob("multi_four_diff_m47-43-37-23_*")) if data.is_dir() else []
    if not matches:
        return
    ds = ModularAdditionDataset.from_disk(matches[0], "A", "test", 0)
    sub = filter_by_modulus(ds, 23)
    assert len(sub) > 0
    assert set(int(x) for x in sub.moduli.tolist()) == {23}
