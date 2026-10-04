"""Tests for AnalysisContext path / checkpoint resolution."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from go4cl.analysis.context import resolve_checkpoint, resolve_data_dir, try_resolve_checkpoint
from go4cl.analysis.types import OperationRef


def test_resolve_data_dir_from_config(tmp_path: Path) -> None:
    data = tmp_path / "dataset"
    data.mkdir()
    (data / "manifest.json").write_text("{}\n", encoding="utf-8")
    job = tmp_path / "job"
    job.mkdir()
    (job / "config_resolved.json").write_text(
        json.dumps({"data_dir": str(data)}), encoding="utf-8"
    )
    assert resolve_data_dir(job) == data


def test_resolve_data_dir_fallback_stamp_layout(tmp_path: Path) -> None:
    stamp = tmp_path / "runs" / "phase1" / "multi_op" / "stamp"
    job = stamp / "runs" / "multi_four_diff_m23-29-31-37_pack1__wd0.3_steps1__ms0"
    data = stamp / "data" / "multi_four_diff_m23-29-31-37_pack1"
    data.mkdir(parents=True)
    job.mkdir(parents=True)
    (data / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert resolve_data_dir(job) == data


def test_resolve_data_dir_unique_glob_fallback(tmp_path: Path) -> None:
    stamp = tmp_path / "stamp"
    job = stamp / "runs" / "expfoo__wd0.3"
    data = stamp / "data" / "expfoo_extra"
    data.mkdir(parents=True)
    job.mkdir(parents=True)
    (data / "manifest.json").write_text("{}\n", encoding="utf-8")
    assert resolve_data_dir(job) == data


def test_resolve_data_dir_ambiguous_glob(tmp_path: Path) -> None:
    stamp = tmp_path / "stamp"
    job = stamp / "runs" / "expfoo__wd0.3"
    job.mkdir(parents=True)
    a = stamp / "data" / "expfoo_a"
    b = stamp / "data" / "expfoo_b"
    a.mkdir(parents=True)
    b.mkdir(parents=True)
    (a / "manifest.json").write_text("{}\n", encoding="utf-8")
    (b / "manifest.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="ambiguous data dir"):
        resolve_data_dir(job)


def test_resolve_data_dir_missing(tmp_path: Path) -> None:
    job = tmp_path / "empty_job"
    job.mkdir()
    with pytest.raises(FileNotFoundError, match="cannot resolve data dir"):
        resolve_data_dir(job)


def test_resolve_checkpoint_best_and_final(tmp_path: Path) -> None:
    ckpt = tmp_path / "ckpts"
    ckpt.mkdir()
    (ckpt / "best.pt").write_bytes(b"x")
    (ckpt / "a_only_final.pt").write_bytes(b"y")
    assert resolve_checkpoint(tmp_path, "best").name == "best.pt"
    assert resolve_checkpoint(tmp_path, "final").name == "a_only_final.pt"
    assert try_resolve_checkpoint(tmp_path, "missing") is None


def test_operation_ref_report_key() -> None:
    op = OperationRef(latent_id=1, modulus=29, operand_i=0, operand_j=1, slot=2)
    assert op.report_key() == "lat1/slot2/p29"
