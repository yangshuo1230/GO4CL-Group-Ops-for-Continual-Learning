"""Tests for analysis reporting writers."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from go4cl.analysis.reporting import write_csv_rows, write_json_report, write_markdown


def test_write_csv_rows_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "out.csv"
    rows = [{"a": 1, "b": "x"}, {"a": 2, "b": "y"}]
    write_csv_rows(path, rows, ["a", "b"])
    with path.open(newline="", encoding="utf-8") as handle:
        got = list(csv.DictReader(handle))
    assert got == [{"a": "1", "b": "x"}, {"a": "2", "b": "y"}]


def test_write_csv_empty_rows_writes_header(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    write_csv_rows(path, [], ["operation", "acc"])
    text = path.read_text(encoding="utf-8")
    assert text.splitlines() == ["operation,acc"]


def test_write_json_report_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "rep.json"
    write_json_report(path, {"k": 1, "nested": {"ok": True}})
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["k"] == 1
    assert payload["nested"]["ok"] is True


def test_write_markdown(tmp_path: Path) -> None:
    path = tmp_path / "n.md"
    write_markdown(path, ["# Title", "", "body"])
    text = path.read_text(encoding="utf-8")
    assert text.startswith("# Title")
    assert text.endswith("\n")
