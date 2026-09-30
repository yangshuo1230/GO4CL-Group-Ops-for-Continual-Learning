"""Phase 2 stubs: dual-task behavioral dynamics."""

from __future__ import annotations

import argparse
import sys


def _not_ready(step: str) -> None:
    print(
        f"[phase2/{step}] not implemented yet.\n"
        "Finish phase1 first. See docs/RESEARCH_EXPERIMENT_PLAN.md §阶段二.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def run_protocols(args: argparse.Namespace) -> None:
    _not_ready("protocols")


def run_relation_matrix(args: argparse.Namespace) -> None:
    _not_ready("relation-matrix")


def run_capacity(args: argparse.Namespace) -> None:
    _not_ready("capacity")


def add_protocols_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase2/protocols")


def add_relation_matrix_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase2/relation_matrix")


def add_capacity_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase2/capacity")
