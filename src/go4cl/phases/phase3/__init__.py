"""Phase 3 stubs: continual-learning mechanism analysis."""

from __future__ import annotations

import argparse
import sys


def _not_ready(step: str) -> None:
    print(
        f"[phase3/{step}] not implemented yet.\n"
        "Finish phase2 first. See docs/RESEARCH_EXPERIMENT_PLAN.md §阶段三.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def run_optimize(args: argparse.Namespace) -> None:
    _not_ready("optimize")


def run_route_compute_readout(args: argparse.Namespace) -> None:
    _not_ready("route-compute-readout")


def run_forget_types(args: argparse.Namespace) -> None:
    _not_ready("forget-types")


def add_optimize_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase3/optimize")


def add_route_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase3/route_compute_readout")


def add_forget_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out", type=str, default="runs/phase3/forget_types")
