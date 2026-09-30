"""Phase 1B: multi-operation single-task and same-modulus facilitation."""

from __future__ import annotations

import argparse
import sys


def run_multi_op(_args: argparse.Namespace) -> None:
    print(
        "[phase1/multi-op] not implemented yet.\n"
        "Run after phase1 calibrate + scan-moduli lock a training config.\n"
        "See docs/RESEARCH_EXPERIMENT_PLAN.md §阶段一 1B.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def add_multi_op_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default="runs/phase1/multi_op",
        help="Output root (placeholder)",
    )
