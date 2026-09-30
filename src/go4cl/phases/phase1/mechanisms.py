"""Phase 1C: multi-op / comparative mechanism verification.

After multi-op training: same-modulus facilitation circuits, shared computation
vs separate routing, and transplants between ops. Single-modulus analysis lives
in ``mech-single`` (1A-mech) and should already be done.
"""

from __future__ import annotations

import argparse
import sys


def run_mechanisms(_args: argparse.Namespace) -> None:
    print(
        "[phase1/mechanisms] not implemented yet.\n"
        "Run after phase1 multi-op (and preferably after mech-single).\n"
        "Focus: same-modulus multi-op facilitation, circuit transplant, "
        "shared compute vs separate routing.\n"
        "See docs/PHASE_STEP_GUIDE.md §1C.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def add_mechanisms_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default="runs/phase1/mechanisms",
        help="Output root (placeholder)",
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default="runs/phase1/multi_op",
        help="Directory of multi-op runs / checkpoints to analyze",
    )
