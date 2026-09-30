"""Phase 1A-mech: single-modulus / single-op mechanism analysis.

Run after scan-moduli. Goal: identify whether the model learned an algorithmic
circuit for one modular-addition op (Fourier features, operand routing, readout).
"""

from __future__ import annotations

import argparse
import sys


def run_mech_single(_args: argparse.Namespace) -> None:
    print(
        "[phase1/mech-single] not implemented yet.\n"
        "Run after phase1 scan-moduli finishes.\n"
        "Focus: single-op / single-modulus Fourier, probes, attention routing, "
        "and causal interventions on checkpoints from runs/phase1/scan_moduli/.\n"
        "See docs/PHASE_STEP_GUIDE.md §1A-mech.",
        file=sys.stderr,
    )
    raise SystemExit(2)


def add_mech_single_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default="runs/phase1/mech_single",
        help="Output root for single-modulus mechanism analyses",
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default="runs/phase1/scan_moduli",
        help="Directory of scan-moduli runs / checkpoints to analyze",
    )
    parser.add_argument(
        "--moduli",
        type=int,
        nargs="+",
        default=None,
        help="Subset of moduli to analyze (default: all available under ckpt-root)",
    )
