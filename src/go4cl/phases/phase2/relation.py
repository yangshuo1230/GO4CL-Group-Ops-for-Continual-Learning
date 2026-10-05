"""Phase 2B: task-relation matrix, extreme 8 cells then full 27."""

from __future__ import annotations

import argparse

from go4cl.phases.common import stamp
from go4cl.phases.phase2.grid import RELATION_PROTOCOLS, conditions_by_name, rho_grid
from go4cl.phases.phase2.launch import add_shared_args, launch_grid


def run_relation_matrix(args: argparse.Namespace) -> None:
    if args.conditions:
        cells = conditions_by_name(list(args.conditions))
    else:
        cells = rho_grid(args.grid)
    launch_grid(
        args,
        step="relation-matrix",
        conditions=cells,
        protocols=list(args.protocols),
        sizes=[(int(args.d_model), int(args.n_layers))],
    )


def add_relation_matrix_args(parser: argparse.ArgumentParser) -> None:
    add_shared_args(parser, default_out=f"runs/phase2/relation_matrix/{stamp()}")
    parser.add_argument(
        "--grid",
        type=str,
        default="extreme",
        choices=["extreme", "full"],
        help="extreme: {0,1}^3 (8). full: {0,0.5,1}^3 (27). Run extreme first.",
    )
    parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=list(RELATION_PROTOCOLS),
        help="Default compares a/b-only, joint, and both sequential orders.",
    )
    parser.add_argument(
        "--conditions",
        type=str,
        nargs="+",
        default=None,
        help="Optional cell names from the 27-grid (e.g. s0.5_o0.5_m1). "
        "Overrides --grid when set.",
    )
