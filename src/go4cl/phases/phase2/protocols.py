"""Phase 2A: compare training protocols on one fixed A/B dataset."""

from __future__ import annotations

import argparse

from go4cl.phases.common import stamp
from go4cl.phases.phase2.grid import ALL_PROTOCOLS, Condition
from go4cl.phases.phase2.launch import add_shared_args, launch_grid


def run_protocols(args: argparse.Namespace) -> None:
    cond = Condition(
        f"s{args.rho_slot:g}_o{args.rho_operand:g}_m{args.rho_mod:g}",
        float(args.rho_slot),
        float(args.rho_operand),
        float(args.rho_mod),
    )
    launch_grid(
        args,
        step="protocols",
        conditions=[cond],
        protocols=list(args.protocols),
        sizes=[(int(args.d_model), int(args.n_layers))],
    )


def add_protocols_args(parser: argparse.ArgumentParser) -> None:
    add_shared_args(parser, default_out=f"runs/phase2/protocols/{stamp()}")
    parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=list(ALL_PROTOCOLS),
        help="Protocols to compare on the same manifest.",
    )
    parser.add_argument("--rho-slot", type=float, default=1.0)
    parser.add_argument("--rho-operand", type=float, default=1.0)
    parser.add_argument("--rho-mod", type=float, default=1.0)
