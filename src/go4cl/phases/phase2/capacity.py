"""Phase 2D: width/depth ablation on representative task relations."""

from __future__ import annotations

import argparse

from go4cl.phases.common import stamp
from go4cl.phases.phase2.grid import CAPACITY_CONDITIONS, CAPACITY_PROTOCOLS
from go4cl.phases.phase2.launch import add_shared_args, launch_grid


def run_capacity(args: argparse.Namespace) -> None:
    sizes = [
        (int(width), int(depth))
        for width in args.d_models
        for depth in args.n_layers_list
    ]
    launch_grid(
        args,
        step="capacity",
        conditions=list(CAPACITY_CONDITIONS),
        protocols=list(args.protocols),
        sizes=sizes,
    )


def add_capacity_args(parser: argparse.ArgumentParser) -> None:
    add_shared_args(
        parser,
        default_out=f"runs/phase2/capacity/{stamp()}",
        include_model_size=False,
    )
    parser.add_argument(
        "--protocols",
        type=str,
        nargs="+",
        default=list(CAPACITY_PROTOCOLS),
    )
    parser.add_argument("--d-models", type=int, nargs="+", default=[32, 64, 128])
    parser.add_argument("--n-layers-list", type=int, nargs="+", default=[2, 3, 4])
