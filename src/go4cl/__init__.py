"""GO4CL: modular-addition continual learning for Transformer circuit analysis."""

from __future__ import annotations

import argparse


def _add_phase1(sub: argparse._SubParsersAction) -> None:
    from go4cl.phases.phase1 import (
        add_calibrate_args,
        add_mech_single_args,
        add_mechanisms_args,
        add_multi_op_args,
        add_scan_moduli_args,
        run_calibrate,
        run_mech_single,
        run_mechanisms,
        run_multi_op,
        run_scan_moduli,
    )

    p1 = sub.add_parser("phase1", help="Stage 1: single-task learning / grokking")
    p1_sub = p1.add_subparsers(dest="step", required=True)

    p_cal = p1_sub.add_parser(
        "calibrate",
        help="1A-0: grokking regime calibration on a medium modulus",
    )
    add_calibrate_args(p_cal)
    p_cal.set_defaults(_phase_runner=run_calibrate)

    p_scan = p1_sub.add_parser(
        "scan-moduli",
        help="1A: single-op modulus scan with a locked training config",
    )
    add_scan_moduli_args(p_scan)
    p_scan.set_defaults(_phase_runner=run_scan_moduli)

    p_ms = p1_sub.add_parser(
        "mech-single",
        help="1A-mech: single-modulus mechanism analysis after scan-moduli (stub)",
    )
    add_mech_single_args(p_ms)
    p_ms.set_defaults(_phase_runner=run_mech_single)

    p_multi = p1_sub.add_parser(
        "multi-op",
        help="1B: multi-op single-task / same-modulus facilitation (stub)",
    )
    add_multi_op_args(p_multi)
    p_multi.set_defaults(_phase_runner=run_multi_op)

    p_mech = p1_sub.add_parser(
        "mechanisms",
        help="1C: multi-op / comparative mechanism verification (stub)",
    )
    add_mechanisms_args(p_mech)
    p_mech.set_defaults(_phase_runner=run_mechanisms)


def _add_phase2(sub: argparse._SubParsersAction) -> None:
    from go4cl.phases.phase2 import (
        add_capacity_args,
        add_protocols_args,
        add_relation_matrix_args,
        run_capacity,
        run_protocols,
        run_relation_matrix,
    )

    p2 = sub.add_parser("phase2", help="Stage 2: dual-task behavioral dynamics")
    p2_sub = p2.add_subparsers(dest="step", required=True)

    p_proto = p2_sub.add_parser("protocols", help="2A: joint / sequential protocols (stub)")
    add_protocols_args(p_proto)
    p_proto.set_defaults(_phase_runner=run_protocols)

    p_rel = p2_sub.add_parser(
        "relation-matrix", help="2B: task-relation matrix (stub)"
    )
    add_relation_matrix_args(p_rel)
    p_rel.set_defaults(_phase_runner=run_relation_matrix)

    p_cap = p2_sub.add_parser("capacity", help="2D: capacity ablation (stub)")
    add_capacity_args(p_cap)
    p_cap.set_defaults(_phase_runner=run_capacity)


def _add_phase3(sub: argparse._SubParsersAction) -> None:
    from go4cl.phases.phase3 import (
        add_forget_args,
        add_optimize_args,
        add_route_args,
        run_forget_types,
        run_optimize,
        run_route_compute_readout,
    )

    p3 = sub.add_parser("phase3", help="Stage 3: continual-learning mechanisms")
    p3_sub = p3.add_subparsers(dest="step", required=True)

    p_opt = p3_sub.add_parser("optimize", help="3A: optimization-layer analysis (stub)")
    add_optimize_args(p_opt)
    p_opt.set_defaults(_phase_runner=run_optimize)

    p_route = p3_sub.add_parser(
        "route-compute-readout", help="3B: route/compute/readout (stub)"
    )
    add_route_args(p_route)
    p_route.set_defaults(_phase_runner=run_route_compute_readout)

    p_forget = p3_sub.add_parser("forget-types", help="3C: forgetting types (stub)")
    add_forget_args(p_forget)
    p_forget.set_defaults(_phase_runner=run_forget_types)


def main(argv: list[str] | None = None) -> None:
    from go4cl.runtime_paths import configure_scratch_dirs

    configure_scratch_dirs()

    parser = argparse.ArgumentParser(prog="go4cl", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    _add_phase1(sub)
    _add_phase2(sub)
    _add_phase3(sub)

    p_gen = sub.add_parser("generate-data", help="Generate fixed A/B datasets + manifest")
    p_gen.add_argument("--out", type=str, required=True)
    p_gen.add_argument("--task-seed", type=int, default=0)
    p_gen.add_argument("--data-seed", type=int, default=0)
    p_gen.add_argument("--rho-slot", type=float, default=1.0)
    p_gen.add_argument("--rho-operand", type=float, default=1.0)
    p_gen.add_argument("--rho-mod", type=float, default=1.0)
    p_gen.add_argument("--n-aliases", type=int, default=16)
    p_gen.add_argument("--n-nuisance", type=int, default=4)
    p_gen.add_argument("--experiment-id", type=str, default="default")

    p_train = sub.add_parser("train", help="Run a training protocol on a fixed dataset")
    p_train.add_argument("--data", type=str, required=True)
    p_train.add_argument("--out", type=str, required=True)
    p_train.add_argument(
        "--protocol",
        type=str,
        default="a_only",
        choices=[
            "a_only",
            "b_only",
            "joint",
            "interleaved",
            "sequential_ab",
            "sequential_ba",
            "a_only_continued",
        ],
    )
    p_train.add_argument("--steps", type=int, default=500)
    p_train.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Mini-batch size. Default: full-batch.",
    )
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1.0)
    p_train.add_argument("--model-seed", type=int, default=0)
    p_train.add_argument("--d-model", type=int, default=64)
    p_train.add_argument("--n-layers", type=int, default=3)
    p_train.add_argument("--device", type=str, default=None)
    p_train.add_argument("--wandb-project", type=str, default="go4cl")
    p_train.add_argument("--wandb-name", type=str, default=None)
    p_train.add_argument(
        "--wandb-mode",
        type=str,
        default=None,
        choices=["online", "offline", "disabled"],
    )
    p_train.add_argument("--no-wandb", action="store_true")

    p_smoke = sub.add_parser("smoke", help="Engineering smoke checks")
    p_smoke.add_argument("--out", type=str, default="runs/smoke")
    p_smoke.add_argument("--device", type=str, default=None)
    p_smoke.add_argument("--quick", action="store_true")

    args = parser.parse_args(argv)

    if args.cmd in {"phase1", "phase2", "phase3"}:
        args._phase_runner(args)
    elif args.cmd == "generate-data":
        from go4cl.scripts.generate_data import run_generate

        run_generate(args)
    elif args.cmd == "train":
        from go4cl.scripts.train import run_train

        run_train(args)
    elif args.cmd == "smoke":
        from go4cl.scripts.smoke import run_smoke

        run_smoke(args)
    else:
        parser.error(f"unknown command {args.cmd}")


__all__ = ["main"]
