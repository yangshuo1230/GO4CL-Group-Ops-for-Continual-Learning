"""GO4CL: modular-addition continual learning for Transformer circuit analysis."""

from __future__ import annotations

import argparse


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="go4cl", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_gen = sub.add_parser("generate-data", help="Generate fixed A/B datasets + manifest")
    p_gen.add_argument("--out", type=str, required=True)
    p_gen.add_argument("--task-seed", type=int, default=0)
    p_gen.add_argument("--data-seed", type=int, default=0)
    p_gen.add_argument("--rho-slot", type=float, default=1.0)
    p_gen.add_argument("--rho-operand", type=float, default=1.0)
    p_gen.add_argument("--rho-mod", type=float, default=1.0)
    p_gen.add_argument("--n-aliases", type=int, default=2)
    p_gen.add_argument("--n-nuisance", type=int, default=1)
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
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr", type=float, default=1e-3)
    p_train.add_argument("--weight-decay", type=float, default=1.0)
    p_train.add_argument("--model-seed", type=int, default=0)
    p_train.add_argument("--d-model", type=int, default=64)
    p_train.add_argument("--n-layers", type=int, default=3)
    p_train.add_argument("--device", type=str, default=None)

    p_smoke = sub.add_parser("smoke", help="Run engineering smoke checks")
    p_smoke.add_argument("--out", type=str, default="runs/smoke")
    p_smoke.add_argument("--device", type=str, default=None)
    p_smoke.add_argument("--quick", action="store_true", help="Fewer steps for CI")

    args = parser.parse_args(argv)

    if args.cmd == "generate-data":
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
