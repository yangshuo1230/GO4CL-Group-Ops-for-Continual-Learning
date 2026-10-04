"""CLI parser and command registration for ``go4cl``.

Phase 1/2 commands are implemented. Phase 3 parsers exist but exit as
not-implemented. Analysis pipelines are also available under ``go4cl analyze``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from go4cl.defaults import MODEL


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
        help="1A-mech: single-modulus mechanism analysis after scan-moduli",
    )
    add_mech_single_args(p_ms)
    p_ms.set_defaults(_phase_runner=run_mech_single)

    p_multi = p1_sub.add_parser(
        "multi-op",
        help="1B: multi-op single-task / same-modulus facilitation",
    )
    add_multi_op_args(p_multi)
    p_multi.set_defaults(_phase_runner=run_multi_op)

    p_mech = p1_sub.add_parser(
        "mechanisms",
        help="1C: multi-op / comparative mechanism verification",
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

    p_proto = p2_sub.add_parser(
        "protocols", help="2A: joint / sequential / interleaved protocols"
    )
    add_protocols_args(p_proto)
    p_proto.set_defaults(_phase_runner=run_protocols)

    p_rel = p2_sub.add_parser(
        "relation-matrix", help="2B: task-relation matrix (8 extreme, then 27)"
    )
    add_relation_matrix_args(p_rel)
    p_rel.set_defaults(_phase_runner=run_relation_matrix)

    p_cap = p2_sub.add_parser(
        "capacity", help="2D: width/depth ablation on representative relations"
    )
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

    p_opt = p3_sub.add_parser(
        "optimize", help="3A: optimization-layer analysis (not implemented)"
    )
    add_optimize_args(p_opt)
    p_opt.set_defaults(_phase_runner=run_optimize)

    p_route = p3_sub.add_parser(
        "route-compute-readout",
        help="3B: route/compute/readout (not implemented)",
    )
    add_route_args(p_route)
    p_route.set_defaults(_phase_runner=run_route_compute_readout)

    p_forget = p3_sub.add_parser(
        "forget-types", help="3C: forgetting types (not implemented)"
    )
    add_forget_args(p_forget)
    p_forget.set_defaults(_phase_runner=run_forget_types)


def _add_job_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--job-dir", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--ckpt-kind", type=str, default="best")
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--max-batches", type=int, default=None)


def _add_analyze(sub: argparse._SubParsersAction) -> None:
    from go4cl.phases.phase1 import (
        add_mech_single_args,
        add_mechanisms_args,
        run_mech_single,
        run_mechanisms,
    )

    p = sub.add_parser("analyze", help="Mechanistic analysis on a trained job")
    ps = p.add_subparsers(dest="analyze_cmd", required=True)

    p_s = ps.add_parser("single-op", help="1A-mech (alias of phase1 mech-single)")
    add_mech_single_args(p_s)
    p_s.set_defaults(_analyze_runner=run_mech_single)

    p_m = ps.add_parser("multi-op", help="1C mechanisms (alias of phase1 mechanisms)")
    add_mechanisms_args(p_m)
    p_m.set_defaults(_analyze_runner=run_mechanisms)

    p_c = ps.add_parser("causal-detail", help="Per-op causal knockout / Fourier matrix")
    _add_job_args(p_c)
    p_c.add_argument("--ablation-ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    p_c.add_argument("--report", type=str, default=None)
    p_c.add_argument("--export-only", action="store_true")
    p_c.set_defaults(_analyze_runner=_run_causal_detail)

    p_t = ps.add_parser("transplant", help="Operand patch + query-path transplant")
    _add_job_args(p_t)
    p_t.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    p_t.add_argument("--minibatch", type=int, default=256)
    p_t.set_defaults(_analyze_runner=_run_transplant)

    p_a = ps.add_parser("attention-swap", help="Swap attention mass between operand keys")
    _add_job_args(p_a)
    p_a.add_argument("--layers", type=int, nargs="+", default=[0, 1])
    p_a.add_argument("--head-mode", type=str, default="all", choices=["all", "each"])
    p_a.set_defaults(_analyze_runner=_run_attention_swap)

    p_st = ps.add_parser("steering", help="Multi-op residual steering")
    _add_job_args(p_st)
    p_st.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    p_st.add_argument("--delta", type=int, default=1)
    p_st.add_argument("--alpha", type=float, default=1.0)
    p_st.set_defaults(_analyze_runner=_run_steering)

    p_tok = ps.add_parser("task-token-edit", help="Edit TASK token on packed eval contexts")
    _add_job_args(p_tok)
    p_tok.set_defaults(_analyze_runner=_run_task_token_edit)

    p_u = ps.add_parser("unembed-fourier", help="Unembedding Fourier vs digit-emb / query")
    _add_job_args(p_u)
    p_u.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    p_u.add_argument("--ablation-ks", type=int, nargs="+", default=[1, 2, 3])
    p_u.set_defaults(_analyze_runner=_run_unembed_fourier)


def _run_causal_detail(args: argparse.Namespace) -> None:
    import json

    import torch

    from go4cl.analysis.pipelines.causal_detail import export_from_report, run_live

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        if not args.report:
            raise SystemExit("--export-only needs --report")
        export_from_report(json.loads(Path(args.report).read_text()), out)
        return
    run_live(
        job_dir=Path(args.job_dir),
        out=out,
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        ablation_ks=list(args.ablation_ks),
        max_batches=args.max_batches,
        report_path=Path(args.report) if args.report else None,
    )


def _run_transplant(args: argparse.Namespace) -> None:
    import torch

    from go4cl.analysis.pipelines.transplant import run

    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        max_batches=args.max_batches,
        minibatch=int(args.minibatch),
    )


def _run_attention_swap(args: argparse.Namespace) -> None:
    import torch

    from go4cl.analysis.pipelines.attention_swap import run

    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        max_batches=args.max_batches,
        head_mode=args.head_mode,
    )


def _run_steering(args: argparse.Namespace) -> None:
    import torch

    from go4cl.analysis.pipelines.steering_multi import run

    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        delta=int(args.delta),
        alpha=float(args.alpha),
        max_batches=args.max_batches,
    )


def _run_task_token_edit(args: argparse.Namespace) -> None:
    import torch

    from go4cl.analysis.pipelines.task_token_edit import run

    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        max_batches=args.max_batches,
    )


def _run_unembed_fourier(args: argparse.Namespace) -> None:
    import torch

    from go4cl.analysis.pipelines.unembed_fourier import run

    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        ablation_ks=list(args.ablation_ks),
        max_batches=args.max_batches,
    )


def main(argv: list[str] | None = None) -> None:
    from go4cl.runtime_paths import configure_scratch_dirs

    configure_scratch_dirs()

    parser = argparse.ArgumentParser(
        prog="go4cl",
        description="Transformer continual learning on modular-addition tasks.",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    _add_phase1(sub)
    _add_phase2(sub)
    _add_phase3(sub)
    _add_analyze(sub)

    p_gen = sub.add_parser("generate-data", help="Generate fixed A/B datasets + manifest")
    p_gen.add_argument("--out", type=str, required=True)
    p_gen.add_argument("--task-seed", type=int, default=0)
    p_gen.add_argument("--data-seed", type=int, default=0)
    p_gen.add_argument("--rho-slot", type=float, default=1.0)
    p_gen.add_argument("--rho-operand", type=float, default=1.0)
    p_gen.add_argument("--rho-mod", type=float, default=1.0)
    p_gen.add_argument("--n-aliases", type=int, default=MODEL.n_aliases)
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
    p_train.add_argument("--lr", type=float, default=MODEL.lr)
    p_train.add_argument("--weight-decay", type=float, default=1.0)
    p_train.add_argument("--model-seed", type=int, default=0)
    p_train.add_argument("--d-model", type=int, default=MODEL.d_model)
    p_train.add_argument("--n-layers", type=int, default=MODEL.n_layers)
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
    elif args.cmd == "analyze":
        args._analyze_runner(args)
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
