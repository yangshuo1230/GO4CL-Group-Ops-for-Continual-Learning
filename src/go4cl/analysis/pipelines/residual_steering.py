"""Phase 1A residual-steering grid: L0/L1/L2_post × several δ × conditions.

Class means are estimated on the reference (train) split only. Test is
evaluation-only. Does not change training or existing checkpoint defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from go4cl.analysis.cache import collect_batches, operand_residues
from go4cl.analysis.context import load_analysis_context
from go4cl.analysis.interventions.steering import (
    DEFAULT_SHUFFLE_SEED,
    DEFAULT_STEER_ALPHA,
    DEFAULT_STEER_DELTAS,
    SITE_LAYER,
    STEERING_CONDITIONS,
    estimate_class_means,
    predict_steering_condition,
    sample_records,
    summarize_records,
    transition_rows,
)
from go4cl.analysis.reporting import write_csv_rows, write_json_report

DEFAULT_P31_JOB = Path(
    "runs/phase1/scan_moduli/20260930_223737/runs/"
    "single_p31_tr0.8_ts0_ds0_a16_p1a__wd0.5_steps20000__ms0"
)
DEFAULT_OUT = Path("runs/phase1/mech_single/p31_summary/residual_steering")

SUMMARY_FIELDS = (
    "modulus",
    "checkpoint",
    "site",
    "delta",
    "alpha",
    "condition",
    "n",
    "original_acc",
    "target_acc",
)
PREDICTION_FIELDS = (
    "site",
    "delta",
    "condition",
    "source_residue",
    "predicted_residue",
    "count",
    "probability",
)
SAMPLE_FIELDS = (
    "site",
    "delta",
    "alpha",
    "condition",
    "original_label",
    "target_label",
    "predicted_label",
    "original_correct",
    "target_correct",
)


def run(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str = "final",
    device: torch.device | str | None = None,
    sites: list[str] | None = None,
    deltas: list[int] | None = None,
    alpha: float = DEFAULT_STEER_ALPHA,
    shuffle_seed: int = DEFAULT_SHUFFLE_SEED,
    include_global: bool = True,
    reference_split: str = "train",
    eval_split: str = "test",
    max_batches: int | None = None,
) -> dict[str, Any]:
    sites = list(sites or list(SITE_LAYER.keys()))
    deltas = [int(d) for d in (deltas or list(DEFAULT_STEER_DELTAS))]
    conditions = list(STEERING_CONDITIONS) if include_global else [
        c for c in STEERING_CONDITIONS if c != "global_structured"
    ]
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)

    ctx = load_analysis_context(job_dir, ckpt_kind=ckpt_kind, device=device)
    model = ctx.model
    model.eval()
    op = ctx.operations[0]
    p = int(op.modulus)
    i, j = op.operand_i, op.operand_j

    ref_loader = ctx.operation_loader(op, reference_split, batch_size=256)
    eval_loader = ctx.operation_loader(op, eval_split, batch_size=256)
    ref_cache = collect_batches(
        model, ref_loader, device=ctx.device, max_batches=max_batches
    )
    eval_cache = collect_batches(
        model, eval_loader, device=ctx.device, max_batches=max_batches
    )

    _, _, sum_ref = operand_residues(ref_cache.tokens, i=i, j=j, modulus=p)
    _, _, sum_eval = operand_residues(eval_cache.tokens, i=i, j=j, modulus=p)
    labels_match_sum = bool(
        torch.equal(sum_eval.cpu(), eval_cache.labels.cpu() % p)
    )

    records: list[dict[str, Any]] = []
    for site in sites:
        if site not in SITE_LAYER:
            raise ValueError(f"unknown site {site}; expected {list(SITE_LAYER)}")
        li = SITE_LAYER[site]
        if li >= len(model.blocks):
            raise ValueError(f"{site} needs layer {li}; model has {len(model.blocks)}")
        means = estimate_class_means(
            ref_cache.resid_post[li][:, -1, :],
            sum_ref,
            n_classes=p,
        )
        resid = eval_cache.resid_post[li].to(ctx.device)
        y = sum_eval.to(ctx.device)
        for delta in deltas:
            for cond in conditions:
                pred = predict_steering_condition(
                    model,
                    resid,
                    y,
                    class_means=means.to(ctx.device),
                    layer_idx=li,
                    modulus=p,
                    delta=delta,
                    alpha=alpha,
                    condition=cond,
                    shuffle_seed=shuffle_seed,
                )
                records.extend(
                    sample_records(
                        site=site,
                        delta=delta,
                        alpha=alpha,
                        condition=cond,
                        original=pred["original"],
                        target=pred["target"],
                        predicted=pred["predicted"],
                    )
                )

    ckpt_name = str(ctx.ckpt_path)
    summary = summarize_records(records, modulus=p, checkpoint=ckpt_name)
    transitions = transition_rows(records, modulus=p)

    summary_path = write_csv_rows(out / "steering_summary.csv", summary, SUMMARY_FIELDS)
    pred_path = write_csv_rows(
        out / "steering_predictions.csv", transitions, PREDICTION_FIELDS
    )
    samples_path = write_csv_rows(out / "steering_samples.csv", records, SAMPLE_FIELDS)

    metrics = {
        f"{row['site']}|d{row['delta']}|{row['condition']}": {
            "original_acc": row["original_acc"],
            "target_acc": row["target_acc"],
            "n": row["n"],
        }
        for row in summary
    }
    report = {
        "checkpoint": ckpt_name,
        "ckpt_kind": ckpt_kind,
        "job_dir": str(ctx.job_dir),
        "data_dir": str(ctx.data_dir),
        "reference_split": reference_split,
        "test_split": eval_split,
        "class_means": (
            "per-residue mean of query resid_post on the reference split; "
            "test split is never used to build μ"
        ),
        "steering_formula": "h + alpha * (mu[target] - mu[s]) at query only",
        "global_formula": "h + alpha * mean_s(mu[(s+delta)%p] - mu[s])",
        "shuffled": (
            "permute class-mean rows with torch.randperm; "
            f"seed={shuffle_seed}"
        ),
        "query_only": True,
        "sites": sites,
        "deltas": deltas,
        "alpha": float(alpha),
        "shuffle_seed": int(shuffle_seed),
        "conditions": conditions,
        "include_global_structured": bool(include_global),
        "modulus": p,
        "operand_i": i,
        "operand_j": j,
        "n_reference": int(sum_ref.shape[0]),
        "n_eval": int(sum_eval.shape[0]),
        "eval_labels_match_operand_sum": labels_match_sum,
        "metrics": metrics,
        "outputs": {
            "steering_summary.csv": str(summary_path),
            "steering_predictions.csv": str(pred_path),
            "steering_samples.csv": str(samples_path),
        },
    }
    report_path = write_json_report(out / "steering_report.json", report)
    report["outputs"]["steering_report.json"] = str(report_path)
    write_json_report(out / "steering_report.json", report)
    return report


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--job-dir", type=Path, default=DEFAULT_P31_JOB)
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    p.add_argument("--ckpt", type=str, default="final")
    p.add_argument("--device", type=str, default=None)
    p.add_argument("--sites", nargs="+", default=list(SITE_LAYER.keys()))
    p.add_argument("--deltas", type=int, nargs="+", default=list(DEFAULT_STEER_DELTAS))
    p.add_argument("--alpha", type=float, default=DEFAULT_STEER_ALPHA)
    p.add_argument("--shuffle-seed", type=int, default=DEFAULT_SHUFFLE_SEED)
    p.add_argument("--no-global", action="store_true")
    p.add_argument("--max-batches", type=int, default=None)
    p.add_argument(
        "--plot",
        action="store_true",
        help="Also write fig6 / fig6b next to the p31_summary figures folder",
    )
    p.add_argument(
        "--figures",
        type=Path,
        default=Path("runs/phase1/mech_single/p31_summary/figures"),
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_argparser().parse_args(argv)
    device = torch.device(args.device) if args.device else None
    run(
        job_dir=args.job_dir,
        out=args.out,
        ckpt_kind=args.ckpt,
        device=device,
        sites=list(args.sites),
        deltas=list(args.deltas),
        alpha=float(args.alpha),
        shuffle_seed=int(args.shuffle_seed),
        include_global=not args.no_global,
        max_batches=args.max_batches,
    )
    if args.plot:
        from go4cl.analysis.pipelines.plot_residual_steering import plot_all

        plot_all(data_dir=args.out, figures_dir=args.figures)


if __name__ == "__main__":
    main()
