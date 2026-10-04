#!/usr/bin/env python3
"""Multi-op residual steering: directed edit of intermediate states.

Per operation (and cross-op for same-modulus pairs):
  - estimate class means on a reference split (train or analysis builder)
  - at resid_post L0/L1/L2, add α·(μ_{y+δ} − μ_y) and continue
  - report steered→target acc vs shuffled-mean control

Usage:
  uv run python scripts/phase1/multi_op_steering.py \\
    --job-dir runs/phase1/multi_op/.../runs/<job> \\
    --out runs/phase1/mechanisms/<stamp>/<tag>/steering \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch


from go4cl.analysis.cache import (
    collect_batches,
    operand_residues,
    to_jsonable,
)
from go4cl.analysis.causal import estimate_class_means, steer_at_layer
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.analysis.context import (
    filter_by_operation,
    load_analysis_context,
    op_report_key,
    operations_from_manifest,
)
from go4cl.analysis.reporting import write_csv_rows, write_json_report, write_markdown


def _op_dataset(op, full, *, split: str, analysis_builder):
    if len(full) == 0:
        examples = analysis_builder(split=split, target_latent_ids=[op.latent_id])
        return ModularAdditionDataset.from_examples(examples, task_id=0)
    return filter_by_operation(full, latent_id=op.latent_id, slot=op.slot)


def run(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str,
    device: torch.device,
    layers: list[int],
    delta: int,
    alpha: float,
    max_batches: int | None,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ctx = load_analysis_context(
        job_dir,
        ckpt_kind=ckpt_kind,
        device=device,
        aliases_per_pair=16,
    )
    ckpt_path = ctx.ckpt_path
    data_dir = ctx.data_dir
    manifest = ctx.manifest
    ops = ctx.operations
    model = ctx.model
    payload = ctx.payload or {}

    train_full = ModularAdditionDataset.from_disk(data_dir, "A", "train", 0)
    val_full = ModularAdditionDataset.from_disk(data_dir, "A", "val", 0)
    test_full = ModularAdditionDataset.from_disk(data_dir, "A", "test", 0)

    def analysis_builder(*, split: str, target_latent_ids: list[int]):
        from go4cl.data.context import build_analysis_dataset

        return build_analysis_dataset(
            manifest.task_pair.task_a,
            manifest.residue_splits,
            split=split,  # type: ignore[arg-type]
            context_mode="packed_id",
            analysis_seed=0,
            aliases_per_pair=16,
            contexts_per_pair=1,
            target_latent_ids=target_latent_ids,
        )

    # Reference for class means: prefer non-empty train, else val (packed has
    # empty train.npz), else denser analysis builder.
    if len(train_full) > 0:
        ref_full, ref_name = train_full, "train"
    elif len(val_full) > 0:
        ref_full, ref_name = val_full, "val"
    else:
        ref_full, ref_name = train_full, "analysis_train"

    # Build per-op caches.
    bundles: list[dict[str, Any]] = []
    for op in ops:
        key = f"lat{op.latent_id}/slot{op.slot}/p{op.modulus}"
        ds_ref = _op_dataset(
            op, ref_full, split=ref_name if ref_name != "analysis_train" else "train",
            analysis_builder=analysis_builder,
        )
        ds_te = _op_dataset(
            op, test_full, split="test", analysis_builder=analysis_builder
        )
        loader_ref = make_loader(ds_ref, batch_size=256, shuffle=False)
        loader_te = make_loader(ds_te, batch_size=256, shuffle=False)
        cache_ref = collect_batches(
            model, loader_ref, device=device, max_batches=max_batches
        )
        cache_te = collect_batches(
            model, loader_te, device=device, max_batches=max_batches
        )
        _, _, sum_ref = operand_residues(
            cache_ref.tokens, i=op.operand_i, j=op.operand_j, modulus=op.modulus
        )
        _, _, sum_te = operand_residues(
            cache_te.tokens, i=op.operand_i, j=op.operand_j, modulus=op.modulus
        )
        print(
            f"  {key}: ref_n={len(ds_ref)} test_n={len(ds_te)} ref_split={ref_name}",
            flush=True,
        )
        bundles.append(
            {
                "op": op,
                "key": key,
                "cache_ref": cache_ref,
                "cache_te": cache_te,
                "sum_ref": sum_ref,
                "sum_te": sum_te,
            }
        )

    self_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []

    # Self steering
    for b in bundles:
        op = b["op"]
        for li in layers:
            print(f"  steer self {b['key']} L{li} ...", flush=True)
            out_s = steer_at_layer(
                model,
                b["cache_te"].resid_post[li].to(device),
                b["sum_te"].to(device),
                layer_idx=li,
                modulus=op.modulus,
                delta=delta,
                alpha=alpha,
                shuffle_means_seed=0,
                reference_residuals=b["cache_ref"].resid_post[li].to(device),
                reference_labels=b["sum_ref"].to(device),
            )
            self_rows.append(
                {
                    "operation": b["key"],
                    "latent_id": op.latent_id,
                    "slot": op.slot,
                    "modulus": op.modulus,
                    "layer": li,
                    "delta": delta,
                    "alpha": alpha,
                    "direction_source": out_s.get("direction_source"),
                    "baseline_acc_true": out_s.get("baseline_acc_true"),
                    "steered_acc_target": out_s.get("steered_acc_target"),
                    "steered_acc_true": out_s.get("steered_acc_true"),
                    "shuffled_steered_acc_target": out_s.get(
                        "shuffled_steered_acc_target"
                    ),
                    "steered_minus_shuffled": out_s.get("steered_minus_shuffled"),
                    "mean_logit_target_gain": out_s.get("mean_logit_target_gain"),
                    "n": out_s.get("n"),
                }
            )

    # Cross-op: same modulus only — direction from source ref, apply on target test
    for src in bundles:
        for tgt in bundles:
            if src["op"].modulus != tgt["op"].modulus:
                continue
            if src["key"] == tgt["key"]:
                continue
            p = src["op"].modulus
            for li in layers:
                print(
                    f"  steer cross {src['key']}→{tgt['key']} L{li} ...",
                    flush=True,
                )
                means = estimate_class_means(
                    src["cache_ref"].resid_post[li][:, -1, :].to(device),
                    src["sum_ref"].to(device),
                    n_classes=p,
                )
                out_c = steer_at_layer(
                    model,
                    tgt["cache_te"].resid_post[li].to(device),
                    tgt["sum_te"].to(device),
                    layer_idx=li,
                    modulus=p,
                    delta=delta,
                    alpha=alpha,
                    shuffle_means_seed=0,
                    class_means=means,
                )
                cross_rows.append(
                    {
                        "source_operation": src["key"],
                        "target_operation": tgt["key"],
                        "modulus": p,
                        "layer": li,
                        "delta": delta,
                        "alpha": alpha,
                        "direction_source": out_c.get("direction_source"),
                        "baseline_acc_true": out_c.get("baseline_acc_true"),
                        "steered_acc_target": out_c.get("steered_acc_target"),
                        "steered_acc_true": out_c.get("steered_acc_true"),
                        "shuffled_steered_acc_target": out_c.get(
                            "shuffled_steered_acc_target"
                        ),
                        "steered_minus_shuffled": out_c.get("steered_minus_shuffled"),
                        "mean_logit_target_gain": out_c.get("mean_logit_target_gain"),
                        "n": out_c.get("n"),
                    }
                )

    fields_self = [
        "operation",
        "latent_id",
        "slot",
        "modulus",
        "layer",
        "delta",
        "alpha",
        "direction_source",
        "baseline_acc_true",
        "steered_acc_target",
        "steered_acc_true",
        "shuffled_steered_acc_target",
        "steered_minus_shuffled",
        "mean_logit_target_gain",
        "n",
    ]
    fields_cross = [
        "source_operation",
        "target_operation",
        "modulus",
        "layer",
        "delta",
        "alpha",
        "direction_source",
        "baseline_acc_true",
        "steered_acc_target",
        "steered_acc_true",
        "shuffled_steered_acc_target",
        "steered_minus_shuffled",
        "mean_logit_target_gain",
        "n",
    ]
    write_csv_rows(out / "per_op_steering.csv", self_rows, fields_self)
    write_csv_rows(out / "cross_op_steering.csv", cross_rows, fields_cross)

    # Markdown
    lines = [
        "# Multi-op residual steering\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`) step={payload.get('step')}  ",
        f"**Data:** `{data_dir.name}`  ",
        f"**Protocol:** Δ={delta}, α={alpha}, layers={layers}, "
        f"means from reference split (not eval)\n",
        "## Self-steer (steered→target / shuffled)\n",
        "| op | "
        + " | ".join(f"L{li} target" for li in layers)
        + " | "
        + " | ".join(f"L{li} −shuf" for li in layers)
        + " |",
        "|----|"
        + "|".join(["------------"] * len(layers))
        + "|"
        + "|".join(["----------"] * len(layers))
        + "|",
    ]
    by_op: dict[str, dict[int, dict[str, Any]]] = {}
    for r in self_rows:
        by_op.setdefault(r["operation"], {})[int(r["layer"])] = r
    for op_key, by_l in by_op.items():
        tcells = [
            f"{by_l[li]['steered_acc_target']:.3f}" if li in by_l else ""
            for li in layers
        ]
        scells = [
            f"{by_l[li]['steered_minus_shuffled']:+.3f}" if li in by_l else ""
            for li in layers
        ]
        lines.append(
            f"| {op_key} | " + " | ".join(tcells) + " | " + " | ".join(scells) + " |"
        )

    if cross_rows:
        lines += [
            "\n## Cross-op steer (same modulus; source means → target resid)\n",
            "| source → target | "
            + " | ".join(f"L{li} target" for li in layers)
            + " |",
            "|-----------------|"
            + "|".join(["------------"] * len(layers))
            + "|",
        ]
        by_pair: dict[str, dict[int, dict[str, Any]]] = {}
        for r in cross_rows:
            pk = f"{r['source_operation']} → {r['target_operation']}"
            by_pair.setdefault(pk, {})[int(r["layer"])] = r
        for pk, by_l in by_pair.items():
            cells = [
                f"{by_l[li]['steered_acc_target']:.3f}" if li in by_l else ""
                for li in layers
            ]
            lines.append(f"| {pk} | " + " | ".join(cells) + " |")
    else:
        lines.append(
            "\n## Cross-op steer\n\n"
            "_No same-modulus op pairs (e.g. four_diff) — skipped._\n"
        )

    lines.append(
        "\nFiles: `per_op_steering.csv`, `cross_op_steering.csv`, "
        "`steering_report.json`.\n"
    )
    (out / "STEERING.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "steering_report.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "job_dir": str(job_dir),
                    "ckpt_path": str(ckpt_path),
                    "ckpt_kind": ckpt_kind,
                    "ckpt_step": payload.get("step"),
                    "data_dir": str(data_dir),
                    "layers": layers,
                    "delta": delta,
                    "alpha": alpha,
                    "self": self_rows,
                    "cross": cross_rows,
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"done → {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ckpt-kind", type=str, default="best")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--delta", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
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


if __name__ == "__main__":
    main()
