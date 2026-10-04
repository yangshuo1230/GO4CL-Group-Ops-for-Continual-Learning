#!/usr/bin/env python3
"""Mod-p Fourier on the unembedding (output head), vs digit-emb / query residual.

Per operation:
  - spectrum of head rows 0..p-1, digit embeddings, resid_post query by sum
  - cosine overlap of energy spectra (skip DC)
  - ablate unembed top-k conjugate pairs; measure Δacc (important vs unimportant)

Usage:
  uv run python scripts/phase1/unembed_fourier.py \\
    --job-dir runs/phase1/multi_op/.../runs/<job> \\
    --out runs/phase1/mechanisms/<stamp>/<tag>/unembed_fourier \\
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
from go4cl.analysis.causal import (
    _freq_pairs_by_energy,
    fourier_ablation_on_unembedding,
    project_out_freqs_from_unembed,
    eval_accuracy,
)
from go4cl.analysis.fourier import (
    analyze_digit_embedding_fourier,
    analyze_query_resid_fourier,
    analyze_unembedding_fourier,
    energy_cosine,
)
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.analysis.context import (
    filter_by_operation,
    load_analysis_context,
    op_report_key,
    operations_from_manifest,
)
from go4cl.analysis.reporting import write_csv_rows, write_json_report, write_markdown


def _op_ds(op, full, *, split: str, analysis_builder):
    if len(full) == 0:
        examples = analysis_builder(split=split, target_latent_ids=[op.latent_id])
        return ModularAdditionDataset.from_examples(examples, task_id=0)
    return filter_by_operation(full, latent_id=op.latent_id, slot=op.slot)


def _op_key(op) -> str:
    return f"lat{op.latent_id}/slot{op.slot}/p{op.modulus}"


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def run(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str,
    device: torch.device,
    layers: list[int],
    ablation_ks: list[int],
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

    spec_rows: list[dict[str, Any]] = []
    abl_rows: list[dict[str, Any]] = []
    cross_rows: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []

    for op in ops:
        key = _op_key(op)
        p = op.modulus
        ds = _op_ds(op, test_full, split="test", analysis_builder=analysis_builder)
        loader = make_loader(ds, batch_size=256, shuffle=False)
        cache = collect_batches(model, loader, device=device, max_batches=max_batches)
        _, _, sums = operand_residues(
            cache.tokens, i=op.operand_i, j=op.operand_j, modulus=p
        )
        emb = analyze_digit_embedding_fourier(model.tok_emb, modulus=p)
        unemb = analyze_unembedding_fourier(model.head, modulus=p)
        qf = {}
        for li in layers:
            qf[f"L{li}"] = analyze_query_resid_fourier(
                cache.resid_post[li][:, -1, :],
                sums,
                modulus=p,
                source=f"resid_post_L{li}_query_by_sum",
            )
        print(
            f"  {key}: unembed top={unemb['top_freq']} "
            f"frac={unemb['top_energy_frac']:.3f} "
            f"cos(emb)={energy_cosine(emb['energy_by_freq'], unemb['energy_by_freq']):.3f}",
            flush=True,
        )
        row = {
            "operation": key,
            "latent_id": op.latent_id,
            "slot": op.slot,
            "modulus": p,
            "digit_emb_top_freq": emb["top_freq"],
            "digit_emb_top_energy_frac": emb["top_energy_frac"],
            "unembed_top_freq": unemb["top_freq"],
            "unembed_top_energy_frac": unemb["top_energy_frac"],
            "cosine_digit_emb_unembed": energy_cosine(
                emb["energy_by_freq"], unemb["energy_by_freq"]
            ),
            "n": int(cache.tokens.shape[0]),
        }
        for li in layers:
            q = qf[f"L{li}"]
            row[f"query_L{li}_top_freq"] = q["top_freq"]
            row[f"query_L{li}_top_energy_frac"] = q["top_energy_frac"]
            row[f"cosine_query_L{li}_unembed"] = energy_cosine(
                q["energy_by_freq"], unemb["energy_by_freq"]
            )
            row[f"cosine_query_L{li}_digit_emb"] = energy_cosine(
                q["energy_by_freq"], emb["energy_by_freq"]
            )
        spec_rows.append(row)

        print(f"  {key}: unembed ablation ...", flush=True)
        abl = fourier_ablation_on_unembedding(
            model,
            loader,
            modulus=p,
            device=device,
            top_k=1,
            max_batches=max_batches,
            sweep_ks=ablation_ks,
        )
        for rank, curve_key in (
            ("important", "important_curve"),
            ("unimportant", "unimportant_curve"),
        ):
            for pt in abl.get(curve_key) or []:
                abl_rows.append(
                    {
                        "operation": key,
                        "modulus": p,
                        "rank": rank,
                        "k": pt["k"],
                        "freqs": ",".join(map(str, pt.get("freqs") or [])),
                        "acc": pt["acc"],
                        "delta_acc": pt["delta_acc"],
                        "baseline_acc": abl["baseline_acc"],
                    }
                )
        bundles.append(
            {
                "op": op,
                "key": key,
                "loader": loader,
                "baseline": abl["baseline_acc"],
                "top1_freqs": list((abl.get("important_curve") or [{}])[0].get("freqs") or []),
                "unembed": unemb,
            }
        )

    # Cross-op: ablate source unembed top-1, eval every target loader
    original = model.head.weight.data.clone()
    for src in bundles:
        freqs = src["top1_freqs"]
        if not freqs:
            continue
        ablated = project_out_freqs_from_unembed(
            original, modulus=int(src["op"].modulus), freqs=freqs
        ).to(device=original.device, dtype=original.dtype)
        model.head.weight.data.copy_(ablated)
        for tgt in bundles:
            acc = float(
                eval_accuracy(
                    model, tgt["loader"], device=device, max_batches=max_batches
                )
            )
            cross_rows.append(
                {
                    "source_operation": src["key"],
                    "source_modulus": src["op"].modulus,
                    "ablated_freqs": ",".join(map(str, freqs)),
                    "target_operation": tgt["key"],
                    "target_modulus": tgt["op"].modulus,
                    "same_modulus": int(src["op"].modulus == tgt["op"].modulus),
                    "baseline_acc": tgt["baseline"],
                    "acc": acc,
                    "delta_acc": acc - tgt["baseline"],
                }
            )
        model.head.weight.data.copy_(original)
    model.head.weight.data.copy_(original)

    spec_fields = list(spec_rows[0].keys()) if spec_rows else ["operation"]
    abl_fields = [
        "operation",
        "modulus",
        "rank",
        "k",
        "freqs",
        "acc",
        "delta_acc",
        "baseline_acc",
    ]
    cross_fields = [
        "source_operation",
        "source_modulus",
        "ablated_freqs",
        "target_operation",
        "target_modulus",
        "same_modulus",
        "baseline_acc",
        "acc",
        "delta_acc",
    ]
    write_csv_rows(out / "unembed_spectra.csv", spec_rows, spec_fields)
    write_csv_rows(out / "unembed_ablation.csv", abl_rows, abl_fields)
    write_csv_rows(out / "unembed_cross_op.csv", cross_rows, cross_fields)

    lines = [
        "# Unembedding mod-p Fourier\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`) step={payload.get('step')}  ",
        f"**Data:** `{data_dir.name}`  ",
        "Head rows `0..p-1` are class directions for modulus p; unused classes kept.\n",
        "## Spectra (top freq / energy frac / cosine vs unembed)\n",
        "| op | emb f | unembed f | cos(emb,U) | "
        + " | ".join(f"qL{li} f / cos(U)" for li in layers)
        + " |",
        "|----|-------|-----------|------------|"
        + "|".join(["----------------"] * len(layers))
        + "|",
    ]
    for r in spec_rows:
        qcells = [
            f"{r.get(f'query_L{li}_top_freq')} / "
            f"{_fmt(r.get(f'cosine_query_L{li}_unembed'))}"
            for li in layers
        ]
        lines.append(
            f"| {r['operation']} | {r['digit_emb_top_freq']} | "
            f"{r['unembed_top_freq']} | {_fmt(r['cosine_digit_emb_unembed'])} | "
            + " | ".join(qcells)
            + " |"
        )

    lines += [
        "\n## Unembed ablation (important top-k Δacc)\n",
        "| op | " + " | ".join(f"k={k}" for k in ablation_ks) + " | unimp k=1 |",
        "|----|" + "|".join(["------"] * len(ablation_ks)) + "|-----------|",
    ]
    for r in spec_rows:
        cells = []
        for k in ablation_ks:
            hit = next(
                (
                    a
                    for a in abl_rows
                    if a["operation"] == r["operation"]
                    and a["rank"] == "important"
                    and int(a["k"]) == int(k)
                ),
                None,
            )
            cells.append(_fmt(hit["delta_acc"]) if hit else "")
        u1 = next(
            (
                a
                for a in abl_rows
                if a["operation"] == r["operation"]
                and a["rank"] == "unimportant"
                and int(a["k"]) == 1
            ),
            None,
        )
        lines.append(
            f"| {r['operation']} | " + " | ".join(cells) + f" | {_fmt((u1 or {}).get('delta_acc'))} |"
        )

    if cross_rows:
        keys = [b["key"] for b in bundles]
        lines += [
            "\n## Cross-op: ablate source unembed top-1 (Δacc)\n",
            "| source | " + " | ".join(keys) + " |",
            "|--------|" + "|".join(["---"] * len(keys)) + "|",
        ]
        for src in keys:
            cells = []
            for tgt in keys:
                hit = next(
                    r
                    for r in cross_rows
                    if r["source_operation"] == src and r["target_operation"] == tgt
                )
                cells.append(f"{hit['delta_acc']:+.2f}")
            lines.append(f"| {src} | " + " | ".join(cells) + " |")

    mean_cos = (
        sum(r["cosine_digit_emb_unembed"] for r in spec_rows) / len(spec_rows)
        if spec_rows
        else 0.0
    )
    k1 = [
        a
        for a in abl_rows
        if a["rank"] == "important" and int(a["k"]) == 1
    ]
    mean_k1 = sum(a["delta_acc"] for a in k1) / len(k1) if k1 else 0.0
    u1s = [
        a
        for a in abl_rows
        if a["rank"] == "unimportant" and int(a["k"]) == 1
    ]
    mean_u1 = sum(a["delta_acc"] for a in u1s) / len(u1s) if u1s else 0.0
    lines += [
        "\n## Verdict sketch\n",
        f"- Mean cosine(digit-emb, unembed) spectra (skip DC): {mean_cos:.3f}.\n",
        f"- Mean Δacc ablating unembed top-1 pair: {mean_k1:+.3f} "
        f"(unimportant k=1: {mean_u1:+.3f}).\n",
        "\nFiles: `unembed_spectra.csv`, `unembed_ablation.csv`, "
        "`unembed_cross_op.csv`, `unembed_fourier_report.json`.\n",
    ]
    (out / "UNEMBED.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "unembed_fourier_report.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "job_dir": str(job_dir),
                    "ckpt_path": str(ckpt_path),
                    "ckpt_kind": ckpt_kind,
                    "ckpt_step": payload.get("step"),
                    "data_dir": str(data_dir),
                    "layers": layers,
                    "ablation_ks": ablation_ks,
                    "spectra": spec_rows,
                    "ablation": abl_rows,
                    "cross_op": cross_rows,
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
    ap.add_argument("--ablation-ks", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        ablation_ks=list(args.ablation_ks),
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
