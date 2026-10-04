#!/usr/bin/env python3
"""Per-op detailed causal ablations for multi-op 1B/1C checkpoints.

For each operation:
  - export head / attn·MLP knockout already in a mechanisms report (if present)
  - deepen Fourier important/unimportant curves (default k=1..6)
  - cross-op matrices: ablate each L0 head / L0 zero-attn / L0 zero-MLP /
    each op's top-1 Fourier pair; measure Δacc on every op

Usage:
  uv run python scripts/phase1/causal_detail.py \\
    --job-dir runs/phase1/multi_op/.../runs/<job> \\
    --out runs/phase1/mechanisms/<stamp>/<tag>/causal_detail \\
    --report runs/phase1/mechanisms/<stamp>/<tag>/phase1_mechanisms_report.json \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

# repo root on sys.path when launched via uv run from project root
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from go4cl.analysis.cache import collect_batches  # noqa: E402
from go4cl.analysis.causal import (  # noqa: E402
    eval_accuracy,
    fourier_ablation_on_embeddings,
)
from go4cl.analysis.composition import (  # noqa: E402
    component_knockout,
    head_knockout,
)
from go4cl.analysis.cache import to_jsonable  # noqa: E402
from go4cl.analysis.causal import project_out_freqs_from_digit_emb  # noqa: E402
from go4cl.data.dataset import ModularAdditionDataset, make_loader  # noqa: E402
from go4cl.data.manifest import DataManifest  # noqa: E402
from go4cl.phases.phase1.mechanisms import (  # noqa: E402
    _op_report_key,
    _ops_from_manifest,
    filter_by_operation,
)
from go4cl.utils.checkpoint import load_checkpoint  # noqa: E402


def _resolve_data_dir(job_dir: Path) -> Path:
    cfg_path = job_dir / "config_resolved.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
        for k in ("data_dir", "data_root", "dataset_dir"):
            if k in cfg and cfg[k]:
                return Path(cfg[k])
        for nest in ("data", "task", "dataset"):
            block = cfg.get(nest) or {}
            if isinstance(block, dict):
                for k in ("data_dir", "data_root", "dir", "path"):
                    if block.get(k):
                        return Path(block[k])
    stamp = job_dir.parent.parent
    data_tag = job_dir.name.split("__")[0]
    cand = stamp / "data" / data_tag
    if cand.exists():
        return cand
    hits = list((stamp / "data").glob(data_tag + "*")) if (stamp / "data").exists() else []
    if hits:
        return hits[0]
    raise FileNotFoundError(f"cannot resolve data_dir for {job_dir}")


def _resolve_ckpt(job_dir: Path, kind: str) -> Path:
    ck = job_dir / "ckpts" / f"{kind}.pt"
    if ck.exists():
        return ck
    alt = job_dir / "ckpts" / f"a_only_{kind}.pt"
    if alt.exists():
        return alt
    raise FileNotFoundError(ck)


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def export_from_report(report: dict[str, Any], out: Path) -> None:
    """Flatten existing per-op composition / Fourier into long CSVs."""
    head_rows: list[dict[str, Any]] = []
    ko_rows: list[dict[str, Any]] = []
    fourier_rows: list[dict[str, Any]] = []
    for op in report.get("per_op") or []:
        key = _op_report_key(op)
        base = {
            "operation": key,
            "latent_id": op["latent_id"],
            "slot": op["slot"],
            "modulus": op["modulus"],
            "operand_i": op["operand_i"],
            "operand_j": op["operand_j"],
            "baseline_acc": (op.get("summary") or {}).get("baseline_acc"),
        }
        comp = op.get("composition") or {}
        hk = (comp.get("head_knockout") or {}).get("by_head") or []
        for row in hk:
            head_rows.append({**base, **row})
        by_layer = (comp.get("knockout") or {}).get("by_layer") or []
        if isinstance(by_layer, dict):
            by_layer = list(by_layer.values())
        for row in by_layer:
            ko_rows.append({**base, **row})
        abl = op.get("fourier_ablation") or {}
        for rank, curve_key in (
            ("important", "important_curve"),
            ("unimportant", "unimportant_curve"),
        ):
            for pt in abl.get(curve_key) or []:
                fourier_rows.append(
                    {
                        **base,
                        "rank": rank,
                        "k": pt.get("k"),
                        "freqs": ",".join(map(str, pt.get("freqs") or [])),
                        "acc": pt.get("acc"),
                        "delta_acc": pt.get("delta_acc"),
                        "energy_sum": pt.get("energy_sum"),
                    }
                )
    _write_csv(
        out / "per_op_head_knockout.csv",
        head_rows,
        [
            "operation",
            "latent_id",
            "slot",
            "modulus",
            "operand_i",
            "operand_j",
            "baseline_acc",
            "layer",
            "head",
            "acc",
            "delta_acc",
        ],
    )
    _write_csv(
        out / "per_op_component_knockout.csv",
        ko_rows,
        [
            "operation",
            "latent_id",
            "slot",
            "modulus",
            "operand_i",
            "operand_j",
            "baseline_acc",
            "layer",
            "zero_attn_acc",
            "zero_attn_delta",
            "zero_mlp_acc",
            "zero_mlp_delta",
        ],
    )
    _write_csv(
        out / "per_op_fourier_curve.csv",
        fourier_rows,
        [
            "operation",
            "latent_id",
            "slot",
            "modulus",
            "operand_i",
            "operand_j",
            "baseline_acc",
            "rank",
            "k",
            "freqs",
            "acc",
            "delta_acc",
            "energy_sum",
        ],
    )


def run_live(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str,
    device: torch.device,
    ablation_ks: list[int],
    max_batches: int | None,
    report_path: Path | None,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = _resolve_ckpt(job_dir, ckpt_kind)
    data_dir = _resolve_data_dir(job_dir)
    manifest = DataManifest.load(data_dir / "manifest.json")
    ops = _ops_from_manifest(manifest)

    model, payload = load_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()

    test_full = ModularAdditionDataset.from_disk(data_dir, "A", "test", 0)
    train_full = ModularAdditionDataset.from_disk(data_dir, "A", "train", 0)

    def _analysis_builder(*, split: str, target_latent_ids: list[int]):
        from go4cl.data.context import build_analysis_dataset

        return build_analysis_dataset(
            manifest.task_pair.task_a,
            manifest.residue_splits,
            split=split,  # type: ignore[arg-type]
            context_mode="packed_id",
            analysis_seed=0,
            aliases_per_pair=4,
            contexts_per_pair=1,
            target_latent_ids=target_latent_ids,
        )

    # per-op caches + loaders
    bundles: list[dict[str, Any]] = []
    for op in ops:
        if len(test_full) == 0:
            examples = _analysis_builder(
                split="test", target_latent_ids=[op.latent_id]
            )
            ds = ModularAdditionDataset.from_examples(examples, task_id=0)
        else:
            ds = filter_by_operation(
                test_full, latent_id=op.latent_id, slot=op.slot
            )
        loader = make_loader(ds, batch_size=256, shuffle=False)
        cache = collect_batches(
            model, loader, device=device, max_batches=max_batches
        )
        base = float(eval_accuracy(model, loader, device=device, max_batches=max_batches))
        key = f"lat{op.latent_id}/slot{op.slot}/p{op.modulus}"
        print(f"  cache {key} n={len(ds)} base={base:.4f}", flush=True)

        print(f"  head+component knockout {key} ...", flush=True)
        hk = head_knockout(
            model,
            resid_pre=cache.resid_pre,
            resid_post=cache.resid_post,
            labels=cache.labels,
            device=device,
            baseline_acc=base,
        )
        ko = component_knockout(
            model,
            resid_pre=cache.resid_pre,
            resid_mid=cache.resid_mid,
            resid_post=cache.resid_post,
            labels=cache.labels,
            device=device,
            baseline_acc=base,
        )
        print(f"  fourier sweep {key} ks={ablation_ks} ...", flush=True)
        abl = fourier_ablation_on_embeddings(
            model,
            loader,
            modulus=op.modulus,
            device=device,
            top_k=1,
            max_batches=max_batches,
            sweep_ks=ablation_ks,
        )
        bundles.append(
            {
                "op": op,
                "key": key,
                "loader": loader,
                "cache": cache,
                "baseline": base,
                "head_knockout": hk,
                "component_knockout": ko,
                "fourier": abl,
            }
        )

    # export per-op long tables
    head_rows = []
    ko_rows = []
    fourier_rows = []
    for b in bundles:
        op = b["op"]
        base_row = {
            "operation": b["key"],
            "latent_id": op.latent_id,
            "slot": op.slot,
            "modulus": op.modulus,
            "operand_i": op.operand_i,
            "operand_j": op.operand_j,
            "baseline_acc": b["baseline"],
        }
        for row in b["head_knockout"]["by_head"]:
            head_rows.append({**base_row, **row})
        for row in b["component_knockout"]["by_layer"]:
            ko_rows.append({**base_row, **row})
        for rank, curve_key in (
            ("important", "important_curve"),
            ("unimportant", "unimportant_curve"),
        ):
            for pt in b["fourier"].get(curve_key) or []:
                fourier_rows.append(
                    {
                        **base_row,
                        "rank": rank,
                        "k": pt.get("k"),
                        "freqs": ",".join(map(str, pt.get("freqs") or [])),
                        "acc": pt.get("acc"),
                        "delta_acc": pt.get("delta_acc"),
                        "energy_sum": pt.get("energy_sum"),
                    }
                )
    _write_csv(
        out / "per_op_head_knockout.csv",
        head_rows,
        [
            "operation",
            "latent_id",
            "slot",
            "modulus",
            "operand_i",
            "operand_j",
            "baseline_acc",
            "layer",
            "head",
            "acc",
            "delta_acc",
        ],
    )
    _write_csv(
        out / "per_op_component_knockout.csv",
        ko_rows,
        [
            "operation",
            "latent_id",
            "slot",
            "modulus",
            "operand_i",
            "operand_j",
            "baseline_acc",
            "layer",
            "zero_attn_acc",
            "zero_attn_delta",
            "zero_mlp_acc",
            "zero_mlp_delta",
        ],
    )
    _write_csv(
        out / "per_op_fourier_curve.csv",
        fourier_rows,
        [
            "operation",
            "latent_id",
            "slot",
            "modulus",
            "operand_i",
            "operand_j",
            "baseline_acc",
            "rank",
            "k",
            "freqs",
            "acc",
            "delta_acc",
            "energy_sum",
        ],
    )

    # --- cross-op matrices ---
    print("  cross-op head matrix (all layers×heads) ...", flush=True)
    cross_head: list[dict[str, Any]] = []
    n_layers = len(model.blocks)
    n_heads = model.cfg.n_heads
    for li in range(n_layers):
        for h in range(n_heads):
            for tgt in bundles:
                cache = tgt["cache"]
                block = model.blocks[li]
                pre = cache.resid_pre[li].to(device)
                labels = cache.labels.to(device)
                attn_out, _, _ = block.attn.forward_detailed(
                    block.ln1(pre), ablate_heads=[h]
                )
                mid = pre + attn_out
                post = mid + block.mlp(block.ln2(mid))
                # continue_from_layer
                from go4cl.analysis.composition import _continue_after_edited_post
                from go4cl.analysis.cache import accuracy_from_logits

                logits = _continue_after_edited_post(model, post, layer_idx=li)
                acc = float(accuracy_from_logits(logits, labels))
                cross_head.append(
                    {
                        "ablate_layer": li,
                        "ablate_head": h,
                        "target_operation": tgt["key"],
                        "target_latent_id": tgt["op"].latent_id,
                        "baseline_acc": tgt["baseline"],
                        "acc": acc,
                        "delta_acc": acc - tgt["baseline"],
                    }
                )
    _write_csv(
        out / "cross_op_head_knockout.csv",
        cross_head,
        [
            "ablate_layer",
            "ablate_head",
            "target_operation",
            "target_latent_id",
            "baseline_acc",
            "acc",
            "delta_acc",
        ],
    )

    print("  cross-op L0 component (zero_attn / zero_mlp) ...", flush=True)
    cross_comp: list[dict[str, Any]] = []
    for li in range(n_layers):
        for kind in ("zero_attn", "zero_mlp"):
            for tgt in bundles:
                cache = tgt["cache"]
                block = model.blocks[li]
                pre = cache.resid_pre[li].to(device)
                mid = cache.resid_mid[li].to(device)
                labels = cache.labels.to(device)
                from go4cl.analysis.composition import _continue_after_edited_post
                from go4cl.analysis.cache import accuracy_from_logits

                if kind == "zero_mlp":
                    logits = _continue_after_edited_post(model, mid, layer_idx=li)
                else:
                    post_no_attn = pre + block.mlp(block.ln2(pre))
                    logits = _continue_after_edited_post(
                        model, post_no_attn, layer_idx=li
                    )
                acc = float(accuracy_from_logits(logits, labels))
                cross_comp.append(
                    {
                        "ablate_layer": li,
                        "kind": kind,
                        "target_operation": tgt["key"],
                        "target_latent_id": tgt["op"].latent_id,
                        "baseline_acc": tgt["baseline"],
                        "acc": acc,
                        "delta_acc": acc - tgt["baseline"],
                    }
                )
    _write_csv(
        out / "cross_op_component_knockout.csv",
        cross_comp,
        [
            "ablate_layer",
            "kind",
            "target_operation",
            "target_latent_id",
            "baseline_acc",
            "acc",
            "delta_acc",
        ],
    )

    print("  cross-op Fourier top-1 (per source modulus pair) ...", flush=True)
    original = model.tok_emb.weight.data.clone()
    cross_f: list[dict[str, Any]] = []
    for src in bundles:
        freqs = list((src["fourier"].get("important_curve") or [{}])[0].get("freqs") or [])
        if not freqs:
            continue
        ablated = project_out_freqs_from_digit_emb(
            original, modulus=int(src["op"].modulus), freqs=freqs
        ).to(device=original.device, dtype=original.dtype)
        model.tok_emb.weight.data.copy_(ablated)
        for tgt in bundles:
            acc = float(
                eval_accuracy(
                    model, tgt["loader"], device=device, max_batches=max_batches
                )
            )
            cross_f.append(
                {
                    "source_operation": src["key"],
                    "source_modulus": src["op"].modulus,
                    "ablated_freqs": ",".join(map(str, freqs)),
                    "target_operation": tgt["key"],
                    "target_modulus": tgt["op"].modulus,
                    "baseline_acc": tgt["baseline"],
                    "acc": acc,
                    "delta_acc": acc - tgt["baseline"],
                }
            )
        model.tok_emb.weight.data.copy_(original)
    model.tok_emb.weight.data.copy_(original)
    _write_csv(
        out / "cross_op_fourier_top1.csv",
        cross_f,
        [
            "source_operation",
            "source_modulus",
            "ablated_freqs",
            "target_operation",
            "target_modulus",
            "baseline_acc",
            "acc",
            "delta_acc",
        ],
    )

    # markdown summary
    lines = [
        "# Per-op detailed causal ablation\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`) step={payload.get('step')}  ",
        f"**Data:** `{data_dir.name}`\n",
        "## Per-op component (Δacc)\n",
        "| op | L0 Δattn | L0 Δmlp | L1 Δattn | L1 Δmlp | L2 Δattn | L2 Δmlp | worst head |",
        "|----|----------|---------|----------|---------|----------|---------|------------|",
    ]
    for b in bundles:
        by = {row["layer"]: row for row in b["component_knockout"]["by_layer"]}
        wh = b["head_knockout"].get("worst_head") or {}
        wh_s = (
            f"L{wh.get('layer')}H{wh.get('head')}({wh.get('delta_acc'):+.2f})"
            if wh.get("layer") is not None
            else ""
        )

        def d(li, key):
            row = by.get(li) or {}
            v = row.get(key)
            return f"{v:+.2f}" if v is not None else ""

        lines.append(
            f"| {b['key']} | {d(0,'zero_attn_delta')} | {d(0,'zero_mlp_delta')} | "
            f"{d(1,'zero_attn_delta')} | {d(1,'zero_mlp_delta')} | "
            f"{d(2,'zero_attn_delta')} | {d(2,'zero_mlp_delta')} | {wh_s} |"
        )

    lines += [
        "\n## Fourier important curve (acc after ablating top-k pairs)\n",
        "| op | " + " | ".join(f"k={k}" for k in ablation_ks) + " |",
        "|----|" + "|".join(["------"] * len(ablation_ks)) + "|",
    ]
    for b in bundles:
        curve = {
            pt["k"]: pt["acc"] for pt in (b["fourier"].get("important_curve") or [])
        }
        cells = [f"{curve[k]:.3f}" if k in curve else "" for k in ablation_ks]
        lines.append(f"| {b['key']} | " + " | ".join(cells) + " |")

    # L0 head cross selectivity: for each head, mean self vs others if we had source
    # Summarize: for each head, max-min delta across targets (shared vs selective)
    lines += ["\n## Cross-op L0 head Δacc (columns = target ops)\n"]
    keys = [b["key"] for b in bundles]
    lines.append("| ablate | " + " | ".join(keys) + " |")
    lines.append("|--------|" + "|".join(["---"] * len(keys)) + "|")
    for h in range(n_heads):
        cells = []
        for key in keys:
            hit = next(
                r
                for r in cross_head
                if r["ablate_layer"] == 0
                and r["ablate_head"] == h
                and r["target_operation"] == key
            )
            cells.append(f"{hit['delta_acc']:+.2f}")
        lines.append(f"| L0H{h} | " + " | ".join(cells) + " |")

    lines += [
        "\nFiles: `per_op_*.csv`, `cross_op_*.csv`, `causal_detail_report.json`.\n"
    ]
    (out / "CAUSAL.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    report = {
        "job_dir": str(job_dir),
        "ckpt_path": str(ckpt_path),
        "ckpt_kind": ckpt_kind,
        "ckpt_step": payload.get("step"),
        "data_dir": str(data_dir),
        "ablation_ks": ablation_ks,
        "ops": [
            {
                "key": b["key"],
                "latent_id": b["op"].latent_id,
                "slot": b["op"].slot,
                "modulus": b["op"].modulus,
                "baseline_acc": b["baseline"],
                "worst_head": b["head_knockout"].get("worst_head"),
                "component_knockout": b["component_knockout"],
                "fourier_important": b["fourier"].get("important_curve"),
                "fourier_unimportant": b["fourier"].get("unimportant_curve"),
            }
            for b in bundles
        ],
    }
    (out / "causal_detail_report.json").write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    # optional: merge export from prior mechanisms report if provided
    if report_path and report_path.exists():
        export_from_report(json.loads(report_path.read_text()), out / "from_1c_report")
    print(f"done → {out}", flush=True)
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ckpt-kind", type=str, default="best")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--ablation-ks", type=int, nargs="+", default=[1, 2, 3, 4, 5, 6])
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument(
        "--report",
        type=str,
        default=None,
        help="Optional existing phase1_mechanisms_report.json to also flatten",
    )
    ap.add_argument(
        "--export-only",
        action="store_true",
        help="Only flatten --report; skip live ablations",
    )
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.export_only:
        if not args.report:
            raise SystemExit("--export-only needs --report")
        export_from_report(json.loads(Path(args.report).read_text()), out)
        print(f"exported → {out}")
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


if __name__ == "__main__":
    main()
