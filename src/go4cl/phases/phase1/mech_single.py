"""Phase 1A-mech: single-modulus / single-op mechanism analysis.

Analyzes scan-moduli checkpoints with:
  - mod-p Fourier on digit embeddings / query residuals
  - linear probes for xi%p, xj%p, (xi+xj)%p (+ random-init control)
  - attention mass from query onto operand positions
  - Fourier-frequency ablation on digit embeddings
  - query-residual class-mean steering to change the predicted sum
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch

from go4cl.analysis.attention import query_attention_to_operands
from go4cl.analysis.cache import (
    accuracy_from_logits,
    collect_batches,
    operand_residues,
    to_jsonable,
)
from go4cl.analysis.causal import (
    fourier_ablation_on_embeddings,
    steer_at_layer,
)
from go4cl.analysis.composition import run_composition_analysis
from go4cl.analysis.discover import discover_targets
from go4cl.analysis.fourier import (
    analyze_digit_embedding_fourier,
    analyze_query_resid_fourier,
    analyze_unembedding_fourier,
    energy_cosine,
)
from go4cl.analysis.probes import run_layer_probes_with_random_control
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.phases.common import stamp
from go4cl.utils.checkpoint import load_checkpoint

DEFAULT_CKPT_ROOT = "runs/phase1/scan_moduli/20260930_223737"
# Probe / steer early layers (not final pre-head residual — that is near-tautological).
DEFAULT_MECH_LAYERS: tuple[int, ...] = (0, 1)


def _device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _strip_private(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {
            k: _strip_private(v)
            for k, v in obj.items()
            if not str(k).startswith("_")
        }
    if isinstance(obj, list):
        return [_strip_private(v) for v in obj]
    return obj


def _query_layer_resids(cache, layers: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    return {
        li: cache.resid_post[li][:, -1, :].to(device) for li in layers
    }


def analyze_one(
    *,
    target,
    device: torch.device,
    max_batches: int | None,
    probe_steps: int,
    steer_delta: int = 1,
    steer_alpha: float = 1.0,
    layers: list[int] | None = None,
    ablation_ks: list[int] | None = None,
    skip_composition: bool = False,
) -> dict[str, Any]:
    model, payload = load_checkpoint(target.ckpt_path, map_location=device)
    model.to(device)
    model.eval()
    n_layers = len(model.blocks)
    layers = list(layers if layers is not None else DEFAULT_MECH_LAYERS)
    for li in layers:
        if not (0 <= li < n_layers):
            raise ValueError(f"layer {li} invalid for n_layers={n_layers}")

    train_ds = ModularAdditionDataset.from_disk(target.data_dir, "A", "train", 0)
    val_ds = ModularAdditionDataset.from_disk(target.data_dir, "A", "val", 0)
    test_ds = ModularAdditionDataset.from_disk(target.data_dir, "A", "test", 0)
    train_loader = make_loader(train_ds, batch_size=256, shuffle=False)
    val_loader = make_loader(val_ds, batch_size=256, shuffle=False)
    test_loader = make_loader(test_ds, batch_size=256, shuffle=False)

    train_cache = collect_batches(
        model, train_loader, device=device, max_batches=max_batches
    )
    val_cache = collect_batches(
        model, val_loader, device=device, max_batches=max_batches
    )
    test_cache = collect_batches(
        model, test_loader, device=device, max_batches=max_batches
    )

    baseline_test = accuracy_from_logits(test_cache.logits, test_cache.labels)
    baseline_val = accuracy_from_logits(val_cache.logits, val_cache.labels)

    p = target.modulus
    i, j = target.operand_i, target.operand_j
    _, _, sum_test = operand_residues(test_cache.tokens, i=i, j=j, modulus=p)
    _, _, sum_train = operand_residues(train_cache.tokens, i=i, j=j, modulus=p)

    emb_fourier = analyze_digit_embedding_fourier(model.tok_emb, modulus=p)
    unembed_fourier = analyze_unembedding_fourier(model.head, modulus=p)
    # Layer-wise Fourier on early resid (by sum class)
    fourier_by_layer = {}
    for li in layers:
        fourier_by_layer[f"L{li}"] = analyze_query_resid_fourier(
            test_cache.resid_post[li][:, -1, :],
            sum_test,
            modulus=p,
            source=f"resid_post_L{li}_query_by_sum",
        )
    cosine_emb_unembed = energy_cosine(
        emb_fourier["energy_by_freq"], unembed_fourier["energy_by_freq"]
    )

    def _collect_random_layers(random_model):
        random_model.to(device)
        tr = collect_batches(
            random_model, train_loader, device=device, max_batches=max_batches
        )
        va = collect_batches(
            random_model, val_loader, device=device, max_batches=max_batches
        )
        te = collect_batches(
            random_model, test_loader, device=device, max_batches=max_batches
        )
        return (
            _query_layer_resids(tr, layers, device),
            {
                "val": _query_layer_resids(va, layers, device),
                "test": _query_layer_resids(te, layers, device),
            },
        )

    probes_by_layer = run_layer_probes_with_random_control(
        trained_model=model,
        train_layer_resids=_query_layer_resids(train_cache, layers, device),
        eval_layer_resids={
            "val": _query_layer_resids(val_cache, layers, device),
            "test": _query_layer_resids(test_cache, layers, device),
        },
        train_tokens=train_cache.tokens,
        eval_tokens={"val": val_cache.tokens, "test": test_cache.tokens},
        operand_i=i,
        operand_j=j,
        modulus=p,
        layers=layers,
        steps=probe_steps,
        random_seed=0,
        collect_random_layers_fn=_collect_random_layers,
    )

    attn = query_attention_to_operands(
        test_cache.attn, operand_i=i, operand_j=j
    )

    ablation = fourier_ablation_on_embeddings(
        model,
        test_loader,
        modulus=p,
        device=device,
        top_k=2,
        max_batches=max_batches,
        sweep_ks=ablation_ks,
    )

    # Mid-layer steering: edit resid after layer ℓ, continue remaining blocks.
    # Need full-sequence resid on device for continue_from_layer.
    steering_by_layer: dict[str, Any] = {}
    # For steering we may need more than max_batches-capped cache if large;
    # reuse test_cache resid_post (already collected).
    for li in layers:
        resid = test_cache.resid_post[li].to(device)
        ref = train_cache.resid_post[li].to(device)
        steering_by_layer[f"L{li}"] = steer_at_layer(
            model,
            resid,
            sum_test.to(device),
            layer_idx=li,
            modulus=p,
            delta=steer_delta,
            alpha=steer_alpha,
            shuffle_means_seed=0,
            reference_residuals=ref,
            reference_labels=sum_train.to(device),
        )

    composition: dict[str, Any] | None = None
    if not skip_composition:
        composition = run_composition_analysis(
            model,
            train_cache=train_cache,
            test_cache=test_cache,
            val_cache=val_cache,
            operand_i=i,
            operand_j=j,
            modulus=p,
            device=device,
            probe_steps=probe_steps,
            top_k_pairs=3,
            baseline_acc=baseline_test,
        )

    summary: dict[str, Any] = {
        "baseline_acc": baseline_test,
        "mech_layers": layers,
        "attn_mass_operands": attn.get("mean_operand_mass"),
        "ablation_delta_acc": ablation.get("delta_acc"),
        "ablated_freqs": ablation.get("ablated_freqs"),
        "ablation_important_curve": ablation.get("important_curve"),
        "ablation_unimportant_curve": ablation.get("unimportant_curve"),
        "compose_layer_guess": (
            None if composition is None else composition.get("compose_layer_guess")
        ),
        "ladder_sum_jump_layer": (
            None
            if composition is None
            else composition.get("ladder_sum_jump_layer")
        ),
        "harmonic_jump_layer": (
            None
            if composition is None
            else composition.get("harmonic_jump_layer")
        ),
        "worst_attn_head": (
            None
            if composition is None
            else composition.get("worst_attn_head")
        ),
    }
    for li in layers:
        key = f"L{li}"
        pr = probes_by_layer[key]
        st = steering_by_layer[key]
        summary[f"probe_sum_acc_{key}"] = pr.get("probe_sum_acc")
        summary[f"probe_sum_acc_random_{key}"] = pr.get("probe_sum_acc_random")
        summary[f"probe_sum_acc_delta_{key}"] = pr.get("probe_sum_acc_delta")
        summary[f"steered_acc_target_{key}"] = st.get("steered_acc_target")
        summary[f"shuffled_steered_acc_target_{key}"] = st.get(
            "shuffled_steered_acc_target"
        )
        summary[f"steered_minus_shuffled_{key}"] = st.get("steered_minus_shuffled")
        summary[f"top_fourier_freq_{key}"] = fourier_by_layer[key].get("top_freq")
    summary["unembed_top_freq"] = unembed_fourier.get("top_freq")
    summary["unembed_top_energy_frac"] = unembed_fourier.get("top_energy_frac")
    summary["digit_emb_top_freq"] = emb_fourier.get("top_freq")
    summary["cosine_digit_emb_unembed"] = cosine_emb_unembed

    # Convenience primary columns = first requested layer
    primary = f"L{layers[0]}"
    summary["probe_sum_acc"] = summary.get(f"probe_sum_acc_{primary}")
    summary["probe_sum_acc_random"] = summary.get(f"probe_sum_acc_random_{primary}")
    summary["steered_acc_target"] = summary.get(f"steered_acc_target_{primary}")
    summary["shuffled_steered_acc_target"] = summary.get(
        f"shuffled_steered_acc_target_{primary}"
    )

    return {
        "modulus": p,
        "ckpt_kind": target.ckpt_kind,
        "ckpt_path": str(target.ckpt_path),
        "job_id": target.job_id,
        "experiment_id": target.experiment_id,
        "operand_i": i,
        "operand_j": j,
        "slot": target.slot,
        "ckpt_step": payload.get("step"),
        "n_layers": n_layers,
        "mech_layers": layers,
        "baseline_val_acc": baseline_val,
        "baseline_test_acc": baseline_test,
        "fourier_digit_emb": emb_fourier,
        "fourier_unembed": unembed_fourier,
        "fourier_by_layer": fourier_by_layer,
        "probes_by_layer": probes_by_layer,
        "attention": attn,
        "fourier_ablation": ablation,
        "steering_by_layer": steering_by_layer,
        "composition": composition,
        "summary": summary,
    }


def run_mech_single(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    moduli = list(args.moduli) if args.moduli else [31]
    ckpt_kinds = list(args.ckpt_kinds)
    max_batches = args.max_batches
    probe_steps = int(args.probe_steps)

    print(f"[phase1/mech-single] out={out_root}")
    print(f"[phase1/mech-single] ckpt_root={args.ckpt_root}")
    print(f"[phase1/mech-single] moduli={moduli} kinds={ckpt_kinds} device={device}")
    layers = list(getattr(args, "layers", None) or DEFAULT_MECH_LAYERS)
    skip_composition = bool(getattr(args, "skip_composition", False))
    print(f"[phase1/mech-single] mech_layers={layers} (resid_post query; not final pre-head)")
    print(f"[phase1/mech-single] composition={'off' if skip_composition else 'on'}")

    targets = discover_targets(
        args.ckpt_root, moduli=moduli, ckpt_kinds=ckpt_kinds
    )
    if not targets:
        raise SystemExit(
            f"no targets found under {args.ckpt_root} for moduli={moduli} "
            f"kinds={ckpt_kinds}"
        )

    results: list[dict[str, Any]] = []
    for t in targets:
        print(
            f"  analyzing p={t.modulus} kind={t.ckpt_kind} "
            f"ckpt={t.ckpt_path.name} data={t.data_dir.name}"
        )
        report = analyze_one(
            target=t,
            device=device,
            max_batches=max_batches,
            probe_steps=probe_steps,
            steer_delta=int(args.steer_delta),
            steer_alpha=float(args.steer_alpha),
            layers=layers,
            ablation_ks=(
                list(args.ablation_ks)
                if getattr(args, "ablation_ks", None)
                else None
            ),
            skip_composition=skip_composition,
        )
        report = _strip_private(report)
        per_path = out_root / f"p{t.modulus}_{t.ckpt_kind}_report.json"
        per_path.write_text(
            json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        results.append(report)
        s = report["summary"]
        bits = [f"test_acc={s['baseline_acc']:.4f}"]
        for li in layers:
            key = f"L{li}"
            bits.append(
                f"{key}:probe={s.get(f'probe_sum_acc_{key}')}"
                f"/rand={s.get(f'probe_sum_acc_random_{key}')}"
                f" steer={s.get(f'steered_acc_target_{key}')}"
                f"/shuf={s.get(f'shuffled_steered_acc_target_{key}')}"
            )
        if s.get("compose_layer_guess") is not None:
            bits.append(
                f"compose=L{s['compose_layer_guess']}"
                f"(ladder_jump=L{s.get('ladder_sum_jump_layer')})"
            )
        wh = s.get("worst_attn_head")
        if isinstance(wh, dict) and wh.get("layer") is not None:
            bits.append(
                f"worst_head=L{wh['layer']}H{wh['head']}"
                f"(Δ={wh.get('delta_acc')})"
            )
        print("    " + " | ".join(bits))

    summary_rows = []
    for r in results:
        s = r["summary"]
        row: dict[str, Any] = {
            "modulus": r["modulus"],
            "ckpt": r["ckpt_kind"],
            "baseline_acc": s["baseline_acc"],
            "attn_mass_operands": s["attn_mass_operands"],
            "ablation_delta_acc": s["ablation_delta_acc"],
            "digit_emb_top_freq": s.get("digit_emb_top_freq"),
            "unembed_top_freq": s.get("unembed_top_freq"),
            "cosine_digit_emb_unembed": s.get("cosine_digit_emb_unembed"),
            "compose_layer_guess": s.get("compose_layer_guess"),
            "ladder_sum_jump_layer": s.get("ladder_sum_jump_layer"),
            "harmonic_jump_layer": s.get("harmonic_jump_layer"),
            "job_id": r["job_id"],
        }
        for li in layers:
            key = f"L{li}"
            row[f"probe_sum_acc_{key}"] = s.get(f"probe_sum_acc_{key}")
            row[f"probe_sum_acc_random_{key}"] = s.get(f"probe_sum_acc_random_{key}")
            row[f"steered_acc_target_{key}"] = s.get(f"steered_acc_target_{key}")
            row[f"shuffled_steered_acc_target_{key}"] = s.get(
                f"shuffled_steered_acc_target_{key}"
            )
            row[f"steered_minus_shuffled_{key}"] = s.get(
                f"steered_minus_shuffled_{key}"
            )
        summary_rows.append(row)

    report = {
        "created_at": stamp(),
        "phase": "phase1",
        "step": "mech-single",
        "out": str(out_root),
        "config": {
            "ckpt_root": str(args.ckpt_root),
            "moduli": moduli,
            "ckpt_kinds": ckpt_kinds,
            "max_batches": max_batches,
            "probe_steps": probe_steps,
            "steer_delta": int(args.steer_delta),
            "steer_alpha": float(args.steer_alpha),
            "layers": layers,
            "ablation_ks": (
                list(args.ablation_ks)
                if getattr(args, "ablation_ks", None)
                else "all"
            ),
            "skip_composition": skip_composition,
            "device": str(device),
        },
        "summary": summary_rows,
        "results": to_jsonable(results),
    }
    report_path = out_root / "phase1_mech-single_report.json"
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    csv_fields = [
        "modulus",
        "ckpt",
        "baseline_acc",
    ]
    for li in layers:
        key = f"L{li}"
        csv_fields.extend(
            [
                f"probe_sum_acc_{key}",
                f"probe_sum_acc_random_{key}",
                f"steered_acc_target_{key}",
                f"shuffled_steered_acc_target_{key}",
                f"steered_minus_shuffled_{key}",
            ]
        )
    csv_fields.extend(
        [
            "attn_mass_operands",
            "ablation_delta_acc",
            "digit_emb_top_freq",
            "unembed_top_freq",
            "cosine_digit_emb_unembed",
            "compose_layer_guess",
            "ladder_sum_jump_layer",
            "harmonic_jump_layer",
            "job_id",
        ]
    )
    lines = [",".join(csv_fields)]
    for row in summary_rows:
        lines.append(",".join(str(row.get(f, "")) for f in csv_fields))
    csv_path = out_root / "phase1_mech-single_summary.csv"
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    # Separate ablation sweep CSV: important vs unimportant vs k
    abl_lines = [
        "modulus,ckpt,k,kind,acc,delta_acc,freqs,reps,energy_sum,baseline_acc,job_id"
    ]
    for r in results:
        abl = r.get("fourier_ablation") or {}
        base = abl.get("baseline_acc", "")
        for kind, curve in (
            ("important", abl.get("important_curve") or []),
            ("unimportant", abl.get("unimportant_curve") or []),
        ):
            for pt in curve:
                abl_lines.append(
                    ",".join(
                        str(x)
                        for x in [
                            r["modulus"],
                            r["ckpt_kind"],
                            pt.get("k"),
                            kind,
                            pt.get("acc"),
                            pt.get("delta_acc"),
                            "|".join(str(f) for f in pt.get("freqs") or []),
                            "|".join(str(f) for f in pt.get("reps") or []),
                            pt.get("energy_sum"),
                            base,
                            r["job_id"],
                        ]
                    )
                )
    abl_csv = out_root / "phase1_mech-single_ablation_curve.csv"
    abl_csv.write_text("\n".join(abl_lines) + "\n", encoding="utf-8")

    # Composition long-form CSV
    comp_lines = [
        "modulus,ckpt,kind,site_or_layer,metric,value,job_id"
    ]
    for r in results:
        comp = r.get("composition") or {}
        job = r["job_id"]
        ladder = comp.get("ladder") or {}
        for site, vals in (ladder.get("by_site") or {}).items():
            for metric in ("probe_xi", "probe_xj", "probe_sum"):
                comp_lines.append(
                    ",".join(
                        str(x)
                        for x in [
                            r["modulus"],
                            r["ckpt_kind"],
                            "ladder",
                            site,
                            metric,
                            vals.get(metric),
                            job,
                        ]
                    )
                )
        for row in (comp.get("knockout") or {}).get("by_layer") or []:
            for metric in (
                "zero_mlp_acc",
                "zero_mlp_delta",
                "zero_attn_acc",
                "zero_attn_delta",
            ):
                comp_lines.append(
                    ",".join(
                        str(x)
                        for x in [
                            r["modulus"],
                            r["ckpt_kind"],
                            "knockout",
                            f"L{row.get('layer')}",
                            metric,
                            row.get(metric),
                            job,
                        ]
                    )
                )
        for row in (comp.get("head_knockout") or {}).get("by_head") or []:
            comp_lines.append(
                ",".join(
                    str(x)
                    for x in [
                        r["modulus"],
                        r["ckpt_kind"],
                        "head_knockout",
                        f"L{row.get('layer')}H{row.get('head')}",
                        "delta_acc",
                        row.get("delta_acc"),
                        job,
                    ]
                )
            )
            comp_lines.append(
                ",".join(
                    str(x)
                    for x in [
                        r["modulus"],
                        r["ckpt_kind"],
                        "head_knockout",
                        f"L{row.get('layer')}H{row.get('head')}",
                        "acc",
                        row.get("acc"),
                        job,
                    ]
                )
            )
        harmonic = comp.get("harmonic") or {}
        for site, vals in (harmonic.get("by_site") or {}).items():
            for metric in (
                "R2_operand_mean",
                "R2_sum_mean",
                "sum_minus_operand_R2_mean",
            ):
                comp_lines.append(
                    ",".join(
                        str(x)
                        for x in [
                            r["modulus"],
                            r["ckpt_kind"],
                            "harmonic",
                            site,
                            metric,
                            vals.get(metric),
                            job,
                        ]
                    )
                )
    comp_csv = out_root / "phase1_mech-single_composition.csv"
    comp_csv.write_text("\n".join(comp_lines) + "\n", encoding="utf-8")

    print(f"\nreport: {report_path}")
    print(f"csv:    {csv_path}")
    print(f"ablation curve: {abl_csv}")
    print(f"composition: {comp_csv}")


def add_mech_single_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=f"runs/phase1/mech_single/{stamp()}",
        help="Output root for single-modulus mechanism analyses",
    )
    parser.add_argument(
        "--ckpt-root",
        type=str,
        default=DEFAULT_CKPT_ROOT,
        help="scan-moduli stamp directory containing runs/ and data/",
    )
    parser.add_argument(
        "--moduli",
        type=int,
        nargs="+",
        default=[31],
        help="Moduli to analyze (default: 31)",
    )
    parser.add_argument(
        "--ckpt-kinds",
        type=str,
        nargs="+",
        default=["final", "best"],
        choices=["final", "best"],
        help="Which checkpoints to analyze per modulus",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional cap on batches per split (debug / smoke)",
    )
    parser.add_argument(
        "--probe-steps",
        type=int,
        default=400,
        help="AdamW steps for linear probes",
    )
    parser.add_argument(
        "--layers",
        type=int,
        nargs="+",
        default=list(DEFAULT_MECH_LAYERS),
        help="Transformer layer indices for resid_post probes/steering "
        f"(default {list(DEFAULT_MECH_LAYERS)}; 0-based)",
    )
    parser.add_argument(
        "--steer-delta",
        type=int,
        default=1,
        help="Target sum shift for residual steering: (s+delta) mod p",
    )
    parser.add_argument(
        "--steer-alpha",
        type=float,
        default=1.0,
        help="Scale of class-mean steering vector",
    )
    parser.add_argument(
        "--ablation-ks",
        type=int,
        nargs="+",
        default=None,
        help="Fourier ablation sweep: how many conjugate freq-pairs to remove "
        "(default: all 1..n_pairs). Applied to both important and unimportant ranks.",
    )
    parser.add_argument(
        "--skip-composition",
        action="store_true",
        help="Skip composition-locus analyses (info ladder / knockout / harmonic fit)",
    )
