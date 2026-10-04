"""Phase 1C: multi-op comparative mechanism verification.

Uses a trained multi-op checkpoint and asks, per operation:
  - Does composition still happen at L0 (as in 1A-mech)?
  - Does attention route to that op's operand positions?
  - Are digit-embedding Fourier features modulus-specific?

Cross-op Fourier ablation measures selectivity: ablating p's top freqs
should hurt op-p much more than other moduli if compute is separate.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from go4cl.analysis.attention import query_attention_to_operands
from go4cl.analysis.cache import (
    accuracy_from_logits,
    collect_batches,
    operand_residues,
    to_jsonable,
)
from go4cl.analysis.causal import (
    _freq_pairs_by_energy,
    eval_accuracy,
    fourier_ablation_on_embeddings,
    project_out_freqs_from_digit_emb,
)
from go4cl.analysis.composition import run_composition_analysis
from go4cl.analysis.fourier import (
    analyze_digit_embedding_fourier,
    analyze_query_resid_fourier,
    analyze_unembedding_fourier,
    energy_cosine,
)
from go4cl.analysis.probes import run_layer_probes_with_random_control
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.data.manifest import DataManifest
from go4cl.phases.common import stamp
from go4cl.utils.checkpoint import load_checkpoint

DEFAULT_JOB = (
    "runs/phase1/multi_op/20261003_132221/runs/"
    "multi_four_diff_m31-37-29-23_tr0.8_ts0_ds0_a16_p1b_pack1"
    "__wd0.5_steps100000__ms0"
)
DEFAULT_MECH_LAYERS: tuple[int, ...] = (0, 1)


@dataclass(frozen=True)
class OpSpec:
    latent_id: int
    modulus: int
    operand_i: int
    operand_j: int
    slot: int


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


def _resolve_ckpt(job_dir: Path, kind: str) -> Path:
    ckpt_dir = job_dir / "ckpts"
    if kind == "final":
        candidates = [ckpt_dir / "a_only_final.pt", ckpt_dir / "final.pt"]
    elif kind == "best":
        candidates = [ckpt_dir / "best.pt", ckpt_dir / "a_only_best.pt"]
    else:
        raise ValueError(f"unknown ckpt kind: {kind}")
    for p in candidates:
        if p.is_file():
            return p
    raise FileNotFoundError(f"no {kind} checkpoint under {ckpt_dir}")


def _resolve_data_dir(job_dir: Path) -> Path:
    stamp_root = job_dir.parent.parent  # .../multi_op/<stamp>/runs/<job>
    exp_id = job_dir.name.split("__", 1)[0]
    data_dir = stamp_root / "data" / exp_id
    if (data_dir / "manifest.json").is_file():
        return data_dir
    # job_result may carry absolute/relative data_dir
    jr = job_dir / "job_result.json"
    if jr.is_file():
        d = json.loads(jr.read_text(encoding="utf-8"))
        cand = Path(d.get("data_dir") or "")
        if (cand / "manifest.json").is_file():
            return cand
    raise FileNotFoundError(f"cannot resolve data dir for {job_dir}")


def _ops_from_manifest(manifest: DataManifest) -> list[OpSpec]:
    ops = []
    for op in manifest.task_pair.task_a.operations:
        ops.append(
            OpSpec(
                latent_id=int(op.latent_id),
                modulus=int(op.modulus),
                operand_i=int(op.i),
                operand_j=int(op.j),
                slot=int(op.slot),
            )
        )
    return ops


def filter_by_modulus(
    ds: ModularAdditionDataset, modulus: int
) -> ModularAdditionDataset:
    """Aggregate filter by modulus (derived only). Prefer ``filter_by_operation``."""
    mask = ds.moduli.cpu().numpy() == int(modulus)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"no samples with modulus={modulus}")
    return _subset_dataset(ds, idx)


def filter_by_operation(
    ds: ModularAdditionDataset,
    *,
    latent_id: int | None = None,
    slot: int | None = None,
) -> ModularAdditionDataset:
    """Filter to one operation. Prefer ``latent_id``; ``slot`` is fallback."""
    if latent_id is None and slot is None:
        raise ValueError("need latent_id or slot")
    if latent_id is not None and int(latent_id) >= 0:
        lids = ds.latent_ids.cpu().numpy()
        if np.any(lids >= 0):
            mask = lids == int(latent_id)
        else:
            # legacy disk without latent_ids: fall back to slot
            if slot is None:
                raise ValueError(
                    "dataset has no latent_ids; pass slot= for legacy filter"
                )
            mask = ds.slots.cpu().numpy() == int(slot)
    else:
        mask = ds.slots.cpu().numpy() == int(slot)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(
            f"no samples for latent_id={latent_id} slot={slot}"
        )
    return _subset_dataset(ds, idx)


def _subset_dataset(ds: ModularAdditionDataset, idx: np.ndarray) -> ModularAdditionDataset:
    return ModularAdditionDataset(
        tokens=ds.tokens[idx].cpu().numpy(),
        labels=ds.labels[idx].cpu().numpy(),
        slots=ds.slots[idx].cpu().numpy(),
        moduli=ds.moduli[idx].cpu().numpy(),
        task_ids=ds.task_ids[idx].cpu().numpy(),
        latent_ids=ds.latent_ids[idx].cpu().numpy(),
    )


def _query_layer_resids(cache, layers: list[int], device: torch.device) -> dict[int, torch.Tensor]:
    return {li: cache.resid_post[li][:, -1, :].to(device) for li in layers}


def _routing_separation(
    attn_layers: list[dict[str, Any]],
    *,
    operand_i: int,
    operand_j: int,
    same_modulus_other: list[int],
    different_modulus_other: list[int],
) -> dict[str, Any]:
    """Compare L0 mass: own operands vs same-modulus other op vs different-mod ops."""
    if not attn_layers:
        return {}
    l0 = next((x for x in attn_layers if x.get("layer") == 0), attn_layers[0])
    mass = l0.get("mass_by_pos") or []
    needed = [operand_i, operand_j, *same_modulus_other, *different_modulus_other]
    if len(mass) <= max(needed or [0]):
        return {"operand_mass": l0.get("operand_mass")}
    own = float(mass[operand_i] + mass[operand_j])
    same_m = float(sum(mass[p] for p in same_modulus_other))
    diff_m = float(sum(mass[p] for p in different_modulus_other))
    other = same_m + diff_m
    return {
        "L0_own_operand_mass": own,
        "L0_same_modulus_other_op_mass": same_m,
        "L0_different_modulus_other_ops_mass": diff_m,
        "L0_other_ops_operand_mass": other,
        "L0_own_minus_other": own - other,
        "same_modulus_other_positions": list(same_modulus_other),
        "different_modulus_other_positions": list(different_modulus_other),
    }


def _analyze_op(
    *,
    model,
    op: OpSpec,
    all_ops: list[OpSpec],
    train_full: ModularAdditionDataset,
    val_full: ModularAdditionDataset,
    test_full: ModularAdditionDataset,
    device: torch.device,
    layers: list[int],
    probe_steps: int,
    max_batches: int | None,
    ablation_ks: list[int] | None,
    skip_composition: bool,
    analysis_builder=None,
) -> dict[str, Any]:
    p = op.modulus
    i, j = op.operand_i, op.operand_j

    def _filter_or_build(ds_full, *, split: str):
        if len(ds_full) == 0:
            if analysis_builder is None:
                raise ValueError(
                    f"empty {split} split and no analysis_builder for "
                    f"latent_id={op.latent_id}"
                )
            from go4cl.data.dataset import ModularAdditionDataset as _DS

            examples = analysis_builder(
                split=split,
                target_latent_ids=[op.latent_id],
            )
            return _DS.from_examples(examples, task_id=0)
        return filter_by_operation(
            ds_full, latent_id=op.latent_id, slot=op.slot
        )

    train_ds = _filter_or_build(train_full, split="train")
    val_ds = _filter_or_build(val_full, split="val")
    test_ds = _filter_or_build(test_full, split="test")
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

    emb_fourier = analyze_digit_embedding_fourier(model.tok_emb, modulus=p)
    unembed_fourier = analyze_unembedding_fourier(model.head, modulus=p)
    _, _, sum_test = operand_residues(test_cache.tokens, i=i, j=j, modulus=p)
    query_fourier = {
        f"L{li}": analyze_query_resid_fourier(
            test_cache.resid_post[li][:, -1, :],
            sum_test,
            modulus=p,
            source=f"resid_post_L{li}_query_by_sum",
        )
        for li in layers
    }
    cosine_emb_unembed = energy_cosine(
        emb_fourier["energy_by_freq"], unembed_fourier["energy_by_freq"]
    )
    q_key = "L1" if "L1" in query_fourier else (
        f"L{layers[-1]}" if layers else None
    )
    cosine_L1_unembed = (
        energy_cosine(
            query_fourier[q_key]["energy_by_freq"],
            unembed_fourier["energy_by_freq"],
        )
        if q_key
        else 0.0
    )
    top_pairs = _freq_pairs_by_energy(
        emb_fourier["energy_by_freq"], modulus=p, skip_dc=True
    )
    top1 = top_pairs[0] if top_pairs else None

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

    attn = query_attention_to_operands(test_cache.attn, operand_i=i, operand_j=j)
    same_mod_other = sorted(
        {
            pos
            for o in all_ops
            if o.latent_id != op.latent_id and o.modulus == p
            for pos in (o.operand_i, o.operand_j)
        }
    )
    diff_mod_other = sorted(
        {
            pos
            for o in all_ops
            if o.modulus != p
            for pos in (o.operand_i, o.operand_j)
        }
    )
    routing = _routing_separation(
        attn.get("layers") or [],
        operand_i=i,
        operand_j=j,
        same_modulus_other=same_mod_other,
        different_modulus_other=diff_mod_other,
    )

    ablation = fourier_ablation_on_embeddings(
        model,
        test_loader,
        modulus=p,
        device=device,
        top_k=1,
        max_batches=max_batches,
        sweep_ks=ablation_ks if ablation_ks is not None else [1, 2],
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

    summary = {
        "baseline_acc": baseline_test,
        "baseline_val_acc": baseline_val,
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
            None if composition is None else composition.get("worst_attn_head")
        ),
        "attn_mass_operands": attn.get("mean_operand_mass"),
        "L0_own_operand_mass": routing.get("L0_own_operand_mass"),
        "L0_other_ops_operand_mass": routing.get("L0_other_ops_operand_mass"),
        "L0_own_minus_other": routing.get("L0_own_minus_other"),
        "top1_imp_freqs": (top1 or {}).get("freqs"),
        "top1_imp_acc": (ablation.get("important_curve") or [{}])[0].get("acc"),
        "top1_imp_delta": (ablation.get("important_curve") or [{}])[0].get(
            "delta_acc"
        ),
        "unembed_top_freq": unembed_fourier.get("top_freq"),
        "unembed_top_energy_frac": unembed_fourier.get("top_energy_frac"),
        "digit_emb_top_freq": emb_fourier.get("top_freq"),
        "cosine_digit_emb_unembed": cosine_emb_unembed,
        "cosine_queryL_unembed": cosine_L1_unembed,
        "probe_sum_acc_L0": probes_by_layer.get("L0", {}).get("probe_sum_acc"),
        "probe_sum_acc_L1": probes_by_layer.get("L1", {}).get("probe_sum_acc"),
        "n_test": int(len(test_ds)),
    }

    return {
        "latent_id": op.latent_id,
        "modulus": p,
        "operand_i": i,
        "operand_j": j,
        "slot": op.slot,
        "summary": summary,
        "attention": attn,
        "routing_separation": routing,
        "fourier_digit_emb": emb_fourier,
        "fourier_unembed": unembed_fourier,
        "fourier_query_resid": query_fourier,
        "fourier_ablation": ablation,
        "probes_by_layer": probes_by_layer,
        "composition": composition,
        "_test_loader": test_loader,  # stripped before write
        "_top1_freqs": list((top1 or {}).get("freqs") or []),
    }


def _op_report_key(rep: dict[str, Any]) -> str:
    return f"lat{rep['latent_id']}/slot{rep['slot']}/p{rep['modulus']}"


def _cross_op_fourier_selectivity(
    model,
    *,
    op_reports: list[dict[str, Any]],
    device: torch.device,
    max_batches: int | None,
) -> dict[str, Any]:
    """Ablate each op's top-1 freqs; measure Δacc on every op's test loader.

    Matrix is keyed by operation identity (latent/slot), not modulus alone, so
    pair_same duplicate moduli stay distinct.
    """
    original = model.tok_emb.weight.data.clone()
    matrix: list[dict[str, Any]] = []
    for src in op_reports:
        p = int(src["modulus"])
        freqs = list(src.get("_top1_freqs") or [])
        src_key = _op_report_key(src)
        if not freqs:
            continue
        ablated = project_out_freqs_from_digit_emb(
            original, modulus=p, freqs=freqs
        ).to(device=original.device, dtype=original.dtype)
        model.tok_emb.weight.data.copy_(ablated)
        row: dict[str, Any] = {
            "source_operation": src_key,
            "source_modulus": p,
            "ablated_freqs": freqs,
            "by_target": {},
        }
        for tgt in op_reports:
            tgt_key = _op_report_key(tgt)
            base = float(tgt["summary"]["baseline_acc"])
            acc = eval_accuracy(
                model, tgt["_test_loader"], device=device, max_batches=max_batches
            )
            row["by_target"][tgt_key] = {
                "acc": float(acc),
                "delta_acc": float(acc - base),
                "baseline_acc": base,
                "modulus": int(tgt["modulus"]),
                "latent_id": int(tgt["latent_id"]),
            }
        model.tok_emb.weight.data.copy_(original)
        self_d = row["by_target"][src_key]["delta_acc"]
        others = [
            v["delta_acc"]
            for k, v in row["by_target"].items()
            if k != src_key
        ]
        row["self_delta"] = self_d
        row["mean_other_delta"] = float(sum(others) / max(len(others), 1))
        row["selectivity"] = float(row["mean_other_delta"] - self_d)
        matrix.append(row)
    model.tok_emb.weight.data.copy_(original)
    return {"ablate_top1_matrix": matrix}


def _compare(op_reports: list[dict[str, Any]], selectivity: dict[str, Any]) -> dict[str, Any]:
    n = len(op_reports)
    compose_L0 = sum(
        1 for r in op_reports if r["summary"].get("compose_layer_guess") == 0
    )
    own_gt_other = sum(
        1
        for r in op_reports
        if (r["summary"].get("L0_own_minus_other") or -1) > 0
    )
    top1_hurt = sum(
        1
        for r in op_reports
        if (r["summary"].get("top1_imp_delta") or 0) < -0.3
    )
    mat = selectivity.get("ablate_top1_matrix") or []
    selective = sum(1 for row in mat if row.get("selectivity", 0) > 0.2)
    worst_heads = [
        r["summary"].get("worst_attn_head") for r in op_reports
    ]
    return {
        "n_ops": n,
        "compose_at_L0": f"{compose_L0}/{n}",
        "L0_routes_to_own_operands": f"{own_gt_other}/{n}",
        "top1_fourier_hurts_self": f"{top1_hurt}/{n}",
        "fourier_ablation_selective": f"{selective}/{len(mat)}",
        "worst_heads": worst_heads,
        "verdict": (
            "non_selective_global_sensitivity"
            if len(mat) > 0 and selective == 0
            else "selective"
            if selective >= max(len(mat) - 1, 1)
            else "inconclusive"
        ),
    }


def run_mechanisms(args: argparse.Namespace) -> None:
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    device = _device(args.device)
    job_dir = Path(args.job_dir)
    ckpt_kind = str(args.ckpt_kind)
    layers = list(getattr(args, "layers", None) or DEFAULT_MECH_LAYERS)
    probe_steps = int(args.probe_steps)
    max_batches = args.max_batches
    skip_composition = bool(getattr(args, "skip_composition", False))
    ablation_ks = (
        list(args.ablation_ks)
        if getattr(args, "ablation_ks", None)
        else [1, 2]
    )

    ckpt_path = _resolve_ckpt(job_dir, ckpt_kind)
    data_dir = _resolve_data_dir(job_dir)
    manifest = DataManifest.load(data_dir / "manifest.json")
    ops = _ops_from_manifest(manifest)

    print(f"[phase1/mechanisms] out={out_root}")
    print(f"[phase1/mechanisms] job={job_dir.name}")
    print(f"[phase1/mechanisms] ckpt={ckpt_path.name} kind={ckpt_kind}")
    print(f"[phase1/mechanisms] data={data_dir.name}")
    print(
        f"[phase1/mechanisms] ops="
        + ", ".join(f"p{o.modulus}@slot{o.slot}(i={o.operand_i},j={o.operand_j})" for o in ops)
    )
    print(f"[phase1/mechanisms] layers={layers} composition={'off' if skip_composition else 'on'}")

    model, payload = load_checkpoint(ckpt_path, map_location=device)
    model.to(device)
    model.eval()

    train_full = ModularAdditionDataset.from_disk(data_dir, "A", "train", 0)
    val_full = ModularAdditionDataset.from_disk(data_dir, "A", "val", 0)
    test_full = ModularAdditionDataset.from_disk(data_dir, "A", "test", 0)

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

    op_reports: list[dict[str, Any]] = []
    for op in ops:
        print(f"  analyzing op latent={op.latent_id} p={op.modulus} slot={op.slot} ...")
        rep = _analyze_op(
            model=model,
            op=op,
            all_ops=ops,
            train_full=train_full,
            val_full=val_full,
            test_full=test_full,
            device=device,
            layers=layers,
            probe_steps=probe_steps,
            max_batches=max_batches,
            ablation_ks=ablation_ks,
            skip_composition=skip_composition,
            analysis_builder=_analysis_builder,
        )
        s = rep["summary"]
        print(
            f"    test={s['baseline_acc']:.4f} compose=L{s.get('compose_layer_guess')} "
            f"own-other={s.get('L0_own_minus_other')} "
            f"top1Δ={s.get('top1_imp_delta')} "
            f"worst={s.get('worst_attn_head')}"
        )
        op_reports.append(rep)

    print("  cross-op Fourier selectivity (ablate each top-1) ...")
    selectivity = _cross_op_fourier_selectivity(
        model, op_reports=op_reports, device=device, max_batches=max_batches
    )
    comparison = _compare(op_reports, selectivity)
    print(f"  compare: {comparison}")

    # strip private / loaders before serialize
    clean_ops = []
    for r in op_reports:
        clean = {k: v for k, v in r.items() if not str(k).startswith("_")}
        clean_ops.append(_strip_private(clean))

    report = {
        "created_at": stamp(),
        "phase": "phase1",
        "step": "mechanisms",
        "out": str(out_root),
        "config": {
            "job_dir": str(job_dir),
            "ckpt_path": str(ckpt_path),
            "ckpt_kind": ckpt_kind,
            "data_dir": str(data_dir),
            "layers": layers,
            "probe_steps": probe_steps,
            "ablation_ks": ablation_ks,
            "skip_composition": skip_composition,
            "max_batches": max_batches,
            "ckpt_step": payload.get("step"),
        },
        "ops": [
            {
                "latent_id": o.latent_id,
                "modulus": o.modulus,
                "operand_i": o.operand_i,
                "operand_j": o.operand_j,
                "slot": o.slot,
            }
            for o in ops
        ],
        "per_op": clean_ops,
        "cross_op_fourier_selectivity": selectivity,
        "comparison": comparison,
    }
    report_path = out_root / "phase1_mechanisms_report.json"
    report_path.write_text(
        json.dumps(to_jsonable(report), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    # summary CSV
    csv_path = out_root / "phase1_mechanisms_summary.csv"
    fields = [
        "modulus",
        "slot",
        "operand_i",
        "operand_j",
        "baseline_acc",
        "compose_layer_guess",
        "ladder_sum_jump_layer",
        "harmonic_jump_layer",
        "L0_own_operand_mass",
        "L0_other_ops_operand_mass",
        "L0_own_minus_other",
        "top1_imp_freqs",
        "top1_imp_acc",
        "top1_imp_delta",
        "unembed_top_freq",
        "cosine_digit_emb_unembed",
        "probe_sum_acc_L0",
        "probe_sum_acc_L1",
        "worst_head_layer",
        "worst_head",
        "worst_head_delta",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in clean_ops:
            s = r["summary"]
            wh = s.get("worst_attn_head") or {}
            w.writerow(
                {
                    "modulus": r["modulus"],
                    "slot": r["slot"],
                    "operand_i": r["operand_i"],
                    "operand_j": r["operand_j"],
                    "baseline_acc": s.get("baseline_acc"),
                    "compose_layer_guess": s.get("compose_layer_guess"),
                    "ladder_sum_jump_layer": s.get("ladder_sum_jump_layer"),
                    "harmonic_jump_layer": s.get("harmonic_jump_layer"),
                    "L0_own_operand_mass": s.get("L0_own_operand_mass"),
                    "L0_other_ops_operand_mass": s.get("L0_other_ops_operand_mass"),
                    "L0_own_minus_other": s.get("L0_own_minus_other"),
                    "top1_imp_freqs": ",".join(
                        map(str, s.get("top1_imp_freqs") or [])
                    ),
                    "top1_imp_acc": s.get("top1_imp_acc"),
                    "top1_imp_delta": s.get("top1_imp_delta"),
                    "unembed_top_freq": s.get("unembed_top_freq"),
                    "cosine_digit_emb_unembed": s.get("cosine_digit_emb_unembed"),
                    "probe_sum_acc_L0": s.get("probe_sum_acc_L0"),
                    "probe_sum_acc_L1": s.get("probe_sum_acc_L1"),
                    "worst_head_layer": wh.get("layer"),
                    "worst_head": wh.get("head"),
                    "worst_head_delta": wh.get("delta_acc"),
                }
            )

    # selectivity long CSV (keys are op identities, not bare moduli)
    sel_csv = out_root / "phase1_mechanisms_fourier_selectivity.csv"
    with sel_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames=[
                "source_operation",
                "source_modulus",
                "target_operation",
                "target_modulus",
                "ablated_freqs",
                "baseline_acc",
                "acc",
                "delta_acc",
                "self_delta",
                "mean_other_delta",
                "selectivity",
            ],
        )
        w.writeheader()
        for row in selectivity.get("ablate_top1_matrix") or []:
            for tgt_key, tv in row["by_target"].items():
                w.writerow(
                    {
                        "source_operation": row.get("source_operation", ""),
                        "source_modulus": row["source_modulus"],
                        "target_operation": tgt_key,
                        "target_modulus": int(tv["modulus"]),
                        "ablated_freqs": ",".join(map(str, row["ablated_freqs"])),
                        "baseline_acc": tv["baseline_acc"],
                        "acc": tv["acc"],
                        "delta_acc": tv["delta_acc"],
                        "self_delta": row["self_delta"],
                        "mean_other_delta": row["mean_other_delta"],
                        "selectivity": row["selectivity"],
                    }
                )

    # short conclusions
    md = out_root / "CONCLUSIONS.md"
    lines = [
        "# 1C · multi-op mechanisms\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`)  ",
        f"**Data:** `{data_dir.name}`\n",
        "## Comparison verdict\n",
        f"- compose @ L0: **{comparison['compose_at_L0']}**",
        f"- L0 routes to own operands (own > other): **{comparison['L0_routes_to_own_operands']}**",
        f"- top-1 Fourier hurts self: **{comparison['top1_fourier_hurts_self']}**",
        f"- Fourier ablation selective (self hurt ≫ others): **{comparison['fourier_ablation_selective']}**",
        f"- overall: `{comparison['verdict']}`\n",
        "## Per-op summary\n",
        "| p | slot | test | compose | L0 own−other | top1 Δ | worst head |",
        "|---|------|------|---------|--------------|--------|------------|",
    ]
    for r in clean_ops:
        s = r["summary"]
        wh = s.get("worst_attn_head") or {}
        wh_s = (
            f"L{wh.get('layer')}H{wh.get('head')}({wh.get('delta_acc'):.2f})"
            if wh.get("layer") is not None
            else ""
        )
        lines.append(
            f"| {r['modulus']} | {r['slot']} | {float(s['baseline_acc']):.3f} | "
            f"L{s.get('compose_layer_guess')} | "
            f"{float(s.get('L0_own_minus_other') or 0):+.3f} | "
            f"{float(s.get('top1_imp_delta') or 0):+.3f} | {wh_s} |"
        )
    lines.append("\n## Cross-op Fourier selectivity (ablate source top-1)\n")
    op_keys = [_op_report_key(r) for r in clean_ops]
    lines.append(
        "| source | "
        + " | ".join(f"→{k}" for k in op_keys)
        + " | selectivity |"
    )
    lines.append("|" + "---|" * (len(op_keys) + 2))
    for row in selectivity.get("ablate_top1_matrix") or []:
        cells = [
            f"{row['by_target'][k]['delta_acc']:+.3f}"
            if k in row["by_target"]
            else ""
            for k in op_keys
        ]
        src_label = row.get("source_operation") or str(row["source_modulus"])
        lines.append(
            f"| {src_label} | "
            + " | ".join(cells)
            + f" | {row['selectivity']:+.3f} |"
        )
    lines.append("\nFiles: `phase1_mechanisms_report.json`, `phase1_mechanisms_summary.csv`, "
                 "`phase1_mechanisms_fourier_selectivity.csv`.\n")
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"\nreport: {report_path}")
    print(f"csv:    {csv_path}")
    print(f"selectivity: {sel_csv}")
    print(f"conclusions: {md}")


def add_mechanisms_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--out",
        type=str,
        default=f"runs/phase1/mechanisms/{stamp()}",
        help="Output root for multi-op mechanism analyses",
    )
    parser.add_argument(
        "--job-dir",
        type=str,
        default=DEFAULT_JOB,
        help="multi-op job directory containing ckpts/ (default: four_diff "
        "m47-43-37-23 best-capable run)",
    )
    parser.add_argument(
        "--ckpt-kind",
        type=str,
        default="best",
        choices=["final", "best"],
        help="Which checkpoint to analyze",
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
        help="Layers for resid_post probes (default 0 1)",
    )
    parser.add_argument(
        "--ablation-ks",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Fourier ablation k values (default: 1 2)",
    )
    parser.add_argument(
        "--skip-composition",
        action="store_true",
        help="Skip composition-locus analyses",
    )
