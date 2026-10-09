#!/usr/bin/env python3
"""Three follow-up mechanism experiments on next80_formal representative ckpts.

1. Fourier energy spectra: theta_A (task A) vs phase_b_final (A and B).
2. Component restore: patch theta_A groups into phase_b_final on forget cases.
3. Circuit change: for replay-success (and all cases), compare mech metrics +
   weight-group deltas between theta_A and phase_b_final.

Reads existing mech_suite reports where possible. Does not overwrite training
artifacts. Writes under ``runs/phase2/next80_formal/mech_followup_<stamp>/``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from go4cl.analysis.context import load_analysis_context
from go4cl.analysis.fourier import energy_cosine
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.phases.phase2.param_patch import (
    evaluate_patches,
    load_theta,
    write_patch_csv,
)
from go4cl.utils.checkpoint import write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_FORMAL = ROOT / "runs/phase2/next80_formal"
DEFAULT_MECH = DEFAULT_FORMAL / "mech_suite"

CASES: list[dict[str, Any]] = [
    {
        "id": "F1_forget_m0",
        "role": "forget_fast",
        "rel": "01_core_fresh/runs/sequential_ab_s0.5_o0.5_m0_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh",
        "mech_theta": "F1_forget_m0_thetaA",
        "mech_finalA": "F1_forget_m0_finalA",
        "mech_finalB": "F1_forget_m0_finalB",
        "run_restore": True,
    },
    {
        "id": "F2_partial_m1",
        "role": "partial_forget",
        "rel": "01_core_fresh/runs/sequential_ab_s0.5_o0.5_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh",
        "mech_theta": "F2_partial_m1_thetaA",
        "mech_finalA": "F2_partial_m1_finalA",
        "mech_finalB": "F2_partial_m1_finalB",
        "run_restore": True,
    },
    {
        "id": "Rp_replay_m1",
        "role": "replay_keep_A",
        "rel": "03_replay_overlap/runs/sequential_ab_replay_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh",
        "mech_theta": "Rp_replay_m1_thetaA",
        "mech_finalA": "Rp_replay_m1_finalA",
        "mech_finalB": "Rp_replay_m1_finalB",
        "run_restore": False,
    },
    {
        "id": "Rm_noreplay_m1",
        "role": "forget_fast",
        "rel": "04_no_replay_anchors/runs/sequential_ab_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh",
        "mech_theta": "Rm_noreplay_m1_thetaA",
        "mech_finalA": "Rm_noreplay_m1_finalA",
        "mech_finalB": "Rm_noreplay_m1_finalB",
        "run_restore": True,
    },
    {
        "id": "Rh_replay075",
        "role": "replay_partial_death",
        "rel": "03_replay_overlap/runs/sequential_ab_replay_s0.5_o0.5_m0.5_forward_ts1_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh",
        "mech_theta": "Rh_replay075_thetaA",
        "mech_finalA": "Rh_replay075_finalA",
        "mech_finalB": "Rh_replay075_finalB",
        "run_restore": True,
    },
]

MECH_KEYS = [
    "baseline_acc",
    "probe_sum_acc_L0",
    "probe_sum_acc_L1",
    "L0_own_minus_other",
    "top1_imp_delta",
    "compose_layer_guess",
    "ladder_sum_jump_layer",
    "harmonic_jump_layer",
    "cosine_digit_emb_unembed",
    "cosine_queryL_unembed",
    "digit_emb_top_freq",
    "unembed_top_freq",
]

SKIP_SUFFIXES = (".attn.mask",)


def _style() -> None:
    mpl.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
            "legend.fontsize": 8,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _load_report(mech_root: Path, label: str) -> dict[str, Any]:
    path = mech_root / label / "phase1_mechanisms_report.json"
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _ops_by_mod(report: dict[str, Any]) -> dict[int, dict[str, Any]]:
    return {int(op["modulus"]): op for op in report["per_op"]}


def _spectrum_row(
    case_id: str,
    source: str,
    role: str,
    task: str,
    op: dict[str, Any],
    kind: str,
) -> dict[str, Any]:
    block = op[kind]
    return {
        "case": case_id,
        "source": source,
        "role": role,
        "task": task,
        "modulus": int(op["modulus"]),
        "latent_id": int(op["latent_id"]),
        "kind": kind,
        "top_freq": block.get("top_freq"),
        "top_energy_frac": block.get("top_energy_frac"),
        "total_energy": block.get("total_energy"),
        "energy_by_freq": block.get("energy_by_freq"),
    }


def _compare_spectra(
    left: dict[str, Any], right: dict[str, Any], pair_name: str
) -> dict[str, Any]:
    e0 = left["energy_by_freq"]
    e1 = right["energy_by_freq"]
    return {
        "pair": pair_name,
        "case": left["case"],
        "kind": left["kind"],
        "mod_left": left["modulus"],
        "mod_right": right["modulus"],
        "same_modulus": int(left["modulus"]) == int(right["modulus"]),
        "cos_skip_dc": energy_cosine(e0, e1, skip_dc=True)
        if len(e0) == len(e1)
        else float("nan"),
        "cos_with_dc": energy_cosine(e0, e1, skip_dc=False)
        if len(e0) == len(e1)
        else float("nan"),
        "top_freq_left": left["top_freq"],
        "top_freq_right": right["top_freq"],
        "top_freq_match": left["top_freq"] == right["top_freq"],
        "top_frac_left": left["top_energy_frac"],
        "top_frac_right": right["top_energy_frac"],
        "total_E_left": left["total_energy"],
        "total_E_right": right["total_energy"],
        "log10_E_ratio": (
            math.log10((right["total_energy"] + 1e-30) / (left["total_energy"] + 1e-30))
        ),
    }


def experiment_fourier(cases: list[dict[str, Any]], mech_root: Path, out: Path) -> None:
    spectra_rows: list[dict[str, Any]] = []
    compare_rows: list[dict[str, Any]] = []
    fig_dir = out / "01_fourier" / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _style()

    for case in cases:
        theta = _load_report(mech_root, case["mech_theta"])
        final_a = _load_report(mech_root, case["mech_finalA"])
        final_b = _load_report(mech_root, case["mech_finalB"])
        by_theta = _ops_by_mod(theta)
        by_fa = _ops_by_mod(final_a)
        by_fb = _ops_by_mod(final_b)

        for kind in ("fourier_digit_emb", "fourier_unembed"):
            left_by_mod: dict[int, dict[str, Any]] = {}
            for mod, op in by_theta.items():
                row = _spectrum_row(case["id"], case["mech_theta"], "theta_A", "A", op, kind)
                spectra_rows.append({k: v for k, v in row.items() if k != "energy_by_freq"})
                left_by_mod[mod] = row
            fa_by_mod: dict[int, dict[str, Any]] = {}
            for mod, op in by_fa.items():
                row = _spectrum_row(case["id"], case["mech_finalA"], "phase_b_final", "A", op, kind)
                spectra_rows.append({k: v for k, v in row.items() if k != "energy_by_freq"})
                fa_by_mod[mod] = row
            fb_by_mod: dict[int, dict[str, Any]] = {}
            for mod, op in by_fb.items():
                row = _spectrum_row(case["id"], case["mech_finalB"], "phase_b_final", "B", op, kind)
                spectra_rows.append({k: v for k, v in row.items() if k != "energy_by_freq"})
                fb_by_mod[mod] = row

            # Pre-A vs post-A (same moduli)
            for mod in sorted(set(left_by_mod) & set(fa_by_mod)):
                compare_rows.append(
                    _compare_spectra(left_by_mod[mod], fa_by_mod[mod], "thetaA_vs_finalA")
                )
            # Shared-modulus A vs B after switch
            for mod in sorted(set(fa_by_mod) & set(fb_by_mod)):
                compare_rows.append(
                    _compare_spectra(fa_by_mod[mod], fb_by_mod[mod], "finalA_vs_finalB_shared")
                )
            # Pre-A vs post-B on shared moduli (when rho_mod allows)
            for mod in sorted(set(left_by_mod) & set(fb_by_mod)):
                compare_rows.append(
                    _compare_spectra(left_by_mod[mod], fb_by_mod[mod], "thetaA_vs_finalB_shared")
                )

            # Plot: one panel per A modulus — theta_A vs final A; overlay B if shared
            mods_a = sorted(left_by_mod)
            n = len(mods_a)
            fig, axes = plt.subplots(1, n, figsize=(3.2 * n, 3.0), sharey=True)
            if n == 1:
                axes = [axes]
            short = kind.replace("fourier_", "")
            for ax, mod in zip(axes, mods_a):
                e_th = np.asarray(left_by_mod[mod]["energy_by_freq"], dtype=float)
                e_fa = np.asarray(fa_by_mod[mod]["energy_by_freq"], dtype=float)
                freqs = np.arange(len(e_th))
                ax.plot(freqs, e_th, label="theta_A / A", color="#1f4e79", lw=1.6)
                ax.plot(freqs, e_fa, label="final / A", color="#c45c26", lw=1.6, alpha=0.9)
                if mod in fb_by_mod:
                    e_fb = np.asarray(fb_by_mod[mod]["energy_by_freq"], dtype=float)
                    ax.plot(freqs, e_fb, label="final / B", color="#2a9d8f", lw=1.4, ls="--")
                cos = energy_cosine(e_th, e_fa, skip_dc=True)
                ax.set_title(f"p={mod}  cos(θA,A′)={cos:.3f}")
                ax.set_xlabel("frequency k")
                ax.set_yscale("log")
            axes[0].set_ylabel(f"{short} energy")
            axes[0].legend(loc="upper right", fontsize=7)
            fig.suptitle(f"{case['id']} · {short}", y=1.02)
            fig.tight_layout()
            fig.savefig(fig_dir / f"{case['id']}_{short}.png", bbox_inches="tight")
            plt.close(fig)

            # Separate B-only moduli plot if any
            only_b = sorted(set(fb_by_mod) - set(left_by_mod))
            if only_b:
                fig, axes = plt.subplots(
                    1, len(only_b), figsize=(3.2 * len(only_b), 3.0), sharey=True
                )
                if len(only_b) == 1:
                    axes = [axes]
                for ax, mod in zip(axes, only_b):
                    e_fb = np.asarray(fb_by_mod[mod]["energy_by_freq"], dtype=float)
                    ax.plot(np.arange(len(e_fb)), e_fb, color="#2a9d8f", lw=1.6)
                    ax.set_title(f"B-only p={mod}")
                    ax.set_xlabel("frequency k")
                    ax.set_yscale("log")
                axes[0].set_ylabel(f"{short} energy")
                fig.suptitle(f"{case['id']} · {short} · B-only moduli", y=1.02)
                fig.tight_layout()
                fig.savefig(fig_dir / f"{case['id']}_{short}_B_only.png", bbox_inches="tight")
                plt.close(fig)

    _write_csv(out / "01_fourier" / "spectra_summary.csv", spectra_rows)
    _write_csv(out / "01_fourier" / "spectrum_similarity.csv", compare_rows)

    # Aggregate heatmap: cos(thetaA, finalA) by case × modulus for digit emb
    dig = [r for r in compare_rows if r["pair"] == "thetaA_vs_finalA" and r["kind"] == "fourier_digit_emb"]
    if dig:
        cases_ids = sorted({r["case"] for r in dig})
        mods = sorted({int(r["mod_left"]) for r in dig})
        mat = np.full((len(cases_ids), len(mods)), np.nan)
        for r in dig:
            i = cases_ids.index(r["case"])
            j = mods.index(int(r["mod_left"]))
            mat[i, j] = r["cos_skip_dc"]
        fig, ax = plt.subplots(figsize=(1.1 * len(mods) + 2.5, 0.55 * len(cases_ids) + 1.5))
        im = ax.imshow(mat, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(len(mods)), [f"p{m}" for m in mods])
        ax.set_yticks(range(len(cases_ids)), cases_ids)
        ax.set_title("digit-emb spectrum cosine: theta_A vs final A")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if not np.isnan(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.046)
        fig.tight_layout()
        fig.savefig(fig_dir / "heatmap_digit_emb_thetaA_vs_finalA.png", bbox_inches="tight")
        plt.close(fig)


def _task_loader(ctx, split: str = "test", batch_size: int = 512):
    ds = ctx.disk_dataset(split)  # type: ignore[arg-type]
    if len(ds) == 0:
        # fall back: union of analysis examples for all ops
        examples = []
        for op in ctx.operations:
            examples.extend(
                ctx.analysis_examples(split=split, target_latent_ids=[op.latent_id])  # type: ignore[arg-type]
            )
        ds = ModularAdditionDataset.from_examples(examples, task_id=ctx.task_id)
    return make_loader(ds, batch_size=batch_size, shuffle=False)


def experiment_restore(
    cases: list[dict[str, Any]],
    formal_root: Path,
    out: Path,
    *,
    device: torch.device,
) -> None:
    restore_root = out / "02_component_restore"
    restore_root.mkdir(parents=True, exist_ok=True)
    summary: list[dict[str, Any]] = []

    for case in cases:
        if not case.get("run_restore"):
            continue
        job_dir = formal_root / case["rel"]
        print(f"[restore] {case['id']}  job={job_dir.name}")
        ctx_a = load_analysis_context(
            job_dir, ckpt_kind="phase_b_final", device=device, task="A"
        )
        ctx_b = load_analysis_context(
            job_dir, ckpt_kind="phase_b_final", device=device, task="B", load_model=False
        )
        theta_a = load_theta(job_dir / "ckpts" / "theta_A.pt")
        theta_ab = load_theta(job_dir / "ckpts" / "phase_b_final.pt")
        loader_a = _task_loader(ctx_a)
        loader_b = _task_loader(ctx_b)
        model = ctx_a.model
        rows = evaluate_patches(
            model, theta_a, theta_ab, loader_a, loader_b, device=device
        )
        case_dir = restore_root / case["id"]
        case_dir.mkdir(parents=True, exist_ok=True)
        write_patch_csv(case_dir / "param_patch.csv", rows)
        write_json(
            case_dir / "meta.json",
            {
                "case": case["id"],
                "role": case["role"],
                "job_dir": str(job_dir),
                "theta_A": str(job_dir / "ckpts" / "theta_A.pt"),
                "phase_b_final": str(job_dir / "ckpts" / "phase_b_final.pt"),
                "n_patches": len(rows),
            },
        )
        base = next(r for r in rows if r["patch"] == "empty")
        full = next(r for r in rows if r["patch"] == "full_theta_A")
        best = max(rows, key=lambda r: float(r["A_acc"]))
        # best single-group among named singles (exclude combos / full / empty)
        singles = [
            r
            for r in rows
            if r["patch"]
            not in {"empty", "full_theta_A", "all_mlp", "all_attention"}
            and "+" not in r["patch"]
        ]
        best_single = max(singles, key=lambda r: float(r["A_acc"])) if singles else None
        summary.append(
            {
                "case": case["id"],
                "role": case["role"],
                "A_base": base["A_acc"],
                "B_base": base["B_acc"],
                "A_full_theta_A": full["A_acc"],
                "B_full_theta_A": full["B_acc"],
                "best_patch": best["patch"],
                "best_A": best["A_acc"],
                "best_B": best["B_acc"],
                "best_single": None if best_single is None else best_single["patch"],
                "best_single_A": None if best_single is None else best_single["A_acc"],
                "best_single_B": None if best_single is None else best_single["B_acc"],
                "A_recovery_best": best["A_acc"] - base["A_acc"],
                "A_recovery_best_single": (
                    None
                    if best_single is None
                    else best_single["A_acc"] - base["A_acc"]
                ),
            }
        )
        # bar chart of A recovery
        _style()
        order = sorted(rows, key=lambda r: float(r["A_acc"]), reverse=True)
        fig, ax = plt.subplots(figsize=(8, 0.35 * len(order) + 1.2))
        ys = np.arange(len(order))
        ax.barh(ys, [r["A_acc"] for r in order], color="#1f4e79", alpha=0.85, label="A")
        ax.barh(
            ys,
            [r["B_acc"] for r in order],
            left=0,
            height=0.35,
            color="#2a9d8f",
            alpha=0.55,
            label="B",
        )
        # cleaner: twin bars
        plt.close(fig)
        fig, ax = plt.subplots(figsize=(9, 0.38 * len(order) + 1.4))
        names = [r["patch"] for r in order]
        a_acc = [float(r["A_acc"]) for r in order]
        b_acc = [float(r["B_acc"]) for r in order]
        y = np.arange(len(order))
        ax.barh(y - 0.18, a_acc, height=0.35, color="#1f4e79", label="A acc")
        ax.barh(y + 0.18, b_acc, height=0.35, color="#2a9d8f", label="B acc")
        ax.axvline(base["A_acc"], color="#1f4e79", ls=":", lw=1, alpha=0.7)
        ax.axvline(base["B_acc"], color="#2a9d8f", ls=":", lw=1, alpha=0.7)
        ax.set_yticks(y, names, fontsize=8)
        ax.set_xlim(0, 1.02)
        ax.set_xlabel("accuracy")
        ax.set_title(f"{case['id']}: restore groups from theta_A into phase_b_final")
        ax.legend(loc="lower right")
        fig.tight_layout()
        fig.savefig(case_dir / "restore_bar.png", bbox_inches="tight")
        plt.close(fig)
        print(
            f"  base A={base['A_acc']:.3f}  best_single={best_single['patch'] if best_single else None}"
            f" A={best_single['A_acc'] if best_single else float('nan'):.3f}"
            f"  full A={full['A_acc']:.3f}"
        )

    _write_csv(restore_root / "restore_summary.csv", summary)


def _group_name(key: str) -> str | None:
    if key.endswith(SKIP_SUFFIXES):
        return None
    if key == "tok_emb.weight":
        return "tok_emb"
    if key == "pos_emb.weight":
        return "pos_emb"
    if key.startswith("ln_f."):
        return "ln_f"
    if key.startswith("head."):
        return "head"
    import re

    m = re.match(r"blocks\.(\d+)\.(attn|mlp|ln1|ln2)\.", key)
    if m:
        layer, kind = m.group(1), m.group(2)
        if kind in {"ln1", "ln2"}:
            return f"L{layer}.ln"
        return f"L{layer}.{kind}"
    return f"other:{key}"


def _weight_group_deltas(theta_a: dict[str, torch.Tensor], theta_b: dict[str, torch.Tensor]) -> list[dict[str, Any]]:
    buckets: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = {}
    for key, a in theta_a.items():
        if key not in theta_b:
            continue
        g = _group_name(key)
        if g is None:
            continue
        b = theta_b[key]
        if not torch.is_floating_point(a):
            continue
        buckets.setdefault(g, []).append((a.float().reshape(-1), b.float().reshape(-1)))
    rows = []
    for g, chunks in sorted(buckets.items()):
        a = torch.cat([c[0] for c in chunks])
        b = torch.cat([c[1] for c in chunks])
        d = b - a
        l2_a = float(torch.linalg.vector_norm(a).item())
        l2_d = float(torch.linalg.vector_norm(d).item())
        cos = float(torch.nn.functional.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0)).item())
        rows.append(
            {
                "group": g,
                "rel_l2": l2_d / (l2_a + 1e-12),
                "l2_delta": l2_d,
                "cosine": cos,
                "numel": int(a.numel()),
            }
        )
    return rows


def experiment_circuit(
    cases: list[dict[str, Any]],
    formal_root: Path,
    mech_root: Path,
    out: Path,
) -> None:
    circ = out / "03_circuit_change"
    circ.mkdir(parents=True, exist_ok=True)
    metric_rows: list[dict[str, Any]] = []
    weight_rows: list[dict[str, Any]] = []
    _style()

    for case in cases:
        theta = _load_report(mech_root, case["mech_theta"])
        final_a = _load_report(mech_root, case["mech_finalA"])
        by_th = {int(op["latent_id"]): op for op in theta["per_op"]}
        by_fa = {int(op["latent_id"]): op for op in final_a["per_op"]}
        for lid in sorted(set(by_th) & set(by_fa)):
            s0 = by_th[lid]["summary"]
            s1 = by_fa[lid]["summary"]
            row: dict[str, Any] = {
                "case": case["id"],
                "role": case["role"],
                "latent_id": lid,
                "modulus": by_th[lid]["modulus"],
            }
            for k in MECH_KEYS:
                v0, v1 = s0.get(k), s1.get(k)
                row[f"{k}_theta"] = v0
                row[f"{k}_final"] = v1
                try:
                    row[f"d_{k}"] = float(v1) - float(v0)
                except (TypeError, ValueError):
                    row[f"d_{k}"] = None
            metric_rows.append(row)

        job_dir = formal_root / case["rel"]
        ta = load_theta(job_dir / "ckpts" / "theta_A.pt")
        tb = load_theta(job_dir / "ckpts" / "phase_b_final.pt")
        for g in _weight_group_deltas(ta, tb):
            weight_rows.append({"case": case["id"], "role": case["role"], **g})

        # per-case weight bar
        groups = _weight_group_deltas(ta, tb)
        fig, ax = plt.subplots(figsize=(7, 3.2))
        names = [g["group"] for g in groups]
        vals = [g["rel_l2"] for g in groups]
        ax.bar(range(len(names)), vals, color="#1f4e79")
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("rel L2 (final − theta_A)")
        ax.set_title(f"{case['id']} weight change by group")
        fig.tight_layout()
        fig.savefig(circ / f"{case['id']}_weight_rel_l2.png", bbox_inches="tight")
        plt.close(fig)

    _write_csv(circ / "mech_metric_delta.csv", metric_rows)
    _write_csv(circ / "weight_group_delta.csv", weight_rows)

    # Focus plot for replay_keep_A: probe / routing / compose shifts
    focus = [r for r in metric_rows if r["role"] == "replay_keep_A"]
    if focus:
        fig, axes = plt.subplots(1, 3, figsize=(10, 3.2))
        mods = [f"p{r['modulus']}" for r in focus]
        for ax, key, title in zip(
            axes,
            ["probe_sum_acc_L1", "L0_own_minus_other", "baseline_acc"],
            ["probe_sum L1", "L0 own−other", "A accuracy"],
        ):
            th = [float(r[f"{key}_theta"]) for r in focus]
            fi = [float(r[f"{key}_final"]) for r in focus]
            x = np.arange(len(mods))
            ax.bar(x - 0.18, th, width=0.35, label="theta_A", color="#1f4e79")
            ax.bar(x + 0.18, fi, width=0.35, label="final", color="#c45c26")
            ax.set_xticks(x, mods)
            ax.set_title(title)
            ax.set_ylim(min(0, min(th + fi) - 0.05), max(1.0, max(th + fi) + 0.05) if key != "L0_own_minus_other" else None)
        axes[0].legend(fontsize=8)
        fig.suptitle("Rp_replay_m1: circuit metrics theta_A vs phase_b_final (task A)", y=1.03)
        fig.tight_layout()
        fig.savefig(circ / "Rp_replay_m1_circuit_bars.png", bbox_inches="tight")
        plt.close(fig)

    # Cross-case: mean |Δ probe_sum_L1| and mean rel_l2 of mlp/attn
    rollup = []
    for case in cases:
        mrows = [r for r in metric_rows if r["case"] == case["id"]]
        wrows = [r for r in weight_rows if r["case"] == case["id"]]
        def _mean_abs(key: str) -> float:
            vals = [abs(float(r[key])) for r in mrows if r.get(key) is not None]
            return float(np.mean(vals)) if vals else float("nan")

        def _w(group: str) -> float:
            hit = [r for r in wrows if r["group"] == group]
            return float(hit[0]["rel_l2"]) if hit else float("nan")

        rollup.append(
            {
                "case": case["id"],
                "role": case["role"],
                "mean_abs_d_probe_L1": _mean_abs("d_probe_sum_acc_L1"),
                "mean_abs_d_L0": _mean_abs("d_L0_own_minus_other"),
                "mean_d_acc": float(np.mean([float(r["d_baseline_acc"]) for r in mrows])),
                "rel_l2_L0_mlp": _w("L0.mlp"),
                "rel_l2_L1_mlp": _w("L1.mlp"),
                "rel_l2_L0_attn": _w("L0.attn"),
                "rel_l2_L1_attn": _w("L1.attn"),
                "rel_l2_head": _w("head"),
                "rel_l2_tok_emb": _w("tok_emb"),
                "compose_layer_changed": int(
                    any(
                        r.get("compose_layer_guess_theta") != r.get("compose_layer_guess_final")
                        for r in mrows
                    )
                ),
            }
        )
    _write_csv(circ / "circuit_rollup.csv", rollup)


def write_readme(out: Path, stamp: str) -> None:
    text = f"""# Mechanism follow-up ({stamp})

Representative checkpoints from next80_formal mech suite.

## 1. Fourier energy spectra (`01_fourier/`)
- Per-case plots of digit-embedding and unembed energy vs frequency.
- `spectrum_similarity.csv`: cosine (skip DC) between theta_A and final A/B spectra when moduli match.
- Heatmap: digit-emb cosine theta_A vs final A.

## 2. Component restore (`02_component_restore/`)
- For forget / partial-death cases: copy named parameter groups from `theta_A` into `phase_b_final`, evaluate A and B test accuracy (no training).
- See `restore_summary.csv` and per-case `param_patch.csv`.

## 3. Circuit change (`03_circuit_change/`)
- Mech metric deltas (probes, L0 routing, compose layer, Fourier coupling) from existing reports.
- Weight-group relative L2 between theta_A and phase_b_final.
- Focus figure for replay-success `Rp_replay_m1`.
"""
    (out / "README.md").write_text(text, encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            clean = {
                k: ("" if v is None else v)
                for k, v in row.items()
                if k != "energy_by_freq"
            }
            writer.writerow(clean)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-root", type=Path, default=DEFAULT_FORMAL)
    parser.add_argument("--mech-root", type=Path, default=DEFAULT_MECH)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument(
        "--skip-restore",
        action="store_true",
        help="Skip GPU component-restore eval (Fourier + circuit only).",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        default=None,
        help="Optional subset of case ids.",
    )
    args = parser.parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out = args.out or (args.formal_root / f"mech_followup_{stamp}")
    out.mkdir(parents=True, exist_ok=True)
    cases = CASES
    if args.cases:
        wanted = set(args.cases)
        cases = [c for c in CASES if c["id"] in wanted]
    write_json(
        out / "meta.json",
        {
            "stamp": stamp,
            "formal_root": str(args.formal_root),
            "mech_root": str(args.mech_root),
            "cases": [c["id"] for c in cases],
            "skip_restore": bool(args.skip_restore),
        },
    )
    print(f"[followup] out={out}")
    print("[followup] 1/3 Fourier spectra from mech_suite reports ...")
    experiment_fourier(cases, args.mech_root, out)
    if not args.skip_restore:
        print("[followup] 2/3 Component restore (GPU eval) ...")
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
        experiment_restore(cases, args.formal_root, out, device=device)
    else:
        print("[followup] 2/3 Component restore SKIPPED")
    print("[followup] 3/3 Circuit / weight change ...")
    experiment_circuit(cases, args.formal_root, args.mech_root, out)
    write_readme(out, stamp)
    print(f"[followup] DONE → {out}")


if __name__ == "__main__":
    main()
