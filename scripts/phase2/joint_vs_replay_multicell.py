#!/usr/bin/env python3
"""Replicate joint vs sequential+replay mechanism contrasts across overlap cells.

For each matched (cell, ts, ms) with both protocols high-acc:
  - probe_sum at L0/L1/L2 (task A and B)
  - query-resid Fourier spectrum cosine A↔B at L0/L1/L2
  - joint↔replay spectrum cosine at L0/L1/L2 (task A)

Does not overwrite training artifacts.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from go4cl.analysis.cache import collect_batches, operand_residues
from go4cl.analysis.context import load_analysis_context
from go4cl.analysis.fourier import analyze_query_resid_fourier, energy_cosine
from go4cl.data.dataset import make_loader
from go4cl.utils.checkpoint import write_json

ROOT = Path(__file__).resolve().parents[2]
LAYERS = (0, 1, 2)


def _discover_matched(
    matrix_root: Path,
) -> list[dict[str, Any]]:
    pairs: dict[tuple, dict[str, dict[str, Any]]] = defaultdict(dict)
    for stamp in sorted(matrix_root.glob("*")):
        runs = stamp / "runs"
        if not runs.is_dir():
            continue
        for jr in runs.glob("*/job_result.json"):
            cfgp = jr.parent / "config_resolved.json"
            if not cfgp.is_file():
                continue
            cfg = json.loads(cfgp.read_text(encoding="utf-8"))
            proto = str(cfg.get("protocol", ""))
            if proto not in {"joint", "sequential_ab_replay"}:
                continue
            name = jr.parent.name
            m = re.search(r"(s[\d.]+_o[\d.]+_m[\d.]+)_", name)
            if not m:
                continue
            cell = m.group(1)
            key = (cell, int(cfg.get("task_seed", 0)), int(cfg.get("model_seed", 0)))
            met = json.loads(jr.read_text(encoding="utf-8")).get("metrics") or {}
            a = met.get("A_test_acc")
            b = met.get("B_test_acc")
            ck = jr.parent / "ckpts"
            if proto == "joint" and not (ck / "final.pt").is_file():
                continue
            if proto == "sequential_ab_replay" and not (ck / "phase_b_final.pt").is_file():
                continue
            pairs[key][proto] = {
                "path": jr.parent,
                "A": float(a) if a is not None else float("nan"),
                "B": float(b) if b is not None else float("nan"),
                "stamp": stamp.name,
            }
    out = []
    for (cell, ts, ms), d in sorted(pairs.items()):
        if "joint" not in d or "sequential_ab_replay" not in d:
            continue
        j, r = d["joint"], d["sequential_ab_replay"]
        out.append(
            {
                "cell": cell,
                "task_seed": ts,
                "model_seed": ms,
                "joint_dir": j["path"],
                "replay_dir": r["path"],
                "joint_A": j["A"],
                "joint_B": j["B"],
                "replay_A": r["A"],
                "replay_B": r["B"],
            }
        )
    return out


def _spectra_and_probes(
    job_dir: Path,
    ckpt_kind: str,
    task: str,
    device: torch.device,
    *,
    probe_steps: int = 300,
) -> dict[int, dict[str, Any]]:
    """Per modulus: layer spectra + linear probe_sum acc at each layer."""
    from go4cl.analysis.probes import run_operand_probes

    ctx = load_analysis_context(job_dir, ckpt_kind=ckpt_kind, device=device, task=task)
    model = ctx.model
    # Disk train split is often empty in these stamps; fall back via operation_dataset
    # (analysis examples) and prefer val residuals for fitting probes when needed.
    train_split = "train"
    if len(ctx.disk_dataset("train")) == 0:
        train_split = "val"
    out: dict[int, dict[str, Any]] = {}
    for op in ctx.operations:
        train_ds = ctx.operation_dataset(op, train_split)  # type: ignore[arg-type]
        test_ds = ctx.operation_dataset(op, "test")
        if len(train_ds) == 0:
            train_ds = ctx.operation_dataset(op, "val")
        train_loader = make_loader(train_ds, batch_size=256, shuffle=False)
        test_loader = make_loader(test_ds, batch_size=256, shuffle=False)
        train_cache = collect_batches(model, train_loader, device=device, max_batches=None)
        test_cache = collect_batches(model, test_loader, device=device, max_batches=None)
        _, _, sum_te = operand_residues(
            test_cache.tokens, i=op.operand_i, j=op.operand_j, modulus=op.modulus
        )
        layer_block: dict[str, Any] = {
            "modulus": int(op.modulus),
            "latent_id": int(op.latent_id),
        }
        for li in LAYERS:
            fq = analyze_query_resid_fourier(
                test_cache.resid_post[li][:, -1, :],
                sum_te,
                modulus=op.modulus,
            )
            layer_block[f"L{li}"] = {
                "energy_by_freq": fq["energy_by_freq"],
                "top_freq": fq["top_freq"],
            }
            pr = run_operand_probes(
                train_resid=train_cache.resid_post[li][:, -1, :],
                train_tokens=train_cache.tokens,
                eval_resids={"test": test_cache.resid_post[li][:, -1, :]},
                eval_tokens={"test": test_cache.tokens},
                operand_i=op.operand_i,
                operand_j=op.operand_j,
                modulus=op.modulus,
                steps=probe_steps,
                seed=0,
            )
            layer_block[f"probe_L{li}"] = float(pr["probe_sum_acc"] or 0.0)
        out[int(op.modulus)] = layer_block
    return out


def run_cell(
    cell_rec: dict[str, Any],
    device: torch.device,
    *,
    probe_steps: int,
) -> dict[str, Any]:
    print(f"[cell] {cell_rec['cell']} ts={cell_rec['task_seed']} ms={cell_rec['model_seed']}")
    jA = _spectra_and_probes(cell_rec["joint_dir"], "final", "A", device, probe_steps=probe_steps)
    jB = _spectra_and_probes(cell_rec["joint_dir"], "final", "B", device, probe_steps=probe_steps)
    rA = _spectra_and_probes(
        cell_rec["replay_dir"], "phase_b_final", "A", device, probe_steps=probe_steps
    )
    rB = _spectra_and_probes(
        cell_rec["replay_dir"], "phase_b_final", "B", device, probe_steps=probe_steps
    )
    shared = sorted(set(jA) & set(jB) & set(rA) & set(rB))
    mods_a = sorted(set(jA) & set(rA))
    mods_b = sorted(set(jB) & set(rB))
    per_mod: list[dict[str, Any]] = []

    def _base(m: int) -> dict[str, Any]:
        return {
            "cell": cell_rec["cell"],
            "task_seed": cell_rec["task_seed"],
            "model_seed": cell_rec["model_seed"],
            "modulus": m,
            "joint_A_acc": cell_rec["joint_A"],
            "joint_B_acc": cell_rec["joint_B"],
            "replay_A_acc": cell_rec["replay_A"],
            "replay_B_acc": cell_rec["replay_B"],
        }

    # Task-A rows (joint vs replay probes / spectra); always defined when moduli match across protocols
    for m in mods_a:
        row = _base(m)
        row["scope"] = "taskA"
        for li in LAYERS:
            row[f"joint_probeA_L{li}"] = jA[m][f"probe_L{li}"]
            row[f"replay_probeA_L{li}"] = rA[m][f"probe_L{li}"]
            row[f"JR_A_cos_L{li}"] = energy_cosine(
                jA[m][f"L{li}"]["energy_by_freq"],
                rA[m][f"L{li}"]["energy_by_freq"],
            )
            if m in jB and m in rB:
                row[f"joint_probeB_L{li}"] = jB[m][f"probe_L{li}"]
                row[f"replay_probeB_L{li}"] = rB[m][f"probe_L{li}"]
                row[f"joint_AB_cos_L{li}"] = energy_cosine(
                    jA[m][f"L{li}"]["energy_by_freq"],
                    jB[m][f"L{li}"]["energy_by_freq"],
                )
                row[f"replay_AB_cos_L{li}"] = energy_cosine(
                    rA[m][f"L{li}"]["energy_by_freq"],
                    rB[m][f"L{li}"]["energy_by_freq"],
                )
                row[f"JR_B_cos_L{li}"] = energy_cosine(
                    jB[m][f"L{li}"]["energy_by_freq"],
                    rB[m][f"L{li}"]["energy_by_freq"],
                )
        per_mod.append(row)

    # Task-B-only moduli (rho_mod=0): still record probes
    for m in mods_b:
        if m in mods_a:
            continue
        row = _base(m)
        row["scope"] = "taskB_only"
        for li in LAYERS:
            row[f"joint_probeB_L{li}"] = jB[m][f"probe_L{li}"]
            row[f"replay_probeB_L{li}"] = rB[m][f"probe_L{li}"]
            row[f"JR_B_cos_L{li}"] = energy_cosine(
                jB[m][f"L{li}"]["energy_by_freq"],
                rB[m][f"L{li}"]["energy_by_freq"],
            )
        per_mod.append(row)

    def _mean(key: str, rows: list[dict[str, Any]] | None = None) -> float:
        src = rows if rows is not None else per_mod
        vals = [float(r[key]) for r in src if key in r and r[key] is not None]
        return float(np.mean(vals)) if vals else float("nan")

    shared_rows = [r for r in per_mod if r.get("scope") == "taskA" and "joint_AB_cos_L1" in r]
    a_rows = [r for r in per_mod if r.get("scope") == "taskA"]
    roll = {
        "cell": cell_rec["cell"],
        "task_seed": cell_rec["task_seed"],
        "model_seed": cell_rec["model_seed"],
        "n_shared_mod": len(shared),
        "n_A_mod": len(mods_a),
        "n_B_mod": len(mods_b),
        "joint_A_acc": cell_rec["joint_A"],
        "replay_A_acc": cell_rec["replay_A"],
        "mean_joint_probeA_L1": _mean("joint_probeA_L1", a_rows),
        "mean_replay_probeA_L1": _mean("replay_probeA_L1", a_rows),
        "mean_joint_probeB_L1": _mean("joint_probeB_L1"),
        "mean_replay_probeB_L1": _mean("replay_probeB_L1"),
        "mean_joint_AB_cos_L0": _mean("joint_AB_cos_L0", shared_rows),
        "mean_joint_AB_cos_L1": _mean("joint_AB_cos_L1", shared_rows),
        "mean_joint_AB_cos_L2": _mean("joint_AB_cos_L2", shared_rows),
        "mean_replay_AB_cos_L0": _mean("replay_AB_cos_L0", shared_rows),
        "mean_replay_AB_cos_L1": _mean("replay_AB_cos_L1", shared_rows),
        "mean_replay_AB_cos_L2": _mean("replay_AB_cos_L2", shared_rows),
        "mean_JR_A_cos_L1": _mean("JR_A_cos_L1", a_rows),
        "frac_replay_AB_L1_lt_0.5": (
            float(np.mean([r["replay_AB_cos_L1"] < 0.5 for r in shared_rows]))
            if shared_rows
            else float("nan")
        ),
        "pattern_replay_L1_split_L2_rejoin": int(
            bool(shared_rows)
            and (_mean("replay_AB_cos_L1", shared_rows) < _mean("replay_AB_cos_L2", shared_rows) - 0.15)
            and _mean("replay_AB_cos_L2", shared_rows) > 0.85
        ),
        "pattern_joint_probe_gt_replay": int(
            _mean("joint_probeA_L1", a_rows) > _mean("replay_probeA_L1", a_rows) + 0.2
        ),
    }
    return {"per_mod": per_mod, "rollup": roll}


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for k in row:
            if k not in fields:
                fields.append(k)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def _plot(rollup: list[dict[str, Any]], out: Path) -> None:
    mpl.rcParams.update(
        {
            "font.size": 9,
            "figure.dpi": 140,
            "savefig.dpi": 220,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )
    cells = [r["cell"] for r in rollup]
    x = np.arange(len(cells))
    fig, axes = plt.subplots(1, 2, figsize=(max(8, 0.55 * len(cells) + 3), 3.6))
    ax = axes[0]
    ax.bar(x - 0.2, [r["mean_joint_probeA_L1"] for r in rollup], 0.4, label="joint A", color="#1f4e79")
    ax.bar(x + 0.2, [r["mean_replay_probeA_L1"] for r in rollup], 0.4, label="replay A", color="#c45c26")
    ax.set_xticks(x, cells, rotation=55, ha="right")
    ax.set_ylabel("mean probe_sum L1")
    ax.set_ylim(0, 1.05)
    ax.set_title("Task A: L1 sum probe")
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.bar(x - 0.2, [r["mean_joint_AB_cos_L1"] for r in rollup], 0.4, label="joint A↔B L1", color="#1f4e79")
    ax.bar(x + 0.2, [r["mean_replay_AB_cos_L1"] for r in rollup], 0.4, label="replay A↔B L1", color="#c45c26")
    ax.set_xticks(x, cells, rotation=55, ha="right")
    ax.set_ylabel("mean A↔B spectrum cos @ L1")
    ax.set_ylim(0, 1.05)
    ax.set_title("A↔B activation spectrum @ L1")
    ax.legend(fontsize=8)
    fig.suptitle("Joint vs replay mechanism contrast across cells", y=1.02)
    fig.tight_layout()
    fig.savefig(out / "multicell_probe_and_AB_L1.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(max(8, 0.55 * len(cells) + 3), 3.4))
    ax.plot(x, [r["mean_replay_AB_cos_L0"] if "mean_replay_AB_cos_L0" in r else np.nan for r in rollup], "o-", label="replay A↔B L0")
    # recompute L0 means into rollup if missing — plot from stored if we add them
    ax.plot(x, [r["mean_replay_AB_cos_L1"] for r in rollup], "s-", label="replay A↔B L1")
    ax.plot(x, [r["mean_replay_AB_cos_L2"] for r in rollup], "^-", label="replay A↔B L2")
    ax.set_xticks(x, cells, rotation=55, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("mean A↔B spectrum cos")
    ax.set_title("Replay: does L1 split then L2 rejoin?")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / "multicell_replay_AB_by_layer.png", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix-root", type=Path, default=ROOT / "runs/phase2/relation_matrix")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/phase2/joint_vs_replay_multicell_20261009")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--min-acc", type=float, default=0.9)
    ap.add_argument("--probe-steps", type=int, default=250)
    ap.add_argument(
        "--cells",
        nargs="*",
        default=None,
        help="Optional cell ids like s0_o0_m1 s0.5_o0.5_m1",
    )
    ap.add_argument("--max-cells", type=int, default=12)
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    matched = _discover_matched(args.matrix_root)
    matched = [
        m
        for m in matched
        if m["joint_A"] >= args.min_acc
        and m["joint_B"] >= args.min_acc
        and m["replay_A"] >= args.min_acc
        and m["replay_B"] >= args.min_acc
    ]
    if args.cells:
        wanted = set(args.cells)
        matched = [m for m in matched if m["cell"] in wanted]
    # diversify by rho_mod then take max_cells
    def mod_key(c: str) -> float:
        m = re.search(r"_m([\d.]+)$", c)
        return float(m.group(1)) if m else 0.0

    matched = sorted(matched, key=lambda m: (mod_key(m["cell"]), m["cell"]))
    # prefer spreading: pick unique-ish cells
    if len(matched) > args.max_cells:
        # take evenly spaced
        idx = np.linspace(0, len(matched) - 1, args.max_cells).round().astype(int)
        matched = [matched[i] for i in sorted(set(idx))]

    args.out.mkdir(parents=True, exist_ok=True)
    write_json(
        args.out / "meta.json",
        {
            "n_cells": len(matched),
            "cells": [m["cell"] for m in matched],
            "min_acc": args.min_acc,
            "probe_steps": args.probe_steps,
            "device": str(device),
        },
    )
    print("cells:", [m["cell"] for m in matched])

    all_mod: list[dict[str, Any]] = []
    rollup: list[dict[str, Any]] = []
    for cell_rec in matched:
        # patch rollup to include L0 mean — compute inside run_cell extension
        try:
            result = run_cell(cell_rec, device, probe_steps=args.probe_steps)
        except Exception as e:
            print(f"  FAIL {cell_rec['cell']}: {e}")
            continue
        all_mod.extend(result["per_mod"])
        rollup.append(result["rollup"])
        r = result["rollup"]
        print(
            f"  shared={r['n_shared_mod']}  "
            f"probeA L1 J/R={r['mean_joint_probeA_L1']:.3f}/{r['mean_replay_probeA_L1']:.3f}  "
            f"AB cos L1 J/R={r['mean_joint_AB_cos_L1']:.3f}/{r['mean_replay_AB_cos_L1']:.3f}  "
            f"R L2={r['mean_replay_AB_cos_L2']:.3f}  "
            f"flags probeΔ={r['pattern_joint_probe_gt_replay']} split={r['pattern_replay_L1_split_L2_rejoin']}"
        )

    _write_csv(args.out / "per_modulus.csv", all_mod)
    _write_csv(args.out / "cell_rollup.csv", rollup)
    if rollup:
        _plot(rollup, args.out)
        n = len(rollup)
        summary = {
            "n_cells": n,
            "frac_joint_probeA_gt_replay_plus0.2": float(
                np.mean([r["pattern_joint_probe_gt_replay"] for r in rollup])
            ),
            "frac_replay_L1_split_L2_rejoin": float(
                np.mean([r["pattern_replay_L1_split_L2_rejoin"] for r in rollup])
            ),
            "mean_joint_probeA_L1": float(np.mean([r["mean_joint_probeA_L1"] for r in rollup])),
            "mean_replay_probeA_L1": float(np.mean([r["mean_replay_probeA_L1"] for r in rollup])),
            "mean_joint_AB_cos_L1": float(np.mean([r["mean_joint_AB_cos_L1"] for r in rollup])),
            "mean_replay_AB_cos_L1": float(np.mean([r["mean_replay_AB_cos_L1"] for r in rollup])),
            "mean_replay_AB_cos_L2": float(np.mean([r["mean_replay_AB_cos_L2"] for r in rollup])),
        }
        write_json(args.out / "summary.json", summary)
        md = [
            "# Joint vs Replay multicell replication",
            "",
            f"Cells (A,B≥{args.min_acc}): {n}",
            "",
            f"- Joint mean probeA L1: **{summary['mean_joint_probeA_L1']:.3f}**",
            f"- Replay mean probeA L1: **{summary['mean_replay_probeA_L1']:.3f}**",
            f"- Cells with joint probe ≫ replay (+0.2): "
            f"**{summary['frac_joint_probeA_gt_replay_plus0.2']:.0%}**",
            f"- Replay A↔B L1 cos mean: **{summary['mean_replay_AB_cos_L1']:.3f}** "
            f"(joint **{summary['mean_joint_AB_cos_L1']:.3f}**)",
            f"- Replay A↔B L2 cos mean: **{summary['mean_replay_AB_cos_L2']:.3f}**",
            f"- Cells with L1-split then L2-rejoin pattern: "
            f"**{summary['frac_replay_L1_split_L2_rejoin']:.0%}**",
            "",
            "See `cell_rollup.csv`, `per_modulus.csv`.",
        ]
        (args.out / "CONCLUSIONS.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"[done] {args.out}")


if __name__ == "__main__":
    main()
