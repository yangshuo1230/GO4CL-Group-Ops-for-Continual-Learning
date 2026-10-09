#!/usr/bin/env python3
"""Flatten next80_formal mech_suite reports into one markdown + CSV."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True)
    args = ap.parse_args()
    root = Path(args.root)

    rows: list[dict] = []
    for sub in sorted(root.iterdir()):
        if not sub.is_dir() or sub.name in {"logs"}:
            continue
        report = sub / "phase1_mechanisms_report.json"
        if not report.is_file():
            # also accept any *mechanisms*report*
            cands = list(sub.glob("*mechanisms*report*.json")) + list(
                sub.glob("phase1_mechanisms_report.json")
            )
            if not cands:
                continue
            report = cands[0]
        data = json.loads(report.read_text())
        cfg = data.get("config", {})
        comp = data.get("comparison", {})
        for op in data.get("per_op", []):
            s = op.get("summary", {})
            emb = op.get("digit_emb_fourier") or op.get("emb_fourier") or {}
            rows.append(
                {
                    "label": sub.name,
                    "ckpt_kind": cfg.get("ckpt_kind"),
                    "task": cfg.get("task"),
                    "job_dir": Path(str(cfg.get("job_dir", ""))).name,
                    "latent_id": op.get("latent_id", s.get("latent_id")),
                    "modulus": op.get("modulus", s.get("modulus")),
                    "slot": op.get("slot", s.get("slot")),
                    "baseline_acc": s.get("baseline_acc"),
                    "compose_layer_guess": s.get("compose_layer_guess"),
                    "L0_own_minus_other": s.get("L0_own_minus_other"),
                    "top1_imp_delta": s.get("top1_imp_delta"),
                    "emb_top_k": emb.get("top_k") or emb.get("peak_k"),
                    "emb_top_frac": emb.get("top_frac") or emb.get("peak_frac"),
                    "verdict": comp.get("verdict"),
                    "fourier_selective": comp.get("fourier_ablation_selective"),
                }
            )

    csv_path = root / "mech_suite_summary.csv"
    if rows:
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)

    lines = [
        "# next80_formal mechanism suite",
        "",
        f"Reports under `{root}` ({len(rows)} op-rows).",
        "",
        "| label | ckpt | task | p | slot | test | composeL | L0Δown | top1Δ | emb_k | emb_frac |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        def fmt(x, nd=3):
            if x is None:
                return ""
            if isinstance(x, float):
                return f"{x:.{nd}f}"
            return str(x)

        lines.append(
            f"| {r['label']} | {r['ckpt_kind']} | {r['task']} | {r['modulus']} | "
            f"{r['slot']} | {fmt(r['baseline_acc'])} | {fmt(r['compose_layer_guess'],0)} | "
            f"{fmt(r['L0_own_minus_other'])} | {fmt(r['top1_imp_delta'])} | "
            f"{fmt(r['emb_top_k'],0)} | {fmt(r['emb_top_frac'])} |"
        )
    print("\n".join(lines))
    (root / "SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nwrote {csv_path}")


if __name__ == "__main__":
    main()
