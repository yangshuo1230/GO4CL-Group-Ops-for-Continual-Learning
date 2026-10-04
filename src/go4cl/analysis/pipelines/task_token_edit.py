#!/usr/bin/env python3
"""Edit the TASK token on multi-op eval contexts; measure per-op accuracy.

Phase-1 packed multi-op trains with TASK_A only. This probes whether flipping
to TASK_B (or other controls) changes behavior.

Conditions (token at position SEQ_LEN_OPERANDS):
  - original: keep as-is (TASK_A)
  - task_b: replace with TOKEN_TASK_B
  - task_a: force TOKEN_TASK_A (sanity)
  - swap_to_digit0: replace with digit token 0 (OOD control)
  - random_special: replace with a random unused vocab id if available, else 63

Usage:
  uv run python scripts/phase1/task_token_edit.py \\
    --job-dir runs/phase1/multi_op/.../runs/<job> \\
    --out runs/phase1/mechanisms/<stamp>/<tag>/task_token_edit \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from go4cl.analysis.context import filter_by_operation, load_analysis_context
from go4cl.analysis.reporting import write_csv_rows, write_json_report, write_markdown
from go4cl.constants import (
    SEQ_LEN_OPERANDS,
    TOKEN_TASK_A,
    TOKEN_TASK_B,
)
from go4cl.data.dataset import ModularAdditionDataset, make_loader


def _edit_task(tokens: torch.Tensor, new_id: int | None) -> torch.Tensor:
    out = tokens.clone()
    if new_id is not None:
        out[:, SEQ_LEN_OPERANDS] = int(new_id)
    return out


def run(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str,
    device: torch.device,
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

    conditions: list[tuple[str, int | None]] = [
        ("original", None),
        ("task_a", TOKEN_TASK_A),
        ("task_b", TOKEN_TASK_B),
        ("digit_0", 0),
        ("digit_63", 63),
    ]

    rows: list[dict[str, Any]] = []
    for op in ops:
        key = f"lat{op.latent_id}/slot{op.slot}/p{op.modulus}"
        if len(test_full) == 0:
            examples = analysis_builder(
                split="test", target_latent_ids=[op.latent_id]
            )
            ds = ModularAdditionDataset.from_examples(examples, task_id=0)
        else:
            ds = filter_by_operation(
                test_full, latent_id=op.latent_id, slot=op.slot
            )
        loader = make_loader(ds, batch_size=256, shuffle=False)
        # gather all tokens/labels (optionally capped)
        toks_list = []
        labs_list = []
        n_seen = 0
        for batch in loader:
            toks_list.append(batch["tokens"])
            labs_list.append(batch["labels"])
            n_seen += batch["tokens"].shape[0]
            if max_batches is not None and len(toks_list) >= max_batches:
                break
        tokens = torch.cat(toks_list, dim=0)
        labels = torch.cat(labs_list, dim=0)
        base_task = int(tokens[0, SEQ_LEN_OPERANDS].item())
        print(f"  {key} n={tokens.shape[0]} base_task_tok={base_task}", flush=True)

        for cond_name, new_id in conditions:
            edited = _edit_task(tokens, new_id)
            # chunked eval
            correct = 0
            total = 0
            bs = 256
            for i in range(0, edited.shape[0], bs):
                sl = slice(i, i + bs)
                pred = model(edited[sl].to(device))
                logits = pred["logits"] if isinstance(pred, dict) else pred
                y = labels[sl].to(device).long()
                correct += int((logits.argmax(-1) == y).sum().item())
                total += int(y.numel())
            acc = correct / max(total, 1)
            used = (
                base_task
                if new_id is None
                else int(new_id)
            )
            rows.append(
                {
                    "operation": key,
                    "latent_id": op.latent_id,
                    "slot": op.slot,
                    "modulus": op.modulus,
                    "condition": cond_name,
                    "task_token_id": used,
                    "acc": acc,
                    "n": total,
                }
            )
            print(f"    {cond_name}: acc={acc:.4f}", flush=True)

    fields = [
        "operation",
        "latent_id",
        "slot",
        "modulus",
        "condition",
        "task_token_id",
        "acc",
        "n",
    ]
    write_csv_rows(out / "task_token_edit.csv", rows, fields)

    conds = [c for c, _ in conditions]
    lines = [
        "# TASK token edit\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`) step={payload.get('step')}  ",
        f"**Data:** `{data_dir.name}`  ",
        f"**Note:** phase1 multi-op trains with TASK_A={TOKEN_TASK_A} only; "
        f"TASK_B={TOKEN_TASK_B}.\n",
        "| op | " + " | ".join(conds) + " |",
        "|----|" + "|".join(["------"] * len(conds)) + "|",
    ]
    by_op: dict[str, dict[str, float]] = {}
    for r in rows:
        by_op.setdefault(r["operation"], {})[r["condition"]] = float(r["acc"])
    for op_key, d in by_op.items():
        cells = [f"{d[c]:.3f}" for c in conds]
        lines.append(f"| {op_key} | " + " | ".join(cells) + " |")

    lines += [
        "\n## Δacc vs original\n",
        "| op | " + " | ".join(c for c in conds if c != "original") + " |",
        "|----|"
        + "|".join(["------"] * (len(conds) - 1))
        + "|",
    ]
    for op_key, d in by_op.items():
        base = d["original"]
        cells = [f"{d[c]-base:+.3f}" for c in conds if c != "original"]
        lines.append(f"| {op_key} | " + " | ".join(cells) + " |")
    lines.append("\nFiles: `task_token_edit.csv`, `task_token_edit_report.json`.\n")
    write_markdown(out / "TASK_TOKEN.md", lines)
    write_json_report(
        out / "task_token_edit_report.json",
        {
            "job_dir": str(job_dir),
            "ckpt_path": str(ckpt_path),
            "ckpt_kind": ckpt_kind,
            "ckpt_step": payload.get("step"),
            "data_dir": str(data_dir),
            "TOKEN_TASK_A": TOKEN_TASK_A,
            "TOKEN_TASK_B": TOKEN_TASK_B,
            "rows": rows,
        },
    )
    print(f"done → {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--job-dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--ckpt-kind", type=str, default="best")
    ap.add_argument("--device", type=str, default="cuda:0")
    ap.add_argument("--max-batches", type=int, default=None)
    args = ap.parse_args()
    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        max_batches=args.max_batches,
    )


if __name__ == "__main__":
    main()
