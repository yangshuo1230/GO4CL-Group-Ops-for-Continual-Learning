#!/usr/bin/env python3
"""Causal test: swap attention mass between operand key positions.

For each source op with operands (i, j), swap query-row attention on L0
(and optionally L1) between (i ↔ i′, j ↔ j′). Measure whether predictions
move from (x_i+x_j) mod p to (x_i′+x_j′) mod p.

Conditions:
  - none: identity baseline
  - to_other_op: swap with another op's operand positions (prefer same modulus)
  - to_random: swap with two unused digit positions (control)
  - permute_ops: for all_same, cycle each op's attn to the next op's slots

Usage:
  uv run python scripts/phase1/attn_pos_swap.py \\
    --job-dir runs/phase1/multi_op/.../runs/<job> \\
    --out runs/phase1/mechanisms/<stamp>/<tag>/attn_swap \\
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from go4cl.analysis.cache import collect_batches, to_jsonable  # noqa: E402
from go4cl.constants import SEQ_LEN_OPERANDS  # noqa: E402
from go4cl.data.dataset import ModularAdditionDataset, make_loader  # noqa: E402
from go4cl.data.manifest import DataManifest  # noqa: E402
from go4cl.phases.phase1.mechanisms import (  # noqa: E402
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
    raise FileNotFoundError(job_dir)


def _resolve_ckpt(job_dir: Path, kind: str) -> Path:
    for name in (f"{kind}.pt", f"a_only_{kind}.pt"):
        p = job_dir / "ckpts" / name
        if p.exists():
            return p
    raise FileNotFoundError(job_dir / "ckpts" / f"{kind}.pt")


def _write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _op_ds(op, full, *, split: str, analysis_builder):
    if len(full) == 0:
        examples = analysis_builder(split=split, target_latent_ids=[op.latent_id])
        return ModularAdditionDataset.from_examples(examples, task_id=0)
    return filter_by_operation(full, latent_id=op.latent_id, slot=op.slot)


@torch.no_grad()
def _forward_with_key_swap(
    model,
    *,
    resid_pre: torch.Tensor,
    layer_idx: int,
    swap_pairs: list[tuple[int, int]] | None,
    swap_heads: list[int] | None,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run block ``layer_idx`` with optional attn key swap, then continue."""
    block = model.blocks[layer_idx]
    pre = resid_pre.to(device)
    attn_out, att, _ = block.attn.forward_detailed(
        block.ln1(pre),
        swap_key_pairs=swap_pairs,
        swap_heads=swap_heads,
        swap_query_only=True,
    )
    mid = pre + attn_out
    post = mid + block.mlp(block.ln2(mid))
    logits = model.continue_from_layer(post, layer_idx=layer_idx)["logits"]
    return logits, att


def _acc(logits: torch.Tensor, labels: torch.Tensor) -> float:
    return float((logits.argmax(-1) == labels).float().mean().item())


def _alt_labels(tokens: torch.Tensor, i: int, j: int, modulus: int) -> torch.Tensor:
    return (tokens[:, i].long() + tokens[:, j].long()) % int(modulus)


def _pick_random_pair(rng: torch.Generator, forbidden: set[int]) -> tuple[int, int]:
    avail = [p for p in range(SEQ_LEN_OPERANDS) if p not in forbidden]
    # sample two distinct
    perm = torch.randperm(len(avail), generator=rng).tolist()
    return avail[perm[0]], avail[perm[1]]


def run(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str,
    device: torch.device,
    layers: list[int],
    max_batches: int | None,
    head_mode: str,
) -> None:
    out.mkdir(parents=True, exist_ok=True)
    ckpt_path = _resolve_ckpt(job_dir, ckpt_kind)
    data_dir = _resolve_data_dir(job_dir)
    manifest = DataManifest.load(data_dir / "manifest.json")
    ops = _ops_from_manifest(manifest)

    model, payload = load_checkpoint(ckpt_path, map_location=device)
    model.to(device).eval()
    n_heads = model.cfg.n_heads

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

    bundles: list[dict[str, Any]] = []
    for op in ops:
        key = f"lat{op.latent_id}/slot{op.slot}/p{op.modulus}"
        ds = _op_ds(op, test_full, split="test", analysis_builder=analysis_builder)
        loader = make_loader(ds, batch_size=256, shuffle=False)
        cache = collect_batches(
            model, loader, device=device, max_batches=max_batches
        )
        bundles.append({"op": op, "key": key, "cache": cache})
        print(f"  cache {key} n={len(ds)}", flush=True)

    # head sets to try
    if head_mode == "all":
        head_sets: list[tuple[str, list[int] | None]] = [("all", None)]
    elif head_mode == "each":
        head_sets = [("all", None)] + [(f"H{h}", [h]) for h in range(n_heads)]
    else:
        head_sets = [("all", None)]

    rows: list[dict[str, Any]] = []
    rng = torch.Generator(device="cpu").manual_seed(0)

    for b in bundles:
        op = b["op"]
        cache = b["cache"]
        tokens = cache.tokens
        y_orig = cache.labels.to(device).long()
        # partner: prefer same-modulus other op; else next op
        partners = [
            o
            for o in bundles
            if o["key"] != b["key"] and o["op"].modulus == op.modulus
        ]
        if not partners:
            partners = [o for o in bundles if o["key"] != b["key"]]
        partner = partners[0] if partners else None

        # random control positions
        forbidden = {op.operand_i, op.operand_j}
        if partner is not None:
            forbidden |= {partner["op"].operand_i, partner["op"].operand_j}
        r0, r1 = _pick_random_pair(rng, forbidden)

        conditions: list[tuple[str, list[tuple[int, int]] | None, int, int]] = [
            ("none", None, op.operand_i, op.operand_j),
        ]
        if partner is not None:
            po = partner["op"]
            conditions.append(
                (
                    "to_other_op",
                    [
                        (op.operand_i, po.operand_i),
                        (op.operand_j, po.operand_j),
                    ],
                    po.operand_i,
                    po.operand_j,
                )
            )
        conditions.append(
            (
                "to_random",
                [(op.operand_i, r0), (op.operand_j, r1)],
                r0,
                r1,
            )
        )

        for li in layers:
            for head_name, head_ids in head_sets:
                for cond_name, swaps, ai, aj in conditions:
                    print(
                        f"  {b['key']} L{li} heads={head_name} cond={cond_name} "
                        f"swap={swaps}",
                        flush=True,
                    )
                    logits, att = _forward_with_key_swap(
                        model,
                        resid_pre=cache.resid_pre[li],
                        layer_idx=li,
                        swap_pairs=swaps,
                        swap_heads=head_ids,
                        device=device,
                    )
                    y_alt = _alt_labels(tokens, ai, aj, op.modulus).to(device)
                    # attention mass on orig vs alt positions (query row, mean heads)
                    q = att[:, :, -1, :]  # [B,H,T]
                    if head_ids is None:
                        mass = q.mean(dim=(0, 1))
                    else:
                        mass = q[:, head_ids, :].mean(dim=(0, 1))
                    m_orig = float(
                        (mass[op.operand_i] + mass[op.operand_j]).item()
                    )
                    m_alt = float((mass[ai] + mass[aj]).item())
                    rows.append(
                        {
                            "source_operation": b["key"],
                            "source_latent_id": op.latent_id,
                            "source_modulus": op.modulus,
                            "source_operands": f"{op.operand_i},{op.operand_j}",
                            "layer": li,
                            "heads": head_name,
                            "condition": cond_name,
                            "alt_operands": f"{ai},{aj}",
                            "partner_operation": (
                                partner["key"] if partner is not None else ""
                            ),
                            "acc_orig": _acc(logits, y_orig),
                            "acc_alt": _acc(logits, y_alt),
                            "attn_mass_orig_pos": m_orig,
                            "attn_mass_alt_pos": m_alt,
                            "n": int(y_orig.shape[0]),
                        }
                    )

    fields = [
        "source_operation",
        "source_latent_id",
        "source_modulus",
        "source_operands",
        "layer",
        "heads",
        "condition",
        "alt_operands",
        "partner_operation",
        "acc_orig",
        "acc_alt",
        "attn_mass_orig_pos",
        "attn_mass_alt_pos",
        "n",
    ]
    _write_csv(out / "attn_pos_swap.csv", rows, fields)

    # markdown
    lines = [
        "# Attention key-position swap\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`) step={payload.get('step')}  ",
        f"**Data:** `{data_dir.name}`  ",
        f"**Intervention:** at resid_pre L∈{layers}, swap query-row attn mass "
        f"between key positions, recompute attn@V + MLP, continue.\n",
        "## Results (heads=all)\n",
        "| source | L | condition | alt pos | acc_orig | acc_alt | "
        "attn_orig | attn_alt |",
        "|--------|---|-----------|---------|----------|---------|----------|----------|",
    ]
    for r in rows:
        if r["heads"] != "all":
            continue
        lines.append(
            f"| {r['source_operation']} | {r['layer']} | {r['condition']} | "
            f"{r['alt_operands']} | {r['acc_orig']:.3f} | {r['acc_alt']:.3f} | "
            f"{r['attn_mass_orig_pos']:.3f} | {r['attn_mass_alt_pos']:.3f} |"
        )
    # highlight to_other_op deltas
    lines.append("\n## Redirect effect (to_other_op − none), L0 heads=all\n")
    lines.append("| source | Δacc_orig | Δacc_alt | partner |")
    lines.append("|--------|-----------|----------|---------|")
    by: dict[str, dict[str, dict[str, Any]]] = {}
    for r in rows:
        if r["heads"] != "all" or int(r["layer"]) != layers[0]:
            continue
        by.setdefault(r["source_operation"], {})[r["condition"]] = r
    for src, conds in by.items():
        if "none" not in conds or "to_other_op" not in conds:
            continue
        a, b0 = conds["to_other_op"], conds["none"]
        lines.append(
            f"| {src} | {a['acc_orig']-b0['acc_orig']:+.3f} | "
            f"{a['acc_alt']-b0['acc_alt']:+.3f} | {a['partner_operation']} |"
        )
    lines.append("\nFiles: `attn_pos_swap.csv`, `attn_swap_report.json`.\n")
    (out / "ATTN_SWAP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (out / "attn_swap_report.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "job_dir": str(job_dir),
                    "ckpt_path": str(ckpt_path),
                    "ckpt_kind": ckpt_kind,
                    "ckpt_step": payload.get("step"),
                    "data_dir": str(data_dir),
                    "layers": layers,
                    "head_mode": head_mode,
                    "rows": rows,
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
    ap.add_argument("--layers", type=int, nargs="+", default=[0])
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument(
        "--head-mode",
        type=str,
        default="each",
        choices=["all", "each"],
        help="all=only all-heads swap; each=also per-head",
    )
    args = ap.parse_args()
    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        max_batches=args.max_batches,
        head_mode=args.head_mode,
    )


if __name__ == "__main__":
    main()
