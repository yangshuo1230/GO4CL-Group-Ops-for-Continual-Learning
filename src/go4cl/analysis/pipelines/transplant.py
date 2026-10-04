#!/usr/bin/env python3
"""1C deepening: operand activation patching + same-context circuit transplant.

For each multi-op checkpoint:

  1. Operand patch — at resid_pre of L0/L1/L2, copy donor activations at the
     op's two operand positions (batch-roll donor). If the model reads those
     positions, predictions should follow the donor sum, not the original.

  2. Query-twin transplant — keep the eight digits, swap only the query token
     to a target op, then path-patch the source op's query-position
     residual / attn-write / MLP-write. Same-modulus pairs test shared
     compute; different-modulus pairs are a negative control.

Usage:
  uv run python scripts/phase1/circuit_transplant.py \\
    --job-dir runs/phase1/multi_op/.../runs/<job> \\
    --out runs/phase1/mechanisms/<stamp>/<tag>/transplant \\
    --device cuda:0
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch


from go4cl.analysis.cache import (
    accuracy_from_logits,
    collect_batches,
    operand_residues,
    to_jsonable,
)
from go4cl.analysis.patching import (
    QueryPatchKind,
    operand_patch_from_pre,
    query_path_patch_logits,
    split_acc,
)
from go4cl.constants import QUERY_TOKEN_IDS, SEQ_LEN_OPERANDS
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.analysis.context import (
    filter_by_operation,
    load_analysis_context,
    op_report_key,
    operations_from_manifest,
)
from go4cl.analysis.reporting import write_csv_rows, write_json_report, write_markdown

QUERY_KINDS: tuple[QueryPatchKind, ...] = (
    "resid_pre_query",
    "attn_write_query",
    "mlp_write_query",
    "resid_post_query",
)


def _op_ds(op, full, *, split: str, analysis_builder):
    if len(full) == 0:
        examples = analysis_builder(split=split, target_latent_ids=[op.latent_id])
        return ModularAdditionDataset.from_examples(examples, task_id=0)
    return filter_by_operation(full, latent_id=op.latent_id, slot=op.slot)


def _op_key(op) -> str:
    return f"lat{op.latent_id}/slot{op.slot}/p{op.modulus}"


def _unused_positions(ops, source_op, k: int = 2) -> list[int]:
    own = {source_op.operand_i, source_op.operand_j}
    others: list[int] = []
    for op in ops:
        for p in (op.operand_i, op.operand_j):
            if p not in own and p not in others:
                others.append(p)
    if len(others) < k:
        others.extend(
            p for p in range(SEQ_LEN_OPERANDS) if p not in own and p not in others
        )
    return others[:k]


def _minibatch_slices(n: int, bs: int) -> list[slice]:
    return [slice(i, min(i + bs, n)) for i in range(0, n, bs)]


@torch.no_grad()
def _run_operand_patch(
    model,
    cache,
    *,
    op,
    ops,
    layers: list[int],
    device: torch.device,
    minibatch: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    n = cache.tokens.shape[0]
    donor_idx = torch.roll(torch.arange(n), shifts=1)
    _, _, sum_self = operand_residues(
        cache.tokens, i=op.operand_i, j=op.operand_j, modulus=op.modulus
    )
    donor_tokens = cache.tokens[donor_idx]
    _, _, sum_donor = operand_residues(
        donor_tokens, i=op.operand_i, j=op.operand_j, modulus=op.modulus
    )
    labels_self = sum_self.to(device)
    labels_donor = sum_donor.to(device)
    ctrl_pos = _unused_positions(ops, op, k=2)
    own_pos = [op.operand_i, op.operand_j]

    for li in layers:
        pre = cache.resid_pre[li]
        donor_pre = pre[donor_idx]
        for name, positions in (("own_operands", own_pos), ("other_positions", ctrl_pos)):
            pred_chunks: list[torch.Tensor] = []
            for sl in _minibatch_slices(n, minibatch):
                logits = operand_patch_from_pre(
                    model,
                    pre[sl].to(device),
                    donor_pre[sl].to(device),
                    layer_idx=li,
                    positions=positions,
                )
                pred_chunks.append(logits)
            logits = torch.cat(pred_chunks, dim=0)
            stats = split_acc(logits, labels_self, labels_donor)
            rows.append(
                {
                    "operation": _op_key(op),
                    "latent_id": op.latent_id,
                    "slot": op.slot,
                    "modulus": op.modulus,
                    "layer": li,
                    "condition": name,
                    "positions": ",".join(map(str, positions)),
                    "n": int(stats["n"]),
                    "n_label_diff": int(stats["n_label_diff"]),
                    "acc_orig": stats["acc_a"],
                    "acc_donor": stats["acc_b"],
                    "acc_orig_when_diff": stats["acc_a_when_diff"],
                    "acc_donor_when_diff": stats["acc_b_when_diff"],
                }
            )
    return rows


@torch.no_grad()
def _twin_and_src_caches(
    model,
    src_tokens: torch.Tensor,
    tgt_slot: int,
    *,
    device: torch.device,
    minibatch: int,
):
    """Forward source tokens and query-twins; return stacked residual lists."""
    n = src_tokens.shape[0]
    twin_tokens = src_tokens.clone()
    twin_tokens[:, -1] = int(QUERY_TOKEN_IDS[int(tgt_slot)])

    def _fwd(tokens: torch.Tensor):
        pres: list[list[torch.Tensor]] = []
        mids: list[list[torch.Tensor]] = []
        posts: list[list[torch.Tensor]] = []
        logits: list[torch.Tensor] = []
        n_layers: int | None = None
        for sl in _minibatch_slices(n, minibatch):
            cache = model.forward_with_cache(tokens[sl].to(device))
            if n_layers is None:
                n_layers = len(cache["resid_pre"])
                pres = [[] for _ in range(n_layers)]
                mids = [[] for _ in range(n_layers)]
                posts = [[] for _ in range(n_layers)]
            for li in range(n_layers):
                pres[li].append(cache["resid_pre"][li].cpu())
                mids[li].append(cache["resid_mid"][li].cpu())
                posts[li].append(cache["resid_post"][li].cpu())
            logits.append(cache["logits"].cpu())
        stacked = {
            "resid_pre": [torch.cat(xs, dim=0) for xs in pres],
            "resid_mid": [torch.cat(xs, dim=0) for xs in mids],
            "resid_post": [torch.cat(xs, dim=0) for xs in posts],
            "logits": torch.cat(logits, dim=0),
        }
        return stacked

    return _fwd(src_tokens), _fwd(twin_tokens)


@torch.no_grad()
def _run_query_transplant(
    model,
    src_cache_tokens: torch.Tensor,
    *,
    src_op,
    tgt_op,
    layers: list[int],
    device: torch.device,
    minibatch: int,
) -> list[dict[str, Any]]:
    src_fwd, twin_fwd = _twin_and_src_caches(
        model,
        src_cache_tokens,
        tgt_op.slot,
        device=device,
        minibatch=minibatch,
    )
    _, _, src_sum = operand_residues(
        src_cache_tokens,
        i=src_op.operand_i,
        j=src_op.operand_j,
        modulus=src_op.modulus,
    )
    _, _, tgt_sum = operand_residues(
        src_cache_tokens,
        i=tgt_op.operand_i,
        j=tgt_op.operand_j,
        modulus=tgt_op.modulus,
    )
    src_y = src_sum.to(device)
    tgt_y = tgt_sum.to(device)
    n = src_cache_tokens.shape[0]
    same_mod = int(src_op.modulus) == int(tgt_op.modulus)
    rows: list[dict[str, Any]] = []

    # Twin identity (no patch): should mostly predict target op.
    twin_logits = twin_fwd["logits"].to(device)
    ident = split_acc(twin_logits, tgt_y, src_y)
    rows.append(
        {
            "source_operation": _op_key(src_op),
            "target_operation": _op_key(tgt_op),
            "same_modulus": int(same_mod),
            "source_modulus": src_op.modulus,
            "target_modulus": tgt_op.modulus,
            "layer": -1,
            "kind": "twin_identity",
            "n": int(ident["n"]),
            "n_label_diff": int(ident["n_label_diff"]),
            "acc_target": ident["acc_a"],
            "acc_source": ident["acc_b"],
            "acc_target_when_diff": ident["acc_a_when_diff"],
            "acc_source_when_diff": ident["acc_b_when_diff"],
        }
    )

    for li in layers:
        for kind in QUERY_KINDS:
            pred_chunks: list[torch.Tensor] = []
            for sl in _minibatch_slices(n, minibatch):
                logits = query_path_patch_logits(
                    model,
                    layer_idx=li,
                    kind=kind,
                    recv_pre=twin_fwd["resid_pre"][li][sl].to(device),
                    recv_mid=twin_fwd["resid_mid"][li][sl].to(device),
                    recv_post=twin_fwd["resid_post"][li][sl].to(device),
                    donor_pre=src_fwd["resid_pre"][li][sl].to(device),
                    donor_mid=src_fwd["resid_mid"][li][sl].to(device),
                    donor_post=src_fwd["resid_post"][li][sl].to(device),
                )
                pred_chunks.append(logits)
            logits = torch.cat(pred_chunks, dim=0)
            stats = split_acc(logits, tgt_y, src_y)
            rows.append(
                {
                    "source_operation": _op_key(src_op),
                    "target_operation": _op_key(tgt_op),
                    "same_modulus": int(same_mod),
                    "source_modulus": src_op.modulus,
                    "target_modulus": tgt_op.modulus,
                    "layer": li,
                    "kind": kind,
                    "n": int(stats["n"]),
                    "n_label_diff": int(stats["n_label_diff"]),
                    "acc_target": stats["acc_a"],
                    "acc_source": stats["acc_b"],
                    "acc_target_when_diff": stats["acc_a_when_diff"],
                    "acc_source_when_diff": stats["acc_b_when_diff"],
                }
            )
    return rows


def _fmt(v: float | None, digits: int = 3) -> str:
    if v is None:
        return ""
    return f"{v:.{digits}f}"


def write_markdown(
    *,
    out: Path,
    job_dir: Path,
    ckpt_kind: str,
    ckpt_path: Path,
    payload: dict[str, Any],
    data_dir: Path,
    layers: list[int],
    operand_rows: list[dict[str, Any]],
    transplant_rows: list[dict[str, Any]],
) -> None:
    lines = [
        "# 1C · activation patching / circuit transplant\n",
        f"**Job:** `{job_dir.name}`  ",
        f"**Checkpoint:** `{ckpt_kind}` (`{ckpt_path.name}`) step={payload.get('step')}  ",
        f"**Data:** `{data_dir.name}`  ",
        "**Protocol:** donor = batch-roll; query-twin keeps digits, swaps only Q token.\n",
        "## Operand residual patch (acc when labels differ)\n",
        "| op | L | own acc_orig | own acc_donor | ctrl acc_orig | ctrl acc_donor |",
        "|----|---|--------------|---------------|---------------|----------------|",
    ]
    by: dict[tuple[str, int, str], dict[str, Any]] = {}
    for r in operand_rows:
        by[(r["operation"], int(r["layer"]), r["condition"])] = r
    ops_seen: list[str] = []
    for r in operand_rows:
        if r["operation"] not in ops_seen:
            ops_seen.append(r["operation"])
    for opk in ops_seen:
        for li in layers:
            own = by.get((opk, li, "own_operands"))
            ctrl = by.get((opk, li, "other_positions"))
            if not own:
                continue
            lines.append(
                f"| {opk} | {li} | "
                f"{_fmt(own.get('acc_orig_when_diff'))} | "
                f"{_fmt(own.get('acc_donor_when_diff'))} | "
                f"{_fmt((ctrl or {}).get('acc_orig_when_diff'))} | "
                f"{_fmt((ctrl or {}).get('acc_donor_when_diff'))} |"
            )

    lines += [
        "\n## Same-context query transplant (mean acc when src/tgt labels differ)\n",
        "Twin identity should follow the **target** op. A successful compute "
        "transplant raises **acc_source** and drops **acc_target**.\n",
        "| same p | L | kind | n_pairs | acc_target | acc_source |",
        "|--------|---|------|---------|------------|------------|",
    ]
    grouped: dict[tuple[int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for r in transplant_rows:
        grouped[(int(r["same_modulus"]), int(r["layer"]), str(r["kind"]))].append(r)
    for key in sorted(grouped, key=lambda k: (k[0] == 0, k[1], k[2])):
        rs = grouped[key]
        same_p, li, kind = key
        at = sum(x["acc_target_when_diff"] for x in rs) / len(rs)
        as_ = sum(x["acc_source_when_diff"] for x in rs) / len(rs)
        layer = "—" if li < 0 else str(li)
        lines.append(
            f"| {'yes' if same_p else 'no'} | {layer} | {kind} | {len(rs)} | "
            f"{_fmt(at)} | {_fmt(as_)} |"
        )

    # One-line verdicts
    lines += ["\n## Verdict sketch\n"]
    own_l0 = [
        r
        for r in operand_rows
        if r["condition"] == "own_operands" and int(r["layer"]) == 0
    ]
    if own_l0:
        mean_donor = sum(r["acc_donor_when_diff"] for r in own_l0) / len(own_l0)
        mean_orig = sum(r["acc_orig_when_diff"] for r in own_l0) / len(own_l0)
        lines.append(
            f"- L0 own-operand patch: mean acc_orig={mean_orig:.3f}, "
            f"acc_donor={mean_donor:.3f} "
            f"({'donor-following' if mean_donor > mean_orig + 0.2 else 'not cleanly donor-following'}).\n"
        )
    same_post = [
        r
        for r in transplant_rows
        if int(r["same_modulus"]) == 1
        and r["kind"] == "resid_post_query"
        and int(r["layer"]) == 1
    ]
    if same_post:
        mean_src = sum(r["acc_source_when_diff"] for r in same_post) / len(same_post)
        mean_tgt = sum(r["acc_target_when_diff"] for r in same_post) / len(same_post)
        lines.append(
            f"- Same-mod L1 `resid_post_query` transplant: mean acc_source={mean_src:.3f}, "
            f"acc_target={mean_tgt:.3f} "
            f"({'shared query compute' if mean_src > mean_tgt + 0.2 else 'not a clean residual transplant'}).\n"
        )
    else:
        diff_post = [
            r
            for r in transplant_rows
            if int(r["same_modulus"]) == 0
            and r["kind"] == "mlp_write_query"
            and int(r["layer"]) == 2
        ]
        if diff_post:
            mean_src = sum(r["acc_source_when_diff"] for r in diff_post) / len(diff_post)
            lines.append(
                f"- No same-modulus pair. L2 `mlp_write_query` still yields "
                f"acc_source={mean_src:.3f} (shared head readout of an already-written "
                f"answer; full query residual transplant is nearly tautological).\n"
            )

    lines.append(
        "\nFiles: `operand_patch.csv`, `query_transplant.csv`, `transplant_report.json`.\n"
    )
    (out / "TRANSPLANT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(
    *,
    job_dir: Path,
    out: Path,
    ckpt_kind: str,
    device: torch.device,
    layers: list[int],
    max_batches: int | None,
    minibatch: int,
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

    operand_rows: list[dict[str, Any]] = []
    transplant_rows: list[dict[str, Any]] = []
    bundles: list[dict[str, Any]] = []

    for op in ops:
        key = _op_key(op)
        ds = _op_ds(op, test_full, split="test", analysis_builder=analysis_builder)
        loader = make_loader(ds, batch_size=256, shuffle=False)
        cache = collect_batches(model, loader, device=device, max_batches=max_batches)
        base = float(accuracy_from_logits(cache.logits, cache.labels))
        print(f"  {key}: n={cache.tokens.shape[0]} baseline={base:.3f}", flush=True)
        bundles.append({"op": op, "key": key, "cache": cache, "baseline": base})
        operand_rows.extend(
            _run_operand_patch(
                model,
                cache,
                op=op,
                ops=ops,
                layers=layers,
                device=device,
                minibatch=minibatch,
            )
        )

    for src in bundles:
        for tgt in bundles:
            if src["key"] == tgt["key"]:
                continue
            print(
                f"  transplant {src['key']} → {tgt['key']} ...",
                flush=True,
            )
            transplant_rows.extend(
                _run_query_transplant(
                    model,
                    src["cache"].tokens,
                    src_op=src["op"],
                    tgt_op=tgt["op"],
                    layers=layers,
                    device=device,
                    minibatch=minibatch,
                )
            )

    op_fields = [
        "operation",
        "latent_id",
        "slot",
        "modulus",
        "layer",
        "condition",
        "positions",
        "n",
        "n_label_diff",
        "acc_orig",
        "acc_donor",
        "acc_orig_when_diff",
        "acc_donor_when_diff",
    ]
    tx_fields = [
        "source_operation",
        "target_operation",
        "same_modulus",
        "source_modulus",
        "target_modulus",
        "layer",
        "kind",
        "n",
        "n_label_diff",
        "acc_target",
        "acc_source",
        "acc_target_when_diff",
        "acc_source_when_diff",
    ]
    write_csv_rows(out / "operand_patch.csv", operand_rows, op_fields)
    write_csv_rows(out / "query_transplant.csv", transplant_rows, tx_fields)
    write_markdown(
        out=out,
        job_dir=job_dir,
        ckpt_kind=ckpt_kind,
        ckpt_path=ckpt_path,
        payload=payload,
        data_dir=data_dir,
        layers=layers,
        operand_rows=operand_rows,
        transplant_rows=transplant_rows,
    )
    (out / "transplant_report.json").write_text(
        json.dumps(
            to_jsonable(
                {
                    "job_dir": str(job_dir),
                    "ckpt_path": str(ckpt_path),
                    "ckpt_kind": ckpt_kind,
                    "ckpt_step": payload.get("step"),
                    "data_dir": str(data_dir),
                    "layers": layers,
                    "operand_patch": operand_rows,
                    "query_transplant": transplant_rows,
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
    ap.add_argument("--max-batches", type=int, default=None)
    ap.add_argument("--minibatch", type=int, default=256)
    args = ap.parse_args()
    run(
        job_dir=Path(args.job_dir),
        out=Path(args.out),
        ckpt_kind=args.ckpt_kind,
        device=torch.device(args.device),
        layers=list(args.layers),
        max_batches=args.max_batches,
        minibatch=int(args.minibatch),
    )


if __name__ == "__main__":
    main()
