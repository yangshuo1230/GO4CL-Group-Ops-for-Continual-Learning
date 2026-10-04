"""Lightweight pipeline / AnalysisContext dataset tests (CPU, no training)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from go4cl.analysis.context import AnalysisContext, OperationRef, filter_by_operation
from go4cl.analysis.pipelines.causal_detail import run_live
from go4cl.analysis.pipelines.task_token_edit import run as run_task_token_edit
from go4cl.constants import CONTEXT_LENGTH, TOKEN_TASK_A, TOKEN_Q0, TOKEN_Q1
from go4cl.data.dataset import ModularAdditionDataset
from go4cl.data.generate import Example, generate_task_datasets, save_datasets
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.tasks.relations import TaskPairSpec
from go4cl.tasks.spec import Operation, TaskSpec
from go4cl.utils.checkpoint import save_checkpoint
from go4cl.utils.seed import seed_everything


def _tiny_pair() -> TaskPairSpec:
    ops = (
        Operation(latent_id=0, i=0, j=1, modulus=7, slot=0),
        Operation(latent_id=1, i=2, j=3, modulus=5, slot=1),
    )
    task_a = TaskSpec(name="A", task_id=0, operations=ops)
    task_b = TaskSpec(name="B", task_id=1, operations=ops)
    return TaskPairSpec(
        task_a=task_a,
        task_b=task_b,
        rho_slot=1.0,
        rho_operand=1.0,
        rho_mod=1.0,
        task_seed=0,
        pair_id="tiny2",
    )


def _tiny_job(tmp_path: Path) -> Path:
    seed_everything(0)
    pair = _tiny_pair()
    manifest, datasets = generate_task_datasets(
        pair,
        data_seed=0,
        n_aliases_per_pair=1,
        n_nuisance_contexts=1,
        experiment_id="tiny2",
    )
    stamp = tmp_path / "stamp"
    data_dir = stamp / "data" / "tiny2"
    job = stamp / "runs" / "tiny2__ms0"
    save_datasets(data_dir, manifest, datasets)
    cfg = ModelConfig(n_layers=1, d_model=16, n_heads=2, d_mlp=32)
    model = ModularTransformer(cfg)
    save_checkpoint(job / "ckpts" / "best.pt", model, step=1)
    (job / "config_resolved.json").write_text(
        json.dumps({"data_dir": str(data_dir)}), encoding="utf-8"
    )
    return job


def test_task_token_edit_writes_reports(tmp_path: Path) -> None:
    job = _tiny_job(tmp_path)
    out = tmp_path / "token_out"
    run_task_token_edit(
        job_dir=job,
        out=out,
        ckpt_kind="best",
        device=torch.device("cpu"),
        max_batches=1,
    )
    csv_path = out / "task_token_edit.csv"
    json_path = out / "task_token_edit_report.json"
    md_path = out / "TASK_TOKEN.md"
    assert csv_path.is_file() and csv_path.stat().st_size > 0
    assert md_path.is_file() and md_path.stat().st_size > 0
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["ckpt_kind"] == "best"
    assert payload["rows"]
    assert {r["condition"] for r in payload["rows"]} >= {"original", "task_b"}


def test_causal_detail_cross_op_head_matrix(tmp_path: Path) -> None:
    job = _tiny_job(tmp_path)
    out = tmp_path / "causal_out"
    run_live(
        job_dir=job,
        out=out,
        ckpt_kind="best",
        device=torch.device("cpu"),
        ablation_ks=[1],
        max_batches=1,
        report_path=None,
    )
    head_csv = out / "cross_op_head_knockout.csv"
    assert head_csv.is_file() and head_csv.stat().st_size > 0
    text = head_csv.read_text(encoding="utf-8")
    assert "ablate_layer" in text
    assert "target_operation" in text
    report = json.loads((out / "causal_detail_report.json").read_text(encoding="utf-8"))
    assert len(report["ops"]) == 2


def test_operation_dataset_empty_packed_fallback_keeps_latent() -> None:
    empty = ModularAdditionDataset(
        tokens=np.zeros((0, CONTEXT_LENGTH), dtype=np.int64),
        labels=np.zeros((0,), dtype=np.int64),
        slots=np.zeros((0,), dtype=np.int64),
        moduli=np.zeros((0,), dtype=np.int64),
        task_ids=np.zeros((0,), dtype=np.int64),
        latent_ids=np.zeros((0,), dtype=np.int64),
    )
    op0 = OperationRef(latent_id=0, modulus=7, operand_i=0, operand_j=1, slot=0)
    op1 = OperationRef(latent_id=1, modulus=5, operand_i=2, operand_j=3, slot=1)

    def _examples(*, split: str, target_latent_ids: list[int]):
        lid = int(target_latent_ids[0])
        op = op0 if lid == 0 else op1
        q = TOKEN_Q0 if lid == 0 else TOKEN_Q1
        tokens = [0, 1, 2, 3, 4, 5, 6, 7, TOKEN_TASK_A, q]
        return [
            Example(
                tokens=tuple(tokens),
                label=(tokens[op.operand_i] + tokens[op.operand_j]) % op.modulus,
                task_name="A",
                slot=op.slot,
                modulus=op.modulus,
                residue_pair=(0, 1),
                split=split,
                latent_id=lid,
                task_id=0,
            )
        ]

    ctx = AnalysisContext(
        job_dir=Path("unused"),
        data_dir=Path("unused"),
        ckpt_path=Path("unused"),
        ckpt_kind="best",
        manifest=None,  # type: ignore[arg-type]
        operations=[op0, op1],
        device=torch.device("cpu"),
        aliases_per_pair=1,
    )
    ctx.analysis_examples = _examples  # type: ignore[method-assign]
    ds0 = ctx.operation_dataset(op0, "test", full=empty)
    ds1 = ctx.operation_dataset(op1, "test", full=empty)
    assert len(ds0) == 1 and int(ds0.latent_ids[0]) == 0
    assert len(ds1) == 1 and int(ds1.latent_ids[0]) == 1
    mixed = ModularAdditionDataset.from_examples(
        _examples(split="test", target_latent_ids=[0])
        + _examples(split="test", target_latent_ids=[1]),
        task_id=0,
    )
    only0 = filter_by_operation(mixed, latent_id=0, slot=0)
    assert set(int(x) for x in only0.latent_ids.tolist()) == {0}
