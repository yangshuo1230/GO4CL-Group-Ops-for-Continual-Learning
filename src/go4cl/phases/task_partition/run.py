"""Run AB joint once, then branch C-only and AB-continued from that checkpoint.

ABC joint starts from the same initial weights. C-only training does not
draw A or B examples.
"""

from __future__ import annotations

import hashlib
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

from go4cl.constants import PARTITION_VOCAB_SIZE
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.task_partition.config import FORMAL_LAUNCH, PartitionConfig, load_config, smoke_config
from go4cl.phases.task_partition.data import (
    EqualPackedJointLoader,
    build_eval_loaders,
    make_c_only_loader,
    peek_task_ids,
    prepare_dataset,
)
from go4cl.phases.task_partition.mechanism import analyze_checkpoint, counterfactual_matrix, evaluate_split
from go4cl.phases.task_partition.metrics import assess_partition, gap_to_upper, retention_block
from go4cl.phases.task_partition.plots import plot_counterfactual, write_figures
from go4cl.tasks.partition import PARTITION_ORDER
from go4cl.tasks.spec import TaskSpec
from go4cl.train.loop import TrainConfig, build_optimizer, train_segment
from go4cl.train.protocol_common import write_history
from go4cl.utils.checkpoint import load_checkpoint, save_checkpoint, write_json
from go4cl.utils.seed import seed_everything


def _log(message: str) -> None:
    print(f"[task-partition] {message}", flush=True)


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return _jsonable(float(value))
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    return value


def _write(path: Path, payload: Any) -> None:
    write_json(path, _jsonable(payload))


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_rev() -> str | None:
    root = Path(__file__).resolve().parents[4]
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=root,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _resolve_device(requested: str) -> str:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        _log("CUDA is unavailable; using cpu")
        return "cpu"
    return requested


def _model_config(cfg: PartitionConfig) -> ModelConfig:
    return ModelConfig(
        vocab_size=PARTITION_VOCAB_SIZE,
        n_layers=int(cfg.n_layers),
        d_model=int(cfg.d_model),
        n_heads=int(cfg.n_heads),
        d_mlp=int(cfg.d_mlp),
        dropout=0.0,
        activation="relu",
    )


def _train_config(cfg: PartitionConfig, steps: int) -> TrainConfig:
    return TrainConfig(
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        batch_size=int(cfg.batch_size),
        max_steps=int(steps),
        eval_every=int(cfg.eval_every),
        ckpt_every=int(cfg.ckpt_every),
        device=cfg.device,
        compile_model=bool(cfg.compile_model),
        null_task_tokens=False,
        log_to_wandb=False,
    )


def _new_model(cfg: PartitionConfig, model_cfg: ModelConfig) -> ModularTransformer:
    seed_everything(int(cfg.model_seed))
    return ModularTransformer(model_cfg).to(torch.device(cfg.device))


def _exposure(cfg: PartitionConfig) -> dict[str, Any]:
    ab = float(cfg.ab_steps)
    cont = float(cfg.continuation_steps)
    abc = float(cfg.abc_steps)
    return {
        "ab_joint": {"A": ab / 2.0, "B": ab / 2.0, "C": 0.0},
        "ab_then_c": {"A": ab / 2.0, "B": ab / 2.0, "C": cont},
        "ab_continued": {"A": (ab + cont) / 2.0, "B": (ab + cont) / 2.0, "C": 0.0},
        "abc_joint": {"A": abc / 3.0, "B": abc / 3.0, "C": abc / 3.0},
        "note": (
            "Expected optimizer steps per task under equal mixing. "
            "AB continued is the no-new-task drift control, so A and B "
            "receive the continuation steps as well."
        ),
    }


def _assert_mix(phase: str, task_ids: list[int], expected: dict[int, int]) -> None:
    counts = {task_id: task_ids.count(task_id) for task_id in expected}
    if counts != expected or len(task_ids) != sum(expected.values()):
        raise RuntimeError(f"{phase} batch mix {counts} != {expected}")


class CMilestoneSaver:
    """Save the requested C steps and the first stable C generalization."""

    def __init__(
        self,
        ckpt_dir: Path,
        *,
        start_step: int,
        milestones: tuple[int, ...],
        threshold: float,
        window: int,
    ) -> None:
        self.ckpt_dir = Path(ckpt_dir)
        self.start_step = int(start_step)
        self.milestones = {int(step) for step in milestones}
        self.threshold = float(threshold)
        self.window = int(window)
        self.streak = 0
        self.first_stable_local: int | None = None
        self.first_stable_global: int | None = None
        self.first_stable_acc: float | None = None

    def __call__(self, step: int, record: dict[str, Any], model, opt) -> None:
        local = int(step) - self.start_step
        if local in self.milestones:
            save_checkpoint(
                self.ckpt_dir / f"c_step{local}.pt",
                model,
                optimizer=opt,
                step=int(step),
                meta={"local_step": local, "kind": "c_milestone"},
            )
        acc = record.get("C_val_macro_op_acc")
        if acc is None:
            return
        if float(acc) >= self.threshold:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= self.window and self.first_stable_local is None:
            self.first_stable_local = local
            self.first_stable_global = int(step)
            self.first_stable_acc = float(acc)
            save_checkpoint(
                self.ckpt_dir / "c_first_stable.pt",
                model,
                optimizer=opt,
                step=int(step),
                meta={
                    "local_step": local,
                    "kind": "c_first_stable",
                    "C_val_macro_op_acc": float(acc),
                    "C_test_acc": record.get("C_test_acc"),
                    "threshold": self.threshold,
                    "window": self.window,
                    "split": "C_val",
                },
            )


def _run_phase(
    *,
    phase_dir: Path,
    run_name: str,
    model: ModularTransformer,
    loader,
    cfg: PartitionConfig,
    steps: int,
    eval_loaders: dict,
    optimizer=None,
    start_step: int = 0,
    after_eval=None,
) -> dict[str, Any]:
    phase_dir.mkdir(parents=True, exist_ok=True)
    segment = train_segment(
        model,
        loader,
        cfg=_train_config(cfg, steps),
        optimizer=optimizer,
        start_step=int(start_step),
        optimizer_transition="preserve",
        eval_loaders=eval_loaders,
        ckpt_dir=str(phase_dir / "ckpts"),
        run_name=run_name,
        track_events=False,
        after_eval=after_eval,
    )
    history_path = phase_dir / "eval_history.jsonl"
    write_history(history_path, segment.state.eval_history)
    final_path = phase_dir / "ckpts" / f"{run_name}_final.pt"
    if not final_path.is_file():
        raise RuntimeError(f"missing final checkpoint {final_path}")
    return {
        "history": segment.state.eval_history,
        "history_path": history_path,
        "final": final_path,
        "step": int(segment.state.step),
    }


def _load_for_continue(path: Path, expected_sha: str, cfg: PartitionConfig):
    found = file_sha256(path)
    if found != expected_sha:
        raise RuntimeError(
            f"shared AB checkpoint hash changed: expected {expected_sha}, found {found}"
        )
    model, payload = load_checkpoint(path, map_location=cfg.device)
    model.to(torch.device(cfg.device))
    if "optimizer_state" not in payload:
        raise RuntimeError(f"{path} has no optimizer state; continuations must preserve it")
    optimizer = build_optimizer(model, _train_config(cfg, steps=1))
    optimizer.load_state_dict(payload["optimizer_state"])
    return model, payload, optimizer


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(__import__("json").loads(line))
    return rows


def _last_train_loss(rows: list[dict[str, Any]]) -> float | None:
    for row in reversed(rows):
        if isinstance(row.get("train_loss"), (int, float)):
            return float(row["train_loss"])
    return None


def _acc_map(record: dict[str, Any]) -> dict[str, float]:
    return {task: float(record["test"][task]["accuracy"]) for task in PARTITION_ORDER}


def _per_query_map(record: dict[str, Any]) -> dict[str, Any]:
    return {task: record["test"][task]["per_query"] for task in PARTITION_ORDER}


def _loss_map(record: dict[str, Any]) -> dict[str, float]:
    return {task: float(record["test"][task]["loss"]) for task in PARTITION_ORDER}


def _format_matrix(matrix: list[list[Any]], names: list[str]) -> str:
    header = " ".join(f"{'label ' + name:>10}" for name in names)
    lines = [f"{'':8} {header}"]
    for name, row in zip(names, matrix, strict=True):
        cells = []
        for value in row:
            missing = value is None or (isinstance(value, float) and math.isnan(value))
            cells.append(f"{'n/a':>10}" if missing else f"{float(value):10.3f}")
        lines.append(f"{'TASK_' + name:8} " + " ".join(cells))
    return "\n".join(lines)


def _write_diagnostic(path: Path, gate: dict[str, Any], matrix: list[list[Any]]) -> None:
    reasons = "\n".join(f"- {reason}" for reason in gate["reasons"])
    text = (
        "# Task-partition diagnostic\n\n"
        "AB joint did not learn a task-space partition. "
        "C-only continuation was not started, and the two control protocols "
        "were not started from this failed checkpoint.\n\n"
        f"- A test accuracy: {gate['A_test_acc']:.4f}\n"
        f"- B test accuracy: {gate['B_test_acc']:.4f}\n"
        f"- C test accuracy: {gate.get('C_test_acc')}\n"
        f"- samples with y_A != y_B: {gate['n_disagree_ab']}\n"
        f"- task-token flip rate: {gate['task_token_flip_rate']}\n\n"
        "Counterfactual matrix on disagreeing labels "
        "(row = task token presented, column = whose label was predicted):\n\n"
        "```\n"
        f"{_format_matrix(matrix, ['A', 'B', 'C'])}\n"
        "```\n\n"
        "AB slice used by the gate:\n\n"
        "```\n"
        f"{_format_matrix(gate['ab_matrix'], ['A', 'B'])}\n"
        "```\n\n"
        "Failed checks:\n\n"
        f"{reasons}\n"
    )
    path.write_text(text, encoding="utf-8")


def _analyze_file(
    path: Path,
    *,
    tag: str,
    role: str,
    title: str,
    routing_step: int | None,
    cfg: PartitionConfig,
    bundle: dict[str, Any],
    loaders: dict,
    seed: int,
) -> dict[str, Any]:
    model, payload = load_checkpoint(path, map_location=cfg.device)
    model.to(torch.device(cfg.device))
    analyzed = analyze_checkpoint(
        model,
        bundle["tasks"],  # type: ignore[arg-type]
        bundle["splits"],  # type: ignore[arg-type]
        loaders,
        device=torch.device(cfg.device),
        n_contexts=int(cfg.mech_n_contexts),
        n_per_operation=int(cfg.mech_n_per_operation),
        probe_steps=int(cfg.probe_steps),
        seed=seed,
    )
    record = {
        "tag": tag,
        "role": role,
        "title": title,
        "routing_step": routing_step,
        "global_step": int(payload.get("step", -1)),
        "checkpoint": str(path),
        **analyzed,
    }
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return record


def run_task_partition(cfg: PartitionConfig) -> dict[str, Any]:
    # Local figures only. Do not open a W&B run or upload charts.
    os.environ["WANDB_MODE"] = "disabled"
    os.environ["WANDB_DISABLED"] = "true"
    cfg.device = _resolve_device(cfg.device)
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    bundle = prepare_dataset(out / "data", cfg)
    tasks: dict[str, TaskSpec] = bundle["tasks"]  # type: ignore[assignment]
    model_cfg = _model_config(cfg)
    resolved = {
        "experiment": "task_partition",
        "git_rev": _git_rev(),
        "config": cfg.to_dict(),
        "model": model_cfg.to_dict(),
        "vocab_size": PARTITION_VOCAB_SIZE,
        "dataset_hash": bundle["dataset_hash"],
        "data_manifest": str(bundle["manifest_path"]),
        "split_sizes": bundle["split_sizes"],
        "ratios": bundle["ratios"],
        "operations": bundle["operations"],
        "exposure_steps": _exposure(cfg),
        "protocols": {
            "ab_then_c": "AB joint, then C only, from AB_joint_checkpoint",
            "ab_continued": "AB joint continued 1:1, from the same AB_joint_checkpoint",
            "abc_joint": "A/B/C 1:1:1 from the same initial weights as AB joint",
        },
        "c_only_mixes_other_tasks": False,
        "wandb": "disabled",
        "figures": "local_png",
        "checkpoint_selection_note": (
            "Reported metrics use the named checkpoints (step 0, milestones, "
            "first stable C val, final). train_steps may also write a best.pt "
            "from A/B val; that file is not the scientific endpoint."
        ),
        "stable_definition": (
            "C first stable = C_val macro operation accuracy "
            f">= {cfg.stable_threshold} for {cfg.stable_window} consecutive evals. "
            "Forgetting uses test accuracy."
        ),
    }
    _write(out / "config_resolved.json", resolved)

    _log("initializing model")
    model = _new_model(cfg, model_cfg)
    init_path = out / "init_checkpoint.pt"
    save_checkpoint(
        init_path,
        model,
        step=0,
        meta={"stage": "init", "model_seed": int(cfg.model_seed)},
    )
    init_sha = file_sha256(init_path)

    nested, flat = build_eval_loaders(tasks, bundle["splits"], cfg)  # type: ignore[arg-type]
    ab_loader = EqualPackedJointLoader(
        [tasks["A"], tasks["B"]],
        bundle["splits"],  # type: ignore[arg-type]
        batch_size=int(cfg.batch_size),
        seed=cfg.phase_sampler_seed("ab_joint"),
    )
    ab_ids = peek_task_ids(ab_loader)
    each = int(cfg.batch_size) // 2
    _assert_mix("ab_joint", ab_ids, {0: each, 1: each})

    _log(f"AB joint for {cfg.ab_steps} steps")
    ab_dir = out / "ab_joint"
    ab_result = _run_phase(
        phase_dir=ab_dir,
        run_name="ab_joint",
        model=model,
        loader=ab_loader,
        cfg=cfg,
        steps=int(cfg.ab_steps),
        eval_loaders=flat,
    )
    canonical = out / "AB_joint_checkpoint.pt"
    shutil.copy2(ab_result["final"], canonical)
    ab_sha = file_sha256(canonical)
    _write(
        ab_dir / "phase.json",
        {
            "phase": "ab_joint",
            "steps": int(cfg.ab_steps),
            "sampler_seed": cfg.phase_sampler_seed("ab_joint"),
            "final_step": ab_result["step"],
            "checkpoint": str(canonical),
            "sha256": ab_sha,
            "init_checkpoint": str(init_path),
            "init_sha256": init_sha,
            "batch_task_id_counts": {"0": each, "1": each},
        },
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log("checking AB task-token partition")
    gate_model, _payload = load_checkpoint(canonical, map_location=cfg.device)
    gate_model.to(torch.device(cfg.device))
    device = torch.device(cfg.device)
    test_eval = evaluate_split(gate_model, tasks, nested, device, "test")
    factual = counterfactual_matrix(
        gate_model,
        tasks,
        bundle["splits"],  # type: ignore[arg-type]
        n_contexts=int(cfg.mech_n_contexts),
        seed=int(cfg.eval_seed) + 7001,
        device=device,
    )
    gate = assess_partition(
        acc_a=test_eval["A"]["accuracy"],
        acc_b=test_eval["B"]["accuracy"],
        ab_matrix=factual["ab_matrix"],
        flip_rate=factual["task_token_flip_rate"],
        n_disagree=int(factual["n_disagree_ab"]),
        min_acc=float(cfg.partition_min_acc),
        diag_min=float(cfg.counterfactual_diag_min),
        off_max=float(cfg.counterfactual_off_max),
        flip_min=float(cfg.counterfactual_flip_min),
    )
    gate["C_test_acc"] = test_eval["C"]["accuracy"]
    gate["per_query_test"] = {name: test_eval[name]["per_query"] for name in PARTITION_ORDER}
    gate["counterfactual_matrix"] = factual["matrix"]
    _write(out / "partition_gate.json", gate)
    del gate_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if not gate["passed"] and cfg.require_partition:
        _log("AB partition gate failed; stopping before C")
        _write_diagnostic(out / "PARTITION_DIAGNOSTIC.md", gate, factual["matrix"])
        failed = {
            "tag": "ab_joint",
            "role": "ab_pretrain",
            "title": "AB joint (gate failed)",
            "routing_step": 0,
            "counterfactual": factual,
        }
        mech_dir = out / "mechanism"
        mech_dir.mkdir(parents=True, exist_ok=True)
        _write(mech_dir / "ab_joint.json", failed)
        figure = out / "figures" / "task_token_counterfactual.png"
        figure.parent.mkdir(parents=True, exist_ok=True)
        plot_counterfactual([failed], figure)
        summary = {
            "status": "stopped",
            "reason": "ab_partition_gate_failed",
            "gate": gate,
            "diagnostic": str(out / "PARTITION_DIAGNOSTIC.md"),
            "ab_joint_checkpoint": str(canonical),
            "ab_joint_sha256": ab_sha,
            "init_checkpoint": str(init_path),
            "formal_command": FORMAL_LAUNCH,
        }
        _write(out / "summary.json", summary)
        return summary

    branch = {
        "ab_joint_checkpoint": str(canonical),
        "ab_joint_sha256": ab_sha,
        "ab_joint_step": ab_result["step"],
        "init_checkpoint": str(init_path),
        "init_sha256": init_sha,
        "continuations": ["c_only", "ab_continued"],
        "abc_joint_starts_from": str(init_path),
        "optimizer_preserved_for_continuations": True,
    }
    _write(out / "branch_manifest.json", branch)

    c_dir = out / "c_only"
    c_dir.mkdir(parents=True, exist_ok=True)
    (c_dir / "ckpts").mkdir(parents=True, exist_ok=True)
    shutil.copy2(canonical, c_dir / "ckpts" / "c_step0.pt")
    c_model, c_payload, c_opt = _load_for_continue(canonical, ab_sha, cfg)
    c_loader = make_c_only_loader(tasks["C"], bundle["splits"], cfg)  # type: ignore[arg-type]
    c_ids = peek_task_ids(c_loader)
    _assert_mix("c_only", c_ids, {2: int(cfg.batch_size)})
    saver = CMilestoneSaver(
        c_dir / "ckpts",
        start_step=int(c_payload["step"]),
        milestones=cfg.c_milestones,
        threshold=float(cfg.stable_threshold),
        window=int(cfg.stable_window),
    )
    _log(f"C-only continuation for {cfg.continuation_steps} steps")
    c_result = _run_phase(
        phase_dir=c_dir,
        run_name="c_only",
        model=c_model,
        loader=c_loader,
        cfg=cfg,
        steps=int(cfg.continuation_steps),
        eval_loaders=flat,
        optimizer=c_opt,
        start_step=int(c_payload["step"]),
        after_eval=saver,
    )
    _write(
        c_dir / "phase.json",
        {
            "phase": "c_only",
            "parent_checkpoint": str(canonical),
            "parent_sha256": ab_sha,
            "parent_step": int(c_payload["step"]),
            "sampler_seed": cfg.phase_sampler_seed("c_only"),
            "steps": int(cfg.continuation_steps),
            "mixes_other_tasks": False,
            "batch_task_id_counts": {"2": int(cfg.batch_size)},
            "first_stable_local_step": saver.first_stable_local,
            "first_stable_global_step": saver.first_stable_global,
            "first_stable_C_val_macro_op_acc": saver.first_stable_acc,
            "stable_threshold": float(cfg.stable_threshold),
            "stable_window": int(cfg.stable_window),
            "milestones": list(cfg.c_milestones),
        },
    )
    del c_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log(f"AB continued for {cfg.continuation_steps} steps")
    ab2_model, ab2_payload, ab2_opt = _load_for_continue(canonical, ab_sha, cfg)
    ab2_loader = EqualPackedJointLoader(
        [tasks["A"], tasks["B"]],
        bundle["splits"],  # type: ignore[arg-type]
        batch_size=int(cfg.batch_size),
        seed=cfg.phase_sampler_seed("ab_continued"),
    )
    ab2_ids = peek_task_ids(ab2_loader)
    _assert_mix("ab_continued", ab2_ids, {0: each, 1: each})
    ab2_dir = out / "ab_continued"
    ab2_result = _run_phase(
        phase_dir=ab2_dir,
        run_name="ab_continued",
        model=ab2_model,
        loader=ab2_loader,
        cfg=cfg,
        steps=int(cfg.continuation_steps),
        eval_loaders=flat,
        optimizer=ab2_opt,
        start_step=int(ab2_payload["step"]),
    )
    _write(
        ab2_dir / "phase.json",
        {
            "phase": "ab_continued",
            "parent_checkpoint": str(canonical),
            "parent_sha256": ab_sha,
            "parent_step": int(ab2_payload["step"]),
            "sampler_seed": cfg.phase_sampler_seed("ab_continued"),
            "steps": int(cfg.continuation_steps),
            "includes_task_c": False,
            "batch_task_id_counts": {"0": each, "1": each},
        },
    )
    del ab2_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    _log(f"ABC joint for {cfg.abc_steps} steps from the shared init")
    abc_model = _new_model(cfg, model_cfg)
    reseeded = {
        key: value.detach().cpu().clone() for key, value in abc_model.state_dict().items()
    }
    _loaded, _init_payload = load_checkpoint(init_path, model=abc_model, map_location=cfg.device)
    abc_model.to(torch.device(cfg.device))
    init_matches = all(
        torch.allclose(reseeded[key], value.detach().cpu())
        for key, value in abc_model.state_dict().items()
    )
    abc_loader = EqualPackedJointLoader(
        [tasks[name] for name in PARTITION_ORDER],
        bundle["splits"],  # type: ignore[arg-type]
        batch_size=int(cfg.batch_size),
        seed=cfg.phase_sampler_seed("abc_joint"),
    )
    third = int(cfg.batch_size) // 3
    abc_ids = peek_task_ids(abc_loader)
    _assert_mix("abc_joint", abc_ids, {0: third, 1: third, 2: third})
    abc_dir = out / "abc_joint"
    abc_result = _run_phase(
        phase_dir=abc_dir,
        run_name="abc_joint",
        model=abc_model,
        loader=abc_loader,
        cfg=cfg,
        steps=int(cfg.abc_steps),
        eval_loaders=flat,
    )
    _write(
        abc_dir / "phase.json",
        {
            "phase": "abc_joint",
            "init_checkpoint": str(init_path),
            "init_sha256": init_sha,
            "init_matches_reseed": bool(init_matches),
            "sampler_seed": cfg.phase_sampler_seed("abc_joint"),
            "steps": int(cfg.abc_steps),
            "batch_task_id_counts": {"0": third, "1": third, "2": third},
        },
    )
    del abc_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    specs: list[dict[str, Any]] = [
        {
            "tag": "ab_joint",
            "role": "ab_pretrain",
            "title": "AB joint",
            "path": canonical,
            "routing_step": 0,
        }
    ]
    seen_steps = {0}
    for milestone in cfg.c_milestones:
        path = c_dir / "ckpts" / f"c_step{milestone}.pt"
        if path.is_file():
            specs.append(
                {
                    "tag": f"c_step{milestone}",
                    "role": "c_milestone",
                    "title": f"C step {milestone}",
                    "path": path,
                    "routing_step": int(milestone),
                }
            )
            seen_steps.add(int(milestone))
    if (
        saver.first_stable_local is not None
        and saver.first_stable_local not in seen_steps
        and (c_dir / "ckpts" / "c_first_stable.pt").is_file()
    ):
        specs.append(
            {
                "tag": "c_first_stable",
                "role": "c_first_stable",
                "title": "C first stable",
                "path": c_dir / "ckpts" / "c_first_stable.pt",
                "routing_step": int(saver.first_stable_local),
            }
        )
        seen_steps.add(int(saver.first_stable_local))
    specs.append(
        {
            "tag": "c_final",
            "role": "c_final",
            "title": "C final",
            "path": c_result["final"],
            "routing_step": int(cfg.continuation_steps),
        }
    )
    specs.append(
        {
            "tag": "abc_final",
            "role": "abc_final",
            "title": "ABC joint final",
            "path": abc_result["final"],
            "routing_step": None,
        }
    )
    specs.append(
        {
            "tag": "ab_continued_final",
            "role": "ab_continued_final",
            "title": "AB continued final",
            "path": ab2_result["final"],
            "routing_step": None,
        }
    )

    mech_dir = out / "mechanism"
    mech_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    index: list[dict[str, Any]] = []
    for spec_index, spec in enumerate(specs):
        _log(f"analyzing {spec['tag']}")
        record = _analyze_file(
            Path(spec["path"]),
            tag=spec["tag"],
            role=spec["role"],
            title=spec["title"],
            routing_step=spec["routing_step"],
            cfg=cfg,
            bundle=bundle,
            loaders=nested,
            seed=int(cfg.eval_seed) + 8000 + 13 * spec_index,
        )
        relative = f"mechanism/{spec['tag']}.json"
        _write(mech_dir / f"{spec['tag']}.json", record)
        records.append(record)
        index.append({"tag": spec["tag"], "file": relative, "role": spec["role"]})
    if saver.first_stable_local in seen_steps and saver.first_stable_local in set(cfg.c_milestones):
        index.append(
            {
                "tag": "c_first_stable",
                "file": f"mechanism/c_step{saver.first_stable_local}.json",
                "role": "c_first_stable_alias",
                "same_weights_as": f"c_step{saver.first_stable_local}",
            }
        )
    _write(out / "mechanism_index.json", index)

    ab_rows = _read_jsonl(ab_result["history_path"])
    c_rows = _read_jsonl(c_result["history_path"])
    ab2_rows = _read_jsonl(ab2_result["history_path"])
    abc_rows = _read_jsonl(abc_result["history_path"])
    curves = {
        "switch_step": int(ab_result["step"]),
        "protocols": {
            "ab_then_c": ab_rows + c_rows,
            "abc_joint": abc_rows,
            "ab_continued": ab_rows + ab2_rows,
        },
    }
    _write(out / "curves.json", curves)

    by_tag = {record["tag"]: record for record in records}
    before = _acc_map(by_tag["ab_joint"])
    after_c = _acc_map(by_tag["c_final"])
    after_drift = _acc_map(by_tag["ab_continued_final"])
    after_abc = _acc_map(by_tag["abc_final"])
    c_block = retention_block(before, after_c)
    drift_block = retention_block(before, after_drift)
    summary = {
        "status": "ok",
        "smoke": bool(cfg.smoke),
        "gate": {
            "passed": bool(gate["passed"]),
            "reasons": gate["reasons"],
            "A_test_acc": gate["A_test_acc"],
            "B_test_acc": gate["B_test_acc"],
            "task_token_flip_rate": gate["task_token_flip_rate"],
        },
        "checkpoints": {
            "init": str(init_path),
            "init_sha256": init_sha,
            "ab_joint": str(canonical),
            "ab_joint_sha256": ab_sha,
            "c_only_parent": str(canonical),
            "ab_continued_parent": str(canonical),
            "abc_init": str(init_path),
            "init_matches_reseed": bool(init_matches),
            "c_first_stable_local_step": saver.first_stable_local,
        },
        "final_test_accuracy": {
            "ab_then_c": after_c,
            "abc_joint": after_abc,
            "ab_continued": after_drift,
        },
        "final_test_loss": {
            "ab_joint": _loss_map(by_tag["ab_joint"]),
            "ab_then_c": _loss_map(by_tag["c_final"]),
            "abc_joint": _loss_map(by_tag["abc_final"]),
            "ab_continued": _loss_map(by_tag["ab_continued_final"]),
        },
        "train_loss_at_last_eval": {
            "ab_joint": _last_train_loss(ab_rows),
            "c_only": _last_train_loss(c_rows),
            "ab_continued": _last_train_loss(ab2_rows),
            "abc_joint": _last_train_loss(abc_rows),
        },
        "per_query_test": {
            tag: _per_query_map(by_tag[tag])
            for tag in ("ab_joint", "c_final", "abc_final", "ab_continued_final")
        },
        "comparison": {
            "ab_then_c": c_block,
            "ab_continued_drift": {
                "drift_A": drift_block["forgetting_A"],
                "drift_B": drift_block["forgetting_B"],
                "mean_old_retention": drift_block["mean_old_retention"],
                "worst_old_retention": drift_block["worst_old_retention"],
                "acc_A": drift_block["acc_A"],
                "acc_B": drift_block["acc_B"],
                "acc_C": drift_block["acc_C"],
            },
            "abc_joint": {
                "acc_A": after_abc["A"],
                "acc_B": after_abc["B"],
                "acc_C": after_abc["C"],
                "mean_ab_accuracy": (after_abc["A"] + after_abc["B"]) / 2.0,
                "worst_ab_accuracy": min(after_abc["A"], after_abc["B"]),
            },
            "gap_to_abc_joint": gap_to_upper(after_c, after_abc),
        },
        "mechanism_summary": [
            {
                "tag": record["tag"],
                "role": record["role"],
                "routing_step": record["routing_step"],
                "probe_acc_ab": record["probe"]["probe_acc_ab"],
                "probe_acc_3way": record["probe"]["probe_acc_3way"],
                "task_mass": {
                    task: record["routing"][task]["task_mass"] for task in PARTITION_ORDER
                },
                "operand_mass": {
                    task: record["routing"][task]["operand_mass"] for task in PARTITION_ORDER
                },
                "counterfactual_matrix": record["counterfactual"]["matrix"],
            }
            for record in records
        ],
        "exposure_steps": _exposure(cfg),
        "formal_command": FORMAL_LAUNCH,
        "data_seed": int(cfg.data_seed),
        "model_seed": int(cfg.model_seed),
        "sampler_seed": int(cfg.sampler_seed),
        "eval_seed": int(cfg.eval_seed),
        "dataset_hash": bundle["dataset_hash"],
    }
    _write(out / "summary.json", summary)
    write_figures(out)
    if cfg.smoke:
        (out / "FORMAL_COMMAND.txt").write_text(FORMAL_LAUNCH + "\n", encoding="utf-8")
    _log(f"finished status=ok out={out}")
    return summary


def run_from_args(args) -> None:
    if args.tp_cmd == "smoke":
        cfg = smoke_config(args.out)
    elif args.tp_cmd == "run":
        cfg = load_config(
            args.config,
            out_dir=args.out,
            device=args.device,
            model_seed=getattr(args, "model_seed", None),
        )
    else:
        raise SystemExit(f"unknown task-partition command {args.tp_cmd}")
    result = run_task_partition(cfg)
    if cfg.smoke:
        print(
            "\nSmoke passed. Formal run was not started.\n"
            f"Command: {FORMAL_LAUNCH}\n",
            flush=True,
        )
    if result.get("status") != "ok":
        raise SystemExit(2)
