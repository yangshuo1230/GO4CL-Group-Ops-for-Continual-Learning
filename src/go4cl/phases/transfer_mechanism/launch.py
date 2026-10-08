"""Launch modulus-specificity and checkpoint-mixing jobs.

Dry-run is the default. ``--execute`` is the only switch that trains.
Phase-2 ``sequential_ab`` keeps Adam state across the A→B boundary. These
runs do not, unless ``carry_optimizer_state`` is set. There is no LR scheduler;
B always starts at exposure 0 with a new AdamW when the flag is false.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import random
import shlex
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import torch

from go4cl.data.manifest import DataManifest
from go4cl.metrics.behavioral import evaluate
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.phases.phase2.data import prepare_phase2_dataset
from go4cl.phases.transfer_mechanism.analysis import (
    analyze_component_dir,
    analyze_modulus_dir,
    assert_shared_task_a,
    b_operation_table,
    summarize_series,
    task_a_hash,
)
from go4cl.phases.transfer_mechanism.checkpoint_mix import (
    DEFAULT_INTERVENTIONS,
    build_b_optimizer,
    build_hybrid,
    load_theta_checkpoint,
    origin_labels,
    parse_intervention,
    save_hybrid_checkpoint,
    save_theta_checkpoint,
)
from go4cl.phases.transfer_mechanism.configs import (
    MODULUS_CONDITIONS,
    TransferConfig,
)
from go4cl.phases.transfer_mechanism.parameter_groups import (
    build_registry,
    parameter_group_manifest,
)
from go4cl.tasks.relations import build_task_pair
from go4cl.train.loaders import task_loaders
from go4cl.train.loop import TrainConfig, train_segment
from go4cl.train.protocol_common import write_history
from go4cl.utils.checkpoint import write_json
from go4cl.utils.wandb_log import operation_acc_metrics

_MODULUS_CONFIG = "configs/transfer_mechanism/modulus_specificity.yaml"
_COMPONENT_CONFIG = "configs/transfer_mechanism/component_reset.yaml"


def register_parser(sub: argparse._SubParsersAction) -> None:
    parser = sub.add_parser(
        "transfer-mechanism",
        help="Forward-transfer mechanism grid (no replay; dry-run by default)",
    )
    commands = parser.add_subparsers(dest="tm_cmd", required=True)

    modulus = commands.add_parser("modulus", help="Modulus-specificity grid")
    _add_grid_args(modulus, _MODULUS_CONFIG)

    component = commands.add_parser("component", help="Checkpoint-mixing grid")
    _add_grid_args(component, _COMPONENT_CONFIG)

    one = commands.add_parser("run-modulus", help="Train one modulus-specificity job")
    one.add_argument("--config", type=str, required=True)
    one.add_argument("--condition", type=str, required=True)
    one.add_argument("--protocol", type=str, required=True)
    one.add_argument("--task-seed", type=int, required=True)
    one.add_argument("--model-seed", type=int, required=True)

    source = commands.add_parser("run-source", help="Train theta_0 and theta_A")
    source.add_argument("--config", type=str, required=True)
    source.add_argument("--task-seed", type=int, required=True)
    source.add_argument("--model-seed", type=int, required=True)

    intervention = commands.add_parser(
        "run-intervention", help="Train B from one hybrid initialization"
    )
    intervention.add_argument("--config", type=str, required=True)
    intervention.add_argument("--intervention", type=str, required=True)
    intervention.add_argument("--task-seed", type=int, required=True)
    intervention.add_argument("--model-seed", type=int, required=True)

    analyze = commands.add_parser("analyze", help="Write CSVs from finished jobs")
    analyze.add_argument("--config", type=str, required=True)
    analyze.add_argument("--out", type=str, default=None)

    smoke = commands.add_parser("smoke", help="Tiny CPU smoke; not a scientific run")
    smoke.add_argument("--out", type=str, default="runs/transfer_mechanism/smoke")


def _add_grid_args(parser: argparse.ArgumentParser, default_config: str) -> None:
    parser.add_argument("--config", type=str, default=default_config)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--seed-set", choices=["pilot", "final"], default=None)


def run_from_args(args: argparse.Namespace) -> None:
    command = args.tm_cmd
    if command == "modulus":
        _run_grid(args, experiment="modulus_specificity")
    elif command == "component":
        _run_grid(args, experiment="component_reset")
    elif command == "run-modulus":
        cfg = TransferConfig.from_yaml(args.config)
        cond = _condition(cfg, args.condition)
        run_modulus_job(
            cfg,
            cond,
            args.protocol,
            int(args.task_seed),
            int(args.model_seed),
        )
    elif command == "run-source":
        cfg = TransferConfig.from_yaml(args.config)
        run_source_job(cfg, int(args.task_seed), int(args.model_seed))
    elif command == "run-intervention":
        cfg = TransferConfig.from_yaml(args.config)
        run_intervention_job(
            cfg,
            args.intervention,
            int(args.task_seed),
            int(args.model_seed),
        )
    elif command == "analyze":
        cfg = TransferConfig.from_yaml(args.config)
        root = Path(args.out) if args.out else Path(cfg.out_dir)
        if cfg.experiment == "modulus_specificity":
            paths = analyze_modulus_dir(root)
        else:
            paths = analyze_component_dir(root)
        for name, path in paths.items():
            print(f"[analyze] {name} -> {path}")
    elif command == "smoke":
        run_smoke(Path(args.out))
    else:
        raise SystemExit(f"unknown transfer-mechanism command {command}")


def modulus_jobs(cfg: TransferConfig, seed_set: str | None = None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for cond in cfg.conditions:
        for protocol in cfg.protocols:
            for task_seed in cfg.task_seeds:
                for model_seed in cfg.model_seeds(seed_set):
                    jobs.append(
                        {
                            "kind": "modulus",
                            "condition": cond["name"],
                            "protocol": protocol,
                            "task_seed": int(task_seed),
                            "model_seed": int(model_seed),
                        }
                    )
    return jobs


def component_jobs(cfg: TransferConfig, seed_set: str | None = None) -> list[dict[str, Any]]:
    jobs: list[dict[str, Any]] = []
    for task_seed in cfg.task_seeds:
        for model_seed in cfg.model_seeds(seed_set):
            jobs.append(
                {
                    "kind": "source",
                    "task_seed": int(task_seed),
                    "model_seed": int(model_seed),
                }
            )
    for intervention in cfg.active_interventions():
        for task_seed in cfg.task_seeds:
            for model_seed in cfg.model_seeds(seed_set):
                jobs.append(
                    {
                        "kind": "intervention",
                        "intervention": intervention,
                        "task_seed": int(task_seed),
                        "model_seed": int(model_seed),
                    }
                )
    return jobs


def format_job_command(config_path: str, job: dict[str, Any]) -> str:
    if job["kind"] == "modulus":
        argv = [
            "uv",
            "run",
            "go4cl",
            "transfer-mechanism",
            "run-modulus",
            "--config",
            config_path,
            "--condition",
            str(job["condition"]),
            "--protocol",
            str(job["protocol"]),
            "--task-seed",
            str(job["task_seed"]),
            "--model-seed",
            str(job["model_seed"]),
        ]
    elif job["kind"] == "source":
        argv = [
            "uv",
            "run",
            "go4cl",
            "transfer-mechanism",
            "run-source",
            "--config",
            config_path,
            "--task-seed",
            str(job["task_seed"]),
            "--model-seed",
            str(job["model_seed"]),
        ]
    else:
        argv = [
            "uv",
            "run",
            "go4cl",
            "transfer-mechanism",
            "run-intervention",
            "--config",
            config_path,
            "--intervention",
            str(job["intervention"]),
            "--task-seed",
            str(job["task_seed"]),
            "--model-seed",
            str(job["model_seed"]),
        ]
    return shlex.join(argv)


def run_modulus_job(
    cfg: TransferConfig,
    condition: dict[str, Any],
    protocol: str,
    task_seed: int,
    model_seed: int,
) -> Path:
    if protocol not in {"b_only", "sequential_ab"}:
        raise ValueError(f"refusing protocol {protocol}")
    cfg.validate()
    shared = assert_shared_task_a(task_seed, MODULUS_CONDITIONS)
    meta = _prepare(cfg, condition, task_seed)
    if meta["task_a_hash"] != shared:
        raise RuntimeError("prepared Task A hash != fixed_a identity")
    job_dir = (
        Path(cfg.out_dir)
        / "jobs"
        / str(condition["name"])
        / protocol
        / f"ts{task_seed}_ms{model_seed}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    manifest = DataManifest.load(Path(meta["data_dir"]) / "manifest.json")
    _require_stamped(manifest)
    pair = manifest.task_pair
    operations = b_operation_table(pair)
    device = torch.device(cfg.device)
    loaders = _loaders(Path(meta["data_dir"]), cfg)
    _seed(model_seed, device)
    model = _new_model(cfg, device)
    write_json(job_dir / "parameter_group_manifest.json", parameter_group_manifest(model))
    history: list[dict[str, Any]] = []
    if protocol == "sequential_ab":
        _save_theta(
            job_dir / "theta_0.pt",
            model,
            manifest=manifest,
            task_seed=task_seed,
            model_seed=model_seed,
            step=0,
        )
        model, _opt_a, a_history, a_step = _train_phase(
            model, loaders, cfg, device, task="A", run_name="phase_a"
        )
        write_history(job_dir / "history_a.jsonl", a_history)
        _save_theta(
            job_dir / "theta_A.pt",
            model,
            manifest=manifest,
            task_seed=task_seed,
            model_seed=model_seed,
            step=a_step,
            optimizer=_opt_a,
            carry_optimizer_state=cfg.carry_optimizer_state,
        )
        if cfg.carry_optimizer_state:
            model = _train_b_from_optimizer(
                model, loaders, cfg, device, optimizer=_opt_a, history=history
            )
        else:
            _train_b_fresh(model, loaders, cfg, device, history)
    else:
        _train_b_fresh(model, loaders, cfg, device, history)
    _finalize_job(
        job_dir,
        cfg,
        history,
        condition=condition["name"],
        protocol=protocol,
        task_seed=task_seed,
        model_seed=model_seed,
        operations=operations,
        manifest=manifest,
        intervention=None,
    )
    print(f"[modulus] ok {job_dir}")
    return job_dir


def run_source_job(cfg: TransferConfig, task_seed: int, model_seed: int) -> Path:
    cfg.validate()
    return _ensure_source(cfg, task_seed, model_seed)


def run_intervention_job(
    cfg: TransferConfig,
    intervention: str,
    task_seed: int,
    model_seed: int,
) -> Path:
    cfg.validate()
    spec = parse_intervention(intervention, n_layers=cfg.n_layers)
    source = _ensure_source(cfg, task_seed, model_seed)
    meta = json.loads((source / "source_complete.json").read_text(encoding="utf-8"))
    manifest = DataManifest.load(Path(meta["data_dir"]) / "manifest.json")
    _require_stamped(manifest)
    before = {
        "theta_0.pt": _sha256(source / "theta_0.pt"),
        "theta_A.pt": _sha256(source / "theta_A.pt"),
    }
    _payload_0, theta_0 = load_theta_checkpoint(source / "theta_0.pt")
    payload_a, theta_a = load_theta_checkpoint(source / "theta_A.pt")
    device = torch.device(cfg.device)
    _seed(model_seed, device)
    model = _new_model(cfg, device)
    registry = build_registry(model)
    hybrid, hybrid_manifest = build_hybrid(
        theta_0,
        theta_a,
        intervention,
        registry,
        n_layers=cfg.n_layers,
        task_seed=task_seed,
        model_seed=model_seed,
        dataset_hash=manifest.dataset_hash,
    )
    after = {
        "theta_0.pt": _sha256(source / "theta_0.pt"),
        "theta_A.pt": _sha256(source / "theta_A.pt"),
    }
    if after != before:
        raise RuntimeError("hybrid construction modified theta_0.pt or theta_A.pt")
    _load_state(model, hybrid, device)
    from_init, from_a = origin_labels(spec, cfg.n_layers)
    job_dir = (
        Path(cfg.out_dir) / "jobs" / intervention / f"ts{task_seed}_ms{model_seed}"
    )
    job_dir.mkdir(parents=True, exist_ok=True)
    write_json(job_dir / "hybrid_manifest.json", hybrid_manifest)
    save_hybrid_checkpoint(
        job_dir / "hybrid.pt",
        hybrid,
        model_config=model.cfg.to_dict(),
        manifest=hybrid_manifest,
    )
    loaders = _loaders(Path(meta["data_dir"]), cfg)
    history: list[dict[str, Any]] = []
    if cfg.carry_optimizer_state:
        tcfg = _train_config(cfg, device)
        opt = build_b_optimizer(
            model,
            tcfg,
            carry_optimizer_state=True,
            payload=payload_a,
        )
        _train_b_from_optimizer(
            model, loaders, cfg, device, optimizer=opt, history=history
        )
    else:
        _train_b_fresh(model, loaders, cfg, device, history)
    _finalize_job(
        job_dir,
        cfg,
        history,
        condition=cfg.conditions[0]["name"],
        protocol="hybrid_b",
        task_seed=task_seed,
        model_seed=model_seed,
        operations=b_operation_table(manifest.task_pair),
        manifest=manifest,
        intervention=intervention,
        kind=spec.kind,
        groups_from_A=list(from_a),
        groups_from_init=list(from_init),
    )
    print(f"[intervention] ok {job_dir}")
    return job_dir


def run_smoke(out: Path) -> None:
    """CPU pipeline check. Numbers in this directory are not results."""
    out = Path(out)
    if out.name != "smoke":
        raise ValueError(f"refusing to wipe {out}; smoke output must be named 'smoke'")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    cfg = TransferConfig.from_yaml(_COMPONENT_CONFIG)
    cfg.out_dir = str(out)
    cfg.device = "cpu"
    cfg.compile_model = False
    cfg.steps = 3
    cfg.eval_every = 1
    cfg.batch_size = 8
    cfg.n_aliases = 1
    cfg.eval_n_per_operation = 2
    cfg.d_model = 8
    cfg.n_layers = 1
    cfg.n_heads = 2
    cfg.d_mlp = 16
    cfg.task_seeds = [0]
    cfg.pilot_model_seeds = [0]
    cfg.final_model_seeds = [0]
    cfg.seed_set = "final"
    cfg.replay_ratio = 0.0
    cfg.carry_optimizer_state = False
    cfg.include_reserved = False
    cfg.validate()
    if cfg.device != "cpu":
        raise RuntimeError("smoke must stay on CPU")
    shared = assert_shared_task_a(0, MODULUS_CONDITIONS)
    write_json(
        out / "task_a_identity.json",
        {"fixed_a": True, "task_seed": 0, "task_a_hash": shared, "scientific_result": False},
    )
    cond = next(row for row in MODULUS_CONDITIONS if row["name"] == "s0_o0_m1")
    run_modulus_job(cfg, cond, "b_only", 0, 0)
    run_modulus_job(cfg, cond, "sequential_ab", 0, 0)
    source = _ensure_source(cfg, 0, 0)
    before = {
        name: _sha256(source / name) for name in ("theta_0.pt", "theta_A.pt")
    }
    _materialize_default_hybrids(cfg, source, out / "hybrids")
    after = {name: _sha256(source / name) for name in ("theta_0.pt", "theta_A.pt")}
    if before != after:
        raise RuntimeError("smoke hybrid build modified theta checkpoints")
    built = sorted(p.name for p in (out / "hybrids").iterdir() if p.is_dir())
    if built != sorted(DEFAULT_INTERVENTIONS):
        raise RuntimeError(f"expected 12 hybrids, found {built}")
    run_intervention_job(cfg, "full_fresh", 0, 0)
    run_intervention_job(cfg, "reset_mlp", 0, 0)
    modulus_paths = analyze_modulus_dir(out)
    component_paths = analyze_component_dir(out)
    history = _read_jsonl(
        out / "jobs" / "s0_o0_m1" / "sequential_ab" / "ts0_ms0" / "eval_history.jsonl"
    )
    if not any("B_test_acc/op0" in row for row in history):
        raise RuntimeError("smoke sequential history has no per-op B accuracy")
    if not (source / "theta_0.pt").is_file() or not (source / "theta_A.pt").is_file():
        raise RuntimeError("smoke did not save theta_0 and theta_A")
    print("[smoke] ok (not a scientific result)")
    print(f"[smoke] {modulus_paths}")
    print(f"[smoke] {component_paths}")


def _run_grid(args: argparse.Namespace, *, experiment: str) -> None:
    cfg = TransferConfig.from_yaml(args.config)
    if cfg.experiment != experiment:
        raise SystemExit(
            f"{args.config} is experiment {cfg.experiment}, not {experiment}"
        )
    mode = _mode(args)
    seed_set = args.seed_set or cfg.seed_set
    for task_seed in cfg.task_seeds:
        digest = assert_shared_task_a(int(task_seed), MODULUS_CONDITIONS)
        print(f"[dry-check] task_seed={task_seed} fixed_a task_a_hash={digest}")
    jobs = (
        modulus_jobs(cfg, seed_set)
        if experiment == "modulus_specificity"
        else component_jobs(cfg, seed_set)
    )
    pilot_n = len(
        modulus_jobs(cfg, "pilot")
        if experiment == "modulus_specificity"
        else component_jobs(cfg, "pilot")
    )
    final_n = len(
        modulus_jobs(cfg, "final")
        if experiment == "modulus_specificity"
        else component_jobs(cfg, "final")
    )
    print(
        f"[transfer] {experiment} mode={mode} seed_set={seed_set} "
        f"jobs={len(jobs)} pilot_jobs={pilot_n} final_jobs={final_n} "
        f"out={cfg.out_dir}"
    )
    commands = [format_job_command(str(args.config), job) for job in jobs]
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    header = [
        f"# {experiment} mode={mode} seed_set={seed_set} n_jobs={len(jobs)}",
        f"# pilot_jobs={pilot_n} final_jobs={final_n}",
        "# dry-run does not train. Pass --execute to run these commands.",
    ]
    (out / "dry_run_commands.txt").write_text(
        "\n".join(header + commands) + "\n", encoding="utf-8"
    )
    for line in commands:
        print(line)
    if mode != "execute":
        print("[transfer] dry-run only; not starting training")
        return
    print(f"[transfer] executing {len(jobs)} jobs on device={cfg.device}")
    for job in jobs:
        if job["kind"] == "modulus":
            run_modulus_job(
                cfg,
                _condition(cfg, job["condition"]),
                job["protocol"],
                job["task_seed"],
                job["model_seed"],
            )
        elif job["kind"] == "source":
            run_source_job(cfg, job["task_seed"], job["model_seed"])
        else:
            run_intervention_job(
                cfg, job["intervention"], job["task_seed"], job["model_seed"]
            )
    if experiment == "modulus_specificity":
        analyze_modulus_dir(cfg.out_dir)
    else:
        analyze_component_dir(cfg.out_dir)


def _mode(args: argparse.Namespace) -> str:
    execute = bool(getattr(args, "execute", False))
    dry = bool(getattr(args, "dry_run", False))
    if execute and dry:
        raise SystemExit("pass only one of --dry-run and --execute")
    if execute:
        return "execute"
    return "dry-run"


def _condition(cfg: TransferConfig, name: str) -> dict[str, Any]:
    for cond in cfg.conditions:
        if cond["name"] == name:
            return cond
    raise KeyError(f"condition {name} is not in the config")


def _prepare(cfg: TransferConfig, condition: dict[str, Any], task_seed: int) -> dict[str, Any]:
    meta = prepare_phase2_dataset(
        Path(cfg.out_dir),
        rho_slot=float(condition["rho_slot"]),
        rho_operand=float(condition["rho_operand"]),
        rho_mod=float(condition["rho_mod"]),
        task_seed=int(task_seed),
        data_seed=int(cfg.data_seed),
        n_aliases=int(cfg.n_aliases),
        train_frac=float(cfg.train_frac),
        direction="forward",
        fixed_a=True,
    )
    data_dir = Path(meta["data_dir"])
    manifest = DataManifest.load(data_dir / "manifest.json")
    digest = task_a_hash(manifest.task_pair)
    pair = build_task_pair(
        rho_slot=float(condition["rho_slot"]),
        rho_operand=float(condition["rho_operand"]),
        rho_mod=float(condition["rho_mod"]),
        task_seed=int(task_seed),
        fixed_a=True,
    )
    if task_a_hash(pair) != digest:
        raise RuntimeError("on-disk Task A does not match fixed_a construction")
    manifest.fixed_a = True
    manifest.task_a_hash = digest
    manifest.save(data_dir / "manifest.json")
    loaded = DataManifest.load(data_dir / "manifest.json")
    if not loaded.fixed_a or loaded.task_a_hash != digest:
        raise RuntimeError("failed to record fixed_a=true on the data manifest")
    meta["task_a_hash"] = digest
    meta["fixed_a"] = True
    return meta


def _require_stamped(manifest: DataManifest) -> None:
    if not manifest.fixed_a or not manifest.task_a_hash:
        raise RuntimeError("dataset manifest must record fixed_a=true and task_a_hash")


def _ensure_source(cfg: TransferConfig, task_seed: int, model_seed: int) -> Path:
    cond = cfg.conditions[0]
    meta = _prepare(cfg, cond, task_seed)
    source = Path(cfg.out_dir) / "sources" / f"ts{task_seed}_ms{model_seed}"
    source.mkdir(parents=True, exist_ok=True)
    lock_path = source / ".lock"
    with lock_path.open("w", encoding="utf-8") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        marker = source / "source_complete.json"
        signature = _source_signature(cfg, task_seed, model_seed, meta)
        if marker.is_file():
            saved = json.loads(marker.read_text(encoding="utf-8"))
            if saved.get("signature") == signature and (source / "theta_A.pt").is_file():
                return source
        manifest = DataManifest.load(Path(meta["data_dir"]) / "manifest.json")
        _require_stamped(manifest)
        device = torch.device(cfg.device)
        loaders = _loaders(Path(meta["data_dir"]), cfg)
        _seed(model_seed, device)
        model = _new_model(cfg, device)
        write_json(source / "parameter_group_manifest.json", parameter_group_manifest(model))
        _save_theta(
            source / "theta_0.pt",
            model,
            manifest=manifest,
            task_seed=task_seed,
            model_seed=model_seed,
            step=0,
        )
        model, opt_a, a_history, a_step = _train_phase(
            model, loaders, cfg, device, task="A", run_name="phase_a"
        )
        write_history(source / "history_a.jsonl", a_history)
        _save_theta(
            source / "theta_A.pt",
            model,
            manifest=manifest,
            task_seed=task_seed,
            model_seed=model_seed,
            step=a_step,
            optimizer=opt_a,
            carry_optimizer_state=cfg.carry_optimizer_state,
        )
        write_json(
            marker,
            {
                "signature": signature,
                "data_dir": meta["data_dir"],
                "dataset_hash": manifest.dataset_hash,
                "task_a_hash": manifest.task_a_hash,
                "fixed_a": True,
                "task_seed": task_seed,
                "model_seed": model_seed,
            },
        )
    return source


def _source_signature(
    cfg: TransferConfig, task_seed: int, model_seed: int, meta: dict[str, Any]
) -> str:
    body = {
        "task_seed": task_seed,
        "model_seed": model_seed,
        "dataset_hash": meta.get("dataset_hash") or meta.get("task_a_hash"),
        "task_a_hash": meta.get("task_a_hash"),
        "steps": cfg.steps,
        "lr": cfg.lr,
        "weight_decay": cfg.weight_decay,
        "batch_size": cfg.batch_size,
        "d_model": cfg.d_model,
        "n_layers": cfg.n_layers,
        "n_heads": cfg.n_heads,
        "d_mlp": cfg.d_mlp,
        "carry_optimizer_state": cfg.carry_optimizer_state,
        "b_sampler_seed": cfg.b_sampler_seed,
    }
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode("utf-8")).hexdigest()


def _materialize_default_hybrids(
    cfg: TransferConfig, source: Path, dest_root: Path
) -> None:
    meta = json.loads((source / "source_complete.json").read_text(encoding="utf-8"))
    manifest = DataManifest.load(Path(meta["data_dir"]) / "manifest.json")
    _payload_0, theta_0 = load_theta_checkpoint(source / "theta_0.pt")
    _payload_a, theta_a = load_theta_checkpoint(source / "theta_A.pt")
    model = _new_model(cfg, torch.device("cpu"))
    registry = build_registry(model)
    for name in DEFAULT_INTERVENTIONS:
        hybrid, hybrid_manifest = build_hybrid(
            theta_0,
            theta_a,
            name,
            registry,
            n_layers=cfg.n_layers,
            task_seed=int(meta["task_seed"]),
            model_seed=int(meta["model_seed"]),
            dataset_hash=manifest.dataset_hash,
        )
        dest = dest_root / name
        write_json(dest / "hybrid_manifest.json", hybrid_manifest)
        save_hybrid_checkpoint(
            dest / "hybrid.pt",
            hybrid,
            model_config=model.cfg.to_dict(),
            manifest=hybrid_manifest,
        )


def _train_b_fresh(
    model: ModularTransformer,
    loaders: dict[str, Any],
    cfg: TransferConfig,
    device: torch.device,
    history: list[dict[str, Any]],
) -> None:
    pre = _eval_point(model, loaders, device)
    history.append(_b_row(0.0, pre, pre_b=True))
    tcfg = _train_config(cfg, device)
    seg = train_segment(
        model,
        loaders["B"]["train"],
        cfg=tcfg,
        optimizer=None,
        start_step=0,
        optimizer_transition="fresh",
        eval_loaders=_eval_loaders(loaders),
        ckpt_dir=None,
        run_name="phase_b",
        track_events=False,
    )
    for row in seg.state.eval_history:
        history.append(_b_row(float(row["step"]), row, pre_b=False))


def _train_b_from_optimizer(
    model: ModularTransformer,
    loaders: dict[str, Any],
    cfg: TransferConfig,
    device: torch.device,
    *,
    optimizer: torch.optim.Optimizer,
    history: list[dict[str, Any]],
) -> ModularTransformer:
    pre = _eval_point(model, loaders, device)
    history.append(_b_row(0.0, pre, pre_b=True))
    tcfg = _train_config(cfg, device)
    seg = train_segment(
        model,
        loaders["B"]["train"],
        cfg=tcfg,
        optimizer=optimizer,
        start_step=0,
        optimizer_transition="preserve",
        eval_loaders=_eval_loaders(loaders),
        ckpt_dir=None,
        run_name="phase_b",
        track_events=False,
    )
    for row in seg.state.eval_history:
        history.append(_b_row(float(row["step"]), row, pre_b=False))
    return seg.model


def _train_phase(
    model: ModularTransformer,
    loaders: dict[str, Any],
    cfg: TransferConfig,
    device: torch.device,
    *,
    task: str,
    run_name: str,
) -> tuple[ModularTransformer, torch.optim.Optimizer, list[dict[str, Any]], int]:
    tcfg = _train_config(cfg, device)
    seg = train_segment(
        model,
        loaders[task]["train"],
        cfg=tcfg,
        optimizer=None,
        start_step=0,
        optimizer_transition="fresh",
        eval_loaders=_eval_loaders(loaders) if task == "B" else {
            "A_val": loaders["A"]["val"],
            "A_test": loaders["A"]["test"],
        },
        ckpt_dir=None,
        run_name=run_name,
        track_events=False,
    )
    return seg.model, seg.optimizer, list(seg.state.eval_history), int(seg.state.step)


def _finalize_job(
    job_dir: Path,
    cfg: TransferConfig,
    history: list[dict[str, Any]],
    *,
    condition: str,
    protocol: str,
    task_seed: int,
    model_seed: int,
    operations: list[dict[str, Any]],
    manifest: DataManifest,
    intervention: str | None,
    kind: str | None = None,
    groups_from_A: list[str] | None = None,
    groups_from_init: list[str] | None = None,
) -> None:
    write_history(job_dir / "eval_history.jsonl", history)
    threshold = float(cfg.gen_threshold)
    window = int(cfg.stable_window)
    # Dynamic curves use validation. Test stays an endpoint, recorded below.
    curve_split = (
        "validation" if any("B_val_acc" in row for row in history) else "test"
    )
    curve_prefix = "B_val_acc" if curve_split == "validation" else "B_test_acc"

    def _curve(key: str) -> list[tuple[float, float]]:
        points: list[tuple[float, float]] = []
        for row in history:
            value = row.get(key)
            if "b_exposure" not in row:
                continue
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            points.append((float(row["b_exposure"]), float(value)))
        return points

    aggregate = summarize_series(_curve(curve_prefix), threshold=threshold, window=window)
    per_op: dict[str, Any] = {}
    for op in operations:
        per_op[op["operation"]] = {
            **op,
            **summarize_series(
                _curve(f"{curve_prefix}/{op['operation']}"),
                threshold=threshold,
                window=window,
            ),
        }
    pre = next((row for row in history if row.get("pre_b")), {})
    last = history[-1] if history else {}
    metrics: dict[str, Any] = {
        "status": "ok",
        "condition": condition,
        "protocol": protocol,
        "task_seed": int(task_seed),
        "model_seed": int(model_seed),
        "fixed_a": True,
        "task_a_hash": manifest.task_a_hash,
        "dataset_hash": manifest.dataset_hash,
        "replay_ratio": 0.0,
        "carry_optimizer_state": bool(cfg.carry_optimizer_state),
        "optimizer_transition": "preserve" if cfg.carry_optimizer_state else "fresh",
        "b_sampler_seed": int(cfg.b_sampler_seed),
        "gen_threshold": threshold,
        "stable_window": window,
        "curve_split": curve_split,
        "selection_protocol": "validation_curve_test_endpoint",
        "operations": operations,
        "per_op": per_op,
        "b_curve": aggregate,
        "b_exposure_auc": aggregate["b_exposure_auc"],
        "first_reach_steps_to_gen": aggregate["first_reach_steps_to_gen"],
        "stable_steps_to_gen": aggregate["stable_steps_to_gen"],
        "b_exposure_steps_to_gen": aggregate["b_exposure_steps_to_gen"],
        "pre_B/A_test_acc": pre.get("A_test_acc"),
        "pre_B/B_test_acc": pre.get("B_test_acc"),
        "pre_B/B_test_loss": pre.get("B_test_loss"),
        "B_test_acc": last.get("B_test_acc"),
        "B_test_loss": last.get("B_test_loss"),
        "B_test_macro_op_acc": last.get("B_test_macro_op_acc"),
        "final_B_acc": last.get("B_test_acc"),
        "scientific_result": False if "smoke" in str(job_dir) else True,
    }
    if intervention is not None:
        metrics["intervention"] = intervention
        metrics["kind"] = kind
        metrics["groups_from_A"] = groups_from_A or []
        metrics["groups_from_init"] = groups_from_init or []
    write_json(job_dir / "metrics.json", metrics)
    write_json(
        job_dir / "config_resolved.json",
        {
            "experiment": cfg.experiment,
            "condition": condition,
            "protocol": protocol,
            "intervention": intervention,
            "steps": cfg.steps,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "batch_size": cfg.batch_size,
            "eval_every": cfg.eval_every,
            "device": cfg.device,
            "fixed_a": True,
            "replay_ratio": 0.0,
            "carry_optimizer_state": cfg.carry_optimizer_state,
        },
    )


def _save_theta(
    path: Path,
    model: ModularTransformer,
    *,
    manifest: DataManifest,
    task_seed: int,
    model_seed: int,
    step: int,
    optimizer: torch.optim.Optimizer | None = None,
    carry_optimizer_state: bool = False,
) -> None:
    save_theta_checkpoint(
        path,
        model,
        task_pair=manifest.task_pair.to_dict(),
        dataset_hash=manifest.dataset_hash,
        task_seed=task_seed,
        model_seed=model_seed,
        step=step,
        optimizer=optimizer,
        carry_optimizer_state=carry_optimizer_state,
    )


def _eval_point(model: ModularTransformer, loaders: dict[str, Any], device: torch.device) -> dict[str, float]:
    row: dict[str, float] = {}
    for task in ("A", "B"):
        for split in ("val", "test"):
            result = evaluate(model, loaders[task][split], device)
            name = f"{task}_{split}"
            row[f"{name}_acc"] = float(result.accuracy)
            row[f"{name}_loss"] = float(result.loss)
            row[f"{name}_macro_op_acc"] = float(result.macro_operation_accuracy)
            row.update(operation_acc_metrics(result.by_operation, prefix=f"{name}_acc"))
    return row


def _b_row(exposure: float, source: dict[str, Any], *, pre_b: bool) -> dict[str, Any]:
    row: dict[str, Any] = {
        "step": exposure,
        "b_exposure": float(exposure),
        "curve": "B",
        "pre_b": bool(pre_b),
    }
    for key, value in source.items():
        if key in {"step", "b_exposure", "curve", "pre_b", "segment"}:
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            continue
        row[key] = value
    return row


def _eval_loaders(loaders: dict[str, Any]) -> dict[str, Any]:
    return {
        "A_val": loaders["A"]["val"],
        "A_test": loaders["A"]["test"],
        "B_val": loaders["B"]["val"],
        "B_test": loaders["B"]["test"],
    }


def _loaders(data_dir: Path, cfg: TransferConfig) -> dict[str, Any]:
    return task_loaders(
        data_dir,
        batch_size=int(cfg.batch_size),
        train_replacement=True,
        train_seed=int(cfg.b_sampler_seed),
        sampler_seed=int(cfg.b_sampler_seed),
        eval_n_per_operation=int(cfg.eval_n_per_operation),
        eval_seed=int(cfg.eval_seed),
    )


def _train_config(cfg: TransferConfig, device: torch.device) -> TrainConfig:
    return TrainConfig(
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
        batch_size=int(cfg.batch_size),
        train_replacement=True,
        max_steps=int(cfg.steps),
        eval_every=int(cfg.eval_every),
        ckpt_every=max(int(cfg.steps), 1),
        grad_clip=1.0,
        device=str(device),
        compile_model=bool(cfg.compile_model),
        log_to_wandb=False,
        null_task_tokens=False,
    )


def _new_model(cfg: TransferConfig, device: torch.device) -> ModularTransformer:
    model = ModularTransformer(
        ModelConfig(
            d_model=int(cfg.d_model),
            n_layers=int(cfg.n_layers),
            n_heads=int(cfg.n_heads),
            d_mlp=int(cfg.d_mlp),
            dropout=0.0,
        )
    )
    return model.to(device)


def _seed(seed: int, device: torch.device) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)


def _load_state(
    model: ModularTransformer, state: dict[str, torch.Tensor], device: torch.device
) -> None:
    moved = {key: value.to(device) for key, value in state.items()}
    incompatible = model.load_state_dict(moved, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "hybrid did not load: "
            f"missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
