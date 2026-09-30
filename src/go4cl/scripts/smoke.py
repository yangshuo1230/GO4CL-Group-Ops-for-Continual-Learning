"""Engineering smoke checks (pre-science validation from the experiment plan)."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.data.generate import generate_task_datasets, save_datasets
from go4cl.data.residue_pairs import assert_disjoint
from go4cl.metrics.behavioral import evaluate
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.tasks.relations import build_task_pair
from go4cl.train.loop import TrainConfig, train_steps
from go4cl.utils.seed import seed_everything


def _check_residue_disjoint(manifest) -> None:
    for split in manifest.residue_splits.values():
        assert_disjoint(split)


def _single_batch_overfit(
    model: ModularTransformer,
    loader: DataLoader,
    device: torch.device,
    steps: int = 200,
) -> float:
    """Overfit a single batch; expect near-perfect train accuracy."""
    batch = next(iter(loader))
    tokens = batch["tokens"][:32].to(device)
    labels = batch["labels"][:32].to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    model.train()
    for _ in range(steps):
        out = model(tokens, labels)
        opt.zero_grad(set_to_none=True)
        out["loss"].backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        preds = model(tokens)["logits"].argmax(-1)
        acc = float((preds == labels).float().mean().item())
    return acc


def run_smoke(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    quick = bool(args.quick)
    seed_everything(0)

    report: dict = {"device": str(device), "checks": {}}

    # 1) Task construction + residue-pair disjointness
    pair = build_task_pair(
        rho_slot=0.5, rho_operand=0.5, rho_mod=0.5, task_seed=0
    )
    manifest, datasets = generate_task_datasets(
        pair,
        data_seed=0,
        n_aliases_per_pair=1 if quick else 2,
        n_nuisance_contexts=1,
        experiment_id="smoke",
    )
    _check_residue_disjoint(manifest)
    data_root = out / "data"
    save_datasets(data_root, manifest, datasets)
    report["checks"]["residue_disjoint"] = "pass"
    report["checks"]["dataset_hash"] = manifest.dataset_hash
    report["pair"] = {
        "rho_slot": pair.rho_slot,
        "rho_operand": pair.rho_operand,
        "rho_mod": pair.rho_mod,
        "pair_id": pair.pair_id,
    }

    # 2) Model forward shape
    cfg = ModelConfig()
    model = ModularTransformer(cfg).to(device)
    ds = ModularAdditionDataset.from_examples(datasets["A"]["train"][:64], task_id=0)
    loader = make_loader(ds, batch_size=16, shuffle=True)
    batch = next(iter(loader))
    logits = model(batch["tokens"].to(device))["logits"]
    assert logits.shape == (16, cfg.n_classes)
    report["checks"]["forward_shape"] = "pass"
    report["n_params"] = model.num_parameters()

    # 3) Single-batch overfit
    overfit_model = ModularTransformer(cfg).to(device)
    overfit_acc = _single_batch_overfit(
        overfit_model, loader, device, steps=100 if quick else 300
    )
    report["checks"]["single_batch_overfit_acc"] = overfit_acc
    report["checks"]["single_batch_overfit"] = (
        "pass" if overfit_acc >= 0.95 else "warn"
    )

    # 4) Short single-task training
    short_steps = 50 if quick else 200
    train_cfg = TrainConfig(
        lr=1e-3,
        weight_decay=0.1,
        batch_size=64,
        max_steps=short_steps,
        eval_every=max(short_steps // 2, 1),
        ckpt_every=short_steps,
        device=str(device),
    )
    train_ds = ModularAdditionDataset.from_disk(data_root, "A", "train", 0)
    val_ds = ModularAdditionDataset.from_disk(data_root, "A", "val", 0)
    train_loader = make_loader(train_ds, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size=train_cfg.batch_size, shuffle=False)
    short_model = ModularTransformer(cfg).to(device)
    state = train_steps(
        short_model,
        train_loader,
        cfg=train_cfg,
        eval_loaders={"A_val": val_loader},
        ckpt_dir=str(out / "ckpts"),
        run_name="smoke_a",
        log_every=max(short_steps // 5, 1),
    )
    final = evaluate(short_model, val_loader, device)
    report["checks"]["short_single_task"] = {
        "steps": state.step,
        "val_acc": final.accuracy,
        "val_loss": final.loss,
        "status": "pass",
    }

    # 5) Tiny joint / sequential stubs (very short; just verify wiring)
    from go4cl.train.protocols import run_protocol

    joint = run_protocol(
        "joint",
        data_root,
        out / "joint",
        model_cfg=cfg,
        train_cfg=TrainConfig(
            lr=1e-3,
            weight_decay=0.1,
            batch_size=64,
            max_steps=20 if quick else 40,
            eval_every=20 if quick else 40,
            ckpt_every=1000,
            device=str(device),
        ),
        model_seed=0,
        phase_steps=10 if quick else 20,
    )
    report["checks"]["joint_short"] = {
        "metrics_keys": sorted(joint.metrics.keys()),
        "status": "pass",
    }

    seq = run_protocol(
        "sequential_ab",
        data_root,
        out / "sequential_ab",
        model_cfg=cfg,
        train_cfg=TrainConfig(
            lr=1e-3,
            weight_decay=0.1,
            batch_size=64,
            max_steps=20 if quick else 40,
            eval_every=20 if quick else 40,
            ckpt_every=1000,
            device=str(device),
        ),
        model_seed=0,
        phase_steps=10 if quick else 20,
    )
    report["checks"]["sequential_ab_short"] = {
        "has_forgetting": "forgetting_A" in seq.metrics,
        "status": "pass",
    }

    (out / "smoke_report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    print(f"\nsmoke report written to {out / 'smoke_report.json'}")
