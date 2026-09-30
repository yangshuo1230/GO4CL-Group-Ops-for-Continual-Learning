"""Train entrypoint for continual-learning protocols."""

from __future__ import annotations

import torch

from go4cl.model.transformer import ModelConfig
from go4cl.train.loop import TrainConfig
from go4cl.train.protocols import run_protocol


def run_train(args) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model_cfg = ModelConfig(d_model=args.d_model, n_layers=args.n_layers)
    # Keep d_mlp = 4 * d_model as in the plan's 64->256 ratio
    model_cfg.d_mlp = 4 * model_cfg.d_model
    train_cfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        batch_size=args.batch_size,
        max_steps=args.steps,
        eval_every=max(args.steps // 10, 1),
        ckpt_every=max(args.steps // 2, 1),
        device=device,
    )
    wandb_enabled = not bool(getattr(args, "no_wandb", False))
    result = run_protocol(
        args.protocol,
        args.data,
        args.out,
        model_cfg=model_cfg,
        train_cfg=train_cfg,
        model_seed=args.model_seed,
        phase_steps=args.steps,
        wandb_enabled=wandb_enabled,
        wandb_project=getattr(args, "wandb_project", "go4cl"),
        wandb_name=getattr(args, "wandb_name", None),
        wandb_mode=getattr(args, "wandb_mode", None),
    )
    print(f"protocol={result.protocol}")
    if result.wandb_url:
        print(f"wandb: {result.wandb_url}")
    for k, v in sorted(result.metrics.items()):
        if isinstance(v, float):
            print(f"  {k}: {v:.4f}")
        elif isinstance(v, dict):
            print(f"  {k}:")
            for kk, vv in v.items():
                if isinstance(vv, float):
                    print(f"    {kk}: {vv:.4f}")
                else:
                    print(f"    {kk}: {vv}")
        else:
            print(f"  {k}: {v}")
