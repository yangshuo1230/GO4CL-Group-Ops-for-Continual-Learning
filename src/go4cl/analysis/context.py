"""Load a trained job for analysis: paths, checkpoint, operations, datasets.

Allowed dependencies: data, model, utils.checkpoint. Must not import phases.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch

from go4cl.analysis.types import OperationRef
from go4cl.data.dataset import ModularAdditionDataset, make_loader
from go4cl.data.manifest import DataManifest
from go4cl.utils.checkpoint import load_checkpoint

SplitName = Literal["train", "val", "test"]

# Legacy alias used by older tests and scripts.
OpSpec = OperationRef


def resolve_device(name: str | None = None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


SEQUENTIAL_ROLES: dict[str, tuple[str, ...]] = {
    "theta_A": ("theta_A.pt",),
    "phase_b_final": ("phase_b_final.pt",),
    "B_first_stable": ("B_first_stable.pt",),
    "B_best_val": ("B_best_val.pt",),
    # Explicit tradeoff role may fall back to the legacy best.pt copy.
    "AB_tradeoff_best": ("AB_tradeoff_best.pt", "best.pt"),
}

_AMBIGUOUS_SEQUENTIAL = (
    "This sequential job needs an explicit checkpoint role: "
    "theta_A, phase_b_final, B_first_stable, B_best_val, or AB_tradeoff_best. "
    "best.pt is the legacy A/B-tradeoff checkpoint and is not a default."
)


def _is_sequential_job(job_dir: Path) -> bool:
    ckpt = job_dir / "ckpts"
    if any((ckpt / name).is_file() for name in ("theta_A.pt", "phase_b_final.pt", "B_best_val.pt")):
        return True
    cfg_path = job_dir / "config_resolved.json"
    if not cfg_path.is_file():
        return False
    try:
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    protocol = str(cfg.get("protocol", ""))
    return protocol.startswith("sequential")


def resolve_checkpoint(job_dir: Path | str, kind: str) -> Path:
    """Resolve a checkpoint under ``job_dir/ckpts``.

    Phase-1 search order is unchanged:
    final → ``a_only_final.pt`` then ``final.pt``;
    best → ``best.pt`` then ``a_only_best.pt``.

    A sequential job must name a role. ``best`` and ``final`` are rejected
    there because ``best.pt`` is the A/B tradeoff checkpoint.
    """
    job_dir = Path(job_dir)
    ckpt_dir = job_dir / "ckpts"
    if _is_sequential_job(job_dir) and kind in {"best", "final", ""}:
        raise ValueError(_AMBIGUOUS_SEQUENTIAL)
    if kind in SEQUENTIAL_ROLES:
        for name in SEQUENTIAL_ROLES[kind]:
            path = ckpt_dir / name
            if path.is_file():
                return path
        raise FileNotFoundError(
            f"no {kind} checkpoint under {ckpt_dir} "
            f"(looked for {', '.join(SEQUENTIAL_ROLES[kind])})"
        )
    if kind == "final":
        candidates = [ckpt_dir / "a_only_final.pt", ckpt_dir / "final.pt"]
    elif kind == "best":
        candidates = [ckpt_dir / "best.pt", ckpt_dir / "a_only_best.pt"]
    else:
        candidates = [
            ckpt_dir / f"{kind}.pt",
            ckpt_dir / f"a_only_{kind}.pt",
            Path(kind),
        ]
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(f"no {kind} checkpoint under {ckpt_dir}")


def try_resolve_checkpoint(job_dir: Path | str, kind: str) -> Path | None:
    try:
        return resolve_checkpoint(job_dir, kind)
    except FileNotFoundError:
        return None


def resolve_data_dir(job_dir: Path | str) -> Path:
    """Find the dataset directory that produced ``job_dir``.

    Order: ``config_resolved.json`` paths, ``job_result.json``, then
    ``<stamp>/data/<experiment_id>`` inferred from the job folder name.
    """
    job_dir = Path(job_dir)
    cfg_path = job_dir / "config_resolved.json"
    if cfg_path.is_file():
        cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        for key in ("data_dir", "data_root", "dataset_dir"):
            found = _manifest_dir(cfg.get(key))
            if found is not None:
                return found
        for nest in ("data", "task", "dataset", "train"):
            block = cfg.get(nest) or {}
            if isinstance(block, dict):
                for key in ("data_dir", "data_root", "dir", "path"):
                    found = _manifest_dir(block.get(key))
                    if found is not None:
                        return found

    result_path = job_dir / "job_result.json"
    if result_path.is_file():
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        found = _manifest_dir(payload.get("data_dir"))
        if found is not None:
            return found

    stamp_root = job_dir.parent.parent
    exp_id = job_dir.name.split("__", 1)[0]
    data_dir = stamp_root / "data" / exp_id
    if (data_dir / "manifest.json").is_file():
        return data_dir
    data_root = stamp_root / "data"
    candidates: list[Path] = []
    if data_root.is_dir():
        candidates = sorted(
            hit
            for hit in data_root.glob(exp_id + "*")
            if (hit / "manifest.json").is_file()
        )
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        listed = "\n".join(f"  - {p}" for p in candidates)
        raise FileNotFoundError(
            f"ambiguous data dir for {job_dir}: glob {exp_id!r}* matched "
            f"{len(candidates)} manifests. Specify data_dir in "
            f"config_resolved.json.\n{listed}"
        )
    raise FileNotFoundError(
        f"cannot resolve data dir for {job_dir}: no manifest under "
        f"{data_root / exp_id} and no unique {exp_id}* fallback"
    )


def _manifest_dir(value: Any) -> Path | None:
    if not value:
        return None
    cand = Path(str(value))
    if (cand / "manifest.json").is_file():
        return cand
    return None


def operations_from_manifest(manifest: DataManifest) -> list[OperationRef]:
    return [
        OperationRef(
            latent_id=int(op.latent_id),
            modulus=int(op.modulus),
            operand_i=int(op.i),
            operand_j=int(op.j),
            slot=int(op.slot),
        )
        for op in manifest.task_pair.task_a.operations
    ]


def subset_dataset(
    ds: ModularAdditionDataset, idx: np.ndarray
) -> ModularAdditionDataset:
    return ModularAdditionDataset(
        tokens=ds.tokens[idx].cpu().numpy(),
        labels=ds.labels[idx].cpu().numpy(),
        slots=ds.slots[idx].cpu().numpy(),
        moduli=ds.moduli[idx].cpu().numpy(),
        task_ids=ds.task_ids[idx].cpu().numpy(),
        latent_ids=ds.latent_ids[idx].cpu().numpy(),
    )


def filter_by_modulus(
    ds: ModularAdditionDataset, modulus: int
) -> ModularAdditionDataset:
    """Aggregate filter by modulus (derived only). Prefer ``filter_by_operation``."""
    mask = ds.moduli.cpu().numpy() == int(modulus)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"no samples with modulus={modulus}")
    return subset_dataset(ds, idx)


def filter_by_operation(
    ds: ModularAdditionDataset,
    *,
    latent_id: int | None = None,
    slot: int | None = None,
) -> ModularAdditionDataset:
    """Filter to one operation. Prefer ``latent_id``; ``slot`` is fallback."""
    if latent_id is None and slot is None:
        raise ValueError("need latent_id or slot")
    if latent_id is not None and int(latent_id) >= 0:
        lids = ds.latent_ids.cpu().numpy()
        if np.any(lids >= 0):
            mask = lids == int(latent_id)
        else:
            if slot is None:
                raise ValueError(
                    "dataset has no latent_ids; pass slot= for legacy filter"
                )
            mask = ds.slots.cpu().numpy() == int(slot)
    else:
        mask = ds.slots.cpu().numpy() == int(slot)
    idx = np.where(mask)[0]
    if len(idx) == 0:
        raise ValueError(f"no samples for latent_id={latent_id} slot={slot}")
    return subset_dataset(ds, idx)


def op_report_key(rep: OperationRef | dict[str, Any]) -> str:
    if isinstance(rep, OperationRef):
        return rep.report_key()
    return (
        f"lat{rep['latent_id']}/slot{rep['slot']}/p{rep['modulus']}"
    )


@dataclass
class AnalysisContext:
    """Resolved job + optional loaded model for multi-op / single-op analysis."""

    job_dir: Path
    data_dir: Path
    ckpt_path: Path
    ckpt_kind: str
    manifest: DataManifest
    operations: list[OperationRef]
    device: torch.device
    model: Any = None
    payload: dict[str, Any] | None = None
    aliases_per_pair: int = 4

    def disk_dataset(self, split: SplitName) -> ModularAdditionDataset:
        return ModularAdditionDataset.from_disk(self.data_dir, "A", split, 0)

    def analysis_examples(
        self, *, split: SplitName, target_latent_ids: list[int]
    ):
        from go4cl.data.context import build_analysis_dataset

        return build_analysis_dataset(
            self.manifest.task_pair.task_a,
            self.manifest.residue_splits,
            split=split,
            context_mode="packed_id",
            analysis_seed=0,
            aliases_per_pair=self.aliases_per_pair,
            contexts_per_pair=1,
            target_latent_ids=target_latent_ids,
        )

    def operation_dataset(
        self,
        op: OperationRef,
        split: SplitName,
        full: ModularAdditionDataset | None = None,
    ) -> ModularAdditionDataset:
        ds_full = full if full is not None else self.disk_dataset(split)
        if len(ds_full) == 0:
            examples = self.analysis_examples(
                split=split, target_latent_ids=[op.latent_id]
            )
            return ModularAdditionDataset.from_examples(examples, task_id=0)
        return filter_by_operation(ds_full, latent_id=op.latent_id, slot=op.slot)

    def operation_loader(
        self,
        op: OperationRef,
        split: SplitName,
        *,
        batch_size: int = 256,
        full: ModularAdditionDataset | None = None,
    ):
        return make_loader(
            self.operation_dataset(op, split, full=full),
            batch_size=batch_size,
            shuffle=False,
        )


def load_analysis_context(
    job_dir: Path | str,
    *,
    ckpt_kind: str = "best",
    device: torch.device | str | None = None,
    aliases_per_pair: int = 4,
    load_model: bool = True,
) -> AnalysisContext:
    job_dir = Path(job_dir)
    if isinstance(device, str):
        device = torch.device(device)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = resolve_checkpoint(job_dir, ckpt_kind)
    data_dir = resolve_data_dir(job_dir)
    manifest = DataManifest.load(data_dir / "manifest.json")
    ops = operations_from_manifest(manifest)
    model = None
    payload: dict[str, Any] | None = None
    if load_model:
        model, payload = load_checkpoint(ckpt_path, map_location=device)
        model.to(device)
        model.eval()
    return AnalysisContext(
        job_dir=job_dir,
        data_dir=data_dir,
        ckpt_path=ckpt_path,
        ckpt_kind=ckpt_kind,
        manifest=manifest,
        operations=ops,
        device=device,
        model=model,
        payload=payload,
        aliases_per_pair=int(aliases_per_pair),
    )


def analysis_builder_for(
    ctx: AnalysisContext,
) -> Callable[..., Any]:
    def _builder(*, split: str, target_latent_ids: list[int]):
        return ctx.analysis_examples(
            split=split,  # type: ignore[arg-type]
            target_latent_ids=target_latent_ids,
        )

    return _builder
