"""Task-token counterfactuals, attention routing, and a residual probe."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from go4cl.analysis.attention import query_attention_to_operands
from go4cl.analysis.probes import fit_linear_probe
from go4cl.constants import QUERY_TOKEN_IDS, SEQ_LEN_OPERANDS
from go4cl.data.context import ContextBuilder
from go4cl.data.eval_contexts import build_eval_examples
from go4cl.data.residue_pairs import ResiduePairSplit
from go4cl.metrics.behavioral import evaluate
from go4cl.model.transformer import ModularTransformer
from go4cl.tasks.partition import PARTITION_ORDER
from go4cl.tasks.spec import TaskSpec


def _per_query(task: TaskSpec, result) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for op in sorted(task.operations, key=lambda item: item.slot):
        key = f"task{task.task_id}/lat{op.latent_id}/slot{op.slot}"
        stats = result.by_operation.get(key, {})
        rows.append(
            {
                "query": f"Q{op.slot}",
                "slot": int(op.slot),
                "operands": [int(op.i), int(op.j)],
                "modulus": int(op.modulus),
                "accuracy": float(stats.get("accuracy", float("nan"))),
                "loss": float(stats.get("loss", float("nan"))),
                "n": float(stats.get("n", 0.0)),
            }
        )
    return rows


@torch.no_grad()
def evaluate_split(
    model: ModularTransformer,
    tasks: dict[str, TaskSpec],
    loaders: dict[str, dict],
    device: torch.device,
    split: str,
) -> dict[str, Any]:
    model.eval()
    out: dict[str, Any] = {}
    for name in PARTITION_ORDER:
        result = evaluate(model, loaders[name][split], device)
        out[name] = {
            "accuracy": float(result.accuracy),
            "loss": float(result.loss),
            "macro_operation_accuracy": float(result.macro_operation_accuracy),
            "per_query": _per_query(tasks[name], result),
        }
    return out


def _predict(model: ModularTransformer, tokens: np.ndarray, device: torch.device) -> np.ndarray:
    preds: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, tokens.shape[0], 256):
            batch = torch.as_tensor(tokens[start : start + 256], device=device)
            logits = model(batch)["logits"]
            preds.append(logits.argmax(dim=-1).detach().cpu().numpy())
    return np.concatenate(preds, axis=0)


def counterfactual_matrix(
    model: ModularTransformer,
    tasks: dict[str, TaskSpec],
    splits: dict[int, ResiduePairSplit],
    *,
    n_contexts: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """M[u, v] = P(model(x, Q, TASK_u) predicts the label of task v).

    Off-diagonal cells use samples whose two labels differ. Diagonal cells
    use samples whose label differs from at least one other task, so shared
    answers do not count as evidence of routing.
    """
    builder = ContextBuilder(tasks["A"], splits)
    rng = np.random.default_rng(int(seed))
    digits = np.asarray(
        [
            builder.sample_packed_digits(rng, default_split="test").digits
            for _ in range(int(n_contexts))
        ],
        dtype=np.int64,
    )
    n_ctx = int(digits.shape[0])
    n_tasks = len(PARTITION_ORDER)
    n_queries = len(QUERY_TOKEN_IDS)
    labels = np.zeros((n_tasks, n_queries, n_ctx), dtype=np.int64)
    tokens = np.zeros((n_tasks, n_queries, n_ctx, SEQ_LEN_OPERANDS + 2), dtype=np.int64)
    for task_index, name in enumerate(PARTITION_ORDER):
        task = tasks[name]
        for op in task.operations:
            labels[task_index, op.slot] = (digits[:, op.i] + digits[:, op.j]) % op.modulus
            tokens[task_index, op.slot, :, :SEQ_LEN_OPERANDS] = digits
            tokens[task_index, op.slot, :, SEQ_LEN_OPERANDS] = task.task_token
            tokens[task_index, op.slot, :, SEQ_LEN_OPERANDS + 1] = QUERY_TOKEN_IDS[op.slot]
    preds = _predict(model, tokens.reshape(-1, SEQ_LEN_OPERANDS + 2), device).reshape(
        n_tasks, n_queries, n_ctx
    )

    matrix = np.full((n_tasks, n_tasks), np.nan)
    counts = np.zeros((n_tasks, n_tasks), dtype=np.int64)
    for u in range(n_tasks):
        for v in range(n_tasks):
            if u == v:
                mask = np.zeros((n_queries, n_ctx), dtype=bool)
                for other in range(n_tasks):
                    if other != u:
                        mask |= labels[u] != labels[other]
            else:
                mask = labels[u] != labels[v]
            counts[u, v] = int(mask.sum())
            if counts[u, v] == 0:
                continue
            matrix[u, v] = float((preds[u][mask] == labels[v][mask]).mean())

    ab_mask = labels[0] != labels[1]
    ab_n = int(ab_mask.sum())
    ab = np.full((2, 2), np.nan)
    flip = float("nan")
    if ab_n:
        for u in range(2):
            for v in range(2):
                ab[u, v] = float((preds[u][ab_mask] == labels[v][ab_mask]).mean())
        flip = float((preds[0][ab_mask] != preds[1][ab_mask]).mean())

    by_query: list[dict[str, Any]] = []
    for slot in range(n_queries):
        slot_mask = labels[0, slot] != labels[1, slot]
        slot_n = int(slot_mask.sum())
        slot_ab = np.full((2, 2), np.nan)
        if slot_n:
            for u in range(2):
                for v in range(2):
                    slot_ab[u, v] = float(
                        (preds[u, slot][slot_mask] == labels[v, slot][slot_mask]).mean()
                    )
        by_query.append(
            {
                "query": f"Q{slot}",
                "n_disagree_ab": slot_n,
                "ab_matrix": slot_ab.tolist(),
            }
        )
    return {
        "task_names": list(PARTITION_ORDER),
        "matrix": matrix.tolist(),
        "counts": counts.tolist(),
        "ab_matrix": ab.tolist(),
        "n_disagree_ab": ab_n,
        "task_token_flip_rate": flip,
        "n_contexts": n_ctx,
        "by_query": by_query,
        "definition": (
            "M[u, v] = P(model(x, Q, TASK_u) predicts label of task v), "
            "restricted to samples whose compared labels differ"
        ),
    }


def _layer_mean(layers: list[dict[str, Any]], key: str) -> float:
    if not layers:
        return float("nan")
    return float(sum(float(layer[key]) for layer in layers) / len(layers))


@torch.no_grad()
def routing_summary(
    model: ModularTransformer,
    tasks: dict[str, TaskSpec],
    splits: dict[int, ResiduePairSplit],
    *,
    n_per_operation: int,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    """Query-row attention onto the task token and onto that task's operands."""
    model.eval()
    per_task: dict[str, Any] = {}
    for name in PARTITION_ORDER:
        task = tasks[name]
        examples = build_eval_examples(
            task,
            splits,
            target_split="test",
            context_mode="packed_id",
            distractor_split="train",
            n_per_operation=int(n_per_operation),
            seed=int(seed) + 100 * int(task.task_id),
        )
        grouped: dict[int, list] = {op.slot: [] for op in task.operations}
        for example in examples:
            grouped[int(example.slot)].append(example)
        rows: list[dict[str, Any]] = []
        for slot, group in sorted(grouped.items()):
            op = task.by_slot()[slot]
            tokens = torch.tensor([example.tokens for example in group], device=device)
            cache = model.forward_with_cache(tokens)
            stats = query_attention_to_operands(
                cache["attn"], operand_i=op.i, operand_j=op.j
            )
            layers = stats["layers"]
            rows.append(
                {
                    "query": f"Q{slot}",
                    "operands": [int(op.i), int(op.j)],
                    "task_mass": _layer_mean(layers, "task_mass"),
                    "operand_mass": float(stats["mean_operand_mass"]),
                    "other_digit_mass": _layer_mean(layers, "other_digit_mass"),
                    "last_layer_task_mass": float(layers[-1]["task_mass"]) if layers else None,
                    "last_layer_operand_mass": float(stats["last_layer_operand_mass"]),
                }
            )
        per_task[name] = {
            "task_mass": float(np.mean([row["task_mass"] for row in rows])),
            "operand_mass": float(np.mean([row["operand_mass"] for row in rows])),
            "other_digit_mass": float(np.mean([row["other_digit_mass"] for row in rows])),
            "per_query": rows,
        }
    return per_task


def _probe_split(
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    n_classes: int,
    steps: int,
    seed: int,
) -> dict[str, float | int]:
    rng = np.random.default_rng(int(seed))
    order = rng.permutation(int(features.shape[0]))
    n_train = max(int(round(0.75 * len(order))), 1)
    train_idx = order[:n_train]
    held_idx = order[n_train:]
    if len(held_idx) == 0:
        held_idx = train_idx
    fitted = fit_linear_probe(
        features[train_idx],
        labels[train_idx],
        {"heldout": features[held_idx]},
        {"heldout": labels[held_idx]},
        n_classes=n_classes,
        steps=int(steps),
    )
    return {
        "train_acc": float(fitted["train_acc"]),
        "heldout_acc": float(fitted["eval_acc"]["heldout"]),
        "n_train": int(len(train_idx)),
        "n_heldout": int(len(held_idx)),
    }


def task_identity_probe(
    model: ModularTransformer,
    tasks: dict[str, TaskSpec],
    splits: dict[int, ResiduePairSplit],
    *,
    n_per_operation: int,
    seed: int,
    steps: int,
    device: torch.device,
) -> dict[str, Any]:
    """Linear readout of task identity from the final query residual."""
    model.eval()
    features: list[torch.Tensor] = []
    labels: list[torch.Tensor] = []
    with torch.no_grad():
        for name in PARTITION_ORDER:
            task = tasks[name]
            examples = build_eval_examples(
                task,
                splits,
                target_split="test",
                context_mode="packed_id",
                distractor_split="train",
                n_per_operation=int(n_per_operation),
                seed=int(seed) + 900 + int(task.task_id),
            )
            tokens = torch.tensor([example.tokens for example in examples], device=device)
            residual = model.forward_with_cache(tokens)["query_resid"].detach().cpu()
            features.append(residual)
            labels.append(
                torch.full((residual.shape[0],), int(task.task_id), dtype=torch.long)
            )
    x = torch.cat(features, dim=0)
    y = torch.cat(labels, dim=0)
    three_way = _probe_split(x, y, n_classes=3, steps=steps, seed=seed)
    ab = y < 2
    two_way = _probe_split(
        x[ab],
        y[ab],
        n_classes=2,
        steps=steps,
        seed=int(seed) + 1,
    )
    return {
        "probe_acc_3way": three_way["heldout_acc"],
        "probe_acc_ab": two_way["heldout_acc"],
        "probe_3way": three_way,
        "probe_ab": two_way,
    }


def analyze_checkpoint(
    model: ModularTransformer,
    tasks: dict[str, TaskSpec],
    splits: dict[int, ResiduePairSplit],
    loaders: dict[str, dict],
    *,
    device: torch.device,
    n_contexts: int,
    n_per_operation: int,
    probe_steps: int,
    seed: int,
) -> dict[str, Any]:
    factual = counterfactual_matrix(
        model,
        tasks,
        splits,
        n_contexts=n_contexts,
        seed=seed,
        device=device,
    )
    routing = routing_summary(
        model,
        tasks,
        splits,
        n_per_operation=n_per_operation,
        seed=seed + 17,
        device=device,
    )
    probe = task_identity_probe(
        model,
        tasks,
        splits,
        n_per_operation=n_per_operation,
        seed=seed + 29,
        steps=probe_steps,
        device=device,
    )
    return {
        "test": evaluate_split(model, tasks, loaders, device, "test"),
        "val": evaluate_split(model, tasks, loaders, device, "val"),
        "counterfactual": factual,
        "routing": routing,
        "probe": probe,
    }
