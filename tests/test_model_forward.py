"""Tests for model forward pass and data labels."""

from __future__ import annotations

import torch

from go4cl.constants import CONTEXT_LENGTH, NUM_OUTPUT_CLASSES, VOCAB_SIZE
from go4cl.data.generate import generate_task_datasets
from go4cl.model.transformer import ModelConfig, ModularTransformer
from go4cl.tasks.relations import build_task_pair


def test_model_forward_shapes() -> None:
    model = ModularTransformer(ModelConfig())
    tokens = torch.randint(0, VOCAB_SIZE, (8, CONTEXT_LENGTH))
    labels = torch.randint(0, NUM_OUTPUT_CLASSES, (8,))
    out = model(tokens, labels)
    assert out["logits"].shape == (8, NUM_OUTPUT_CLASSES)
    assert out["loss"].ndim == 0


def test_generated_labels_match_ops() -> None:
    pair = build_task_pair(task_seed=1)
    manifest, datasets = generate_task_datasets(
        pair, data_seed=1, n_aliases_per_pair=1, n_nuisance_contexts=1
    )
    assert manifest.dataset_hash
    for task in (pair.task_a, pair.task_b):
        for ex in datasets[task.name]["train"][:20]:
            op = task.by_slot()[ex.slot]
            x = ex.tokens[:8]
            assert op.evaluate(x) == ex.label
            assert ex.label < NUM_OUTPUT_CLASSES
