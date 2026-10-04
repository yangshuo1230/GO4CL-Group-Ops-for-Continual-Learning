"""Causal interventions: ablation, patching, residual steering.

This module re-exports the historical ``go4cl.analysis.causal`` public names.
New code should import from ``go4cl.analysis.interventions``.
"""

from go4cl.analysis.interventions.ablation import (
    eval_accuracy,
    fourier_ablation_on_embeddings,
    fourier_ablation_on_unembedding,
    project_out_freqs_from_digit_emb,
    project_out_freqs_from_unembed,
)
from go4cl.analysis.interventions.steering import (
    estimate_class_means,
    evaluate_steering,
    steer_at_layer,
    steer_query_resid_sum,
)
from go4cl.analysis.interventions.knockout import (
    component_knockout,
    continue_after_edited_post,
    head_knockout,
)

# Private helper still imported by phase-1 pipelines.
from go4cl.analysis.interventions.ablation import _freq_pairs_by_energy

__all__ = [
    "project_out_freqs_from_digit_emb",
    "project_out_freqs_from_unembed",
    "eval_accuracy",
    "_freq_pairs_by_energy",
    "fourier_ablation_on_embeddings",
    "fourier_ablation_on_unembedding",
    "estimate_class_means",
    "evaluate_steering",
    "steer_query_resid_sum",
    "steer_at_layer",
    "component_knockout",
    "continue_after_edited_post",
    "head_knockout",
]
