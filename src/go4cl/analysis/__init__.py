"""Mechanistic analysis: Fourier, probes, attention routing, causal ablations."""

from go4cl.analysis.attention import query_attention_to_operands
from go4cl.analysis.cache import collect_batches, operand_residues
from go4cl.analysis.causal import (
    estimate_class_means,
    evaluate_steering,
    fourier_ablation_on_embeddings,
    steer_at_layer,
    steer_query_resid_sum,
)
from go4cl.analysis.composition import (
    component_knockout,
    freq_harmonic_fit,
    head_knockout,
    probe_info_ladder,
    run_composition_analysis,
)
from go4cl.analysis.discover import MechTarget, discover_targets
from go4cl.analysis.fourier import (
    analyze_digit_embedding_fourier,
    analyze_query_resid_fourier,
)
from go4cl.analysis.probes import (
    run_layer_probes_with_random_control,
    run_operand_probes,
    run_operand_probes_with_random_control,
)

__all__ = [
    "MechTarget",
    "discover_targets",
    "collect_batches",
    "operand_residues",
    "analyze_digit_embedding_fourier",
    "analyze_query_resid_fourier",
    "run_operand_probes",
    "run_operand_probes_with_random_control",
    "run_layer_probes_with_random_control",
    "query_attention_to_operands",
    "fourier_ablation_on_embeddings",
    "estimate_class_means",
    "evaluate_steering",
    "steer_query_resid_sum",
    "steer_at_layer",
    "probe_info_ladder",
    "component_knockout",
    "head_knockout",
    "freq_harmonic_fit",
    "run_composition_analysis",
]
