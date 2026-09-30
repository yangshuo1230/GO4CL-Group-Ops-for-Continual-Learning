"""Scientific experiment phases aligned with docs/RESEARCH_EXPERIMENT_PLAN.md."""

from __future__ import annotations

PHASES = ("phase1", "phase2", "phase3")

PHASE_STEPS: dict[str, tuple[str, ...]] = {
    "phase1": ("calibrate", "scan-moduli", "mech-single", "multi-op", "mechanisms"),
    "phase2": ("protocols", "relation-matrix", "capacity"),
    "phase3": ("optimize", "route-compute-readout", "forget-types"),
}
