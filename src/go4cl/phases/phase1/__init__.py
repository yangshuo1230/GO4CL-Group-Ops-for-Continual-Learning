"""Phase 1: single-task learning, grokking, and algorithm formation."""

from __future__ import annotations

from go4cl.phases.phase1.calibrate import add_calibrate_args, run_calibrate
from go4cl.phases.phase1.mech_single import add_mech_single_args, run_mech_single
from go4cl.phases.phase1.mechanisms import add_mechanisms_args, run_mechanisms
from go4cl.phases.phase1.modulus_scan import add_scan_moduli_args, run_scan_moduli
from go4cl.phases.phase1.multi_op import add_multi_op_args, run_multi_op

__all__ = [
    "add_calibrate_args",
    "run_calibrate",
    "add_scan_moduli_args",
    "run_scan_moduli",
    "add_mech_single_args",
    "run_mech_single",
    "add_multi_op_args",
    "run_multi_op",
    "add_mechanisms_args",
    "run_mechanisms",
]
