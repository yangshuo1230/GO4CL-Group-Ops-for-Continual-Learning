"""Forward-transfer mechanism experiments (no replay).

Modulus specificity and checkpoint mixing live here so they stay independent
of the phase-2 trainer. Training still calls ``train_segment``.
"""

from __future__ import annotations

PARAMETER_GROUP_SCHEMA_VERSION = 1

__all__ = ["PARAMETER_GROUP_SCHEMA_VERSION"]
