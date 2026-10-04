"""Compatibility re-export. Prefer go4cl.analysis.interventions.patching."""

from go4cl.analysis.interventions.patching import *  # noqa: F403
from go4cl.analysis.interventions.patching import (  # noqa: F401
    QueryPatchKind,
    continue_from_pre,
    operand_patch_from_pre,
    patch_token_positions,
    query_path_patch_logits,
    split_acc,
)
