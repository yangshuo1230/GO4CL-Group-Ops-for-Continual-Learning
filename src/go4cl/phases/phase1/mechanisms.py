"""Phase 1C CLI adapter. Core analysis is in ``go4cl.analysis.pipelines.multi_op``."""

from go4cl.analysis.context import (
    OpSpec,
    filter_by_modulus,
    filter_by_operation,
    op_report_key,
    operations_from_manifest,
    resolve_checkpoint,
    resolve_data_dir,
)
from go4cl.analysis.pipelines.multi_op import (
    DEFAULT_JOB,
    add_mechanisms_args,
    run_mechanisms,
)

_ops_from_manifest = operations_from_manifest
_resolve_ckpt = resolve_checkpoint
_resolve_data_dir = resolve_data_dir
_op_report_key = op_report_key

__all__ = [
    "DEFAULT_JOB",
    "OpSpec",
    "add_mechanisms_args",
    "run_mechanisms",
    "filter_by_modulus",
    "filter_by_operation",
    "_ops_from_manifest",
    "_resolve_ckpt",
    "_resolve_data_dir",
    "_op_report_key",
]
