"""Dependency-boundary tests for the structural refactor."""

from __future__ import annotations

import sys


def test_analysis_context_does_not_import_phases() -> None:
    banned = [k for k in list(sys.modules) if k.startswith("go4cl.phases")]
    for key in banned:
        del sys.modules[key]
    import go4cl.analysis.context  # noqa: F401
    import go4cl.analysis.reporting  # noqa: F401
    import go4cl.analysis.types  # noqa: F401

    leaked = [k for k in sys.modules if k.startswith("go4cl.phases")]
    assert leaked == []


def test_analysis_pipelines_do_not_import_phases() -> None:
    banned = [k for k in list(sys.modules) if k.startswith("go4cl.phases")]
    for key in banned:
        del sys.modules[key]
    import go4cl.analysis.pipelines.multi_op  # noqa: F401
    import go4cl.analysis.pipelines.single_op  # noqa: F401

    leaked = [k for k in sys.modules if k.startswith("go4cl.phases")]
    assert leaked == []
