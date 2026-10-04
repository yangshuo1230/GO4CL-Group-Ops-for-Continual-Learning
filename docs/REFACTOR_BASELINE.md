# Refactor baseline (R0)

Captured 2026-10-04 before structural edits. Working tree was clean at
`36b1355` (`1C Update`). No source files were reverted.

## Tests (pre-refactor)

```text
uv run pytest tests/ -q
1 failed, 103 passed
FAILED tests/test_mechanisms.py::test_resolve_default_four_diff_job
```

That failure is **pre-existing**: `DEFAULT_JOB` points at
`multi_four_diff_m31-37-29-23_...` while the test still asserts moduli
`[47, 43, 37, 23]`. Do not “fix” it as part of the structure refactor.

## CLI help snapshot

Saved under `/tmp/go4cl_*_help.txt` at R0. Phase 1 `mech-single` /
`multi-op` / `mechanisms` help still contained the word `stub`.

## Constraints

- Do not run full scientific experiments.
- Do not delete `logs/` contents; only ignore them going forward.
- Keep training metrics schema, checkpoint names, and job IDs unchanged.
