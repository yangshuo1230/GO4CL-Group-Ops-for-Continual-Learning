#!/usr/bin/env bash
# Phase 1B: multi-op / same-modulus facilitation training.
#
# Default variants: one | four_diff | pair_same
# Locked cfg: train_frac=0.8 wd=0.5 steps=100000 aliases=16 (val/test)
#   bs=8192 query examples/step, packed online train (equal op exposure)
#
# Usage:
#   bash scripts/phase1/multi_op.sh --gpus 0,1,2,3,4,5 --workers-per-gpu 1
#   bash scripts/phase1/multi_op.sh --variants one four_diff pair_same all_same

set -euo pipefail
cd "$(dirname "$0")/../.."

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase1 multi-op "$@"
