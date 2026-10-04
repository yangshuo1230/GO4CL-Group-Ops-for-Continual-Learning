#!/usr/bin/env bash
# Phase 2D: capacity ablation.
#
# d_model in {32,64,128}, depth in {2,3,4}, on six representative overlaps
# (identical, slot-only, operand-only, modulus-only, disjoint, partial).
# Default protocols: a_only b_only joint sequential_ab.
#
# Usage:
#   bash scripts/phase2/capacity.sh --dry-run
#   bash scripts/phase2/capacity.sh --d-models 64 --n-layers-list 3 --gpus 0,1

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase2 capacity "$@"
