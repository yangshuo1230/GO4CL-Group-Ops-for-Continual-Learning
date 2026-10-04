#!/usr/bin/env bash
# Phase 2B: task-relation matrix.
#
# Default grid is the 8 extreme cells ({0,1}^3). Pass --grid full for 27.
# Default protocols: a_only b_only joint sequential_ab sequential_ba
# on one shared manifest per cell (swap is optional via --directions).
#
# Usage:
#   bash scripts/phase2/relation_matrix.sh --dry-run
#   bash scripts/phase2/relation_matrix.sh --grid full --gpus 0,1,2,3,4,5

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase2 relation-matrix "$@"
