#!/usr/bin/env bash
# Phase 2A: protocol comparison on one fixed A/B dataset.
#
# Default overlap is fully shared (rho_slot=rho_operand=rho_mod=1).
# Locked cfg: train_frac=0.8 wd=0.3 steps=100000 aliases=16
#   bs=8192 query examples/step, packed online, primary eval = packed_id.
#
# Usage:
#   bash scripts/phase2/protocols.sh --gpus 0,1,2,3 --workers-per-gpu 1
#   bash scripts/phase2/protocols.sh --dry-run --rho-slot 0 --rho-operand 0 --rho-mod 0

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase2 protocols "$@"
