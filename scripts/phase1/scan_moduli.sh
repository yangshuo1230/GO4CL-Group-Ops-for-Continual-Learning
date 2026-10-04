#!/usr/bin/env bash
# Phase 1 / step scan-moduli — single-op scan with a locked training config.
#
# Usage (after calibration):
#   # ratio split (default)
#   bash scripts/phase1/scan_moduli.sh \
#     --train-frac 0.8 --weight-decay 0.3 --steps 100000 \
#     --gpus 0,1 --workers-per-gpu 4
#   # optional fixed train residue-pair count
#   bash scripts/phase1/scan_moduli.sh \
#     --n-train-pairs 150 --weight-decay 0.3 --steps 100000 \
#     --gpus 0,1

set -euo pipefail
cd "$(dirname "$0")/../.."
# NFS scratch (avoid system /tmp ENOSPC)
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase1 scan-moduli "$@"
