#!/usr/bin/env bash
# Phase 1 / step scan-moduli — single-op scan with a locked training config.
#
# Usage (after calibration):
#   bash scripts/phase1/scan_moduli.sh \
#     --train-frac 0.6 --weight-decay 0.3 --steps 100000 \
#     --gpus 4,5

set -euo pipefail
cd "$(dirname "$0")/../.."

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase1 scan-moduli "$@"
