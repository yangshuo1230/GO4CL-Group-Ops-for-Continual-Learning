#!/usr/bin/env bash
# Phase 1C: multi-op comparative mechanisms.
#
# Default: four_diff [47,43,37,23] best ckpt from concat 1B stamp.
#
# Usage:
#   bash scripts/phase1/mechanisms.sh
#   bash scripts/phase1/mechanisms.sh --ckpt-kind final
#   bash scripts/phase1/mechanisms.sh --max-batches 2 --ablation-ks 1  # smoke

set -euo pipefail
cd "$(dirname "$0")/../.."

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase1 mechanisms "$@"
