#!/usr/bin/env bash
# Phase 1 / step calibrate — grokking regime on a medium modulus (default p=17).
#
# Usage:
#   bash scripts/phase1/calibrate.sh
#   bash scripts/phase1/calibrate.sh --gpus 4,5 --workers-per-gpu 2

set -euo pipefail
cd "$(dirname "$0")/../.."

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run go4cl phase1 calibrate "$@"
