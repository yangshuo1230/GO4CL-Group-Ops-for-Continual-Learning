#!/usr/bin/env bash
# Phase 1C: multi-op comparative mechanisms.
#
# Default: four_diff [31,37,29,23] best from packed 1B stamp 20261003_132221.
#
# Usage:
#   bash scripts/phase1/mechanisms.sh
#   bash scripts/phase1/mechanisms.sh --ckpt-kind final
#   bash scripts/phase1/mechanisms.sh --max-batches 2 --ablation-ks 1  # smoke

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

exec uv run go4cl phase1 mechanisms "$@"
