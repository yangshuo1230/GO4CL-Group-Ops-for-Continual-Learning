#!/usr/bin/env bash
# Phase 1A-mech: single-op / single-modulus mechanism analysis.
#
# Default: analyze p=31 final+best under the wd=0.5 scan-moduli stamp.
#
# Usage:
#   bash scripts/phase1/mech_single.sh
#   bash scripts/phase1/mech_single.sh --moduli 31 19 41 --ckpt-kinds final best
#   bash scripts/phase1/mech_single.sh --ablation-ks 1 2 4 8
#   bash scripts/phase1/mech_single.sh --max-batches 4   # smoke

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

exec uv run go4cl phase1 mech-single "$@"
