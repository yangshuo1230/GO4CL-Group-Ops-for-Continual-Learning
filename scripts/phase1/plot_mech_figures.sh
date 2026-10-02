#!/usr/bin/env bash
# Regenerate 1A-mech summary figures from mech_single stamp directories.
#
# Usage:
#   bash scripts/phase1/plot_mech_figures.sh
#   bash scripts/phase1/plot_mech_figures.sh \
#     --composition-stamp runs/phase1/mech_single/20261002_105034 \
#     --head-stamp runs/phase1/mech_single/20261002_110852 \
#     --out runs/phase1/mech_single/p31_summary/figures

set -euo pipefail
cd "$(dirname "$0")/../.."

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

exec uv run python scripts/phase1/plot_mech_figures.py "$@"
