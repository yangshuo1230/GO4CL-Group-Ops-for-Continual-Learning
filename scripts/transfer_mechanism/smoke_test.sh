#!/usr/bin/env bash
# Tiny CPU smoke for the transfer-mechanism framework.
# Does not start the formal GPU grid. Outputs are not scientific results.

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

# Keep this process off GPUs that other jobs are using.
export CUDA_VISIBLE_DEVICES=""

exec uv run go4cl transfer-mechanism smoke --out runs/transfer_mechanism/smoke
