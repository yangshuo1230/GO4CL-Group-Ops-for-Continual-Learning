#!/usr/bin/env bash
# CPU analysis of finished transfer-mechanism jobs. Does not train.
# Default prints the commands. --execute writes the CSVs.

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

mode=dry-run
if [[ "${1:-}" == "--execute" ]]; then
  mode=execute
elif [[ "${1:-}" == "--dry-run" || -z "${1:-}" ]]; then
  mode=dry-run
fi

modulus=(uv run go4cl transfer-mechanism analyze --config configs/transfer_mechanism/modulus_specificity.yaml)
component=(uv run go4cl transfer-mechanism analyze --config configs/transfer_mechanism/component_reset.yaml)

if [[ "${mode}" == "dry-run" ]]; then
  printf '%q ' "${modulus[@]}"
  printf '\n'
  printf '%q ' "${component[@]}"
  printf '\n'
  echo "[analyze] dry-run only; pass --execute to write CSVs"
  exit 0
fi

"${modulus[@]}"
"${component[@]}"
