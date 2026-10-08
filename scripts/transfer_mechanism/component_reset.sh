#!/usr/bin/env bash
# Checkpoint-mixing grid. Default is dry-run (prints commands, does not train).
#   bash scripts/transfer_mechanism/component_reset.sh --dry-run
#   bash scripts/transfer_mechanism/component_reset.sh --execute

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

mode=(--dry-run)
extra=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --execute)
      mode=(--execute)
      shift
      ;;
    --dry-run)
      mode=(--dry-run)
      shift
      ;;
    *)
      extra+=("$1")
      shift
      ;;
  esac
done

exec uv run go4cl transfer-mechanism component "${mode[@]}" "${extra[@]}"
