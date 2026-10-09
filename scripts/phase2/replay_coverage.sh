#!/usr/bin/env bash
# Fixed replay-ratio coverage grid.
# Default: dry-run. Formal training is `bash scripts/phase2/replay_coverage.sh --execute`.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

args=()
for arg in "$@"; do
  if [[ "${arg}" != "--dry-run" ]]; then
    args+=("${arg}")
  fi
done
exec uv run go4cl phase2 replay-coverage "${args[@]}"
