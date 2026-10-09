#!/usr/bin/env bash
# Fixed-B modulus causal grid.
# Default: dry-run. Formal training is `bash scripts/phase2/causal_modulus.sh --execute`.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

if [[ " $* " != *" --execute "* ]]; then
  set -- --dry-run "$@"
fi
# --dry-run is implied by omitting --execute; the python flag is --execute only.
args=()
for arg in "$@"; do
  if [[ "${arg}" != "--dry-run" ]]; then
    args+=("${arg}")
  fi
done
exec uv run go4cl phase2 causal-modulus "${args[@]}"
