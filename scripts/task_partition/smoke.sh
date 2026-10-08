#!/usr/bin/env bash
# Tiny end-to-end check of the three task-partition protocols.
# Does not start the 200k/100k/300k run.

set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"

export WANDB_MODE=disabled
export WANDB_DISABLED=true
export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

rm -rf runs/task_partition/smoke
exec uv run go4cl task-partition smoke --out runs/task_partition/smoke
