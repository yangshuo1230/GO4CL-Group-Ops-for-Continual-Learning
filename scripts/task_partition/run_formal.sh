#!/usr/bin/env bash
# Formal task-partition run. The smoke test prints this command and does not execute it.
#
#   AB joint 200k
#     ├── C-only 100k          (no A/B examples)
#     └── AB continued 100k    (same AB checkpoint, no C)
#   ABC joint 300k from the same initial weights

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

STAMP="$(date +%Y%m%d_%H%M%S)"
exec uv run go4cl task-partition run \
  --config configs/task_partition/default.yaml \
  --out "runs/task_partition/formal_${STAMP}"
