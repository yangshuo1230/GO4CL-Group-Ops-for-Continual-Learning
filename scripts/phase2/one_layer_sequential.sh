#!/usr/bin/env bash
# Plan by default. Training starts only with --execute; see the README.
set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
cd "$PROJECT_ROOT"
source scripts/env.sh
export PATH="${HOME}/.local/bin:${PATH}"
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled
exec uv run --no-sync python -u -m go4cl.phases.phase2.one_layer_sequential \
  --gpus "${GPUS:-0}" --root "${RUN_ROOT:-runs/phase2/one_layer_sequential_20261011}" "$@"
