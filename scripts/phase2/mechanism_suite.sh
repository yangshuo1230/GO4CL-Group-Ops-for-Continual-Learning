#!/usr/bin/env bash
# Explicit execution only. Defaults: seeds 0/1, single GPU 0, all stages.
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled
exec .venv/bin/python -m go4cl.phases.phase2.mechanism_suite \
  --gpus "${GPUS:-0}" --root "${RUN_ROOT:-runs/phase2/mechanism_suite_20261009}" "$@"
