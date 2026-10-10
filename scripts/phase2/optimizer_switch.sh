#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
source scripts/env.sh
export PYTHONPATH="${PWD}/src${PYTHONPATH:+:${PYTHONPATH}}"
export WANDB_MODE=disabled
exec .venv/bin/python scripts/phase2/optimizer_switch.py \
  --gpus "${GPUS:-0}" \
  --source-root "${SOURCE_ROOT:-runs/phase2/mechanism_suite_20261009}" \
  --root "${RUN_ROOT:-runs/phase2/optimizer_switch_20261010}" "$@"
