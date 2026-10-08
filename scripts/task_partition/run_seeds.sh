#!/usr/bin/env bash
# Four model seeds, one GPU each. Data / sampler / eval seeds stay at the yaml values.
#
#   GPU 0  model_seed 0
#   GPU 1  model_seed 1
#   GPU 2  model_seed 2
#   GPU 3  model_seed 3
#
# Each process runs AB joint, then C-only and AB-continued from that seed's
# checkpoint, then ABC joint from that seed's initial weights.

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
ROOT="runs/task_partition/seeds_${STAMP}"
mkdir -p "${ROOT}"

cat > "${ROOT}/seeds.json" <<EOF
{
  "model_seeds": [0, 1, 2, 3],
  "gpus": [0, 1, 2, 3],
  "data_seed": 0,
  "sampler_seed": 0,
  "eval_seed": 0,
  "config": "configs/task_partition/default.yaml",
  "wandb": "disabled"
}
EOF

for seed in 0 1 2 3; do
  out="${ROOT}/ms${seed}"
  mkdir -p "${out}"
  CUDA_VISIBLE_DEVICES="${seed}" setsid nohup uv run go4cl task-partition run \
    --config configs/task_partition/default.yaml \
    --out "${out}" \
    --device cuda:0 \
    --model-seed "${seed}" \
    > "${out}/train.log" 2>&1 < /dev/null &
  echo $! > "${out}/pid"
  echo "started model_seed=${seed} gpu=${seed} pid=$(cat "${out}/pid") out=${out}"
done

echo "${ROOT}"
