#!/usr/bin/env bash
# 1C deepening: operand patching + query-twin circuit transplant
# on packed 1B stamp 20261003_132221 (six jobs → six GPUs).
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source "$(dirname "$0")/../env.sh"
export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi

STAMP="${STAMP:-runs/phase1/multi_op/20261003_132221}"
OUT_ROOT="${OUT_ROOT:-runs/phase1/mechanisms/20261004_1b_132221}"
LOG_DIR="${LOG_DIR:-logs}"
mkdir -p "${LOG_DIR}" "${OUT_ROOT}"

declare -a TAGS=(
  all_same_ts0
  all_same_ts1
  four_diff_ts0
  four_diff_ts1
  pair_same_ts0
  pair_same_ts1
)
declare -a JOBS=(
  multi_all_same_m31-31-31-31_tr0.8_ts0_ds0_a16_p1b_pack1__wd0.5_steps100000__ms0
  multi_all_same_m53-53-53-53_tr0.8_ts1_ds0_a16_p1b_pack1__wd0.5_steps100000__ms0
  multi_four_diff_m31-37-29-23_tr0.8_ts0_ds0_a16_p1b_pack1__wd0.5_steps100000__ms0
  multi_four_diff_m53-47-41-29_tr0.8_ts1_ds0_a16_p1b_pack1__wd0.5_steps100000__ms0
  multi_pair_same_m31-31-29-23_tr0.8_ts0_ds0_a16_p1b_pack1__wd0.5_steps100000__ms0
  multi_pair_same_m53-53-41-29_tr0.8_ts1_ds0_a16_p1b_pack1__wd0.5_steps100000__ms0
)

pids=()
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  job="${JOBS[$i]}"
  gpu="${i}"
  echo "launch ${tag} on cuda:${gpu}"
  uv run python scripts/phase1/circuit_transplant.py \
    --job-dir "${STAMP}/runs/${job}" \
    --out "${OUT_ROOT}/${tag}/transplant" \
    --device "cuda:${gpu}" \
    > "${LOG_DIR}/1c_transplant_${tag}.log" 2>&1 &
  pids+=("$!")
done

fail=0
for i in "${!pids[@]}"; do
  if ! wait "${pids[$i]}"; then
    echo "FAILED ${TAGS[$i]} — see ${LOG_DIR}/1c_transplant_${TAGS[$i]}.log" >&2
    fail=1
  fi
done
exit "${fail}"
