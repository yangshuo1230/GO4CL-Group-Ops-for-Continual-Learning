#!/usr/bin/env bash
# Unembedding mod-p Fourier on packed 1B 1C jobs, then 1A scan-moduli.
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

fail=0
pids=()
for i in "${!TAGS[@]}"; do
  tag="${TAGS[$i]}"
  job="${JOBS[$i]}"
  echo "launch ${tag} on cuda:${i}"
  uv run python scripts/phase1/unembed_fourier.py \
    --job-dir "${STAMP}/runs/${job}" \
    --out "${OUT_ROOT}/${tag}/unembed_fourier" \
    --device "cuda:${i}" \
    > "${LOG_DIR}/1c_unembed_${tag}.log" 2>&1 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    echo "FAILED 1C pid ${pid}" >&2
    fail=1
  fi
done

ONEA_ROOT="${ONEA_ROOT:-runs/phase1/scan_moduli/20260930_223737}"
if [[ -d "${ONEA_ROOT}/runs" ]]; then
  mapfile -t ONEA_JOBS < <(ls -1 "${ONEA_ROOT}/runs" | grep '^single_p' | sort)
  i=0
  while [[ ${i} -lt ${#ONEA_JOBS[@]} ]]; do
    batch_pids=()
    for g in 6 7; do
      if [[ ${i} -ge ${#ONEA_JOBS[@]} ]]; then
        break
      fi
      job="${ONEA_JOBS[$i]}"
      p="$(echo "${job}" | sed -n 's/^single_p\([0-9]*\)_.*/\1/p')"
      echo "launch 1A p=${p} on cuda:${g}"
      uv run python scripts/phase1/unembed_fourier.py \
        --job-dir "${ONEA_ROOT}/runs/${job}" \
        --out "runs/phase1/mech_single/unembed_fourier_1a/p${p}" \
        --device "cuda:${g}" \
        --layers 0 1 \
        > "${LOG_DIR}/1a_unembed_p${p}.log" 2>&1 &
      batch_pids+=("$!")
      i=$((i + 1))
    done
    for pid in "${batch_pids[@]}"; do
      if ! wait "${pid}"; then
        echo "FAILED 1A pid ${pid}" >&2
        fail=1
      fi
    done
  done
fi
exit "${fail}"
