#!/usr/bin/env bash
# Phase-1 style multi-op mechanisms on next80_formal representative jobs.
#
# For each sequential job:
#   theta_A + task A          (pre-switch A)
#   phase_b_final + task A    (post-switch, A ops)
#   phase_b_final + task B    (post-switch, B ops)
# Plus shared-A / b_only baselines.
#
# Usage:
#   bash scripts/phase2/run_next80_mech_suite.sh --dry-run
#   bash scripts/phase2/run_next80_mech_suite.sh --execute
#   GPUS=0,1,2,3 bash scripts/phase2/run_next80_mech_suite.sh --execute --smoke

set -Eeuo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source scripts/env.sh

export PATH="${HOME}/.local/bin:${PATH}"
if [[ -f "${HOME}/.local/bin/env" ]]; then
  # shellcheck disable=SC1091
  source "${HOME}/.local/bin/env"
fi
export WANDB_MODE=disabled
export WANDB_DISABLED=true

MODE="${1:---dry-run}"
shift || true
SMOKE=0
EXTRA=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --smoke) SMOKE=1; shift ;;
    --execute|--dry-run) MODE="$1"; shift ;;
    *) EXTRA+=("$1"); shift ;;
  esac
done

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
ROOT="${ROOT:-runs/phase2/next80_formal}"
OUT_ROOT="${OUT_ROOT:-${ROOT}/mech_suite}"
LOG_ROOT="${OUT_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

COMMON=(
  --probe-steps 400
  --layers 0 1
  --ablation-ks 1 2
)
if [[ "${SMOKE}" -eq 1 ]]; then
  COMMON+=(--max-batches 2 --skip-composition --probe-steps 50)
fi

# label|rel_job_dir|ckpt_kind|task
JOBS=(
  "A0_sharedA_finalA|01_core_fresh/runs/shared_a_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA|final|A"
  "B1_bonly_finalB|01_core_fresh/runs/b_only_s0.5_o0.5_m0_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA|final|B"

  "F1_forget_m0_thetaA|01_core_fresh/runs/sequential_ab_s0.5_o0.5_m0_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|theta_A|A"
  "F1_forget_m0_finalA|01_core_fresh/runs/sequential_ab_s0.5_o0.5_m0_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|A"
  "F1_forget_m0_finalB|01_core_fresh/runs/sequential_ab_s0.5_o0.5_m0_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|B"

  "F2_partial_m1_thetaA|01_core_fresh/runs/sequential_ab_s0.5_o0.5_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|theta_A|A"
  "F2_partial_m1_finalA|01_core_fresh/runs/sequential_ab_s0.5_o0.5_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|A"
  "F2_partial_m1_finalB|01_core_fresh/runs/sequential_ab_s0.5_o0.5_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|B"

  "Rp_replay_m1_thetaA|03_replay_overlap/runs/sequential_ab_replay_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|theta_A|A"
  "Rp_replay_m1_finalA|03_replay_overlap/runs/sequential_ab_replay_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|A"
  "Rp_replay_m1_finalB|03_replay_overlap/runs/sequential_ab_replay_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|B"

  "Rm_noreplay_m1_thetaA|04_no_replay_anchors/runs/sequential_ab_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|theta_A|A"
  "Rm_noreplay_m1_finalA|04_no_replay_anchors/runs/sequential_ab_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|A"
  "Rm_noreplay_m1_finalB|04_no_replay_anchors/runs/sequential_ab_s0_o0_m1_forward_ts0_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|B"

  "Rh_replay075_thetaA|03_replay_overlap/runs/sequential_ab_replay_s0.5_o0.5_m0.5_forward_ts1_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|theta_A|A"
  "Rh_replay075_finalA|03_replay_overlap/runs/sequential_ab_replay_s0.5_o0.5_m0.5_forward_ts1_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|A"
  "Rh_replay075_finalB|03_replay_overlap/runs/sequential_ab_replay_s0.5_o0.5_m0.5_forward_ts1_ms0_d64_L3_wd0.3_steps100000_fixedA_fromSharedA_optfresh|phase_b_final|B"
)

echo "OUT_ROOT=${OUT_ROOT}"
echo "MODE=${MODE} SMOKE=${SMOKE} GPUS=${GPUS}"
echo "n_jobs=${#JOBS[@]}"

run_one() {
  local label="$1" rel="$2" kind="$3" task="$4" gpu="$5"
  local job_dir="${ROOT}/${rel}"
  local out="${OUT_ROOT}/${label}"
  local log="${LOG_ROOT}/${label}.log"
  mkdir -p "${out}"
  echo "[gpu=${gpu}] START ${label} kind=${kind} task=${task}"
  CUDA_VISIBLE_DEVICES="${gpu}" uv run go4cl phase1 mechanisms \
    --job-dir "${job_dir}" \
    --ckpt-kind "${kind}" \
    --task "${task}" \
    --out "${out}" \
    "${COMMON[@]}" \
    "${EXTRA[@]+"${EXTRA[@]}"}" \
    >"${log}" 2>&1
  echo "[gpu=${gpu}] DONE  ${label}"
}

if [[ "${MODE}" == "--dry-run" ]]; then
  for spec in "${JOBS[@]}"; do
    IFS='|' read -r label rel kind task <<<"${spec}"
    echo "would run: ${label}  job=${ROOT}/${rel}  kind=${kind} task=${task}"
  done
  exit 0
fi

if [[ "${MODE}" != "--execute" ]]; then
  echo "Usage: $0 [--dry-run|--execute] [--smoke]"
  exit 2
fi

PIDS=()
idx=0
fail=0
for spec in "${JOBS[@]}"; do
  IFS='|' read -r label rel kind task <<<"${spec}"
  gpu="${GPU_LIST[$((idx % ${#GPU_LIST[@]}))]}"
  (
    run_one "${label}" "${rel}" "${kind}" "${task}" "${gpu}"
  ) &
  PIDS+=("$!")
  idx=$((idx + 1))
  if [[ "${#PIDS[@]}" -eq "${#GPU_LIST[@]}" ]]; then
    for pid in "${PIDS[@]}"; do
      if ! wait "${pid}"; then fail=1; fi
    done
    PIDS=()
  fi
done
for pid in "${PIDS[@]+"${PIDS[@]}"}"; do
  if ! wait "${pid}"; then fail=1; fi
done

uv run python scripts/phase2/summarize_next80_mech_suite.py --root "${OUT_ROOT}" \
  | tee "${OUT_ROOT}/SUMMARY.md"

if [[ "${fail}" -ne 0 ]]; then
  echo "Some mechanism jobs failed; see ${LOG_ROOT}"
  exit 1
fi
echo "All mechanism jobs finished → ${OUT_ROOT}"
