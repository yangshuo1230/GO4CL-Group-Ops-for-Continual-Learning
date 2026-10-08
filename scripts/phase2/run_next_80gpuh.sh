#!/usr/bin/env bash
# Approximately 80 GPU-hours on 8 GPUs. Dry-run by default.
#
# Usage:
#   bash scripts/phase2/run_next_80gpuh.sh --dry-run
#   RUN_ROOT=runs/phase2/next80_formal bash scripts/phase2/run_next_80gpuh.sh --execute

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
export PYTHONDONTWRITEBYTECODE=1

MODE="${1:---dry-run}"
if [[ "${MODE}" != "--dry-run" && "${MODE}" != "--execute" ]]; then
    echo "Usage: $0 [--dry-run|--execute]"
    exit 2
fi

GPUS="${GPUS:-0,1,2,3,4,5,6,7}"
IFS=',' read -r -a GPU_LIST <<< "${GPUS}"
if [[ "${#GPU_LIST[@]}" -ne 8 ]]; then
    echo "Expected exactly 8 GPU ids, got: ${GPUS}"
    exit 2
fi

STAMP="$(date +%Y%m%d_%H%M%S)"
RUN_ROOT="${RUN_ROOT:-runs/phase2/next80_${STAMP}}"
LOG_ROOT="${RUN_ROOT}/logs"
mkdir -p "${LOG_ROOT}" "${RUN_ROOT}/generated_configs"

if [[ "${MODE}" == "--execute" ]]; then
    if ! command -v nvidia-smi >/dev/null 2>&1; then
        echo "nvidia-smi is unavailable"
        exit 2
    fi
    AVAILABLE_GPUS="$(nvidia-smi -L | wc -l)"
    if [[ "${AVAILABLE_GPUS}" -lt 8 ]]; then
        echo "Need 8 visible GPUs, found ${AVAILABLE_GPUS}"
        exit 2
    fi
    if [[ "${ALLOW_DIRTY:-0}" != "1" ]]; then
        if ! git diff --quiet ||
           ! git diff --cached --quiet ||
           [[ -n "$(git ls-files --others --exclude-standard)" ]]; then
            echo "Refusing a formal run from a dirty worktree."
            echo "Commit the validity fixes first, or set ALLOW_DIRTY=1 explicitly."
            git status --short
            exit 2
        fi
    fi
fi

GIT_REV="$(git rev-parse HEAD)"
{
    printf 'run_root=%s\n' "${RUN_ROOT}"
    printf 'git_rev=%s\n' "${GIT_REV}"
    printf 'mode=%s\n' "${MODE}"
    printf 'gpus=%s\n' "${GPUS}"
    printf 'started_at=%s\n' "$(date --iso-8601=seconds)"
} > "${RUN_ROOT}/RUN_INFO.txt"

echo "Run root: ${RUN_ROOT}"
echo "Git revision: ${GIT_REV}"
echo "GPUs: ${GPUS}"
echo "Mode: ${MODE}"

run_logged() {
    local name="$1"
    shift
    echo
    echo "============================================================"
    echo "START ${name}: $(date --iso-8601=seconds)"
    echo "============================================================"
    "$@" 2>&1 | tee "${LOG_ROOT}/${name}.log"
    echo "END ${name}: $(date --iso-8601=seconds)"
}

if [[ "${MODE}" == "--execute" ]]; then
    run_logged tests uv run python -m pytest -q -p no:cacheprovider
    run_logged transfer_smoke bash scripts/transfer_mechanism/smoke_test.sh
fi

PHASE2_MODE=()
if [[ "${MODE}" == "--dry-run" ]]; then
    PHASE2_MODE=(--dry-run)
fi

COMMON_ARGS=(
    --steps 100000
    --train-frac 0.8
    --weight-decay 0.3
    --batch-size 8192
    --lr 0.001
    --d-model 64
    --n-layers 3
    --n-heads 4
    --task-seeds 0 1
    --model-seeds 0 1 2
    --data-seed 0
    --directions forward
    --switch-on fixed
    --fixed-a
    --share-a
    --gpus "${GPUS}"
    --workers-per-gpu 1
    --wandb-mode disabled
)

CORE_CONDITIONS=(
    s0.5_o0.5_m0
    s0.5_o0.5_m0.5
    s0.5_o0.5_m1
)

REPLAY_CONDITIONS=(
    s0.5_o0.5_m0
    s0.5_o0.5_m0.5
    s0.5_o0.5_m1
    s0_o0_m0
    s0_o0_m1
    s0_o1_m0
    s1_o0_m0
    s1_o1_m1
)

NO_REPLAY_ANCHORS=(
    s0_o0_m0
    s0_o0_m1
    s0_o1_m0
    s1_o0_m0
)

# Stage 1: fresh optimizer plus matched B-only. ~18.5 GPU-hours.
run_logged 01_core_fresh \
    bash scripts/phase2/relation_matrix.sh \
    "${PHASE2_MODE[@]}" \
    --out "${RUN_ROOT}/01_core_fresh" \
    "${COMMON_ARGS[@]}" \
    --optimizer-transition fresh \
    --protocols b_only sequential_ab \
    --conditions "${CORE_CONDITIONS[@]}"

# Stage 2: carried-optimizer control. ~11.6 GPU-hours.
run_logged 02_core_preserve \
    bash scripts/phase2/relation_matrix.sh \
    "${PHASE2_MODE[@]}" \
    --out "${RUN_ROOT}/02_core_preserve" \
    "${COMMON_ARGS[@]}" \
    --optimizer-transition preserve \
    --protocols sequential_ab \
    --conditions "${CORE_CONDITIONS[@]}"

# Stage 3: fixed 10% replay across representative overlap cells. ~23 GPU-hours.
run_logged 03_replay_overlap \
    bash scripts/phase2/relation_matrix.sh \
    "${PHASE2_MODE[@]}" \
    --out "${RUN_ROOT}/03_replay_overlap" \
    "${COMMON_ARGS[@]}" \
    --optimizer-transition fresh \
    --protocols sequential_ab_replay \
    --conditions "${REPLAY_CONDITIONS[@]}"

# Stage 4: matched no-replay controls for non-core anchors. ~13.9 GPU-hours.
run_logged 04_no_replay_anchors \
    bash scripts/phase2/relation_matrix.sh \
    "${PHASE2_MODE[@]}" \
    --out "${RUN_ROOT}/04_no_replay_anchors" \
    "${COMMON_ARGS[@]}" \
    --optimizer-transition fresh \
    --protocols sequential_ab \
    --conditions "${NO_REPLAY_ANCHORS[@]}"

# Stage 5: parameter-source localization on s0_o0_m1. ~13.8 GPU-hours.
COMPONENT_OUT="${RUN_ROOT}/05_component_reset"
COMPONENT_CONFIG="${RUN_ROOT}/generated_configs/component_reset.yaml"
sed -E \
    "s|^out_dir:.*$|out_dir: ${COMPONENT_OUT}|" \
    configs/transfer_mechanism/component_reset.yaml \
    > "${COMPONENT_CONFIG}"

COMPONENT_INTERVENTIONS=(
    full_A
    full_fresh
    reset_digit_embedding
    reset_control_embedding
    reset_attention
    reset_mlp
    keep_digit_embedding
    keep_control_embedding
    keep_attention
    keep_mlp
)
COMPONENT_MODEL_SEEDS=(0 1 2)
COMPONENT_TASK_SEED=0

run_background_batch() {
    local -a pids=("$@")
    local failed=0
    local pid
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=1
        fi
    done
    if [[ "${failed}" -ne 0 ]]; then
        echo "At least one background GPU job failed."
        exit 1
    fi
}

if [[ "${MODE}" == "--dry-run" ]]; then
    echo
    echo "===== COMPONENT SOURCES ====="
    for model_seed in "${COMPONENT_MODEL_SEEDS[@]}"; do
        echo "CUDA_VISIBLE_DEVICES=<gpu> uv run go4cl transfer-mechanism run-source --config ${COMPONENT_CONFIG} --task-seed ${COMPONENT_TASK_SEED} --model-seed ${model_seed}"
    done
    echo
    echo "===== COMPONENT INTERVENTIONS ====="
    for intervention in "${COMPONENT_INTERVENTIONS[@]}"; do
        for model_seed in "${COMPONENT_MODEL_SEEDS[@]}"; do
            echo "CUDA_VISIBLE_DEVICES=<gpu> uv run go4cl transfer-mechanism run-intervention --config ${COMPONENT_CONFIG} --intervention ${intervention} --task-seed ${COMPONENT_TASK_SEED} --model-seed ${model_seed}"
        done
    done
else
    SOURCE_PIDS=()
    source_index=0
    for model_seed in "${COMPONENT_MODEL_SEEDS[@]}"; do
        gpu="${GPU_LIST[$((source_index % ${#GPU_LIST[@]}))]}"
        log="${LOG_ROOT}/05_source_ts${COMPONENT_TASK_SEED}_ms${model_seed}.log"
        (
            export CUDA_VISIBLE_DEVICES="${gpu}"
            uv run go4cl transfer-mechanism run-source \
                --config "${COMPONENT_CONFIG}" \
                --task-seed "${COMPONENT_TASK_SEED}" \
                --model-seed "${model_seed}"
        ) > "${log}" 2>&1 &
        SOURCE_PIDS+=("$!")
        source_index=$((source_index + 1))
    done
    run_background_batch "${SOURCE_PIDS[@]}"

    BATCH_PIDS=()
    job_index=0
    for intervention in "${COMPONENT_INTERVENTIONS[@]}"; do
        for model_seed in "${COMPONENT_MODEL_SEEDS[@]}"; do
            gpu="${GPU_LIST[$((job_index % ${#GPU_LIST[@]}))]}"
            log="${LOG_ROOT}/05_${intervention}_ts${COMPONENT_TASK_SEED}_ms${model_seed}.log"
            (
                export CUDA_VISIBLE_DEVICES="${gpu}"
                uv run go4cl transfer-mechanism run-intervention \
                    --config "${COMPONENT_CONFIG}" \
                    --intervention "${intervention}" \
                    --task-seed "${COMPONENT_TASK_SEED}" \
                    --model-seed "${model_seed}"
            ) > "${log}" 2>&1 &
            BATCH_PIDS+=("$!")
            job_index=$((job_index + 1))
            if [[ "${#BATCH_PIDS[@]}" -eq "${#GPU_LIST[@]}" ]]; then
                run_background_batch "${BATCH_PIDS[@]}"
                BATCH_PIDS=()
            fi
        done
    done
    if [[ "${#BATCH_PIDS[@]}" -gt 0 ]]; then
        run_background_batch "${BATCH_PIDS[@]}"
    fi

    run_logged 05_component_analysis \
        uv run go4cl transfer-mechanism analyze \
        --config "${COMPONENT_CONFIG}" \
        --out "${COMPONENT_OUT}"
fi

printf 'finished_at=%s\n' "$(date --iso-8601=seconds)" >> "${RUN_ROOT}/RUN_INFO.txt"

echo
echo "============================================================"
echo "Finished: ${RUN_ROOT}"
echo "============================================================"
echo "01_core_fresh: fresh optimizer + B-only"
echo "02_core_preserve: carried-optimizer control"
echo "03_replay_overlap: fixed 10% replay"
echo "04_no_replay_anchors: matched no-replay controls"
echo "05_component_reset: parameter-source localization"
