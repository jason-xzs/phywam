#!/usr/bin/bash

set -euo pipefail
set -x

REPO_ROOT=${REPO_ROOT:-"/data/worldmodel_xzs/phywam_v3_adapter_exp"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_phywam_freeze_phys_only"}
SAVE_ROOT=${SAVE_ROOT:-"${REPO_ROOT}/train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36"}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36"}
NGPU=${NGPU:-"4"}
MASTER_PORT=${MASTER_PORT:-"29571"}
RUN_MODE=${RUN_MODE:-"bg"}
RESUME=${RESUME:-"auto"}
WANDB_PROJECT=${WANDB_PROJECT:-"phywam_v3_s2_adapter_late8_conf_taskgate_phys36"}
WANDB_MODE=${WANDB_MODE:-"online"}
PHYS_MONITOR_INTERVAL=${PHYS_MONITOR_INTERVAL:-"20"}

cd "${REPO_ROOT}"
mkdir -p "${LOG_DIR}"

resume_args=()
if [ "${RESUME}" = "auto" ]; then
    latest_checkpoint=""
    if [ -d "${SAVE_ROOT}/checkpoints" ]; then
        mapfile -t checkpoint_candidates < <(
            find "${SAVE_ROOT}/checkpoints" -maxdepth 1 -type d -name 'checkpoint_step_*' | sort -Vr
        )
        for checkpoint in "${checkpoint_candidates[@]}"; do
            if [ -f "${checkpoint}/transformer/diffusion_pytorch_model.safetensors" ] \
                && [ -f "${checkpoint}/transformer/config.json" ] \
                && [ -f "${checkpoint}/training_state.pt" ]; then
                latest_checkpoint="${checkpoint}"
                break
            fi
            echo "[WARN] Skip incomplete checkpoint: ${checkpoint}"
        done
    fi
    if [ -n "${latest_checkpoint}" ]; then
        resume_args=(--resume-from "${latest_checkpoint}" --resume-optimizer-state)
        echo "[INFO] Auto resume from ${latest_checkpoint}"
    else
        echo "[INFO] No checkpoint found under ${SAVE_ROOT}/checkpoints; starting a new run."
    fi
elif [ "${RESUME}" != "none" ]; then
    resume_args=(--resume-from "${RESUME}" --resume-optimizer-state)
    echo "[INFO] Resume from ${RESUME}"
else
    echo "[INFO] RESUME=none; starting a new run."
fi

CONFIG_NAME="${CONFIG_NAME}" \
SAVE_ROOT="${SAVE_ROOT}" \
LOG_DIR="${LOG_DIR}" \
NGPU="${NGPU}" \
MASTER_PORT="${MASTER_PORT}" \
RUN_MODE="${RUN_MODE}" \
WANDB_PROJECT="${WANDB_PROJECT}" \
WANDB_MODE="${WANDB_MODE}" \
bash script/run_va_posttrain.sh \
    --config-name "${CONFIG_NAME}" \
    --save-root "${SAVE_ROOT}" \
    --freeze-backbone \
    --phys-monitor-interval "${PHYS_MONITOR_INTERVAL}" \
    --phys-monitor-jsonl "${LOG_DIR}/phys_train_stats.jsonl" \
    "${resume_args[@]}" \
    "$@"
