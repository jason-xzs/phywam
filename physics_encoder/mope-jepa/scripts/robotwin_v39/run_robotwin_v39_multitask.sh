#!/usr/bin/env bash
set -euo pipefail

# New-architecture RobotWin adaptation.
#
# Stage A: full-visible event-head warmup
#   bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh head 0,1,2,3 fg
#
# Stage B: joint JEPA + v35 physics + v39 event adaptation
#   bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh joint 0,1,2,3 bg
#
# Smoke checks:
#   bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh smoke_head 0 fg
#   FINETUNE_CKPT=/data2/mope-jepa/output/stage1_wisa7k_physics_only/checkpoint-100.pth \
#     bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh smoke_joint 0 fg

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

MODE="${1:-head}"  # head | joint | smoke_head | smoke_joint
GPUS="${2:-0,1,2,3}"
RUN_MODE="${3:-fg}"  # fg | bg
NUM_GPUS="$(tr ',' '\n' <<<"${GPUS}" | wc -l)"
MASTER_PORT="${MASTER_PORT:-29706}"
RUN_TS="${RUN_TS:-$(date +%Y%m%d_%H%M%S)}"

PYTHON="${PYTHON:-/data1/miniconda3/envs/worldmodel_mopeva/bin/python}"
TORCHRUN="${TORCHRUN:-/data1/miniconda3/envs/worldmodel_mopeva/bin/torchrun}"

DATASETS_ROOT="${DATASETS_ROOT:-/data/public_data/xzs_data/lingbotva-post-training-dataset/robotwin-clean-and-aug-lerobot}"
EVENT_LABEL_PATH="${EVENT_LABEL_PATH:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/event_labels_full50_qwen_v39_task_canonical/robotwin_full50_c12_a24_event_segments_task_canonical_mope_jepa.json}"
PHYSICS_SOFT_PATH="${PHYSICS_SOFT_PATH:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/physics_labels_full50_qwen_v35/robotwin_full50_c12_a24_physics_scores_qwen.json}"
BASE_CKPT="${BASE_CKPT:-/data2/mope-jepa/output/stage1_wisa7k_physics_only/checkpoint-100.pth}"

HEAD_EPOCHS="${HEAD_EPOCHS:-10}"
JOINT_EPOCHS="${JOINT_EPOCHS:-100}"
HEAD_SAVE_CKPT_FREQ="${HEAD_SAVE_CKPT_FREQ:-${HEAD_EPOCHS}}"
JOINT_SAVE_CKPT_FREQ="${JOINT_SAVE_CKPT_FREQ:-10}"
HEAD_OUTPUT_DIR="${HEAD_OUTPUT_DIR:-${ROOT_DIR}/output/mope_jepa_0706_robotwin_v39_head_warmup}"
JOINT_OUTPUT_DIR="${JOINT_OUTPUT_DIR:-${ROOT_DIR}/output/mope_jepa_0706_robotwin_v39_joint}"
HEAD_CKPT="${HEAD_CKPT:-${HEAD_OUTPUT_DIR}/checkpoint-${HEAD_EPOCHS}.pth}"

case "${MODE}" in
  head)
    FINETUNE_CKPT="${FINETUNE_CKPT:-${BASE_CKPT}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${HEAD_OUTPUT_DIR}}"
    EPOCHS="${EPOCHS:-${HEAD_EPOCHS}}"
    BATCH_SIZE="${BATCH_SIZE:-32}"
    LR="${LR:-1.0e-4}"
    MASK_RATIO="${MASK_RATIO:-0.9}"
    SAVE_FREQ="${HEAD_SAVE_CKPT_FREQ}"
    STAGE_ARGS=(--event_head_only)
    ;;
  joint)
    FINETUNE_CKPT="${FINETUNE_CKPT:-${HEAD_CKPT}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${JOINT_OUTPUT_DIR}}"
    EPOCHS="${EPOCHS:-${JOINT_EPOCHS}}"
    BATCH_SIZE="${BATCH_SIZE:-16}"
    LR="${LR:-5.0e-5}"
    MASK_RATIO="${MASK_RATIO:-0.5}"
    SAVE_FREQ="${JOINT_SAVE_CKPT_FREQ}"
    STAGE_ARGS=(
      --freeze_encoder_except_moe
      --train_predictor
      --freeze_phys_router
    )
    ;;
  smoke_head)
    FINETUNE_CKPT="${FINETUNE_CKPT:-${BASE_CKPT}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/smoke_robotwin_v39_head}"
    EPOCHS=1
    BATCH_SIZE="${BATCH_SIZE:-2}"
    LR="${LR:-1.0e-4}"
    MASK_RATIO=0.9
    SAVE_FREQ=1
    STAGE_ARGS=(--event_head_only --max_train_steps_per_epoch 2)
    ;;
  smoke_joint)
    FINETUNE_CKPT="${FINETUNE_CKPT:-${BASE_CKPT}}"
    OUTPUT_DIR="${OUTPUT_DIR:-${ROOT_DIR}/output/smoke_robotwin_v39_joint}"
    EPOCHS=1
    BATCH_SIZE="${BATCH_SIZE:-2}"
    LR="${LR:-5.0e-5}"
    MASK_RATIO=0.5
    SAVE_FREQ=1
    STAGE_ARGS=(
      --freeze_encoder_except_moe
      --train_predictor
      --freeze_phys_router
      --max_train_steps_per_epoch 2
    )
    ;;
  *)
    echo "[ERROR] MODE must be head, joint, smoke_head, or smoke_joint; got ${MODE}" >&2
    exit 2
    ;;
esac

for path in "${PYTHON}" "${EVENT_LABEL_PATH}" "${PHYSICS_SOFT_PATH}" "${FINETUNE_CKPT}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] required path not found: ${path}" >&2
    exit 1
  fi
done
if [[ "${NUM_GPUS}" -gt 1 && ! -x "${TORCHRUN}" ]]; then
  echo "[ERROR] torchrun not executable: ${TORCHRUN}" >&2
  exit 1
fi

LOG_DIR="${LOG_DIR:-${ROOT_DIR}/logs/robotwin_v39_${MODE}_${RUN_TS}}"
LOG_FILE="${LOG_DIR}/train_${RUN_TS}.log"
PID_FILE="${LOG_DIR}/train.pid"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="1661825351-beijing-institute-of-technology"
export WANDB_ENTITY="1661825351-beijing-institute-of-technology"
export WANDB_PROJECT="${WANDB_PROJECT:-mope_jepa_0706_robotwin_v39}"
export WANDB_MODE="${WANDB_MODE:-online}"
WANDB_API_KEY_DEFAULT="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"
export WANDB_API_KEY="${WANDB_API_KEY:-${WANDB_API_KEY_DEFAULT}}"
export WANDB_NAME="${WANDB_NAME:-mope_jepa_0706_${MODE}_${RUN_TS}}"
export WANDB_DIR="${WANDB_DIR:-${LOG_DIR}/wandb}"
mkdir -p "${WANDB_DIR}"

if [ "${WANDB_MODE}" != "offline" ] && [ "${WANDB_MODE}" != "disabled" ] && [ -z "${WANDB_API_KEY}" ]; then
  echo "[ERROR] WANDB_API_KEY is empty. Set WANDB_API_KEY or use WANDB_MODE=offline/disabled." >&2
  exit 1
fi
if [ "${WANDB_MODE}" != "disabled" ]; then
  if ! "${PYTHON}" -c "import wandb" >/dev/null 2>&1; then
    echo "[ERROR] WANDB_MODE=${WANDB_MODE}, but wandb is not installed in ${PYTHON}." >&2
    echo "[ERROR] Install it with: ${PYTHON} -m pip install wandb" >&2
    echo "[ERROR] Or launch with WANDB_MODE=disabled to run without W&B." >&2
    exit 1
  fi
fi

export MOPE_FFMPEG_BIN="${MOPE_FFMPEG_BIN:-/data/worldmodel_xzs/ffmpeg-7.0.2-amd64-static/ffmpeg}"
export MOPE_DECODE_CACHE_DIR="${MOPE_DECODE_CACHE_DIR:-/data/worldmodel_xzs/phywam_v3/tmp/mope_event_decode_cache_v3.6}"
mkdir -p "${MOPE_DECODE_CACHE_DIR}"

COMMON_ARGS=(
  --model pretrain_mope_jepa_base_patch16_224
  --datasets_root "${DATASETS_ROOT}"
  --event_label_path "${EVENT_LABEL_PATH}"
  --physics_soft_path "${PHYSICS_SOFT_PATH}"
  --finetune "${FINETUNE_CKPT}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --num_frames 16
  --sampling_rate 4
  --input_size 224
  --mask_type tube
  --mask_ratio "${MASK_RATIO}"
  --tubelet_size 2
  --use_mope
  --num_physics_experts 17
  --num_general_experts 10
  --num_shared_experts 4
  --candidate_k 5
  --gate_threshold 0.0
  --gate_hidden 0
  --physics_cls_weight "${PHYSICS_CLS_WEIGHT:-0.5}"
  --event_loss_weight "${EVENT_LOSS_WEIGHT:-1.0}"
  --moe_balance_weight "${MOE_BALANCE_WEIGHT:-0.01}"
  --sigreg_weight "${SIGREG_WEIGHT:-0.3}"
  --predictor_dim 384
  --predictor_depth 6
  --predictor_num_heads 6
  --batch_size "${BATCH_SIZE}"
  --epochs "${EPOCHS}"
  --warmup_epochs "${WARMUP_EPOCHS:-1}"
  --lr "${LR}"
  --min_lr "${MIN_LR:-1.0e-6}"
  --weight_decay "${WEIGHT_DECAY:-0.05}"
  --save_ckpt_freq "${SAVE_CKPT_FREQ:-${SAVE_FREQ}}"
  --num_workers "${NUM_WORKERS:-8}"
  --no_auto_resume
  "${STAGE_ARGS[@]}"
)

if [[ "${NUM_GPUS}" -gt 1 ]]; then
  CMD=(
    "${TORCHRUN}"
    --nproc_per_node="${NUM_GPUS}"
    --master_port="${MASTER_PORT}"
    run_jepa_pretraining.py
    "${COMMON_ARGS[@]}"
  )
else
  CMD=("${PYTHON}" run_jepa_pretraining.py "${COMMON_ARGS[@]}")
fi

echo "[INFO] mode=${MODE} gpus=${GPUS} run_mode=${RUN_MODE}"
echo "[INFO] finetune=${FINETUNE_CKPT}"
echo "[INFO] output=${OUTPUT_DIR}"
echo "[INFO] event_labels=${EVENT_LABEL_PATH}"
echo "[INFO] physics_labels=${PHYSICS_SOFT_PATH}"
echo "[INFO] save_ckpt_freq=${SAVE_CKPT_FREQ:-${SAVE_FREQ}}"
echo "[INFO] wandb_project=${WANDB_PROJECT} wandb_name=${WANDB_NAME} wandb_mode=${WANDB_MODE}"
echo "[INFO] enable_general=false physics_router_frozen=$([[ "${MODE}" == *joint ]] && echo true || echo false)"

if [[ "${RUN_MODE}" == "bg" ]]; then
  nohup env CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 \
    "${CMD[@]}" >"${LOG_FILE}" 2>&1 < /dev/null &
  echo "$!" >"${PID_FILE}"
  echo "[INFO] started pid=$(<"${PID_FILE}") log=${LOG_FILE}"
else
  env CUDA_VISIBLE_DEVICES="${GPUS}" PYTHONUNBUFFERED=1 \
    "${CMD[@]}" 2>&1 | tee "${LOG_FILE}"
fi
