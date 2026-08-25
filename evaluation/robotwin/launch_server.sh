START_PORT=${START_PORT:-29056}
MASTER_PORT=${MASTER_PORT:-29061}
CONFIG_NAME=${CONFIG_NAME:-robotwin_infer}
PHYWAM_INFER_CHECKPOINT=${PHYWAM_INFER_CHECKPOINT:-/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/checkpoints/checkpoint_step_16000}
export PHYWAM_INFER_CHECKPOINT
WAIT_FOR_CHILDREN=${WAIT_FOR_CHILDREN:-0}
trap '' HUP

LOG_DIR='./logs'
mkdir -p $LOG_DIR

USE_PHYS_MEMORY=${USE_PHYS_MEMORY:-1}
PHYS_MEMORY_INFER_MODE=${PHYS_MEMORY_INFER_MODE:-phase}
PHYS_MEMORY_DIM=${PHYS_MEMORY_DIM:-768}
PHYS_MEMORY_BLOCK_SIZE=${PHYS_MEMORY_BLOCK_SIZE:-16}
INFER_PHYS_MEMORY_NPY=${INFER_PHYS_MEMORY_NPY:-}
MOPE_REPO=${MOPE_REPO:-/data/worldmodel_xzs/phywam_v3/mope-jepa}
MOPE_CKPT=${MOPE_CKPT:-/data/worldmodel_xzs/phywam_v3/mope-jepa/output/mope_jepa_v39_task_canonical_event_physics_c12a24_freeze150/checkpoint-400.pth}
PHYS_EVENT_LABEL_PATH=${PHYS_EVENT_LABEL_PATH:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical/robotwin_s2_8tasks_c50_a500_event_segments_task_canonical_mope_jepa.json}
PHYS_PHASE_SWITCH_PATIENCE=${PHYS_PHASE_SWITCH_PATIENCE:-1}
PHYS_PHASE_BUFFER_FRAMES=${PHYS_PHASE_BUFFER_FRAMES:-16}
PHYS_ENABLE_IMAGE_DELTA_EVENT=${PHYS_ENABLE_IMAGE_DELTA_EVENT:-0}
PHYS_GRIPPER_EVENT_THRESHOLD=${PHYS_GRIPPER_EVENT_THRESHOLD:-0.2}
PHYS_GRIPPER_OPEN_THRESHOLD=${PHYS_GRIPPER_OPEN_THRESHOLD:-0.2}
PHYS_GRIPPER_CLOSED_THRESHOLD=${PHYS_GRIPPER_CLOSED_THRESHOLD:-0.8}
PHYS_PHASE_CONFIDENCE_GATE=${PHYS_PHASE_CONFIDENCE_GATE:-1}
PHYS_DEFAULT_TASK_GATE=${PHYS_DEFAULT_TASK_GATE:-1.0}
PHYS_TASK_GATES_JSON=${PHYS_TASK_GATES_JSON:-}
if [ "$USE_PHYS_MEMORY" != "1" ]; then
    PHYS_MEMORY_INFER_MODE=none
fi

EXPERIMENT_NAME=${EXPERIMENT_NAME:-phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_phys36}
CHECKPOINT_TAG=${CHECKPOINT_TAG:-$(basename "$PHYWAM_INFER_CHECKPOINT")}
INFER_VARIANT=${INFER_VARIANT:-${PHYS_MEMORY_INFER_MODE}}
save_root=${SAVE_ROOT:-"/data/public_data/xzs_data/${EXPERIMENT_NAME}/inference/${CHECKPOINT_TAG}/${INFER_VARIANT}/server"}
mkdir -p $save_root

# Prefer cuDNN bundled with the active Python env over any previously sourced env.
PYTHON_CUDNN_LIB=$(python - <<'PY'
import os
import site

for base in site.getsitepackages():
    path = os.path.join(base, "nvidia", "cudnn", "lib")
    if os.path.isdir(path):
        print(path)
        break
PY
)
if [ -n "$PYTHON_CUDNN_LIB" ]; then
    export LD_LIBRARY_PATH="${PYTHON_CUDNN_LIB}:${LD_LIBRARY_PATH:-}"
fi

batch_time=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${LOG_DIR}/server_${START_PORT}_${batch_time}.log"

phys_args=(--phys-memory-infer-mode "$PHYS_MEMORY_INFER_MODE")
if [ "$USE_PHYS_MEMORY" = "1" ]; then
    phys_args+=(--use-phys-memory)
    phys_args+=(--phys-memory-dim "$PHYS_MEMORY_DIM")
    phys_args+=(--phys-memory-block-size "$PHYS_MEMORY_BLOCK_SIZE")
    phys_args+=(--mope-repo "$MOPE_REPO")
    phys_args+=(--mope-ckpt "$MOPE_CKPT")
    phys_args+=(--phys-event-label-path "$PHYS_EVENT_LABEL_PATH")
    phys_args+=(--phys-phase-switch-patience "$PHYS_PHASE_SWITCH_PATIENCE")
    phys_args+=(--phys-phase-buffer-frames "$PHYS_PHASE_BUFFER_FRAMES")
    phys_args+=(--phys-gripper-event-threshold "$PHYS_GRIPPER_EVENT_THRESHOLD")
    phys_args+=(--phys-gripper-open-threshold "$PHYS_GRIPPER_OPEN_THRESHOLD")
    phys_args+=(--phys-gripper-closed-threshold "$PHYS_GRIPPER_CLOSED_THRESHOLD")
    phys_args+=(--phys-default-task-gate "$PHYS_DEFAULT_TASK_GATE")
    if [ "$PHYS_PHASE_CONFIDENCE_GATE" = "0" ]; then
        phys_args+=(--no-phys-phase-confidence-gate)
    fi
    if [ -n "$PHYS_TASK_GATES_JSON" ]; then
        phys_args+=(--phys-task-gates-json "$PHYS_TASK_GATES_JSON")
    fi
    if [ "$PHYS_ENABLE_IMAGE_DELTA_EVENT" = "1" ]; then
        phys_args+=(--enable-phys-image-delta-event)
    fi
    if [ -n "$INFER_PHYS_MEMORY_NPY" ]; then
        phys_args+=(--infer-phys-memory-npy "$INFER_PHYS_MEMORY_NPY")
    fi
fi

nohup python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port $MASTER_PORT \
    wan_va/wan_va_server.py \
    --config-name $CONFIG_NAME \
    --port $START_PORT \
    --save_root $save_root \
    "${phys_args[@]}" > $LOG_FILE 2>&1 < /dev/null &

pid=$!
echo "$pid" > pids.txt
echo "Launched server PID=${pid}, log=${LOG_FILE}"
echo "PYTHON_CUDNN_LIB=${PYTHON_CUDNN_LIB}"

if [ "$WAIT_FOR_CHILDREN" = "1" ]; then
    wait "$pid"
fi
