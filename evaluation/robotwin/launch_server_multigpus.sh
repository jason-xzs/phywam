START_PORT=${START_PORT:-31556}
MASTER_PORT=${MASTER_PORT:-29661}
CONFIG_NAME=${CONFIG_NAME:-robotwin_infer}
PHYWAM_INFER_CHECKPOINT=${PHYWAM_INFER_CHECKPOINT:-/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/checkpoints/checkpoint_step_16000}
export PHYWAM_INFER_CHECKPOINT
NUM_GPUS=${NUM_GPUS:-}
GPU_OFFSET=${GPU_OFFSET:-0}
GPU_IDS=${GPU_IDS:-}
WAIT_FOR_CHILDREN=${WAIT_FOR_CHILDREN:-0}
trap '' HUP

USE_PHYS_MEMORY=${USE_PHYS_MEMORY:-1}
PHYS_MEMORY_INFER_MODE=${PHYS_MEMORY_INFER_MODE:-phase}
PHYS_MEMORY_DIM=${PHYS_MEMORY_DIM:-768}
PHYS_MEMORY_BLOCK_SIZE=${PHYS_MEMORY_BLOCK_SIZE:-16}
INFER_PHYS_MEMORY_NPY=${INFER_PHYS_MEMORY_NPY:-}
MOPE_REPO=${MOPE_REPO:-/data/worldmodel_xzs/phywam_v3/mope-jepa_0706}
MOPE_CKPT=${MOPE_CKPT:-/data/worldmodel_xzs/phywam_v3/mope-jepa_0706/output/mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth}
PHYS_EVENT_LABEL_PATH=${PHYS_EVENT_LABEL_PATH:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical/robotwin_s2_8tasks_c50_a500_event_segments_task_canonical_mope_jepa.json}
PHYS_PHASE_SWITCH_PATIENCE=${PHYS_PHASE_SWITCH_PATIENCE:-1}
PHYS_PHASE_BUFFER_FRAMES=${PHYS_PHASE_BUFFER_FRAMES:-16}
PHYS_EVENT_THRESHOLD=${PHYS_EVENT_THRESHOLD:-0.5}
PHYS_EVENT_WINDOW_BLOCKS=${PHYS_EVENT_WINDOW_BLOCKS:-0}
PHYS_EVENT_DETECTOR=${PHYS_EVENT_DETECTOR:-mope}
PHYS_GRIPPER_EVENT_THRESHOLD=${PHYS_GRIPPER_EVENT_THRESHOLD:-0.2}
PHYS_IMAGE_EVENT_THRESHOLD=${PHYS_IMAGE_EVENT_THRESHOLD:-0.05}
PHYS_ENABLE_IMAGE_DELTA_EVENT=${PHYS_ENABLE_IMAGE_DELTA_EVENT:-0}
PHYS_GRIPPER_OPEN_THRESHOLD=${PHYS_GRIPPER_OPEN_THRESHOLD:-0.2}
PHYS_GRIPPER_CLOSED_THRESHOLD=${PHYS_GRIPPER_CLOSED_THRESHOLD:-0.8}
PHYS_PHASE_CONFIDENCE_GATE=${PHYS_PHASE_CONFIDENCE_GATE:-1}
PHYS_DEFAULT_TASK_GATE=${PHYS_DEFAULT_TASK_GATE:-1.0}
PHYS_TASK_GATES_JSON=${PHYS_TASK_GATES_JSON:-}
if [ "$USE_PHYS_MEMORY" != "1" ]; then
    PHYS_MEMORY_INFER_MODE=none
fi

LOG_DIR='./logs/late8_conf_taskgate_phys36'
mkdir -p $LOG_DIR

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
pid_file="${LOG_DIR}/server_pids_${batch_time}.txt"
> "$pid_file"

echo "[Launch Config] CONFIG_NAME=${CONFIG_NAME}"
echo "[Launch Config] PHYWAM_INFER_CHECKPOINT=${PHYWAM_INFER_CHECKPOINT}"
echo "[Launch Config] USE_PHYS_MEMORY=${USE_PHYS_MEMORY} PHYS_MEMORY_INFER_MODE=${PHYS_MEMORY_INFER_MODE}"
echo "[Launch Config] INFER_PHYS_MEMORY_NPY=${INFER_PHYS_MEMORY_NPY}"
echo "[Launch Config] PHYS_ENABLE_IMAGE_DELTA_EVENT=${PHYS_ENABLE_IMAGE_DELTA_EVENT}"
echo "[Launch Config] PHYS_GRIPPER_OPEN_THRESHOLD=${PHYS_GRIPPER_OPEN_THRESHOLD} PHYS_GRIPPER_CLOSED_THRESHOLD=${PHYS_GRIPPER_CLOSED_THRESHOLD}"
echo "[Launch Config] PHYS_PHASE_CONFIDENCE_GATE=${PHYS_PHASE_CONFIDENCE_GATE} PHYS_DEFAULT_TASK_GATE=${PHYS_DEFAULT_TASK_GATE} PHYS_TASK_GATES_JSON=${PHYS_TASK_GATES_JSON}"
echo "[Launch Config] SAVE_ROOT=${save_root}"
echo "[Launch Config] PYTHON_CUDNN_LIB=${PYTHON_CUDNN_LIB}"

gpu_ids=()
if [ -n "$GPU_IDS" ]; then
    IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
    if [ ${#gpu_ids[@]} -eq 0 ]; then
        echo "GPU_IDS is set but empty: $GPU_IDS" >&2
        exit 1
    fi

    for idx in "${!gpu_ids[@]}"; do
        gpu_ids[$idx]=$(echo "${gpu_ids[$idx]}" | xargs)
        if [[ ! "${gpu_ids[$idx]}" =~ ^[0-9]+$ ]]; then
            echo "Invalid GPU id in GPU_IDS: ${gpu_ids[$idx]}" >&2
            exit 1
        fi
    done

    if [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=${#gpu_ids[@]}
    fi

    if [ "$NUM_GPUS" -le 0 ] || [ "$NUM_GPUS" -gt ${#gpu_ids[@]} ]; then
        echo "NUM_GPUS must be in 1..${#gpu_ids[@]} when GPU_IDS is used, got: ${NUM_GPUS}" >&2
        exit 1
    fi
else
    if [ -z "$NUM_GPUS" ]; then
        NUM_GPUS=8
    fi
fi

for ((i=0; i<NUM_GPUS; i++)); do
    if [ -n "$GPU_IDS" ]; then
        gpu_id=${gpu_ids[$i]}
    else
        gpu_id=$((GPU_OFFSET + i))
    fi
    CURRENT_PORT=$((START_PORT + i))
    CURRENT_MASTER_PORT=$((MASTER_PORT + i))

    LOG_FILE="${LOG_DIR}/server_${gpu_id}_${batch_time}.log"
    echo "[Task ${i}] GPU: ${gpu_id} | PORT: ${CURRENT_PORT} | MASTER_PORT: ${CURRENT_MASTER_PORT} | Log: ${LOG_FILE}"

    phys_args=()
    if [ "$USE_PHYS_MEMORY" = "1" ] && [ "$PHYS_MEMORY_INFER_MODE" != "none" ]; then
        phys_args+=(--use-phys-memory)
    fi
    phys_args+=(--phys-memory-infer-mode "$PHYS_MEMORY_INFER_MODE")
    phys_args+=(--phys-memory-dim "$PHYS_MEMORY_DIM")
    phys_args+=(--phys-memory-block-size "$PHYS_MEMORY_BLOCK_SIZE")
    phys_args+=(--mope-repo "$MOPE_REPO")
    phys_args+=(--mope-ckpt "$MOPE_CKPT")
    phys_args+=(--phys-event-label-path "$PHYS_EVENT_LABEL_PATH")
    phys_args+=(--phys-phase-switch-patience "$PHYS_PHASE_SWITCH_PATIENCE")
    phys_args+=(--phys-phase-buffer-frames "$PHYS_PHASE_BUFFER_FRAMES")
    phys_args+=(--phys-event-threshold "$PHYS_EVENT_THRESHOLD")
    phys_args+=(--phys-event-window-blocks "$PHYS_EVENT_WINDOW_BLOCKS")
    phys_args+=(--phys-event-detector "$PHYS_EVENT_DETECTOR")
    phys_args+=(--phys-gripper-event-threshold "$PHYS_GRIPPER_EVENT_THRESHOLD")
    phys_args+=(--phys-image-event-threshold "$PHYS_IMAGE_EVENT_THRESHOLD")
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

    CUDA_VISIBLE_DEVICES=$gpu_id  \
    nohup python -m torch.distributed.run \
        --nproc_per_node 1 \
        --master_port $CURRENT_MASTER_PORT \
        wan_va/wan_va_server.py \
        --config-name $CONFIG_NAME \
        --save_root $save_root \
        --port $CURRENT_PORT \
        "${phys_args[@]}" > $LOG_FILE 2>&1 &
    echo "$!" >> "$pid_file"
    sleep 2;
done

if [ -n "$GPU_IDS" ]; then
    echo "All ${NUM_GPUS} instances have been launched in the background. GPU_IDS=${GPU_IDS}"
else
    echo "All ${NUM_GPUS} instances have been launched in the background."
fi
echo "PIDs saved to ${pid_file}"
echo "To terminate all processes, run: kill \$(cat ${pid_file})"
if [ "$WAIT_FOR_CHILDREN" = "1" ]; then
    wait
fi

# NUM_GPUS=4 GPU_OFFSET=0 START_PORT=30556 MASTER_PORT=30661 bash evaluation/robotwin/launch_server_multigpus.sh
# GPU_IDS=0,2,3,7 NUM_GPUS=4 START_PORT=30556 MASTER_PORT=30661 bash evaluation/robotwin/launch_server_multigpus.sh
# NUM_GPUS=4 GPU_OFFSET=0 START_PORT=30556 bash evaluation/robotwin/launch_client_multigpus.sh "" 0
# RESUME=1 GPU_IDS=0,2,3,7 NUM_GPUS=4 START_PORT=30556 bash evaluation/robotwin/launch_client_multigpus.sh "" 0
#
# Inference ablation modes:
# 1) 无物理 token:
#    USE_PHYS_MEMORY=0 PHYS_MEMORY_INFER_MODE=none bash evaluation/robotwin/launch_server_multigpus.sh
# 2) 在线 phase token:
#    USE_PHYS_MEMORY=1 PHYS_MEMORY_INFER_MODE=phase bash evaluation/robotwin/launch_server_multigpus.sh
