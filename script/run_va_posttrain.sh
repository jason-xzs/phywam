#!/usr/bin/bash

set -x

umask 007
 
NGPU=${NGPU:-"8"}
MASTER_PORT=${MASTER_PORT:-"29551"}
PORT=${PORT:-"1106"}
LOG_RANK=${LOG_RANK:-""}
TORCHFT_LIGHTHOUSE=${TORCHFT_LIGHTHOUSE:-"http://localhost:29510"}
CONFIG_NAME=${CONFIG_NAME:-"robotwin_train_phywam_freeze_phys_only"}
RUN_MODE=${RUN_MODE:-"fg"}
LOG_DIR=${LOG_DIR:-"/data/worldmodel_xzs/phywam_v3_adapter_exp/logs/train_late8_conf_taskgate"}

overrides=""
if [ $# -ne 0 ]; then
    overrides="$*"
fi

config_name=${CONFIG_NAME}

train_args="${overrides}"
if [[ " ${overrides} " != *" --config-name "* ]]; then
    train_args="--config-name ${config_name} ${overrides}"
fi

export WANDB_BASE_URL="https://api.wandb.ai"
export WANDB_TEAM_NAME="1661825351-beijing-institute-of-technology"
export WANDB_ENTITY="1661825351-beijing-institute-of-technology"
export WANDB_PROJECT="${WANDB_PROJECT:-phywam_2_sub}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_API_KEY="${WANDB_API_KEY:-}"

if [ "${WANDB_MODE}" != "offline" ] && [ -z "${WANDB_API_KEY}" ]; then
    echo "[ERROR] WANDB_API_KEY is empty. Set WANDB_API_KEY or use WANDB_MODE=offline."
    exit 1
fi

mkdir -p "${LOG_DIR}"
LOG_TS=$(date +"%Y%m%d_%H%M%S")
TRAIN_LOG="${LOG_DIR}/train_${LOG_TS}.log"
PID_FILE="${LOG_DIR}/train.pid"

# 手动修改
export HF_HOME=/data/worldmodel_xzs/.cache/huggingface
export HF_DATASETS_CACHE=/data/worldmodel_xzs/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/data/worldmodel_xzs/.cache/huggingface/hub
export TORCH_HOME=/data/worldmodel_xzs/.cache/torch

## node setting
num_gpu=${NGPU}
master_port=${MASTER_PORT}
log_rank=${LOG_RANK}
torchft_lighthouse=${TORCHFT_LIGHTHOUSE}

rank_filter_arg=""
if [ -n "${log_rank}" ]; then
    rank_filter_arg="--local-ranks-filter=${log_rank}"
fi

if [ "${RUN_MODE}" != "fg" ] && [ "${RUN_MODE}" != "bg" ]; then
    echo "[ERROR] RUN_MODE must be fg or bg, got: ${RUN_MODE}"
    exit 1
fi

## cmd setting
export TOKENIZERS_PARALLELISM=false
if [ "${RUN_MODE}" = "bg" ]; then
    nohup env PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
    python -m torch.distributed.run \
        --nproc_per_node=${num_gpu} \
        ${rank_filter_arg} \
        --master_port ${master_port} \
        --tee 3 \
        -m wan_va.train ${train_args} \
        > "${TRAIN_LOG}" 2>&1 < /dev/null &
    pid=$!
    echo "${pid}" > "${PID_FILE}"
    echo "[INFO] train started in background, pid=${pid}"
    echo "[INFO] log: ${TRAIN_LOG}"
    echo "[INFO] stop: kill ${pid}"
else
    PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True" TORCHFT_LIGHTHOUSE=${torchft_lighthouse} \
    python -m torch.distributed.run \
        --nproc_per_node=${num_gpu} \
        ${rank_filter_arg} \
        --master_port ${master_port} \
        --tee 3 \
        -m wan_va.train ${train_args} 2>&1 | tee -a "${TRAIN_LOG}"
fi

## CUDA_VISIBLE_DEVICES=6,7 NGPU=2 bash script/run_va_posttrain.sh
## 后台运行示例（SSH 断开后继续）：
## CUDA_VISIBLE_DEVICES=0,1,2,3 NGPU=4 RUN_MODE=bg \
##   LOG_DIR=/data/worldmodel_xzs/phywam_v3_adapter_exp/logs/train_late8_conf_taskgate \
##   bash script/run_va_posttrain.sh --config-name robotwin_train_phywam_freeze_phys_only
## 注：不传 --phys-memory-path 时，默认从每个任务目录下的 physics_features3.4/ 自动读取。
