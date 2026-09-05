#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_mope_wisa7k.sh
# MoPE 在 WISA-7K 上预训练启动脚本
#
# 用法：
#   单卡 smoke test：  bash scripts/run_mope_wisa7k.sh smoke
#   单机多卡训练：     bash scripts/run_mope_wisa7k.sh train
# ──────────────────────────────────────────────────────────────────────────────

set -e

MOPE_ROOT="/home/nvme04/mope"
DATASETS_ROOT="${MOPE_ROOT}/datasets"
ANNO_PATH="${DATASETS_ROOT}/wisa_7k.json"
# Qwen 软标签（router soft CE）；不需要时设为空字符串
PHYSICS_SOFT_PATH="${DATASETS_ROOT}/wisa_7k_qwen3vl8b_scores_all.json"
OUTPUT_DIR="${MOPE_ROOT}/output/mope_wisa7k_vitb_2"
LOG_DIR="${OUTPUT_DIR}/log"

# GPU 数量（自动检测）
GPUS=${2:-"0"}                          # 第二个参数，默认用GPU 0
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)   # 自动数有几张
MASTER_PORT=29513

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# ── 公共参数 ──────────────────────────────────────────────────────────────────
COMMON_ARGS="
  --model pretrain_videomae_base_patch16_224
  --datasets_root ${DATASETS_ROOT}
  --anno_path ${ANNO_PATH}
  --physics_soft_path ${PHYSICS_SOFT_PATH}
  --output_dir ${OUTPUT_DIR}
  --log_dir ${LOG_DIR}
  --num_frames 16
  --sampling_rate 4
  --input_size 224
  --mask_type tube
  --mask_ratio 0.9
  --decoder_mask_type run_cell
  --decoder_mask_ratio 0.0
  --decoder_depth 4
  --tubelet_size 2
  --num_workers 8
  --lr 1.5e-4
  --min_lr 1e-5
  --weight_decay 0.05
  --save_ckpt_freq 50
  --use_mope
  --num_routable_experts 17
  --num_shared_experts 4
  --top_k 5
  --moe_balance_weight 0.01
  --physics_cls_weight 1.0
"

# ── Smoke Test（2 步验证链路）────────────────────────────────────────────────
if [ "$1" = "smoke" ]; then
    echo "=== Smoke Test ==="
    CUDA_VISIBLE_DEVICES=${GPUS} python ${MOPE_ROOT}/run_mae_pretraining.py \
        ${COMMON_ARGS} \
        --batch_size 2 \
        --epochs 1 \
        --warmup_epochs 0 \
        --num_workers 2 \
        --save_ckpt_freq 1 \
        2>&1 | tee "${OUTPUT_DIR}/smoke.log"
    echo "=== Smoke Test Done ==="
    exit 0
fi

# ── 正式训练 ──────────────────────────────────────────────────────────────────
echo "=== Training on ${NUM_GPUS} GPU(s) ==="

if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPUS torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${MOPE_ROOT}/run_mae_pretraining.py \
        ${COMMON_ARGS} \
        --batch_size 128 \
        --epochs 400 \
        --warmup_epochs 20 \
        --with_checkpoint \
        --num_workers 16 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
else
    CUDA_VISIBLE_DEVICES=$GPUS python ${MOPE_ROOT}/run_mae_pretraining.py \
        ${COMMON_ARGS} \
        --batch_size 128 \
        --epochs 400 \
        --warmup_epochs 20 \
        --num_workers 16 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

# 启动训练
# conda activate mope
# cd /home/nvme04/mope
# bash scripts/run_mope_wisa7k.sh train 0,1,2,3
# bash scripts/run_mope_wisa7k.sh train 4,5,6,7 --resume /home/nvme04/mope/output/mope_wisa7k_vitb_1/checkpoint-299.pth


