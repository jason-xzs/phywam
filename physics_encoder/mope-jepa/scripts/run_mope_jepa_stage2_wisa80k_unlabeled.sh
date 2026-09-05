#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_mope_jepa_wisa7k.sh
# MoPE-JEPA 18类(17物理+1通用) 端到端预训练
#
# 用法：
#   smoke（单卡快速验证）： bash scripts/run_mope_jepa_wisa7k.sh smoke 0
#   正式训练（4卡）：       bash scripts/run_mope_jepa_wisa7k.sh train 4,5,6,7
#                          bash scripts/run_mope_jepa_wisa7k.sh train 2,3
# OpenVid 通用数据：
#   - OPENVID_DIR 为空 → 只用 WISA（先 debug 用）
#   - 下好后把 OPENVID_DIR 填上路径，自动混合训练
# ──────────────────────────────────────────────────────────────────────────────

set -e

# ── 路径配置（要改就改这里）─────────────────────────────────────────────────
MOPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="/data2/WISA-80K"
ANNO_PATH=""
PHYSICS_SOFT_PATH=""
PRETRAIN_CKPT="/data2/mope-jepa/output/stage1_wisa7k_physics_only/checkpoint-100.pth"

# OpenVid 通用数据目录：为空="" → 只用WISA；下好后填 "/data2/openvid"
OPENVID_DIR=""
OPENVID_MAX=0          # openvid-7k

# 门控阈值（通用维分数 >= 此值 → 通用组）
GATE_THRESHOLD=0.5

OUTPUT_NAME="stage2_wisa80k_unlabeled"   # 官方VideoMAE初始化重训

# ── 解析参数 ──────────────────────────────────────────────────────────────────
MODE="${1:-train}"          # smoke | train
GPUS="${2:-0}"              # GPU 列表，如 0 或 0,1,2,3

NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
MASTER_PORT=29885

OUTPUT_DIR="${MOPE_ROOT}/output/${OUTPUT_NAME}"
LOG_DIR="${OUTPUT_DIR}/log"
mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 检查预训练权重
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: 找不到预训练权重 ${PRETRAIN_CKPT}"
    exit 1
fi

echo ">>> 输出目录：${OUTPUT_DIR}"
echo ">>> GPU：${GPUS}（共 ${NUM_GPUS} 张）"
echo ">>> OpenVid：${OPENVID_DIR:-（空，只用WISA）}  门控阈值：${GATE_THRESHOLD}"

# ── OpenVid 参数（为空则不传）────────────────────────────────────────────────
OPENVID_ARGS=()
if [ -n "${OPENVID_DIR}" ]; then
    OPENVID_ARGS=(--openvid_dir "${OPENVID_DIR}" --openvid_max "${OPENVID_MAX}")
fi

# ── 标注参数（为空则不传）────────────────────────────────────────────────────
ANNO_ARGS=(--anno_path "${ANNO_PATH}")

# ── 公共参数 ──────────────────────────────────────────────────────────────────
COMMON_ARGS=(
  --model pretrain_mope_jepa_base_patch16_224
  --datasets_root "${DATASETS_ROOT}"
  "${ANNO_ARGS[@]}"
  --output_dir "${OUTPUT_DIR}"
  --log_dir "${LOG_DIR}"
  --finetune "${PRETRAIN_CKPT}"
  --freeze_encoder_except_moe
  --train_predictor
  --gate_threshold "${GATE_THRESHOLD}"
  --num_frames 16
  --sampling_rate 4
  --input_size 224
  --mask_type tube
  --mask_ratio 0.9
  --tubelet_size 2
  --lr 5e-5
  --min_lr 1e-6
  --weight_decay 0.05
  --use_mope
  --num_physics_experts 17
  --num_general_experts 10
  --num_shared_experts 4
  --moe_balance_weight 0.01
  --predictor_dim 384
  --predictor_depth 6
  --predictor_num_heads 6
  --sigreg_weight 0.3
  --no_auto_resume
  "${OPENVID_ARGS[@]}"
)

# ── Smoke Test（单卡快速验证）────────────────────────────────────────────────
if [ "${MODE}" = "smoke" ]; then
    echo "=== Smoke Test ==="
    CUDA_VISIBLE_DEVICES=${GPUS} python ${MOPE_ROOT}/run_jepa_pretraining.py \
        "${COMMON_ARGS[@]}" \
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
echo "=== 开始训练 ==="
if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPUS torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${MOPE_ROOT}/run_jepa_pretraining.py \
        "${COMMON_ARGS[@]}" \
        --batch_size 32 \
        --epochs 100 \
        --warmup_epochs 1 \
        --num_workers 4 \
        --save_ckpt_freq 3 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
else
    CUDA_VISIBLE_DEVICES=$GPUS python ${MOPE_ROOT}/run_jepa_pretraining.py \
        "${COMMON_ARGS[@]}" \
        --batch_size 32 \
        --epochs 100 \
        --warmup_epochs 1 \
        --num_workers 4 \
        --save_ckpt_freq 3 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

# ── 启动速查 ──────────────────────────────────────────────────────────────────
# conda activate /data1/miniconda3/envs/worldmodel_mopeva
# cd /data2/mope-jepa
#   bash scripts/run_mope_jepa_wisa7k.sh smoke 0          # 先单卡验证
#   bash scripts/run_mope_jepa_wisa7k.sh train 0,1,2,3    # 4卡正式训
# OpenVid下好后：把脚本里 OPENVID_DIR="/data2/openvid" 填上，再 train
