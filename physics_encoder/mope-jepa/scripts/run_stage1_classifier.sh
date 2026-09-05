#!/bin/bash
# ──────────────────────────────────────────────────────────────────────────────
# scripts/run_stage1_classifier.sh
# 阶段一：单独训练【全局分类器/门控器】
#   - 加载 VideoMAEv2 预训练权重当 backbone（block0-7 提供真实特征）
#   - 冻结全部参数，只训 global_router.classifier，仅用 router_loss
#
# 用法：
#   单卡 smoke test：  bash scripts/run_stage1_classifier.sh smoke 0
#   单机多卡训练：     bash scripts/run_stage1_classifier.sh train 0,2,3,4
# ──────────────────────────────────────────────────────────────────────────────

set -e

MOPE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASETS_ROOT="${MOPE_ROOT}/datasets"
ANNO_PATH="${DATASETS_ROOT}/wisa_7k.json"
PHYSICS_SOFT_PATH="${DATASETS_ROOT}/wisa_7k_qwen3vl8b_scores_all.json"
PRETRAIN_CKPT="${MOPE_ROOT}/pretrained/videomaev2_base.pth"
OUTPUT_DIR="${MOPE_ROOT}/output/stage1_classifier_mlp_256_128"
LOG_DIR="${OUTPUT_DIR}/log"

GPUS=${2:-"0"}
NUM_GPUS=$(echo $GPUS | tr ',' '\n' | wc -l)
MASTER_PORT=29525

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

# 检查预训练权重
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: 找不到 VideoMAEv2 预训练权重 ${PRETRAIN_CKPT}"
    exit 1
fi

# ── 公共参数（新架构 + 阶段一开关）────────────────────────────────────────────
COMMON_ARGS="
  --model pretrain_mope_jepa_base_patch16_224
  --finetune ${PRETRAIN_CKPT}
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
  --tubelet_size 2
  --weight_decay 0.05
  --save_ckpt_freq 1
  --no_auto_resume
  --use_mope
  --stage1_train_classifier
  --num_physics_experts 17
  --num_general_experts 10
  --num_shared_experts 4
  --candidate_k 5
  --gate_threshold 0.0
  --moe_balance_weight 0.01
  --physics_cls_weight 1.0
  --min_lr 1.5e-4
  --gate_hidden 256
  --gate_layers 2
  --gate_dims 256,128
"
# 注：阶段一不加 --enable_general（通用组不参与）；不加 --with_checkpoint。
#     --finetune 加载 VideoMAEv2 权重：block0-7 普通层正常加载（分类器吃的就是
#     block7 特征）；block8-11 的 MoE/专家在权重里没有，strict=False 忽略、随机
#     初始化，但阶段一不影响分类器（分类器在 block8 之前出结果）。

# ── Smoke Test（2 步验证链路）────────────────────────────────────────────────
if [ "$1" = "smoke" ]; then
    echo "=== Stage1 Smoke Test ==="
    CUDA_VISIBLE_DEVICES=${GPUS} python ${MOPE_ROOT}/run_jepa_pretraining.py \
        ${COMMON_ARGS} \
        --batch_size 2 \
        --epochs 1 \
        --warmup_epochs 0 \
        --lr 1e-3 \
        --num_workers 0 \
        --max_train_steps_per_epoch 5 \
        2>&1 | tee "${OUTPUT_DIR}/smoke.log"
    echo "=== Stage1 Smoke Test Done ==="
    exit 0
fi

# ── 正式训练 ──────────────────────────────────────────────────────────────────
echo "=== Stage1 Training on ${NUM_GPUS} GPU(s) ==="

if [ "${NUM_GPUS}" -gt 1 ]; then
    CUDA_VISIBLE_DEVICES=$GPUS torchrun \
        --nproc_per_node=${NUM_GPUS} \
        --master_port=${MASTER_PORT} \
        ${MOPE_ROOT}/run_jepa_pretraining.py \
        ${COMMON_ARGS} \
        --batch_size 64 \
        --epochs 40 \
        --warmup_epochs 0 \
        --lr 1.5e-4 \
        --num_workers 8 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
else
    CUDA_VISIBLE_DEVICES=$GPUS python ${MOPE_ROOT}/run_jepa_pretraining.py \
        ${COMMON_ARGS} \
        --batch_size 64 \
        --epochs 40 \
        --warmup_epochs 0 \
        --lr 1.5e-4 \
        --num_workers 8 \
        2>&1 | tee "${OUTPUT_DIR}/train.log"
fi

# ── 阶段一训完后，验证分类器准确率 ────────────────────────────────────────────
# python infer_mope.py \
#     --ckpt ${OUTPUT_DIR}/checkpoint-XX.pth \
#     --video_dir ${DATASETS_ROOT}/test/rigid_body_motion/ \
#     --save_dir ${MOPE_ROOT}/features/rigid_body_motion/ \
#     --use_mope --show_cls --gate_threshold 0.0
