#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

PYTHON="${PYTHON:-/data1/miniconda3/envs/worldmodel_mopeva/bin/python}"
CKPT="${CKPT:-${ROOT_DIR}/output/mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth}"
DATASETS_ROOT="${DATASETS_ROOT:-/data/public_data/xzs_data/lingbotva-post-training-dataset/robotwin-clean-and-aug-lerobot}"
EVENT_LABEL_PATH="${EVENT_LABEL_PATH:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/event_labels_full50_qwen_v39_task_canonical_test_c1a2/robotwin_full50_c1_a2_event_segments_task_canonical_mope_jepa.json}"
PHYSICS_SOFT_PATH="${PHYSICS_SOFT_PATH:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/physics_labels_full50_qwen_v35_test_c1a2/robotwin_full50_c1_a2_physics_scores_qwen.json}"
OUTPUT_JSON="${OUTPUT_JSON:-${ROOT_DIR}/validation/robotwin_v39_c1a2_metrics—100.json}"
GPU="${GPU:-0}"

CUDA_VISIBLE_DEVICES="${GPU}" "${PYTHON}" tools/eval_robotwin_v39.py \
  --ckpt "${CKPT}" \
  --datasets-root "${DATASETS_ROOT}" \
  --event-label-path "${EVENT_LABEL_PATH}" \
  --physics-soft-path "${PHYSICS_SOFT_PATH}" \
  --output-json "${OUTPUT_JSON}" \
  --batch-size "${BATCH_SIZE:-16}" \
  --num-workers "${NUM_WORKERS:-4}" \
  "$@"
