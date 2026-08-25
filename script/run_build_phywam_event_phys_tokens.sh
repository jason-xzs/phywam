#!/usr/bin/bash
set -euo pipefail
set -x

REPO_ROOT=${REPO_ROOT:-"/data/worldmodel_xzs/phywam_v3"}
DATASET_ROOT=${DATASET_ROOT:-"/data/public_data/xzs_data/lingbotva-post-training-dataset_s4"}
MOPE_REPO=${MOPE_REPO:-"${REPO_ROOT}/mope-jepa"}
EVENT_LABEL_JSON=${EVENT_LABEL_JSON:-"${MOPE_REPO}/datasets/event_labels_s4_qwen/robotwin_s4_event_duration_labels_qwen.json"}
OUTPUT_DIRNAME=${OUTPUT_DIRNAME:-"event_physics_features3.4"}
SEGMENT_MODE=${SEGMENT_MODE:-"event"}
MISSING_EVENT_POLICY=${MISSING_EVENT_POLICY:-"error"}
TEMP_ROOT=${TEMP_ROOT:-"${REPO_ROOT}/tmp/phywam_event_phys_tokens"}
LOG_DIR=${LOG_DIR:-"${REPO_ROOT}/logs"}

cd "${REPO_ROOT}"

DATASET_ROOT="${DATASET_ROOT}" \
MOPE_REPO="${MOPE_REPO}" \
EVENT_LABEL_JSON="${EVENT_LABEL_JSON}" \
OUTPUT_DIRNAME="${OUTPUT_DIRNAME}" \
SEGMENT_MODE="${SEGMENT_MODE}" \
MISSING_EVENT_POLICY="${MISSING_EVENT_POLICY}" \
TEMP_ROOT="${TEMP_ROOT}" \
LOG_DIR="${LOG_DIR}" \
  bash script/run_build_phywam_phys_tokens.sh "$@"

