#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-benchmarks/public_videoqa}"
MODEL_NAME_OR_PATH="${MODEL_NAME_OR_PATH:-Qwen/Qwen3-VL-8B-Instruct}"
MAX_SAMPLES="${MAX_SAMPLES:-0}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-128}"

echo "[INFO] output root: ${OUTPUT_ROOT}"
echo "[INFO] model: ${MODEL_NAME_OR_PATH}"
echo "[INFO] max samples per benchmark: ${MAX_SAMPLES}"

python dataset/run_public_videoqa_benchmarks.py \
  --output_root "${OUTPUT_ROOT}" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --max_samples "${MAX_SAMPLES}" \
  --max_new_tokens "${MAX_NEW_TOKENS}" \
  "$@"

echo "[DONE] Leaderboard: ${OUTPUT_ROOT}/leaderboard.csv"
