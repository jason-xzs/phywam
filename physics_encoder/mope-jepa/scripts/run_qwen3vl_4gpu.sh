#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"

mkdir -p score_labels datasets

INPUT_JSON="datasets/wisa_7k.json"
VIDEO_ROOT="datasets"
# Use Hub repo id (uses ~/.cache/huggingface/hub/models--Qwen--Qwen3-VL-8B-Instruct/ automatically).
MODEL_NAME_OR_PATH="Qwen/Qwen3-VL-8B-Instruct"
FINAL_JSON="datasets/wisa_7k_qwen3vl8b_scores_all.json"

echo "[INFO] Start 4-GPU scoring..."

CUDA_VISIBLE_DEVICES=1 python -u dataset/batch_score_physics_labels.py \
  --input_json "${INPUT_JSON}" \
  --video_root "${VIDEO_ROOT}" \
  --output_json "datasets/wisa_7k_qwen3vl8b_scores_rank0.json" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --rank 0 --world_size 4 \
  > "score_labels/gpu1_rank0.log" 2>&1 &

CUDA_VISIBLE_DEVICES=2 python -u dataset/batch_score_physics_labels.py \
  --input_json "${INPUT_JSON}" \
  --video_root "${VIDEO_ROOT}" \
  --output_json "datasets/wisa_7k_qwen3vl8b_scores_rank1.json" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --rank 1 --world_size 4 \
  > "score_labels/gpu2_rank1.log" 2>&1 &

CUDA_VISIBLE_DEVICES=3 python -u dataset/batch_score_physics_labels.py \
  --input_json "${INPUT_JSON}" \
  --video_root "${VIDEO_ROOT}" \
  --output_json "datasets/wisa_7k_qwen3vl8b_scores_rank2.json" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --rank 2 --world_size 4 \
  > "score_labels/gpu3_rank2.log" 2>&1 &

CUDA_VISIBLE_DEVICES=4 python -u dataset/batch_score_physics_labels.py \
  --input_json "${INPUT_JSON}" \
  --video_root "${VIDEO_ROOT}" \
  --output_json "datasets/wisa_7k_qwen3vl8b_scores_rank3.json" \
  --model_name_or_path "${MODEL_NAME_OR_PATH}" \
  --rank 3 --world_size 4 \
  > "score_labels/gpu4_rank3.log" 2>&1 &

wait
echo "[INFO] 4 workers finished. Merging outputs..."

python - <<'PY'
import json

parts = [
    "datasets/wisa_7k_qwen3vl8b_scores_rank0.json",
    "datasets/wisa_7k_qwen3vl8b_scores_rank1.json",
    "datasets/wisa_7k_qwen3vl8b_scores_rank2.json",
    "datasets/wisa_7k_qwen3vl8b_scores_rank3.json",
]
merged = []
for p in parts:
    with open(p, "r", encoding="utf-8") as f:
        merged.extend(json.load(f))

with open("datasets/wisa_7k_qwen3vl8b_scores_all.json", "w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"[INFO] merged samples: {len(merged)}")
print("[INFO] final file: datasets/wisa_7k_qwen3vl8b_scores_all.json")
PY

echo "[INFO] Done. Final result: ${FINAL_JSON}"
