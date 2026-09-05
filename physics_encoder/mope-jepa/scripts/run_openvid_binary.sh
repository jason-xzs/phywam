#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT_DIR}"
mkdir -p score_labels datasets

INPUT="datasets/openvid_clean.txt"
MODEL="Qwen/Qwen3-VL-8B-Instruct"

echo "[INFO] 4-GPU OpenVid binary classify (cards 4-7)..."
for i in 0 1 2 3; do
  CARD=$((4+i))
  CUDA_VISIBLE_DEVICES=${CARD} python -u dataset/classify_openvid_binary.py \
    --input_list "${INPUT}" \
    --output_json "datasets/openvid_binary_rank${i}.json" \
    --model_name_or_path "${MODEL}" \
    --rank ${i} --world_size 4 \
    > "score_labels/openvid_binary_rank${i}.log" 2>&1 &
done
wait
echo "[INFO] merging..."
python - <<'PY'
import json
merged=[]
for i in range(4):
    merged.extend(json.load(open(f"datasets/openvid_binary_rank{i}.json")))
json.dump(merged, open("datasets/openvid_binary_all.json","w"), ensure_ascii=False, indent=2)
g=sum(1 for x in merged if x.get("category")=="general")
p=sum(1 for x in merged if x.get("category")=="physics")
print(f"total={len(merged)} general={g} physics={p}")
PY
echo "[INFO] Done -> datasets/openvid_binary_all.json"
