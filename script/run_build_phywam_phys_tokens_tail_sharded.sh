#!/usr/bin/bash
set -euo pipefail
set -x

DATASET_ROOT=${DATASET_ROOT:-"/data/public_data/xzs_data/lingbotva-post-training-dataset_s"}
MOPE_REPO=${MOPE_REPO:-"/data/worldmodel_xzs/phywam_v3/mope-jepa"}
CKPT=${CKPT:-"/data/public_data/xzs_data/mope_jepa_robotwin_event_ft_from_freeze150/checkpoint-100.pth"}
CAMERA_KEY=${CAMERA_KEY:-"observation.images.cam_high"}
OUTPUT_DIRNAME=${OUTPUT_DIRNAME:-"physics_features3.4"}
BLOCK_SIZE=${BLOCK_SIZE:-"16"}
NUM_FRAMES=${NUM_FRAMES:-"16"}
SAMPLING_RATE=${SAMPLING_RATE:-"4"}
INPUT_SIZE=${INPUT_SIZE:-"224"}
DEVICE=${DEVICE:-"cuda"}
FFMPEG_BIN=${FFMPEG_BIN:-"/data/worldmodel_xzs/ffmpeg-7.0.2-amd64-static/ffmpeg"}
TEMP_ROOT=${TEMP_ROOT:-"/data/worldmodel_xzs/phywam_v3/tmp/phywam_phys_tokens_tail"}
PYTHON=${PYTHON:-"python"}

GPUS=${GPUS:-${GPU:-"0,1,2,3,4,5,6,7"}}
SHARDS_PER_TASK=${SHARDS_PER_TASK:-"8"}
MAX_JOBS=${MAX_JOBS:-"8"}
DETACH=${DETACH:-"1"}
LOG_DIR=${LOG_DIR:-"/data/worldmodel_xzs/phywam_v3/logs"}
TASKS=${TASKS:-"lerobot_robotwin_eef_aug_500_s/put_object_cabinet,lerobot_robotwin_eef_aug_500_s/stack_bowls_three-aloha-agilex_randomized_500-1000"}

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

if [[ "${DETACH}" == "1" && "${_PHYWAM_DETACHED_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/build_phywam_phys_tokens_tail_sharded_${ts}.log"
  pid_file="${LOG_DIR}/build_phywam_phys_tokens_tail_sharded_${ts}.pid"

  echo "[INFO] launch detached tail sharded process"
  echo "[INFO] log: ${log_file}"

  _PHYWAM_DETACHED_CHILD=1 DETACH=0 nohup bash "${SCRIPT_PATH}" "$@" >"${log_file}" 2>&1 < /dev/null &
  pid=$!
  echo "${pid}" > "${pid_file}"

  echo "[INFO] pid: ${pid}"
  echo "[INFO] pid file: ${pid_file}"
  echo "[INFO] follow log: tail -f ${log_file}"
  exit 0
fi

cd "${REPO_ROOT}"
mkdir -p "${TEMP_ROOT}"

IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
IFS=',' read -r -a TASK_ARR <<< "${TASKS}"

if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
  echo "[ERR] no GPUs configured: ${GPUS}"
  exit 1
fi
if [[ ${#TASK_ARR[@]} -eq 0 ]]; then
  echo "[ERR] no tasks configured: ${TASKS}"
  exit 1
fi
if [[ "${SHARDS_PER_TASK}" -lt 1 ]]; then
  echo "[ERR] SHARDS_PER_TASK must be >= 1"
  exit 1
fi
if [[ "${MAX_JOBS}" -lt 1 ]]; then
  echo "[ERR] MAX_JOBS must be >= 1"
  exit 1
fi

run_one() {
  local task_root="$1"
  local gpu="$2"
  local shard_index="$3"
  shift 3

  echo "[INFO] launch task=${task_root##*/} shard=${shard_index}/${SHARDS_PER_TASK} on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" PYTHONUNBUFFERED=1 "${PYTHON}" -u script/build_phywam_phys_tokens.py \
    --dataset-root "${task_root}" \
    --mope-repo "${MOPE_REPO}" \
    --ckpt "${CKPT}" \
    --camera-key "${CAMERA_KEY}" \
    --output-dirname "${OUTPUT_DIRNAME}" \
    --block-size "${BLOCK_SIZE}" \
    --num-frames "${NUM_FRAMES}" \
    --sampling-rate "${SAMPLING_RATE}" \
    --input-size "${INPUT_SIZE}" \
    --device "${DEVICE}" \
    --ffmpeg-bin "${FFMPEG_BIN}" \
    --temp-root "${TEMP_ROOT}" \
    --episode-num-shards "${SHARDS_PER_TASK}" \
    --episode-shard-index "${shard_index}" \
    "$@"
}

i=0
for shard_index in $(seq 0 $((SHARDS_PER_TASK - 1))); do
  for task_rel in "${TASK_ARR[@]}"; do
    task_root="${DATASET_ROOT}/${task_rel}"
    if [[ ! -f "${task_root}/meta/episodes.jsonl" || ! -d "${task_root}/videos" ]]; then
      echo "[ERR] invalid task root: ${task_root}"
      exit 1
    fi

    while [[ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]]; do
      wait -n
    done

    gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
    run_one "${task_root}" "${gpu}" "${shard_index}" "$@" &
    i=$((i + 1))
  done
done

wait
echo "[SUMMARY] tail sharded jobs complete"
