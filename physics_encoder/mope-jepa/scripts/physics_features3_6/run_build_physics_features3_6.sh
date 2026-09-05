#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT_DIR}"

DATASET_ROOT="${DATASET_ROOT:-/data/public_data/xzs_data/lingbotva-post-training-dataset_s2}"
CKPT="${CKPT:-${ROOT_DIR}/output/mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth}"
EVENT_LABEL_DIR="${EVENT_LABEL_DIR:-/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical}"
EVENT_LABEL_JSON="${EVENT_LABEL_JSON:-}"
CAMERA_KEY="${CAMERA_KEY:-observation.images.cam_high}"
OUTPUT_DIRNAME="${OUTPUT_DIRNAME:-physics_features3.6}"
PYTHON="${PYTHON:-/data1/miniconda3/envs/worldmodel_mopeva/bin/python}"
FFMPEG_BIN="${FFMPEG_BIN:-/data/worldmodel_xzs/ffmpeg-7.0.2-amd64-static/ffmpeg}"
DECODE_CACHE_DIR="${DECODE_CACHE_DIR:-/data/worldmodel_xzs/phywam_v3/tmp/physics_features3.6_decode_cache}"

GPUS="${GPUS:-${GPU:-}}"
MAX_JOBS="${MAX_JOBS:-0}"
DETACH="${DETACH:-1}"
OVERWRITE="${OVERWRITE:-0}"
LOG_DIR="${LOG_DIR:-/data/worldmodel_xzs/phywam_v3/logs/physics_features3.6}"

if [[ -z "${EVENT_LABEL_JSON}" ]]; then
  mapfile -t candidates < <(
    find "${EVENT_LABEL_DIR}" -maxdepth 1 -type f -name '*.json' | sort
  )
  if [[ ${#candidates[@]} -ne 1 ]]; then
    echo "[ERROR] expected one event JSON in ${EVENT_LABEL_DIR}, found ${#candidates[@]}" >&2
    printf '[ERROR] candidate: %s\n' "${candidates[@]}" >&2
    exit 1
  fi
  EVENT_LABEL_JSON="${candidates[0]}"
fi

for path in "${PYTHON}" "${CKPT}" "${EVENT_LABEL_JSON}" "${DATASET_ROOT}" "${FFMPEG_BIN}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[ERROR] required path not found: ${path}" >&2
    exit 1
  fi
done

SCRIPT_PATH="$(readlink -f "$0")"
if [[ "${DETACH}" == "1" && "${_PHYS36_DETACHED_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  run_ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/build_${run_ts}.log"
  pid_file="${LOG_DIR}/build_${run_ts}.pid"
  _PHYS36_DETACHED_CHILD=1 DETACH=0 nohup bash "${SCRIPT_PATH}" "$@" \
    >"${log_file}" 2>&1 < /dev/null &
  echo "$!" >"${pid_file}"
  echo "[INFO] started pid=$(<"${pid_file}") log=${log_file}"
  exit 0
fi

run_one() {
  local task_root="$1"
  local device="$2"
  shift 2
  local extra_args=()
  if [[ "${OVERWRITE}" == "1" ]]; then
    extra_args+=(--overwrite)
  fi
  PYTHONUNBUFFERED=1 "${PYTHON}" -u tools/build_phywam_phys_tokens_v36.py \
    --dataset-root "${task_root}" \
    --mope-repo "${ROOT_DIR}" \
    --ckpt "${CKPT}" \
    --camera-key "${CAMERA_KEY}" \
    --output-dirname "${OUTPUT_DIRNAME}" \
    --segment-mode event \
    --event-label-json "${EVENT_LABEL_JSON}" \
    --event-min-segment-frames "${EVENT_MIN_SEGMENT_FRAMES:-4}" \
    --event-max-segment-frames "${EVENT_MAX_SEGMENT_FRAMES:-0}" \
    --missing-event-policy "${MISSING_EVENT_POLICY:-error}" \
    --num-frames "${NUM_FRAMES:-16}" \
    --input-size "${INPUT_SIZE:-224}" \
    --device "${device}" \
    --ffmpeg-bin "${FFMPEG_BIN}" \
    --decode-cache-dir "${DECODE_CACHE_DIR}" \
    --num-physics-experts 17 \
    --num-general-experts 10 \
    --num-shared-experts 4 \
    --candidate-k 5 \
    --gate-threshold 0.0 \
    "${extra_args[@]}" \
    "$@"
}

if [[ -n "${GPUS}" ]]; then
  IFS=',' read -r -a gpu_array <<<"${GPUS}"
  if [[ "${MAX_JOBS}" == "0" ]]; then
    MAX_JOBS="${#gpu_array[@]}"
  fi
  mapfile -t task_roots < <(
    {
      if [[ -f "${DATASET_ROOT}/meta/episodes.jsonl" && -d "${DATASET_ROOT}/videos" ]]; then
        echo "${DATASET_ROOT}"
      fi
      find "${DATASET_ROOT}" -type f -path '*/meta/episodes.jsonl' \
        | sed 's#/meta/episodes.jsonl$##'
    } | sort -u
  )
  if [[ ${#task_roots[@]} -eq 0 ]]; then
    echo "[ERROR] no LeRobot task roots found under ${DATASET_ROOT}" >&2
    exit 1
  fi
  index=0
  for task_root in "${task_roots[@]}"; do
    while [[ "$(jobs -rp | wc -l)" -ge "${MAX_JOBS}" ]]; do
      wait -n
    done
    gpu="${gpu_array[$((index % ${#gpu_array[@]}))]}"
    echo "[INFO] task=${task_root##*/} gpu=${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" run_one "${task_root}" cuda "$@" &
    index=$((index + 1))
  done
  wait
else
  run_one "${DATASET_ROOT}" "${DEVICE:-cuda}" "$@"
fi
