#!/usr/bin/bash
set -euo pipefail
set -x

DATASET_ROOT=${DATASET_ROOT:-"/data/public_data/xzs_data/lingbotva-post-training-dataset_s2"}
MOPE_REPO=${MOPE_REPO:-"/data/worldmodel_xzs/phywam_v3/mope-jepa"}
CKPT=${CKPT:-"/data/worldmodel_xzs/phywam_v3/mope-jepa/output/mope_jepa_v39_task_canonical_event_physics_c12a24_freeze150/checkpoint-400.pth"}
CAMERA_KEY=${CAMERA_KEY:-"observation.images.cam_high"}
OUTPUT_DIRNAME=${OUTPUT_DIRNAME:-"physics_features3.5"}
SEGMENT_MODE=${SEGMENT_MODE:-"event"}
EVENT_LABEL_DIR=${EVENT_LABEL_DIR:-"${MOPE_REPO}/datasets/robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical"}
EVENT_LABEL_JSON=${EVENT_LABEL_JSON:-""}
EVENT_MIN_SEGMENT_FRAMES=${EVENT_MIN_SEGMENT_FRAMES:-"4"}
EVENT_MAX_SEGMENT_FRAMES=${EVENT_MAX_SEGMENT_FRAMES:-"0"}
MISSING_EVENT_POLICY=${MISSING_EVENT_POLICY:-"error"}
BLOCK_SIZE=${BLOCK_SIZE:-"16"}
NUM_FRAMES=${NUM_FRAMES:-"16"}
SAMPLING_RATE=${SAMPLING_RATE:-"4"}
INPUT_SIZE=${INPUT_SIZE:-"224"}
DEVICE=${DEVICE:-"cuda"}
FFMPEG_BIN=${FFMPEG_BIN:-"/data/worldmodel_xzs/ffmpeg-7.0.2-amd64-static/ffmpeg"}
TEMP_ROOT=${TEMP_ROOT:-"/data/worldmodel_xzs/phywam_v3/tmp/phywam_phys_tokens"}
PYTHON=${PYTHON:-"python"}

GPUS=${GPUS:-${GPU:-""}}
MAX_JOBS=${MAX_JOBS:-"0"}
DETACH=${DETACH:-"1"}
LOG_DIR=${LOG_DIR:-"/data/worldmodel_xzs/phywam_v3/logs"}

if [[ -z "${EVENT_LABEL_JSON}" && "${SEGMENT_MODE}" == "event" ]]; then
  if [[ ! -d "${EVENT_LABEL_DIR}" ]]; then
    echo "[ERR] event label dir does not exist: ${EVENT_LABEL_DIR}"
    exit 1
  fi

  mapfile -t EVENT_LABEL_JSON_CANDIDATES < <(find "${EVENT_LABEL_DIR}" -maxdepth 1 -type f -name "*.json" | sort)
  if [[ ${#EVENT_LABEL_JSON_CANDIDATES[@]} -ne 1 ]]; then
    echo "[ERR] expected exactly one event label json in ${EVENT_LABEL_DIR}, found ${#EVENT_LABEL_JSON_CANDIDATES[@]}"
    printf '[ERR] candidate: %s\n' "${EVENT_LABEL_JSON_CANDIDATES[@]}"
    echo "[ERR] set EVENT_LABEL_JSON=/path/to/file.json explicitly if needed"
    exit 1
  fi
  EVENT_LABEL_JSON="${EVENT_LABEL_JSON_CANDIDATES[0]}"
fi

SCRIPT_PATH="$(readlink -f "$0")"
SCRIPT_DIR="$(dirname "${SCRIPT_PATH}")"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"

if [[ "${DETACH}" == "1" && "${_PHYWAM_DETACHED_CHILD:-0}" != "1" ]]; then
  mkdir -p "${LOG_DIR}"
  ts="$(date +%Y%m%d_%H%M%S)"
  log_file="${LOG_DIR}/build_phywam_phys_tokens_${ts}.log"
  pid_file="${LOG_DIR}/build_phywam_phys_tokens_${ts}.pid"

  echo "[INFO] launch detached process"
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

run_one() {
  local dataset_root="$1"
  local device="$2"
  shift 2
  PYTHONUNBUFFERED=1 "${PYTHON}" -u script/build_phywam_phys_tokens.py \
    --dataset-root "${dataset_root}" \
    --mope-repo "${MOPE_REPO}" \
    --ckpt "${CKPT}" \
    --camera-key "${CAMERA_KEY}" \
    --output-dirname "${OUTPUT_DIRNAME}" \
    --segment-mode "${SEGMENT_MODE}" \
    --event-label-json "${EVENT_LABEL_JSON}" \
    --event-min-segment-frames "${EVENT_MIN_SEGMENT_FRAMES}" \
    --event-max-segment-frames "${EVENT_MAX_SEGMENT_FRAMES}" \
    --missing-event-policy "${MISSING_EVENT_POLICY}" \
    --block-size "${BLOCK_SIZE}" \
    --num-frames "${NUM_FRAMES}" \
    --sampling-rate "${SAMPLING_RATE}" \
    --input-size "${INPUT_SIZE}" \
    --device "${device}" \
    --ffmpeg-bin "${FFMPEG_BIN}" \
    --temp-root "${TEMP_ROOT}" \
    "$@"
}

if [[ -n "${GPUS}" ]]; then
  IFS=',' read -r -a GPU_ARR <<< "${GPUS}"
  if [[ ${#GPU_ARR[@]} -eq 0 ]]; then
    echo "[ERR] GPUS is set but empty after parsing: ${GPUS}"
    exit 1
  fi

  if [[ "${MAX_JOBS}" == "0" ]]; then
    MAX_JOBS=${#GPU_ARR[@]}
  fi

  mapfile -t TASK_ROOTS < <(
    {
      if [[ -f "${DATASET_ROOT}/meta/episodes.jsonl" && -d "${DATASET_ROOT}/videos" ]]; then
        echo "${DATASET_ROOT}"
      fi
      find "${DATASET_ROOT}" -type f -path "*/meta/episodes.jsonl" -print | sed 's#/meta/episodes.jsonl$##'
    } | sort -u
  )
  if [[ ${#TASK_ROOTS[@]} -eq 0 ]]; then
    echo "[ERR] no task dirs found under ${DATASET_ROOT}"
    exit 1
  fi

  i=0
  launched=0
  for task_root in "${TASK_ROOTS[@]}"; do
    if [[ ! -f "${task_root}/meta/episodes.jsonl" || ! -d "${task_root}/videos" ]]; then
      continue
    fi

    while [[ $(jobs -rp | wc -l) -ge ${MAX_JOBS} ]]; do
      wait -n
    done

    gpu=${GPU_ARR[$((i % ${#GPU_ARR[@]}))]}
    echo "[INFO] launch task=${task_root##*/} on GPU ${gpu}"
    CUDA_VISIBLE_DEVICES="${gpu}" run_one "${task_root}" "cuda" "$@" &
    i=$((i + 1))
    launched=$((launched + 1))
  done

  if [[ ${launched} -eq 0 ]]; then
    echo "[ERR] found candidate dirs but none are valid task roots"
    exit 1
  fi

  wait
else
  run_one "${DATASET_ROOT}" "${DEVICE}" "$@"
fi
