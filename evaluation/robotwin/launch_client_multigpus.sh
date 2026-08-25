#!/bin/bash
set -euo pipefail

PYTHON=${PYTHON:-/data1/miniconda3/envs/worldmodel_phyva/bin/python}
ROBOWIN_ROOT=${ROBOWIN_ROOT:-/data/worldmodel_xzs/RoboTwin}
CUROBO_SRC=${CUROBO_SRC:-${ROBOWIN_ROOT}/envs/curobo/src}
PYTORCH_LIB=$("${PYTHON}" - <<'PY'
import os
import torch

print(os.path.join(os.path.dirname(torch.__file__), "lib"))
PY
)
export PYTHONPATH="${CUROBO_SRC}:${PYTHONPATH:-}"
export LD_LIBRARY_PATH="${PYTORCH_LIB}:/usr/local/cuda-12.8/lib64:/usr/lib64:/usr/lib:${LD_LIBRARY_PATH:-}"
trap '' HUP

PHYS_MEMORY_INFER_MODE=${PHYS_MEMORY_INFER_MODE:-phase}
PHYWAM_INFER_CHECKPOINT=${PHYWAM_INFER_CHECKPOINT:-/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/checkpoints/checkpoint_step_16000}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_phys36}
CHECKPOINT_TAG=${CHECKPOINT_TAG:-$(basename "$PHYWAM_INFER_CHECKPOINT")}
INFER_VARIANT=${INFER_VARIANT:-${PHYS_MEMORY_INFER_MODE}}
save_root=${SAVE_ROOT:-"/data/public_data/xzs_data/${EXPERIMENT_NAME}/inference/${CHECKPOINT_TAG}/${INFER_VARIANT}/client"}

# General parameters
policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=${SEED:-0}
test_num=${TEST_NUM:-100}
start_port=${START_PORT:-31556}
gpu_ids_str=${2:-${GPU_IDS:-}}
num_gpus=${3:-${NUM_GPUS:-}}
gpu_offset=${GPU_OFFSET:-0}
RESUME=${RESUME:-1}

task_list_id=${1:-${TASK_LIST_ID:-6}}

task_groups=(
  "put_object_cabinet place_bread_skillet place_burger_fries stack_bowls_three"
  "hanging_mug place_can_basket handover_block pick_diverse_bottles"
  "hanging_mug stack_bowls_three put_object_cabinet place_can_basket"
  "pick_diverse_bottles move_stapler_pad handover_block turn_switch"
  "hanging_mug stack_bowls_three put_object_cabinet place_bread_skillet handover_block"
  "hanging_mug stack_bowls_three put_object_cabinet place_can_basket pick_diverse_bottles handover_block"
  "pick_diverse_bottles"
  "hanging_mug stack_bowls_three put_object_cabinet place_can_basket pick_diverse_bottles place_bread_skillet handover_block place_burger_fries"
  "hanging_mug stack_bowls_three put_object_cabinet place_can_basket pick_diverse_bottles move_stapler_pad handover_block turn_switch"
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

if (( task_list_id < 0 || task_list_id >= ${#task_groups[@]} )); then
  echo "task_list_id out of range: $task_list_id (0..$(( ${#task_groups[@]} - 1 )))" >&2
  exit 1
fi

read -r -a task_names <<< "${task_groups[$task_list_id]}"

gpu_mode="offset"
gpu_ids=()
if [[ -n "$gpu_ids_str" ]]; then
  gpu_mode="manual"
  IFS=',' read -r -a gpu_ids <<< "$gpu_ids_str"
  if (( ${#gpu_ids[@]} == 0 )); then
    echo "GPU_IDS is set but empty: $gpu_ids_str" >&2
    exit 1
  fi

  for idx in "${!gpu_ids[@]}"; do
    gpu_ids[$idx]=$(echo "${gpu_ids[$idx]}" | xargs)
    if [[ ! "${gpu_ids[$idx]}" =~ ^[0-9]+$ ]]; then
      echo "Invalid GPU id in GPU_IDS: ${gpu_ids[$idx]}" >&2
      exit 1
    fi
  done

  if [[ -z "$num_gpus" ]]; then
    num_gpus=${#gpu_ids[@]}
  fi

  if (( num_gpus <= 0 || num_gpus > ${#gpu_ids[@]} )); then
    echo "NUM_GPUS must be in 1..${#gpu_ids[@]} when GPU_IDS is used, got: ${num_gpus}" >&2
    exit 1
  fi
else
  if [[ -z "$num_gpus" ]]; then
    num_gpus=8
  fi
fi

echo "task_list_id=$task_list_id"
echo "RESUME=$RESUME"
echo "PYTHON=$PYTHON"
echo "PYTORCH_LIB=$PYTORCH_LIB"
printf 'task_names (%d): %s\n' "${#task_names[@]}" "${task_names[*]}"

log_dir="./logs/cfgdrop_zeroinit_gate_freeze_phys36"
mkdir -p "$log_dir"
mkdir -p "$save_root"

"${PYTHON}" - <<'PY'
from curobo.curobolib import (
    geom_cu,
    kinematics_fused_cu,
    lbfgs_step_cu,
    line_search_cu,
    tensor_step_cu,
)

print("[Preflight] curobo CUDA extensions loaded")
PY

batch_time=$(date +%Y%m%d_%H%M%S)
pid_file="${log_dir}/client_pids_${batch_time}.txt"
> "$pid_file"

if [[ "$gpu_mode" == "manual" ]]; then
  echo -e "\033[32mLaunching ${#task_names[@]} tasks. Manual GPU_IDS=(${gpu_ids[*]}), using first ${num_gpus}, ports starting from ${start_port} incrementing.\033[0m"
else
  echo -e "\033[32mLaunching ${#task_names[@]} tasks. GPUs assigned by mod ${num_gpus} with GPU_OFFSET=${gpu_offset}, ports starting from ${start_port} incrementing.\033[0m"
fi

for i in "${!task_names[@]}"; do
    task_name="${task_names[$i]}"
    if [[ "$gpu_mode" == "manual" ]]; then
      gpu_id=${gpu_ids[$(( i % num_gpus ))]}
    else
      gpu_id=$(( gpu_offset + (i % num_gpus) ))
    fi
    port=$(( start_port + i ))

    export CUDA_VISIBLE_DEVICES=${gpu_id}

    log_file="${log_dir}/${task_name}_${batch_time}.log"

    echo -e "\033[33m[Task $i] Task: ${task_name}, GPU: ${gpu_id}, PORT: ${port}, Log: ${log_file}\033[0m"

    nohup env \
    PYTHONWARNINGS=ignore::UserWarning \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
    "${PYTHON}" -m evaluation.robotwin.eval_polict_client_openpi --config policy/$policy_name/deploy_policy.yml \
      --resume ${RESUME} \
        --overrides \
        --task_name ${task_name} \
        --task_config ${task_config} \
        --train_config_name ${train_config_name} \
        --model_name ${model_name} \
        --ckpt_setting ${model_name} \
        --seed ${seed} \
        --policy_name ${policy_name} \
        --save_root ${save_root} \
        --video_guidance_scale 5 \
        --action_guidance_scale 1 \
        --test_num ${test_num} \
        --port ${port} > "$log_file" 2>&1 < /dev/null &

    pid=$!
    echo "${pid}" | tee -a "$pid_file"
done

echo -e "\033[32mAll tasks launched. PIDs saved to ${pid_file}\033[0m"
echo -e "\033[36mTo terminate all processes, run: kill \$(cat ${pid_file})\033[0m"
