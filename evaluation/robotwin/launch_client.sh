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

task_groups=(
  "turn_switch hanging_mug stack_bowls_three put_object_cabinet place_can_basket pick_diverse_bottles move_stapler_pad handover_block"
  "stack_bowls_three handover_block hanging_mug scan_object lift_pot put_object_cabinet stack_blocks_three place_shoe"
  "adjust_bottle place_mouse_pad dump_bin_bigbin move_pillbottle_pad pick_dual_bottles shake_bottle place_fan turn_switch"
  "shake_bottle_horizontally place_container_plate rotate_qrcode place_object_stand put_bottles_dustbin move_stapler_pad place_burger_fries place_bread_basket"
  "pick_diverse_bottles open_microwave beat_block_hammer press_stapler click_bell move_playingcard_away open_laptop move_can_pot"
  "stack_bowls_two place_a2b_right stamp_seal place_object_basket handover_mic place_bread_skillet stack_blocks_two place_cans_plasticbox"
  "click_alarmclock blocks_ranking_size place_phone_stand place_can_basket place_object_scale place_a2b_left grab_roller place_dual_shoes"
  "place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb place_empty_cup blocks_ranking_rgb"
)

PHYS_MEMORY_INFER_MODE=${PHYS_MEMORY_INFER_MODE:-phase}
PHYWAM_INFER_CHECKPOINT=${PHYWAM_INFER_CHECKPOINT:-/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000/checkpoints/checkpoint_step_16000}
EXPERIMENT_NAME=${EXPERIMENT_NAME:-phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate}
CHECKPOINT_TAG=${CHECKPOINT_TAG:-$(basename "$PHYWAM_INFER_CHECKPOINT")}
INFER_VARIANT=${INFER_VARIANT:-${PHYS_MEMORY_INFER_MODE}}
save_root=${1:-"/data/public_data/xzs_data/${EXPERIMENT_NAME}/inference/${CHECKPOINT_TAG}/${INFER_VARIANT}/client"}
task_name=${2:-"handover_block"}

policy_name=ACT
task_config=demo_clean
train_config_name=0
model_name=0
seed=0
PORT=${PORT:-29056}
WAIT_FOR_CHILDREN=${WAIT_FOR_CHILDREN:-0}
RESUME=${RESUME:-0}

mkdir -p "${save_root}"
mkdir -p "./logs"

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
log_file="./logs/${task_name}_${batch_time}.log"

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
    --test_num 100 \
    --port ${PORT} > "$log_file" 2>&1 < /dev/null &

pid=$!
echo "$pid" > pids.txt
echo "Launched task ${task_name}, PID=${pid}, log=${log_file}"

if [ "$WAIT_FOR_CHILDREN" = "1" ]; then
  wait "$pid"
fi
