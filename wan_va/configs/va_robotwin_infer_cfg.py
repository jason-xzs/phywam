# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import os

from easydict import EasyDict

from .shared_config import va_shared_cfg


va_robotwin_infer_cfg = EasyDict(__name__='Config: VA robotwin infer PhyWAM')
va_robotwin_infer_cfg.update(va_shared_cfg)

va_robotwin_infer_cfg.wan22_pretrained_model_name_or_path = os.environ.get(
    'PHYWAM_INFER_CHECKPOINT',
    '/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/'
    'phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/'
    'checkpoints/checkpoint_step_16000',
)

va_robotwin_infer_cfg.attn_window = 72
va_robotwin_infer_cfg.frame_chunk_size = 2
va_robotwin_infer_cfg.env_type = 'robotwin_tshape'

va_robotwin_infer_cfg.height = 256
va_robotwin_infer_cfg.width = 320
va_robotwin_infer_cfg.action_dim = 30
va_robotwin_infer_cfg.action_per_frame = 16
va_robotwin_infer_cfg.obs_cam_keys = [
    'observation.images.cam_high', 'observation.images.cam_left_wrist',
    'observation.images.cam_right_wrist'
]
va_robotwin_infer_cfg.guidance_scale = 5
va_robotwin_infer_cfg.action_guidance_scale = 1

va_robotwin_infer_cfg.num_inference_steps = 25
va_robotwin_infer_cfg.video_exec_step = -1
va_robotwin_infer_cfg.action_num_inference_steps = 50

va_robotwin_infer_cfg.snr_shift = 5.0
va_robotwin_infer_cfg.action_snr_shift = 1.0

va_robotwin_infer_cfg.used_action_channel_ids = list(range(0, 7)) + list(
    range(28, 29)) + list(range(7, 14)) + list(range(29, 30))
inverse_used_action_channel_ids = [
    len(va_robotwin_infer_cfg.used_action_channel_ids)
] * va_robotwin_infer_cfg.action_dim
for i, j in enumerate(va_robotwin_infer_cfg.used_action_channel_ids):
    inverse_used_action_channel_ids[j] = i
va_robotwin_infer_cfg.inverse_used_action_channel_ids = inverse_used_action_channel_ids

va_robotwin_infer_cfg.action_norm_method = 'quantiles'
va_robotwin_infer_cfg.norm_stat = {
    "q01": [
        -0.06172713458538055, -3.6716461181640625e-05, -0.08783501386642456,
        -1, -1, -1, -1, -0.3547105032205582, -1.3113021850585938e-06,
        -0.11975435614585876, -1, -1, -1, -1
    ] + [0.] * 16,
    "q99": [
        0.3462600058317184, 0.39966784834861746, 0.14745532035827624, 1, 1, 1,
        1, 0.034201726913452024, 0.39142737388610793, 0.1792279863357542, 1, 1,
        1, 1
    ] + [0.] * 14 + [1.0, 1.0],
}

va_robotwin_infer_cfg.use_phys_memory = True
va_robotwin_infer_cfg.phys_memory_dim = 768
va_robotwin_infer_cfg.phys_memory_block_size = 16
va_robotwin_infer_cfg.phys_memory_align_mode = 'phase_tokens'
va_robotwin_infer_cfg.phys_memory_max_tokens = 8
va_robotwin_infer_cfg.phys_cross_start_layer = 22
va_robotwin_infer_cfg.infer_phys_memory_npy = None
va_robotwin_infer_cfg.phys_memory_infer_mode = 'phase'
va_robotwin_infer_cfg.phys_use_phase_confidence_gate = True
va_robotwin_infer_cfg.phys_default_task_gate = 1.0
va_robotwin_infer_cfg.phys_task_gates = {}
va_robotwin_infer_cfg.mope_repo = '/data/worldmodel_xzs/phywam_v3/mope-jepa'
va_robotwin_infer_cfg.mope_ckpt = (
    '/data/worldmodel_xzs/phywam_v3/mope-jepa/output/'
    'mope_jepa_v39_task_canonical_event_physics_c12a24_freeze150/'
    'checkpoint-400.pth'
)
va_robotwin_infer_cfg.phys_event_label_path = (
    '/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/'
    'robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical/'
    'robotwin_s2_8tasks_c50_a500_event_segments_task_canonical_mope_jepa.json'
)
va_robotwin_infer_cfg.phys_num_event_classes = 28
va_robotwin_infer_cfg.phys_memory_camera_key = 'observation.images.cam_high'
# MoPE always receives 16 frames. The client supplies one key observation every
# four executed actions, so retain up to 16 real key observations per phase.
va_robotwin_infer_cfg.phys_memory_num_frames = 16
va_robotwin_infer_cfg.phys_memory_input_size = 224
va_robotwin_infer_cfg.phys_memory_obs_frames_per_block = 4
va_robotwin_infer_cfg.phys_phase_buffer_frames = 16
va_robotwin_infer_cfg.phys_phase_switch_patience = 1
va_robotwin_infer_cfg.phys_event_no_event_label_id = 0
va_robotwin_infer_cfg.phys_event_threshold = 0.5
va_robotwin_infer_cfg.phys_event_window_blocks = 0
va_robotwin_infer_cfg.phys_event_detector = 'mope'
va_robotwin_infer_cfg.phys_gripper_event_threshold = 0.2
va_robotwin_infer_cfg.phys_image_event_threshold = 0.05
