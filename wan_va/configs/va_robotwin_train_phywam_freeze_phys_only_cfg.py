# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_train_phywam_event_cfg import va_robotwin_train_phywam_event_cfg


va_robotwin_train_phywam_freeze_phys_only_cfg = EasyDict(
    __name__='Config: VA robotwin train S2 PhyWAM freeze-backbone'
)
va_robotwin_train_phywam_freeze_phys_only_cfg.update(
    va_robotwin_train_phywam_event_cfg
)

va_robotwin_train_phywam_freeze_phys_only_cfg.resume_from = None
va_robotwin_train_phywam_freeze_phys_only_cfg.resume_optimizer_state = False
va_robotwin_train_phywam_freeze_phys_only_cfg.wan22_pretrained_model_name_or_path = (
    '/data/worldmodel_xzs/lingbot-va/train_out/'
    'lingbot_va_s2_base_8gpu/checkpoints/checkpoint_step_16000'
)

va_robotwin_train_phywam_freeze_phys_only_cfg.save_root = (
    '/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/'
    'phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000'
)
va_robotwin_train_phywam_freeze_phys_only_cfg.wandb_run_name = (
    'phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36'
)

va_robotwin_train_phywam_freeze_phys_only_cfg.freeze_backbone = True
va_robotwin_train_phywam_freeze_phys_only_cfg.enable_wandb = True
va_robotwin_train_phywam_freeze_phys_only_cfg.attn_mode = 'flex'
va_robotwin_train_phywam_freeze_phys_only_cfg.phys_cross_start_layer = 22
va_robotwin_train_phywam_freeze_phys_only_cfg.phys_monitor_interval = 20
va_robotwin_train_phywam_freeze_phys_only_cfg.phys_monitor_jsonl = (
    '/data/worldmodel_xzs/phywam_v3_adapter_exp/logs/'
    'phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000/'
    'phys_train_stats.jsonl'
)

va_robotwin_train_phywam_freeze_phys_only_cfg.batch_size = 1
va_robotwin_train_phywam_freeze_phys_only_cfg.gradient_accumulation_steps = 2
va_robotwin_train_phywam_freeze_phys_only_cfg.learning_rate = 1e-5
