# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_train_phywam_small_cfg import va_robotwin_train_phywam_small_cfg


va_robotwin_train_phywam_event_cfg = EasyDict(__name__='Config: VA robotwin train S2 phase-cross PhyWAM')
va_robotwin_train_phywam_event_cfg.update(va_robotwin_train_phywam_small_cfg)

va_robotwin_train_phywam_event_cfg.dataset_path = '/data/public_data/xzs_data/lingbotva-post-training-dataset_s2'
va_robotwin_train_phywam_event_cfg.empty_emb_path = '/data/public_data/xzs_data/lingbotva-post-training-dataset_s2/empty_emb.pt'
va_robotwin_train_phywam_event_cfg.use_phys_memory = True
va_robotwin_train_phywam_event_cfg.phys_memory_dirname = 'physics_features3.6'
va_robotwin_train_phywam_event_cfg.phys_memory_path = None
va_robotwin_train_phywam_event_cfg.phys_memory_dim = 768
va_robotwin_train_phywam_event_cfg.phys_memory_block_size = 16
va_robotwin_train_phywam_event_cfg.phys_memory_align_mode = 'phase_tokens'
va_robotwin_train_phywam_event_cfg.phys_memory_max_tokens = 8
va_robotwin_train_phywam_event_cfg.phys_cross_start_layer = 22

va_robotwin_train_phywam_event_cfg.save_root = (
    '/data/worldmodel_xzs/phywam_v3_adapter_exp/train_out/'
    'phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate'
)
va_robotwin_train_phywam_event_cfg.wandb_run_name = (
    'phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate'
)
