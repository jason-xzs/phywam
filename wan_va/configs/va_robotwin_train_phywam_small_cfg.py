# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_train_cfg import va_robotwin_train_cfg


va_robotwin_train_phywam_small_cfg = EasyDict(__name__='Config: VA robotwin train small PhyWAM')
va_robotwin_train_phywam_small_cfg.update(va_robotwin_train_cfg)

va_robotwin_train_phywam_small_cfg.dataset_path = '/data/public_data/xzs_data/lingbotva-post-training-dataset_s4'
va_robotwin_train_phywam_small_cfg.empty_emb_path = '/data/public_data/xzs_data/lingbotva-post-training-dataset_s4/empty_emb.pt'
va_robotwin_train_phywam_small_cfg.use_phys_memory = True
va_robotwin_train_phywam_small_cfg.phys_memory_dirname = 'physics_features3.4'
va_robotwin_train_phywam_small_cfg.phys_memory_path = None
va_robotwin_train_phywam_small_cfg.phys_memory_dim = 768
va_robotwin_train_phywam_small_cfg.phys_memory_block_size = 16

va_robotwin_train_phywam_small_cfg.save_root = '/data/worldmodel_xzs/phywam_v3/train_out/phywam_s4_phys_v3.4'
va_robotwin_train_phywam_small_cfg.enable_wandb = True
va_robotwin_train_phywam_small_cfg.wandb_run_name = 'robotwin_phywam_s4_phys_v3.4'
