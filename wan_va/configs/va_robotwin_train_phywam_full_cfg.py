# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from easydict import EasyDict

from .va_robotwin_train_phywam_small_cfg import va_robotwin_train_phywam_small_cfg


va_robotwin_train_phywam_full_cfg = EasyDict(__name__='Config: VA robotwin train full PhyWAM')
va_robotwin_train_phywam_full_cfg.update(va_robotwin_train_phywam_small_cfg)

va_robotwin_train_phywam_full_cfg.dataset_path = '/data/public_data/xzs_data/lingbotva-post-training-dataset'
va_robotwin_train_phywam_full_cfg.empty_emb_path = (
    '/data/public_data/xzs_data/lingbotva-post-training-dataset/empty_emb.pt'
)
va_robotwin_train_phywam_full_cfg.save_root = (
    '/data/worldmodel_xzs/phywam_v3/train_out/phywam_full_phys_v3.4_8gpu'
)

va_robotwin_train_phywam_full_cfg.enable_wandb = True
va_robotwin_train_phywam_full_cfg.wandb_run_name = 'robotwin_phywam_full_phys_v3.4_8gpu'

va_robotwin_train_phywam_full_cfg.use_phys_memory = True
va_robotwin_train_phywam_full_cfg.phys_memory_dirname = 'physics_features3.4'
va_robotwin_train_phywam_full_cfg.phys_memory_path = None
va_robotwin_train_phywam_full_cfg.phys_memory_dim = 768
va_robotwin_train_phywam_full_cfg.phys_memory_block_size = 16

# Effective global batch = batch_size * world_size * gradient_accumulation_steps.
# With 8 GPUs and batch_size=1, accumulation=8 matches a 64-sample update.
va_robotwin_train_phywam_full_cfg.batch_size = 1
va_robotwin_train_phywam_full_cfg.gradient_accumulation_steps = 8
va_robotwin_train_phywam_full_cfg.num_steps = 50000
