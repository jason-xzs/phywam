# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import torch
from easydict import EasyDict

va_shared_cfg = EasyDict()

va_shared_cfg.host = '0.0.0.0'
va_shared_cfg.port = 29536

va_shared_cfg.param_dtype = torch.bfloat16
va_shared_cfg.save_root = './train_out'
va_shared_cfg.resume_optimizer_state = False

va_shared_cfg.patch_size = (1, 2, 2)

va_shared_cfg.enable_offload = True
va_shared_cfg.dataset_init_worker = 32
va_shared_cfg.dataloader_persistent_workers = True
va_shared_cfg.dataloader_prefetch_factor = 1
va_shared_cfg.dataloader_timeout = 0
va_shared_cfg.dataloader_sharing_strategy = 'file_system'
va_shared_cfg.use_phys_memory = False
va_shared_cfg.phys_memory_path = None
va_shared_cfg.phys_memory_dirname = 'physics_features3.4'
va_shared_cfg.phys_memory_dim = 768
va_shared_cfg.phys_memory_block_size = 16
va_shared_cfg.phys_memory_align_mode = 'per_latent_frame'
va_shared_cfg.phys_memory_max_tokens = 8
