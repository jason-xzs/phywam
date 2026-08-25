# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
from .va_franka_cfg import va_franka_cfg
from .va_robotwin_cfg import va_robotwin_cfg
from .va_robotwin_infer_cfg import va_robotwin_infer_cfg
from .va_franka_i2va import va_franka_i2va_cfg
from .va_robotwin_i2va import va_robotwin_i2va_cfg
from .va_robotwin_train_cfg import va_robotwin_train_cfg
from .va_robotwin_train_phywam_small_cfg import va_robotwin_train_phywam_small_cfg
from .va_robotwin_train_phywam_full_cfg import va_robotwin_train_phywam_full_cfg
from .va_robotwin_train_phywam_event_cfg import va_robotwin_train_phywam_event_cfg
from .va_robotwin_train_phywam_freeze_phys_only_cfg import va_robotwin_train_phywam_freeze_phys_only_cfg
from .va_demo_train_cfg import va_demo_train_cfg
from .va_demo_cfg import va_demo_cfg
from .va_demo_i2va import va_demo_i2va_cfg

VA_CONFIGS = {
    'robotwin': va_robotwin_cfg,
    'robotwin_infer': va_robotwin_infer_cfg,
    'franka': va_franka_cfg,
    'robotwin_i2av': va_robotwin_i2va_cfg,
    'franka_i2av': va_franka_i2va_cfg,
    'robotwin_train': va_robotwin_train_cfg,
    'robotwin_train_phywam_small': va_robotwin_train_phywam_small_cfg,
    'robotwin_train_phywam_full': va_robotwin_train_phywam_full_cfg,
    'robotwin_train_phywam_event': va_robotwin_train_phywam_event_cfg,
    'robotwin_train_phywam_freeze_phys_only': va_robotwin_train_phywam_freeze_phys_only_cfg,
    'demo': va_demo_cfg,
    'demo_train': va_demo_train_cfg,
    'demo_i2av': va_demo_i2va_cfg,
}
