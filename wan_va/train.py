# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import faulthandler
import os
import sys
from pathlib import Path
import wandb

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_optimizer_state_dict,
    StateDictOptions,
)
from safetensors.torch import save_file, load_file
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model, apply_ac
from distributed.util import (
    _configure_model, 
    init_distributed, 
    dist_barrier,
    dist_mean, 
    dist_max
)
from einops import rearrange
from modules.utils import (
    load_transformer,
    sync_transformer_phywam_config,
)
from utils import (
    init_logger, 
    logger, 
    get_mesh_id, 
    sample_timestep_id,
    data_seq_to_patch,
    warmup_constant_lambda,
    FlowMatchScheduler
)

from dataset import MultiLatentLeRobotDataset
import gc


def configure_dataloader_runtime(config):
    strategy = getattr(config, 'dataloader_sharing_strategy', None)
    if not strategy:
        return

    current_strategy = torch.multiprocessing.get_sharing_strategy()
    if current_strategy == strategy:
        return

    torch.multiprocessing.set_sharing_strategy(strategy)
    if getattr(config, 'rank', 0) == 0:
        logger.info(f"Set torch multiprocessing sharing strategy to {strategy}")


class Trainer:
    def __init__(self, config):
        if config.enable_wandb and config.rank == 0:
            self.wandb = wandb
            self.wandb.init(
                entity=os.getenv("WANDB_ENTITY", os.getenv("WANDB_TEAM_NAME")),
                project=os.getenv("WANDB_PROJECT", "phywam"),
                # dir=log_dir,
                config=config,
                mode=os.getenv("WANDB_MODE", "online"),
                name=getattr(config, 'wandb_run_name', 'test_lln')
                # name=os.path.basename(os.path.normpath(job_config.job.dump_folder))
            )
            logger.info("WandB logging enabled")
        self.step = 0
        self.config = config
        self.device = torch.device(f"cuda:{config.local_rank}")
        self.dtype = config.param_dtype
        self.patch_size = config.patch_size
        self.freeze_backbone = bool(getattr(config, 'freeze_backbone', False))
        self.phys_monitor_interval = int(
            getattr(config, 'phys_monitor_interval', 0) or 0)
        self.phys_monitor_jsonl = None
        if config.rank == 0 and self.phys_monitor_interval > 0:
            monitor_path = getattr(config, 'phys_monitor_jsonl', None)
            if not monitor_path:
                monitor_path = (
                    Path(config.save_root)
                    / 'monitor'
                    / 'phys_train_stats.jsonl'
                )
            self.phys_monitor_jsonl = Path(monitor_path)
            self.phys_monitor_jsonl.parent.mkdir(parents=True, exist_ok=True)

        # Load models
        logger.info("Loading models...")

        # Load and shard transformer with FSDP
        logger.info("Loading transformer...")

        if hasattr(config, 'resume_from') and config.resume_from:
            transformer_path = os.path.join(config.resume_from, 'transformer')
            if config.rank == 0:
                logger.info(f"Resuming from checkpoint: {transformer_path}")
        else:
            transformer_path = os.path.join(config.wan22_pretrained_model_name_or_path, 'transformer')

        self.transformer = load_transformer(
            transformer_path,
            torch_dtype=torch.float32,
            torch_device='cpu',
            use_phys_memory=getattr(config, 'use_phys_memory', False),
            phys_memory_dim=getattr(config, 'phys_memory_dim', 768),
            phys_zero_init=getattr(config, 'phys_zero_init', True),
            phys_gate=getattr(config, 'phys_gate', True),
            phys_cross_start_layer=getattr(config, 'phys_cross_start_layer', 0),
            attn_mode=getattr(config, 'attn_mode', None),
        )
        sync_transformer_phywam_config(
            self.transformer,
            use_phys_memory=getattr(config, 'use_phys_memory', False),
            phys_memory_dim=getattr(config, 'phys_memory_dim', 768),
            phys_memory_align_mode=getattr(config, 'phys_memory_align_mode', None),
            phys_memory_max_tokens=getattr(config, 'phys_memory_max_tokens', None),
            phys_zero_init=getattr(config, 'phys_zero_init', True),
            phys_gate=getattr(config, 'phys_gate', True),
            phys_cross_start_layer=getattr(config, 'phys_cross_start_layer', 0),
        )
        self._configure_trainable_parameters()

        logger.info("Setting up activation checkpointing ...")
        apply_ac(self.transformer)

        logger.info("Setting up FSDP...")
        shard_fn = shard_model
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_fn,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=False,
        )
        self.transformer.train()
        if not self.freeze_backbone:
            self.transformer.requires_grad_(True)

            # Keep action text-embedder frozen even after global unfreeze.
            if hasattr(self.transformer, 'condition_embedder_action'):
                for p in self.transformer.condition_embedder_action.text_embedder.parameters():
                    p.requires_grad = False
        self._log_trainable_parameter_summary()

        trainable_params = [
            p for p in self.transformer.parameters() if p.requires_grad
        ]
        if not trainable_params:
            raise RuntimeError("No trainable parameters found.")

        # Optimizer
        self.optimizer = torch.optim.AdamW(
            trainable_params,
            lr=config.learning_rate,
            betas=(config.beta1, config.beta2),
            eps=1e-8,
            weight_decay=config.weight_decay,
            fused=True,
            foreach=False,
        )

        self.lr_scheduler = torch.optim.lr_scheduler.LambdaLR(self.optimizer, 
            lr_lambda=lambda step: warmup_constant_lambda(step, warmup_steps=config.warmup_steps))

        # Setup dataloaders
        logger.info("Setting up datasets...")
        train_dataset = MultiLatentLeRobotDataset(
            config=config,
            num_init_worker=getattr(config, 'dataset_init_worker', 128),
        )
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=config.world_size,
            rank=config.rank,
            shuffle=True,
            seed=42
        ) if config.world_size > 1 else None
        loader_kwargs = dict(
            batch_size=config.batch_size,
            shuffle=(train_sampler is None),
            num_workers=config.load_worker,
            sampler=train_sampler,
            persistent_workers=bool(
                getattr(config, 'dataloader_persistent_workers', False) and config.load_worker > 0
            ),
            timeout=int(getattr(config, 'dataloader_timeout', 0)),
        )
        if config.load_worker > 0:
            loader_kwargs['prefetch_factor'] = int(getattr(config, 'dataloader_prefetch_factor', 2))

        self.train_loader = DataLoader(
            train_dataset,
            **loader_kwargs,
        )
        if config.rank == 0:
            logger.info(
                "DataLoader settings: "
                f"dataset_init_worker={getattr(config, 'dataset_init_worker', 128)}, "
                f"num_workers={config.load_worker}, "
                f"persistent_workers={loader_kwargs['persistent_workers']}, "
                f"prefetch_factor={loader_kwargs.get('prefetch_factor', 'n/a')}, "
                f"timeout={loader_kwargs['timeout']}, "
                f"sharing_strategy={getattr(config, 'dataloader_sharing_strategy', None)}"
            )

        self.train_scheduler_latent = FlowMatchScheduler(shift=self.config.snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_latent.set_timesteps(1000, training=True)
        self.train_scheduler_action = FlowMatchScheduler(shift=self.config.action_snr_shift, sigma_min=0.0, extra_one_step=True)
        self.train_scheduler_action.set_timesteps(1000, training=True)

        self.save_dir = Path(config.save_root) / "checkpoints"
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.gradient_accumulation_steps = getattr(config, 'gradient_accumulation_steps', 1)
        self.train_loader_iter = None
        if hasattr(config, 'resume_from') and config.resume_from:
            self._load_training_state(config.resume_from)

    @staticmethod
    def _is_phys_trainable_name(name):
        phys_keys = (
            'phys_memory_projector',
            'phys_memory_type_embed',
            'attn_phys',
            'norm_phys',
            'phys_gate_logit',
        )
        return any(key in name for key in phys_keys)

    def _configure_trainable_parameters(self):
        if not self.freeze_backbone:
            self.transformer.requires_grad_(True)
            if hasattr(self.transformer, 'condition_embedder_action'):
                for p in self.transformer.condition_embedder_action.text_embedder.parameters():
                    p.requires_grad = False
            return

        self.transformer.requires_grad_(False)
        for name, param in self.transformer.named_parameters():
            if self._is_phys_trainable_name(name):
                param.requires_grad = True

    def _log_trainable_parameter_summary(self):
        trainable_names = []
        trainable_count = 0
        frozen_count = 0
        unexpected_trainable = []
        for name, param in self.transformer.named_parameters():
            param_count = param.numel()
            if param.requires_grad:
                trainable_count += param_count
                trainable_names.append(name)
                if self.freeze_backbone and not self._is_phys_trainable_name(name):
                    unexpected_trainable.append(name)
            else:
                frozen_count += param_count

        if unexpected_trainable:
            preview = ', '.join(unexpected_trainable[:20])
            raise RuntimeError(
                "freeze_backbone=True left non-physics parameters trainable: "
                f"{preview}"
            )

        if self.config.rank == 0:
            logger.info(
                "Trainable parameter summary: "
                f"freeze_backbone={self.freeze_backbone}, "
                f"trainable={trainable_count:,}, frozen={frozen_count:,}"
            )
            logger.info(
                "Trainable parameter names preview: "
                + ', '.join(trainable_names[:80])
            )

    def _should_monitor_phys(self, should_sync):
        return (
            should_sync
            and self.phys_monitor_interval > 0
            and self.step % self.phys_monitor_interval == 0
        )

    def _collect_phys_monitor_stats(self):
        if not hasattr(self.transformer, 'pop_phys_monitor_stats'):
            return {}
        raw_stats = self.transformer.pop_phys_monitor_stats()
        out = {}
        for key, value in raw_stats.items():
            value = value.detach().float().clone()
            if key.endswith('_max'):
                reduced = dist_max(value)
            else:
                reduced = dist_mean(value)
            out[key] = reduced.detach().cpu().item()
        return out

    def _write_phys_monitor_jsonl(self, payload):
        if self.phys_monitor_jsonl is None:
            return
        with self.phys_monitor_jsonl.open('a', encoding='utf-8') as f:
            f.write(json.dumps(payload, sort_keys=True) + '\n')
    
    def _get_next_batch(self):
        """Get next batch from iterator, reset if epoch is finished."""
        if self.train_loader_iter is None:
            self.train_loader_iter = iter(self.train_loader)
        
        try:
            batch = next(self.train_loader_iter)
        except StopIteration:
            # Reset sampler and iterator when epoch finishes
            if hasattr(self.train_loader.sampler, 'set_epoch'):
                self.train_loader.sampler.set_epoch(self.train_loader.sampler.epoch + 1)
            self.train_loader_iter = iter(self.train_loader)
            batch = next(self.train_loader_iter)
        
        return batch

    @torch.no_grad()
    def _sample_phys_cfg_dropout(self, batch):
        """Synchronously drop physics conditioning for one training microbatch."""
        if (
            'phys_mem_feat' not in batch
            or not getattr(self.config, 'use_phys_memory', False)
            or not getattr(self.config, 'phys_cfg_dropout', True)
        ):
            return False

        drop_prob = float(getattr(self.config, 'cfg_prob', 0.0))
        if not 0.0 <= drop_prob <= 1.0:
            raise ValueError(f"cfg_prob must be in [0, 1], got {drop_prob}")
        if drop_prob == 0.0:
            return False

        drop_flag = torch.zeros(1, dtype=torch.int32, device=self.device)
        if not dist.is_initialized() or self.config.rank == 0:
            drop_flag[0] = int(
                torch.rand((), device=self.device).item() < drop_prob
            )
        if dist.is_initialized():
            # FSDP ranks must all execute or skip attn_phys together.
            dist.broadcast(drop_flag, src=0)
        return bool(drop_flag.item())

    @torch.no_grad()
    def _add_noise(self, latent, train_scheduler, action_mask=False, action_mode=False, noisy_cond_prob=0.):
        B, C, F, H, W = latent.shape

        timestep_ids = sample_timestep_id(batch_size=F, num_train_timesteps=train_scheduler.num_train_timesteps)
        noise = torch.zeros_like(latent).normal_()
        timesteps = train_scheduler.timesteps[timestep_ids].to(device=self.device)
        noisy_latents =train_scheduler.add_noise(latent, noise, timesteps, t_dim=2)
        targets =train_scheduler.training_target(latent, noise, timesteps)

        patch_f, patch_h, patch_w = self.patch_size
        if action_mode:
            patch_f = patch_h = patch_w = 1
        
        latent_grid_id = get_mesh_id(
            latent.shape[-3] // patch_f,  # F
            latent.shape[-2] // patch_h,  # H
            latent.shape[-1] // patch_w,  # W
            t=1 if action_mode else 0,  # 1 for action mode (0 for latent), not used
            f_w=1,
            f_shift=0,
            action=action_mode
        ).to(self.device)  # shape: [4, seq_len]
        latent_grid_id = latent_grid_id[None].repeat(B, 1, 1)

        if torch.rand(1).item() < noisy_cond_prob:
            cond_timestep_ids = sample_timestep_id(
                    batch_size=F,
                    min_timestep_bd=0.5, 
                    max_timestep_bd=1.0, 
                    num_train_timesteps=train_scheduler.num_train_timesteps,
                )
            noise = torch.zeros_like(latent).normal_()
            cond_timesteps = train_scheduler.timesteps[cond_timestep_ids].to(device=self.device)
            latent = train_scheduler.add_noise(latent, noise, cond_timesteps, t_dim=2)
        else:
            cond_timesteps = torch.zeros_like(timesteps)

        if action_mask is not None:
            noisy_latents *= action_mask.float()
            targets *= action_mask.float()
            latent *= action_mask.float()

        return dict(
            timesteps=timesteps[None].repeat(B, 1),
            noisy_latents=noisy_latents,
            targets=targets,
            latent=latent,
            cond_timesteps=cond_timesteps[None].repeat(B, 1),
            grid_id=latent_grid_id,
        )

    @torch.no_grad()
    def _prepare_input_dict(self, batch_dict):
        """Prepare input dict following infer code pattern from wan_va_server.py."""
        # Generate grid_id following infer code (no batch dimension yet)
        # For action mode: get_mesh_id(shape[-3], shape[-2], shape[-1], t=1, f_w=1, f_shift, action=True)
        latent_dict = self._add_noise(
            latent=batch_dict['latents'], 
            train_scheduler=self.train_scheduler_latent, 
            action_mask=None, 
            action_mode=False,
            noisy_cond_prob=0.5)
        
        action_dict = self._add_noise(
            latent=batch_dict['actions'], 
            train_scheduler=self.train_scheduler_action, 
            action_mask=batch_dict['actions_mask'], 
            action_mode=True,
            noisy_cond_prob=0.0)

        latent_dict['text_emb'] = batch_dict['text_emb']
        action_dict['text_emb'] = batch_dict['text_emb']
        if 'phys_mem_feat' in batch_dict:
            latent_dict['phys_mem_feat'] = batch_dict['phys_mem_feat']
            action_dict['phys_mem_feat'] = batch_dict['phys_mem_feat']
        if 'phys_mem_mask' in batch_dict:
            latent_dict['phys_mem_mask'] = batch_dict['phys_mem_mask']
            action_dict['phys_mem_mask'] = batch_dict['phys_mem_mask']
        if 'phys_mem_spans' in batch_dict:
            latent_dict['phys_mem_spans'] = batch_dict['phys_mem_spans']
            action_dict['phys_mem_spans'] = batch_dict['phys_mem_spans']
        action_dict['actions_mask'] = batch_dict['actions_mask']

        input_dict = {
            'latent_dict': latent_dict,
            'action_dict': action_dict,
            'chunk_size': torch.randint(1, 5, (1,)).item(),
            'window_size': torch.randint(4, 65, (1,)).item(),
        }
        return input_dict

    def convert_input_format(self, input_dict):
        """Convert input dict to match transformer input format if needed."""
        for key, value in input_dict.items():
            input_dict[key] = value.to(self.device)#.to(self.dtype)
        return input_dict

    def compute_loss(self,
        input_dict,
        pred
    ):
        latent_pred, action_pred = pred
        action_pred = rearrange(action_pred, 'b (f n) c -> b c f n 1', f=input_dict['action_dict']['targets'].shape[-3])
        latent_pred = data_seq_to_patch(
                        self.patch_size, latent_pred,
                        input_dict['latent_dict']['targets'].shape[-3], input_dict['latent_dict']['targets'].shape[-2],
                        input_dict['latent_dict']['targets'].shape[-1], batch_size=latent_pred.shape[0])
        Bn, Fn = input_dict['latent_dict']['timesteps'].shape
        latent_loss_weight = self.train_scheduler_latent.training_weight(input_dict['latent_dict']['timesteps'].flatten()).reshape(Bn, Fn)
        action_loss_weight = self.train_scheduler_action.training_weight(input_dict['action_dict']['timesteps'].flatten()).reshape(Bn, Fn)

        # Frame-wise video loss calculation
        latent_loss = F.mse_loss(latent_pred.float(), input_dict['latent_dict']['targets'].float().detach(), reduction='none')
        latent_loss = latent_loss * latent_loss_weight[:, None, :, None, None]
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        latent_loss = latent_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        latent_loss = latent_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and compute mask per frame
        latent_loss_per_frame = latent_loss.sum(dim=1)  # (B*F,)
        latent_mask_per_frame = torch.ones_like(latent_loss).sum(dim=1)  # (B*F,)
        latent_loss = (latent_loss_per_frame / (latent_mask_per_frame + 1e-6)).mean()

        # Frame-wise action loss calculation
        action_loss = F.mse_loss(action_pred.float(), input_dict['action_dict']['targets'].float().detach(), reduction='none')
        action_loss = action_loss * action_loss_weight[:, None, :, None, None]
        action_loss = action_loss * input_dict['action_dict']['actions_mask'].float()
        # Permute to (B, F, H, W, C) and flatten to (B*F, H*W*C)
        action_loss = action_loss.permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_mask = input_dict['action_dict']['actions_mask'].float().permute(0, 2, 3, 4, 1)  # (B, C, F, H, W) -> (B, F, H, W, C)
        action_loss = action_loss.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        action_mask = action_mask.flatten(0, 1).flatten(1)  # (B, F, H, W, C) -> (B*F, H*W*C)
        # Sum per frame and normalize by mask per frame
        action_loss_per_frame = action_loss.sum(dim=1)  # (B*F,)
        action_mask_per_frame = action_mask.sum(dim=1)  # (B*F,)
        action_loss = (action_loss_per_frame / (action_mask_per_frame + 1e-6)).mean()

        return latent_loss / self.gradient_accumulation_steps, action_loss / self.gradient_accumulation_steps

    def _train_step(self, batch, batch_idx):
        """Train a single batch, returns losses for logging."""
        phys_cfg_dropped = self._sample_phys_cfg_dropout(batch)
        batch = self.convert_input_format(batch)
        input_dict = self._prepare_input_dict(batch)
        input_dict['phys_condition_scale'] = (
            0.0 if phys_cfg_dropped else 1.0
        )
        
        should_sync = (batch_idx + 1) % self.gradient_accumulation_steps == 0
        
        if not should_sync:
            self.transformer.set_requires_gradient_sync(False)
        else:
            self.transformer.set_requires_gradient_sync(True)

        monitor_phys = self._should_monitor_phys(should_sync)
        if hasattr(self.transformer, 'set_phys_monitor_enabled'):
            self.transformer.set_phys_monitor_enabled(monitor_phys)
        output = self.transformer(input_dict, train_mode=True)
        phys_monitor = (
            self._collect_phys_monitor_stats() if monitor_phys else {}
        )
        if hasattr(self.transformer, 'set_phys_monitor_enabled'):
            self.transformer.set_phys_monitor_enabled(False)
        latent_loss, action_loss = self.compute_loss(input_dict, output)
        loss = latent_loss + action_loss

        loss.backward()

        losses = {
            'latent_loss': latent_loss.detach(),
            'action_loss': action_loss.detach(),
            'phys_cfg_dropped': phys_cfg_dropped,
            'phys_monitor': phys_monitor,
        }
        
        # Only update weights after accumulating gradients
        if should_sync:
            total_norm = torch.nn.utils.clip_grad_norm_(self.transformer.parameters(), 2.0)
            self.optimizer.step()
            self.lr_scheduler.step()
            self.optimizer.zero_grad()
            
            losses['total_norm'] = total_norm
            losses['should_log'] = True
        else:
            losses['should_log'] = False

        return losses

    def save_checkpoint(self,):
        """Save model checkpoint in the same format as pretrained model."""
        try:
            state_dict = get_model_state_dict(
                self.transformer,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
            state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}
            optim_state = get_optimizer_state_dict(
                    self.transformer, self.optimizer,
                    options=StateDictOptions(full_state_dict=True, cpu_offload=True),
                )

            # Only rank 0 saves the checkpoint
            if self.config.rank == 0:
                checkpoint_dir = self.save_dir / f"checkpoint_step_{self.step}"
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                # Save transformer in the same format as pretrained model
                transformer_dir = checkpoint_dir / "transformer"
                transformer_dir.mkdir(parents=True, exist_ok=True)

                logger.info(f"Saving transformer to {transformer_dir}")

                # Manually save in diffusers format (outside FSDP context to avoid deadlock)
                # Save model weights
                model_file = transformer_dir / "diffusion_pytorch_model.safetensors"
                save_file(state_dict_bf16, model_file)

                # Save config (copy from original transformer config and update _name_or_path)
                config_file = transformer_dir / "config.json"
                config_dict = dict(self.transformer.config)
                config_dict.pop('_name_or_path', None)
                if getattr(self.config, 'use_phys_memory', False):
                    config_dict['use_phys_memory'] = True
                    config_dict['phys_memory_dim'] = int(
                        getattr(self.config, 'phys_memory_dim', 768))
                    config_dict['phys_memory_align_mode'] = getattr(
                        self.config, 'phys_memory_align_mode', 'per_latent_frame')
                    config_dict['phys_memory_max_tokens'] = int(
                        getattr(self.config, 'phys_memory_max_tokens', 8))
                    config_dict['phys_zero_init'] = bool(
                        getattr(self.config, 'phys_zero_init', True))
                    config_dict['phys_gate'] = bool(
                        getattr(self.config, 'phys_gate', True))
                    config_dict['phys_cross_start_layer'] = int(
                        getattr(self.config, 'phys_cross_start_layer', 0))
                    default_values = config_dict.get('_use_default_values', [])
                    config_dict['_use_default_values'] = [
                        key for key in default_values
                        if key not in (
                            'use_phys_memory',
                            'phys_memory_dim',
                            'phys_memory_align_mode',
                            'phys_memory_max_tokens',
                            'phys_zero_init',
                            'phys_gate',
                            'phys_cross_start_layer',
                        )
                    ]
                with open(config_file, 'w') as f:
                    json.dump(config_dict, f, indent=2)

                # Save optimizer/scheduler state and training metadata.
                training_state_path = checkpoint_dir / "training_state.pt"
                logger.info(f"Saving training state to {training_state_path}")
                torch.save({
                    'step': self.step,
                    'optimizer_state_dict': optim_state,
                    'lr_scheduler_state_dict': self.lr_scheduler.state_dict(),
                    'conditioning_config': {
                        'cfg_prob': float(getattr(self.config, 'cfg_prob', 0.0)),
                        'phys_cfg_dropout': bool(
                            getattr(self.config, 'phys_cfg_dropout', True)
                        ),
                        'phys_zero_init': bool(
                            getattr(self.config, 'phys_zero_init', True)
                        ),
                        'phys_gate': bool(
                            getattr(self.config, 'phys_gate', True)
                        ),
                        'phys_cross_start_layer': int(
                            getattr(self.config, 'phys_cross_start_layer', 0)
                        ),
                        'freeze_backbone': bool(
                            getattr(self.config, 'freeze_backbone', False)
                        ),
                        'phys_monitor_interval': int(
                            getattr(self.config, 'phys_monitor_interval', 0) or 0
                        ),
                    },
                }, training_state_path)

                logger.info(f"Checkpoint saved successfully at step {self.step}")

            # Synchronize all processes after saving
            dist_barrier()

        except Exception as e:
            if self.config.rank == 0:
                logger.error(f"Failed to save checkpoint: {e}")
                import traceback
                logger.error(traceback.format_exc())
            # Ensure all processes stay synchronized even on error
            dist_barrier()

    def _load_training_state(self, checkpoint_path):
        """Load training state (optimizer + step) after FSDP and optimizer creation."""
        checkpoint_dir = Path(checkpoint_path)
        training_state_path = checkpoint_dir / "training_state.pt"

        resume_optimizer_state = bool(getattr(self.config, 'resume_optimizer_state', False))
        if not resume_optimizer_state:
            if self.config.rank == 0:
                logger.info(
                    "Skipping optimizer/training-state restore "
                    "(resume_optimizer_state=False)."
                )
            return

        if not training_state_path.exists():
            if self.config.rank == 0:
                logger.warning(f"Training state not found: {training_state_path}, starting from step 0")
            return

        if self.config.rank == 0:
            logger.info(f"Loading training state from {training_state_path}")

        # All ranks load the training state directly
        training_state = torch.load(training_state_path, map_location='cpu', weights_only=False)

        # All ranks load optimizer state (required for strict optimizer resume).
        try:
            set_optimizer_state_dict(
                self.transformer,
                self.optimizer,
                optim_state_dict=training_state['optimizer_state_dict'],
                options=StateDictOptions(full_state_dict=True, strict=False)
            )
        except KeyError as e:
            raise RuntimeError(
                "Strict optimizer resume failed due to missing optimizer state key. "
                "This means current trainable parameter FQNs do not fully match the "
                "saved optimizer state. Disable strict optimizer resume by setting "
                "resume_optimizer_state=False (default), or resume from a checkpoint "
                "produced by exactly the same trainable-parameter set. "
                f"Missing key: {e}"
            ) from e
        if 'lr_scheduler_state_dict' in training_state:
            self.lr_scheduler.load_state_dict(training_state['lr_scheduler_state_dict'])
        self.step = training_state.get('step', 0)

        if self.config.rank == 0:
            logger.info(f"Training state loaded, resuming from step {self.step}")

        # Synchronize all ranks
        dist_barrier()

    def train(self):
        """Main training loop - train by steps instead of epochs."""
        logger.info(f"Starting training for {self.config.num_steps} steps...")
        self.transformer.train()

        progress_bar = tqdm(
            total=self.config.num_steps,
            desc="Training",
            disable=(self.config.rank != 0),
            leave=True,
            dynamic_ncols=True,
            initial=self.step
        )

        self.optimizer.zero_grad()
        accumulated_latent_losses = []
        accumulated_action_losses = []
        accumulated_phys_cfg_drops = []
        step_in_accumulation = 0

        while self.step < self.config.num_steps:
            # Get next batch (handles epoch reset automatically)
            batch = self._get_next_batch()
            
            losses = self._train_step(batch, step_in_accumulation)
            
            # Accumulate losses for logging
            accumulated_latent_losses.append(losses['latent_loss'])
            accumulated_action_losses.append(losses['action_loss'])
            accumulated_phys_cfg_drops.append(float(losses['phys_cfg_dropped']))
            step_in_accumulation += 1

            # Log and checkpoint when optimizer steps
            if losses['should_log']:
                lr = self.lr_scheduler.get_last_lr()[0]
                phys_monitor = losses.get('phys_monitor') or {}

                # Average accumulated losses
                latent_loss_show = dist_mean(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                action_loss_show = dist_mean(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                max_latent_loss_show = dist_max(torch.stack(accumulated_latent_losses).sum()).detach().cpu().item()
                max_action_loss_show = dist_max(torch.stack(accumulated_action_losses).sum()).detach().cpu().item()
                phys_cfg_drop_rate = (
                    sum(accumulated_phys_cfg_drops)
                    / max(len(accumulated_phys_cfg_drops), 1)
                )

                # Clear accumulated losses
                accumulated_latent_losses = []
                accumulated_action_losses = []
                accumulated_phys_cfg_drops = []
                step_in_accumulation = 0

                torch.cuda.synchronize()
                if self.step % self.config.gc_interval == 0:
                    torch.cuda.empty_cache()
                    gc.collect()

                if self.config.rank == 0:
                    total_norm = losses['total_norm']
                    progress_bar.update(1)
                    progress_bar.set_postfix({
                        'latent_loss': f'{latent_loss_show:.4f}',
                        'action_loss': f'{action_loss_show:.4f}',
                        'step': self.step,
                        'grad_norm': f'{total_norm.item():.2f}',
                        'lr': f'{lr:.2e}',
                        'phys_drop': f'{phys_cfg_drop_rate:.2f}',
                    })
                    if phys_monitor:
                        progress_bar.set_postfix({
                            'latent_loss': f'{latent_loss_show:.4f}',
                            'action_loss': f'{action_loss_show:.4f}',
                            'step': self.step,
                            'grad_norm': f'{total_norm.item():.2f}',
                            'lr': f'{lr:.2e}',
                            'phys_drop': f'{phys_cfg_drop_rate:.2f}',
                            'phys_ratio': f"{phys_monitor.get('phys/residual_ratio_mean', 0.0):.4f}",
                            'phys_gate': f"{phys_monitor.get('phys/gate_mean', 0.0):.4f}",
                        })
                        monitor_payload = {
                            'step': self.step,
                            'lr': lr,
                            'grad_norm': total_norm.detach().cpu().item(),
                            'latent_loss': latent_loss_show,
                            'action_loss': action_loss_show,
                            'phys_cfg_dropout_rate': phys_cfg_drop_rate,
                        }
                        monitor_payload.update(phys_monitor)
                        self._write_phys_monitor_jsonl(monitor_payload)
                    if self.config.enable_wandb:
                        wandb_payload = {
                            'loss_metrics/global_avg_video_loss': latent_loss_show,
                            'loss_metrics/global_avg_action_loss': action_loss_show,
                            'loss_metrics/global_max_video_loss': max_latent_loss_show,
                            'loss_metrics/global_max_action_loss': max_action_loss_show,
                            'conditioning/phys_cfg_dropout_rate': phys_cfg_drop_rate,
                            'grad_norm': total_norm.item(),
                            'lr': lr,
                        }
                        wandb_payload.update(phys_monitor)
                        self.wandb.log(wandb_payload, step=self.step)
                
                self.step += 1
                
                if self.step % self.config.save_interval == 0:
                    if self.config.rank == 0:
                        logger.info(f"Starting save model at step {self.step}")
                    self.save_checkpoint()

            dist_barrier()

        progress_bar.close()
        logger.info("Training completed!")


def run(args):
    """Main entry point."""
    config = VA_CONFIGS[args.config_name]

    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))

    init_distributed(world_size, local_rank, rank)

    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size

    if args.save_root is not None:
        config.save_root = args.save_root
    if args.resume_from is not None:
        config.resume_from = args.resume_from
    if args.use_phys_memory:
        config.use_phys_memory = True
    if args.phys_memory_path is not None:
        config.phys_memory_path = args.phys_memory_path
    if args.phys_memory_dirname is not None:
        config.phys_memory_dirname = args.phys_memory_dirname
    if args.phys_memory_dim is not None:
        config.phys_memory_dim = args.phys_memory_dim
    if args.phys_memory_block_size is not None:
        config.phys_memory_block_size = args.phys_memory_block_size
    if args.phys_memory_align_mode is not None:
        config.phys_memory_align_mode = args.phys_memory_align_mode
    if args.phys_memory_max_tokens is not None:
        config.phys_memory_max_tokens = args.phys_memory_max_tokens
    if args.phys_cfg_dropout is not None:
        config.phys_cfg_dropout = args.phys_cfg_dropout
    if args.phys_zero_init is not None:
        config.phys_zero_init = args.phys_zero_init
    if args.phys_gate is not None:
        config.phys_gate = args.phys_gate
    if args.phys_cross_start_layer is not None:
        config.phys_cross_start_layer = args.phys_cross_start_layer
    if args.freeze_backbone:
        config.freeze_backbone = True
    if args.phys_monitor_interval is not None:
        config.phys_monitor_interval = args.phys_monitor_interval
    if args.phys_monitor_jsonl is not None:
        config.phys_monitor_jsonl = args.phys_monitor_jsonl
    if args.gradient_accumulation_steps is not None:
        config.gradient_accumulation_steps = args.gradient_accumulation_steps
    if args.num_steps is not None:
        config.num_steps = args.num_steps
    if args.load_worker is not None:
        config.load_worker = args.load_worker
    if args.dataset_init_worker is not None:
        config.dataset_init_worker = args.dataset_init_worker
    if args.dataloader_prefetch_factor is not None:
        config.dataloader_prefetch_factor = args.dataloader_prefetch_factor
    if args.dataloader_timeout is not None:
        config.dataloader_timeout = args.dataloader_timeout
    if args.dataloader_sharing_strategy is not None:
        config.dataloader_sharing_strategy = args.dataloader_sharing_strategy
    if args.disable_persistent_workers:
        config.dataloader_persistent_workers = False
    if args.resume_optimizer_state:
        config.resume_optimizer_state = True

    configure_dataloader_runtime(config)

    if rank == 0:
        logger.info(f"Using config: {args.config_name}")
        logger.info(f"World size: {world_size}, Local rank: {local_rank}")
        logger.info(f"Dataset path: {config.dataset_path}")
        logger.info(f"Save root: {config.save_root}")
        logger.info(f"Resume from: {getattr(config, 'resume_from', None)}")
        logger.info(
            "Batch settings: "
            f"per_gpu_batch={config.batch_size}, "
            f"gradient_accumulation_steps={config.gradient_accumulation_steps}, "
            f"effective_global_batch={config.batch_size * world_size * config.gradient_accumulation_steps}"
        )
        logger.info(
            "PhyWAM physics memory: "
            f"use={getattr(config, 'use_phys_memory', False)}, "
            f"dirname={getattr(config, 'phys_memory_dirname', None)}, "
            f"path={getattr(config, 'phys_memory_path', None)}, "
            f"dim={getattr(config, 'phys_memory_dim', None)}, "
            f"block_size={getattr(config, 'phys_memory_block_size', None)}, "
            f"align_mode={getattr(config, 'phys_memory_align_mode', None)}, "
            f"max_tokens={getattr(config, 'phys_memory_max_tokens', None)}, "
            f"cfg_dropout={getattr(config, 'phys_cfg_dropout', True)}, "
            f"cfg_prob={getattr(config, 'cfg_prob', None)}, "
            f"zero_init={getattr(config, 'phys_zero_init', True)}, "
            f"gate={getattr(config, 'phys_gate', True)}, "
            f"cross_start_layer={getattr(config, 'phys_cross_start_layer', 0)}, "
            f"freeze_backbone={getattr(config, 'freeze_backbone', False)}, "
            f"monitor_interval={getattr(config, 'phys_monitor_interval', 0)}, "
            f"monitor_jsonl={getattr(config, 'phys_monitor_jsonl', None)}"
        )
        logger.info(
            "DataLoader overrides: "
            f"load_worker={getattr(config, 'load_worker', None)}, "
            f"dataset_init_worker={getattr(config, 'dataset_init_worker', None)}, "
            f"persistent_workers={getattr(config, 'dataloader_persistent_workers', None)}, "
            f"prefetch_factor={getattr(config, 'dataloader_prefetch_factor', None)}, "
            f"timeout={getattr(config, 'dataloader_timeout', None)}, "
            f"sharing_strategy={getattr(config, 'dataloader_sharing_strategy', None)}"
        )

    trainer = Trainer(config)
    trainer.train()


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description="Train WAN model for robotics")
    parser.add_argument(
        "--config-name",
        type=str,
        default='robotwin_train',
        help="Config name",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        default=None,
        help="Root directory for saving checkpoints",
    )
    parser.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Checkpoint directory to resume from, e.g. .../checkpoints/checkpoint_step_2000",
    )
    parser.add_argument(
        "--use-phys-memory",
        action="store_true",
        help="Enable block-aligned physics memory tokens in self-attention",
    )
    parser.add_argument(
        "--phys-memory-path",
        type=str,
        default=None,
        help="Explicit path to precomputed physics memory token files",
    )
    parser.add_argument(
        "--phys-memory-dirname",
        type=str,
        default=None,
        help="Directory name under each task root for precomputed physics memory tokens",
    )
    parser.add_argument(
        "--phys-memory-dim",
        type=int,
        default=None,
        help="Physics memory feature dimension",
    )
    parser.add_argument(
        "--phys-memory-block-size",
        type=int,
        default=None,
        help="Raw video frames per precomputed physics memory block",
    )
    parser.add_argument(
        "--phys-memory-align-mode",
        type=str,
        default=None,
        choices=["per_latent_frame", "phase_tokens"],
        help="How dataset aligns physics/event memory to video latent frames",
    )
    parser.add_argument(
        "--phys-memory-max-tokens",
        type=int,
        default=None,
        help="Maximum padded physics phase tokens per episode chunk",
    )
    parser.add_argument(
        "--phys-cfg-dropout",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Drop the entire physics condition with cfg_prob during training. "
            "Enabled by default in the training config; use "
            "--no-phys-cfg-dropout for the previous behavior."
        ),
    )
    parser.add_argument(
        "--phys-zero-init",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Zero-initialize each physics cross-attention output projection. "
            "Enabled by default in the training config."
        ),
    )
    parser.add_argument(
        "--phys-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Apply a learnable per-block sigmoid gate to physics residuals. "
            "Enabled by default in the training config."
        ),
    )
    parser.add_argument(
        "--phys-cross-start-layer",
        type=int,
        default=None,
        help=(
            "First transformer block index that receives phase-token "
            "physics cross-attention. Earlier blocks keep the base path."
        ),
    )
    parser.add_argument(
        "--freeze-backbone",
        action="store_true",
        help="Freeze the base transformer and train only PhyWAM physics modules.",
    )
    parser.add_argument(
        "--phys-monitor-interval",
        type=int,
        default=None,
        help="Optimizer-step interval for physics residual monitoring. Use 0 to disable.",
    )
    parser.add_argument(
        "--phys-monitor-jsonl",
        type=str,
        default=None,
        help="Optional JSONL path for rank-0 physics monitor records.",
    )
    parser.add_argument(
        "--gradient-accumulation-steps",
        type=int,
        default=None,
        help="Override gradient_accumulation_steps from config",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=None,
        help="Override num_steps from config",
    )
    parser.add_argument(
        "--load-worker",
        type=int,
        default=None,
        help="Override DataLoader num_workers from config",
    )
    parser.add_argument(
        "--dataset-init-worker",
        type=int,
        default=None,
        help="Override multiprocessing worker count used while building the dataset index",
    )
    parser.add_argument(
        "--dataloader-prefetch-factor",
        type=int,
        default=None,
        help="Override DataLoader prefetch_factor when num_workers > 0",
    )
    parser.add_argument(
        "--dataloader-timeout",
        type=int,
        default=None,
        help="Override DataLoader timeout in seconds",
    )
    parser.add_argument(
        "--dataloader-sharing-strategy",
        type=str,
        default=None,
        choices=["file_descriptor", "file_system"],
        help="Override torch multiprocessing sharing strategy for DataLoader IPC",
    )
    parser.add_argument(
        "--disable-persistent-workers",
        action="store_true",
        help="Disable persistent DataLoader workers for debugging worker lifecycle issues",
    )
    parser.add_argument(
        "--resume-optimizer-state",
        action="store_true",
        help="Strictly restore optimizer/lr/step from training_state.pt. "
             "Default is disabled to avoid resume-time key mismatch errors.",
    )

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    faulthandler.enable()
    init_logger()
    main()
