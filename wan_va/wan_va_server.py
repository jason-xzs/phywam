# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import argparse
import json
import os
import sys
import time
from collections import Counter, defaultdict
from functools import partial
from types import SimpleNamespace
from PIL import Image
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

import numpy as np
import torch
import torch.nn.functional as F
from diffusers.pipelines.wan.pipeline_wan import prompt_clean
from einops import rearrange
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from configs import VA_CONFIGS
from distributed.fsdp import shard_model
from distributed.util import _configure_model, init_distributed
from modules.utils import (
    WanVAEStreamingWrapper,
    load_text_encoder,
    load_tokenizer,
    load_transformer,
    load_vae,
    sync_transformer_phywam_config,
)
from utils import (
    FlowMatchScheduler,
    data_seq_to_patch,
    get_mesh_id,
    init_logger,
    logger,
    run_async_server_mode,
    save_async,
)


class OnlinePhysMemoryEncoder:
    """MoPE-JEPA block encoder used by online PhyWAM inference."""

    def __init__(self, job_config, device):
        init_t0 = time.perf_counter()
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.mope_repo = getattr(
            job_config,
            'mope_repo',
            '/data/worldmodel_xzs/phywam_v3/mope-jepa',
        )
        self.ckpt = getattr(
            job_config,
            'mope_ckpt',
            '/data/worldmodel_xzs/phywam_v3/mope-jepa/output/'
            'mope_jepa_v39_task_canonical_event_physics_c12a24_freeze150/'
            'checkpoint-400.pth',
        )
        self.camera_key = getattr(
            job_config, 'phys_memory_camera_key',
            getattr(job_config, 'obs_cam_keys', ['observation.images.cam_high'])[0])
        self.num_frames = int(getattr(job_config, 'phys_memory_num_frames', 16))
        self.input_size = int(getattr(job_config, 'phys_memory_input_size', 224))
        self.obs_frames_per_block = int(
            getattr(job_config, 'phys_memory_obs_frames_per_block', 4))
        self.no_event_label_id = int(
            getattr(job_config, 'phys_event_no_event_label_id', 0))
        self.event_threshold = float(
            getattr(job_config, 'phys_event_threshold', 0.5))
        self.event_window_blocks = int(
            getattr(job_config, 'phys_event_window_blocks', 0))
        self.event_detector = getattr(job_config, 'phys_event_detector', 'manual')
        self.event_label_path = getattr(
            job_config, 'phys_event_label_path', None)
        self.event_vocabulary = []
        self.task_phase_sequences = {}
        self._load_phase_metadata()

        v36_builder_path = os.path.join(
            self.mope_repo,
            'tools',
            'build_phywam_phys_tokens_v36.py',
        )
        if os.path.isfile(v36_builder_path):
            builder_dir = os.path.dirname(v36_builder_path)
            if builder_dir not in sys.path:
                sys.path.insert(0, builder_dir)
            from build_phywam_phys_tokens_v36 import (  # noqa: WPS433
                build_mope,
                extract_block_feature,
                extract_block_feature_and_event,
            )
            self.mope_builder_source = v36_builder_path
        else:
            script_dir = os.path.join(repo_root, 'script')
            if script_dir not in sys.path:
                sys.path.insert(0, script_dir)
            from build_phywam_phys_tokens import (  # noqa: WPS433
                build_mope,
                extract_block_feature,
                extract_block_feature_and_event,
            )
            self.mope_builder_source = os.path.join(
                script_dir, 'build_phywam_phys_tokens.py')

        args = SimpleNamespace(
            mope_repo=self.mope_repo,
            ckpt=self.ckpt,
            input_size=self.input_size,
            device=str(device),
            model='pretrain_mope_jepa_base_patch16_224',
            tubelet_size=2,
            use_mope=True,
            num_physics_experts=int(
                getattr(job_config, 'phys_mope_num_physics_experts', 17)),
            num_general_experts=int(
                getattr(job_config, 'phys_mope_num_general_experts', 10)),
            num_routable_experts=17,
            num_shared_experts=4,
            candidate_k=int(getattr(job_config, 'phys_mope_candidate_k', 5)),
            gate_threshold=float(
                getattr(job_config, 'phys_mope_gate_threshold', 0.0)),
            gate_hidden=int(getattr(job_config, 'phys_mope_gate_hidden', 0)),
            gate_layers=int(getattr(job_config, 'phys_mope_gate_layers', 2)),
            gate_dims=str(getattr(job_config, 'phys_mope_gate_dims', '')),
            enable_general=bool(
                getattr(job_config, 'phys_mope_enable_general', False)),
            top_k=5,
            num_frames=self.num_frames,
            num_event_classes=int(
                getattr(job_config, 'phys_num_event_classes', 0)
                or len(self.event_vocabulary)
                or 0
            ),
        )
        self.infer_mope, self.model, self.transform, self.device = build_mope(args)
        self.extract_block_feature = extract_block_feature
        self.extract_block_feature_and_event = extract_block_feature_and_event
        logger.info(
            "[PhyWAM Timing] mope_encoder_init_s="
            f"{time.perf_counter() - init_t0:.3f} "
            f"builder={self.mope_builder_source} ckpt={self.ckpt}"
        )

    def _load_phase_metadata(self):
        if not self.event_label_path:
            return
        if not os.path.isfile(self.event_label_path):
            raise FileNotFoundError(
                f"Physics event label metadata not found: {self.event_label_path}")
        with open(self.event_label_path, 'r', encoding='utf-8') as f:
            payload = json.load(f)

        self.event_vocabulary = list(payload.get('event_vocabulary') or [])
        episode_segments = defaultdict(list)
        for sample in payload.get('samples', []):
            task_key = str(
                sample.get('task_key')
                or sample.get('base_task_name')
                or sample.get('task_name')
                or ''
            )
            episode_index = int(sample.get('episode_index', -1))
            label = str(sample.get('event_label', ''))
            if not task_key or episode_index < 0 or not label or label == 'no_event':
                continue
            episode_segments[(task_key, episode_index)].append((
                int(sample.get('start_frame', 0)),
                int(sample.get('segment_index', 0)),
                label,
            ))

        sequence_counts = defaultdict(Counter)
        for (task_key, _), segments in episode_segments.items():
            sequence_labels = []
            for _, _, label in sorted(segments):
                if not sequence_labels or label != sequence_labels[-1]:
                    sequence_labels.append(label)
            sequence = tuple(sequence_labels)
            if sequence:
                sequence_counts[task_key][sequence] += 1
        self.task_phase_sequences = {
            task_key: counts.most_common(1)[0][0]
            for task_key, counts in sequence_counts.items()
        }
        logger.info(
            "[PhyWAM] loaded phase metadata "
            f"classes={len(self.event_vocabulary)} "
            f"tasks={len(self.task_phase_sequences)} "
            f"path={self.event_label_path}"
        )

    def phase_sequence_for_task(self, task_name):
        if not task_name:
            return ()
        task_name = str(task_name)
        candidates = [
            task_name,
            task_name.split('-', 1)[0],
        ]
        for candidate in candidates:
            if candidate in self.task_phase_sequences:
                return self.task_phase_sequences[candidate]
        for task_key, sequence in self.task_phase_sequences.items():
            if task_key in task_name or task_name in task_key:
                return sequence
        return ()

    def _to_pil(self, frame):
        arr = np.asarray(frame)
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255).astype(np.uint8)
        return Image.fromarray(arr).convert("RGB")

    def _sample_to_num_frames(self, images):
        if not images:
            raise ValueError("Cannot build physics memory from an empty image block")
        if len(images) == self.num_frames:
            return images
        if len(images) == 1:
            return images * self.num_frames
        ids = np.linspace(0, len(images) - 1, self.num_frames).round().astype(int)
        return [images[int(i)] for i in ids]

    def _obs_images(self, obs_payload):
        if obs_payload is None:
            return []
        if not isinstance(obs_payload, list):
            obs_payload = [obs_payload]
        images = []
        for item in obs_payload:
            if item is None:
                continue
            if self.camera_key not in item:
                raise KeyError(
                    f"Missing camera key for physics memory: {self.camera_key}")
            images.append(self._to_pil(item[self.camera_key]))
        return images

    @torch.no_grad()
    def encode(self, obs_payload, with_event=False):
        images = self._obs_images(obs_payload)
        if not images:
            return None

        features = []
        events = []
        block_size = max(self.obs_frames_per_block, 1)
        for start in range(0, len(images), block_size):
            block_images = images[start:start + block_size]
            block_images = self._sample_to_num_frames(block_images)
            if with_event:
                feature, event = self.extract_block_feature_and_event(
                    self.infer_mope,
                    self.model,
                    self.transform,
                    self.device,
                    block_images,
                    self.num_frames,
                )
            else:
                feature = self.extract_block_feature(
                    self.infer_mope,
                    self.model,
                    self.transform,
                    self.device,
                    block_images,
                    self.num_frames,
                )
                event = None
            features.append(feature)
            events.append(event)

        feat = torch.from_numpy(np.stack(features, axis=0)).float().to(self.device)
        return feat, events

    @torch.no_grad()
    def encode_phase(self, obs_payload):
        images = self._obs_images(obs_payload)
        if not images:
            return None
        images = self._sample_to_num_frames(images)
        video_tensor = self.infer_mope.frames_to_tensor(
            images, self.transform, self.num_frames).to(self.device)
        mask = self.infer_mope.make_full_visible_mask(
            video_tensor.size(0),
            self.model.encoder.patch_embed.num_patches,
            self.device,
        )
        x_vis = self.model.encoder.forward_features(
            video_tensor,
            mask,
            physics_label=None,
            physics_label_soft=None,
        )
        feature = x_vis.mean(dim=1)[0].float()

        event = None
        event_head = getattr(self.model, 'event_head', None)
        if event_head is not None:
            logits = event_head(x_vis.mean(dim=1))
            probs = torch.softmax(logits.float(), dim=-1)
            confidence, label_id = probs.max(dim=-1)
            event = {
                'label_id': int(label_id.item()),
                'confidence': float(confidence.item()),
                'probs': probs[0].detach().cpu().numpy().astype(np.float32),
            }
            label_id = int(event.get('label_id', -1))
            if 0 <= label_id < len(self.event_vocabulary):
                event['label'] = self.event_vocabulary[label_id]
        return feature, event

    def select_event_blocks(self, events):
        selected = set()
        event_summaries = []
        for idx, event in enumerate(events):
            if event is None:
                continue
            label_id = int(event.get('label_id', self.no_event_label_id))
            confidence = float(event.get('confidence', 0.0))
            triggered = (
                label_id != self.no_event_label_id
                and confidence >= self.event_threshold
            )
            event_summaries.append({
                'block_index': idx,
                'label_id': label_id,
                'confidence': confidence,
                'triggered': triggered,
            })
            if not triggered:
                continue
            for j in range(idx - self.event_window_blocks,
                           idx + self.event_window_blocks + 1):
                if 0 <= j < len(events):
                    selected.add(j)
        return sorted(selected), event_summaries


class VA_Server:

    def __init__(self, job_config):
        self.cache_name = 'pos'
        self.job_config = job_config
        self.save_root = job_config.save_root
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        self.enable_offload = getattr(job_config, 'enable_offload', True)  # offload vae & text_encoder to save vram
        self._logged_frame_start_ids = set()

        self.scheduler = FlowMatchScheduler(shift=self.job_config.snr_shift,
                                            sigma_min=0.0,
                                            extra_one_step=True)
        self.action_scheduler = FlowMatchScheduler(
            shift=self.job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)

        self.vae = load_vae(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'vae'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )
        self.streaming_vae = WanVAEStreamingWrapper(self.vae)

        self.tokenizer = load_tokenizer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'tokenizer'), )

        self.text_encoder = load_text_encoder(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'text_encoder'),
            torch_dtype=self.dtype,
            torch_device='cpu' if self.enable_offload else self.device,
        )

        self.transformer = load_transformer(
            os.path.join(job_config.wan22_pretrained_model_name_or_path,
                         'transformer'),
            torch_dtype=self.dtype,
            torch_device=self.device,
            use_phys_memory=None,
            phys_memory_dim=None,
        )
        checkpoint_has_phys_memory = bool(
            getattr(self.transformer, 'use_phys_memory', False))
        requested_phys_memory = bool(
            getattr(job_config, 'use_phys_memory', False))
        checkpoint_phys_cross_start_layer = getattr(
            self.transformer.config, 'phys_cross_start_layer', None)
        requested_phys_cross_start_layer = int(
            getattr(job_config, 'phys_cross_start_layer', 0) or 0)
        phys_cross_start_layer = (
            checkpoint_phys_cross_start_layer
            if checkpoint_phys_cross_start_layer is not None
            else (0 if checkpoint_has_phys_memory else requested_phys_cross_start_layer)
        )
        updated = sync_transformer_phywam_config(
            self.transformer,
            use_phys_memory=checkpoint_has_phys_memory or requested_phys_memory,
            phys_memory_dim=getattr(job_config, 'phys_memory_dim', 768),
            phys_memory_align_mode=getattr(
                self.transformer.config, 'phys_memory_align_mode', None),
            phys_memory_max_tokens=getattr(
                self.transformer.config, 'phys_memory_max_tokens', None),
            phys_zero_init=getattr(
                self.transformer.config,
                'phys_zero_init',
                getattr(job_config, 'phys_zero_init', True),
            ),
            phys_gate=getattr(
                self.transformer.config,
                'phys_gate',
                getattr(job_config, 'phys_gate', True),
            ),
            phys_cross_start_layer=phys_cross_start_layer,
        )
        if updated:
            logger.info(
                "Synchronized transformer PhyWAM config: "
                f"use_phys_memory={self.transformer.use_phys_memory}, "
                f"phys_memory_dim={self.transformer.phys_memory_dim}, "
                f"phys_cross_start_layer="
                f"{getattr(self.transformer, 'phys_cross_start_layer', 0)}"
            )

        shard_fn = shard_model
        self.transformer = _configure_model(model=self.transformer,
                                            shard_fn=shard_fn,
                                            param_dtype=self.dtype,
                                            device=self.device,
                                            eval_mode=True,
                                            )
        self.phys_memory_infer_mode = getattr(job_config, 'phys_memory_infer_mode', 'event')
        self.use_phys_memory = (
            requested_phys_memory
            and self.phys_memory_infer_mode != 'none'
        )
        if self.use_phys_memory and not self.transformer.use_phys_memory:
            raise ValueError(
                "Physics inference was requested, but the transformer does not "
                "contain physics-memory modules.")
        self.infer_phys_memory_npy = getattr(job_config, 'infer_phys_memory_npy', None)
        self.static_phys_mem_feat = None
        self.active_phys_mem_feat = None
        self.active_phys_mem_frame_indices = None
        self.active_phys_confidence = 0.0
        self.current_task_name = None
        self.online_phys_encoder = None
        self._last_phys_timing = None
        self.phys_use_phase_confidence_gate = bool(
            getattr(job_config, 'phys_use_phase_confidence_gate', True))
        self.phys_default_task_gate = float(
            getattr(job_config, 'phys_default_task_gate', 1.0))
        raw_task_gates = getattr(job_config, 'phys_task_gates', {}) or {}
        if isinstance(raw_task_gates, str):
            raw_task_gates = json.loads(raw_task_gates)
        self.phys_task_gates = dict(raw_task_gates)
        self.phys_event_detector = getattr(job_config, 'phys_event_detector', 'manual')
        self.phys_gripper_event_threshold = float(
            getattr(job_config, 'phys_gripper_event_threshold', 0.2))
        self.phys_image_event_threshold = float(
            getattr(job_config, 'phys_image_event_threshold', 0.05))
        self.phys_enable_image_delta_event = bool(
            getattr(job_config, 'phys_enable_image_delta_event', False))
        self.phys_gripper_open_threshold = float(
            getattr(job_config, 'phys_gripper_open_threshold', 0.2))
        self.phys_gripper_closed_threshold = float(
            getattr(job_config, 'phys_gripper_closed_threshold', 0.8))
        configured_max_tokens = int(
            getattr(job_config, 'phys_memory_max_tokens', 8) or 8)
        checkpoint_max_tokens = int(
            getattr(self.transformer.config, 'phys_memory_max_tokens', 0) or 0)
        if checkpoint_max_tokens and checkpoint_max_tokens != configured_max_tokens:
            logger.warning(
                "[PhyWAM] phys_memory_max_tokens mismatch: "
                f"checkpoint={checkpoint_max_tokens} config={configured_max_tokens}; "
                "using the checkpoint value."
            )
        self.phys_memory_max_tokens = (
            checkpoint_max_tokens or configured_max_tokens)
        self.phys_phase_switch_patience = int(
            getattr(job_config, 'phys_phase_switch_patience', 1) or 1)
        self.phys_phase_buffer_frames = int(
            getattr(job_config, 'phys_phase_buffer_frames', 16) or 16)
        if self.use_phys_memory and self.phys_memory_infer_mode == 'phase':
            checkpoint_align_mode = getattr(
                self.transformer.config, 'phys_memory_align_mode', None)
            if checkpoint_align_mode != 'phase_tokens':
                raise ValueError(
                    "Phase-token inference requires a checkpoint trained with "
                    "phys_memory_align_mode='phase_tokens', got "
                    f"{checkpoint_align_mode!r}."
                )
            if self.phys_memory_max_tokens < 1:
                raise ValueError("phys_memory_max_tokens must be positive")
        self._reset_manual_phys_event_state()
        if self.use_phys_memory and self.infer_phys_memory_npy:
            self.static_phys_mem_feat = self._load_infer_phys_memory(self.infer_phys_memory_npy)
        elif self.use_phys_memory and self.phys_memory_infer_mode == 'static':
            logger.warning(
                "PhyWAM static physics memory is enabled but no "
                "--infer-phys-memory-npy was provided; running without physics tokens."
            )
        if self.use_phys_memory and self.phys_memory_infer_mode in (
            'always', 'event', 'phase'
        ):
            self.online_phys_encoder = OnlinePhysMemoryEncoder(job_config, self.device)
        self._reset_phase_phys_state()

        self.env_type = job_config.env_type
        self.streaming_vae_half = None
        if self.env_type == 'robotwin_tshape':
            vae_half = load_vae(
                os.path.join(job_config.wan22_pretrained_model_name_or_path,
                             'vae'),
                torch_dtype=self.dtype,
                torch_device='cpu' if self.enable_offload else self.device,
            )
            self.streaming_vae_half = WanVAEStreamingWrapper(vae_half)

    def _load_infer_phys_memory(self, path):
        if path.endswith('.npz'):
            payload = np.load(path)
            if 'features' in payload:
                feat = payload['features']
            elif 'phys_feat' in payload:
                feat = payload['phys_feat']
            else:
                raise KeyError(f"Missing 'features' or 'phys_feat' in physics memory file: {path}")
        else:
            feat = np.load(path)
        feat = np.asarray(feat, dtype=np.float32)
        if feat.ndim == 1:
            feat = feat[None, None, :]
        elif feat.ndim == 2:
            feat = feat[None, :, :]
        elif feat.ndim != 3:
            raise ValueError(f"Expected physics memory shape [D], [T,D], or [B,T,D], got {feat.shape}: {path}")

        expected_dim = int(getattr(self.job_config, 'phys_memory_dim', 0) or 0)
        if expected_dim > 0 and feat.shape[-1] != expected_dim:
            raise ValueError(
                f"Physics memory dim mismatch for {path}: expected {expected_dim}, got {feat.shape[-1]}"
            )
        return torch.from_numpy(feat).float().to(self.device)

    def _current_task_gate(self):
        task_name = self.current_task_name
        gate = self.phys_default_task_gate
        if task_name:
            task_text = str(task_name)
            candidates = [
                task_text,
                os.path.basename(task_text),
                self._sanitize_name(task_text, default=''),
            ]
            for key in candidates:
                if key in self.phys_task_gates:
                    gate = self.phys_task_gates[key]
                    break
        try:
            gate = float(gate)
        except (TypeError, ValueError):
            gate = float(self.phys_default_task_gate)
        return max(gate, 0.0)

    def _current_phase_confidence_gate(self):
        if (
            not self.phys_use_phase_confidence_gate
            or self.phys_memory_infer_mode != 'phase'
        ):
            return 1.0
        return float(np.clip(self.active_phys_confidence, 0.0, 1.0))

    def _current_phys_condition_scale(self):
        return self._current_phase_confidence_gate() * self._current_task_gate()

    def _annotate_phys_condition_scale(self, timing):
        timing['phase_confidence_active'] = float(
            np.clip(self.active_phys_confidence, 0.0, 1.0))
        timing['phys_task_gate'] = float(self._current_task_gate())
        timing['phys_condition_scale'] = float(
            self._current_phys_condition_scale())
        return timing

    def _attach_infer_phys_memory(self, input_dict):
        if not self.use_phys_memory:
            return input_dict
        phys_source = self.active_phys_mem_feat
        frame_indices = self.active_phys_mem_frame_indices
        if phys_source is None and self.phys_memory_infer_mode == 'static':
            phys_source = self.static_phys_mem_feat
            frame_indices = None
        if phys_source is None:
            return input_dict
        phys_mem_feat = phys_source.to(
            device=input_dict['noisy_latents'].device,
            dtype=input_dict['noisy_latents'].dtype,
        )
        batch_size = input_dict['noisy_latents'].shape[0]
        if phys_mem_feat.shape[0] != batch_size:
            if phys_mem_feat.shape[0] != 1:
                raise ValueError(
                    f"Physics memory batch mismatch: got {phys_mem_feat.shape[0]}, expected {batch_size}"
                )
            phys_mem_feat = phys_mem_feat.repeat(batch_size, 1, 1)
        input_dict['phys_mem_feat'] = phys_mem_feat
        input_dict['phys_condition_scale'] = float(
            self._current_phys_condition_scale())
        if frame_indices is not None:
            frame_indices = frame_indices.to(device=input_dict['noisy_latents'].device)
            if frame_indices.dim() == 1:
                frame_indices = frame_indices[None]
            if frame_indices.shape[0] != batch_size:
                if frame_indices.shape[0] != 1:
                    raise ValueError(
                        "Physics memory frame-index batch mismatch: "
                        f"got {frame_indices.shape[0]}, expected {batch_size}"
                    )
                frame_indices = frame_indices.repeat(batch_size, 1)
            input_dict['phys_mem_frame_indices'] = frame_indices
        return input_dict

    def _clear_active_phys_memory(self):
        if self.phys_memory_infer_mode == 'phase':
            self._sync_active_phase_memory()
            return
        self.active_phys_mem_feat = None
        self.active_phys_mem_frame_indices = None

    def _reset_phase_phys_state(self, task_name=None):
        self.phase_phys_tokens = []
        self.phase_current_label = None
        self.phase_sequence_index = -1
        self.phase_candidate_label = None
        self.phase_candidate_count = 0
        self.phase_recent_obs = []
        self.phase_task_sequence = (
            self.online_phys_encoder.phase_sequence_for_task(task_name)
            if self.online_phys_encoder is not None else ()
        )
        self.active_phys_mem_feat = None
        self.active_phys_mem_frame_indices = None
        self.active_phys_confidence = 0.0
        if self.phase_task_sequence:
            logger.info(
                "[PhyWAM] task phase sequence "
                f"task={task_name} sequence={list(self.phase_task_sequence)} "
                f"max_tokens={self.phys_memory_max_tokens} "
                f"buffer_frames={self.phys_phase_buffer_frames}"
            )

    def _sync_active_phase_memory(self):
        if not self.phase_phys_tokens:
            self.active_phys_mem_feat = None
            self.active_phys_mem_frame_indices = None
            self.active_phys_confidence = 0.0
            return
        self.active_phys_mem_feat = torch.stack(
            self.phase_phys_tokens[-self.phys_memory_max_tokens:],
            dim=0,
        )[None].to(self.device)
        self.active_phys_mem_frame_indices = None

    def _phase_transition_target(self, predicted_label, confidence):
        if not predicted_label or predicted_label == 'no_event':
            self.phase_candidate_label = None
            self.phase_candidate_count = 0
            return None
        if confidence < self.online_phys_encoder.event_threshold:
            self.phase_candidate_label = None
            self.phase_candidate_count = 0
            return None
        if self.phase_current_label is None:
            if self.phase_task_sequence:
                return self.phase_task_sequence[0]
            return predicted_label
        if predicted_label == self.phase_current_label:
            self.phase_candidate_label = None
            self.phase_candidate_count = 0
            return None

        target = predicted_label
        if self.phase_task_sequence:
            next_index = self.phase_sequence_index + 1
            if next_index >= len(self.phase_task_sequence):
                self.phase_candidate_label = None
                self.phase_candidate_count = 0
                return None
            target = self.phase_task_sequence[next_index]
            if predicted_label != target:
                self.phase_candidate_label = None
                self.phase_candidate_count = 0
                return None

        if self.phase_candidate_label == target:
            self.phase_candidate_count += 1
        else:
            self.phase_candidate_label = target
            self.phase_candidate_count = 1
        if self.phase_candidate_count < self.phys_phase_switch_patience:
            return None
        self.phase_candidate_label = None
        self.phase_candidate_count = 0
        return target

    def _update_phase_phys_memory(self, obs, real_update=False):
        start_t = time.perf_counter()
        timing = self._empty_phys_timing(real_update=real_update)
        timing['update_attempted'] = True
        timing['tokens_before'] = len(self.phase_phys_tokens)
        if not self.use_phys_memory:
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='disabled'))
        if self.online_phys_encoder is None:
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='missing_online_encoder'))

        obs_payload = obs.get('obs')
        if obs_payload is None:
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='empty_obs_for_phys_memory'))
        if not isinstance(obs_payload, list):
            obs_payload = [obs_payload]
        timing['input_obs_frames'] = len(obs_payload)
        timing['buffer_frames_before'] = len(self.phase_recent_obs)
        self.phase_recent_obs.extend(obs_payload)
        self.phase_recent_obs = self.phase_recent_obs[
            -self.phys_phase_buffer_frames:]
        timing['buffer_frames'] = len(self.phase_recent_obs)
        timing['mope_input_frames'] = int(self.online_phys_encoder.num_frames)

        encode_t0 = time.perf_counter()
        encoded = self.online_phys_encoder.encode_phase(self.phase_recent_obs)
        timing['encode_s'] += time.perf_counter() - encode_t0
        if encoded is None:
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='empty_obs_for_phys_memory'))
        feature, event = encoded
        predicted_label = (event or {}).get('label')
        confidence = float((event or {}).get('confidence', 0.0))
        confident_same_phase = (
            self.phase_current_label is not None
            and predicted_label == self.phase_current_label
            and confidence >= self.online_phys_encoder.event_threshold
        )
        target_label = self._phase_transition_target(
            predicted_label, confidence)

        transition = False
        update_action = 'hold'
        update_reason = 'phase_not_changed'
        if self.phase_current_label is None:
            target_label = (
                self.phase_task_sequence[0]
                if self.phase_task_sequence
                else target_label or predicted_label or 'unknown_phase'
            )
            self.phase_current_label = target_label
            self.phase_sequence_index = (
                0 if self.phase_task_sequence
                and target_label == self.phase_task_sequence[0] else -1
            )
            self.phase_phys_tokens.append(feature)
            transition = True
            update_action = 'initialize'
            update_reason = 'first_phase_token'
        elif target_label is not None:
            self.phase_current_label = target_label
            if self.phase_task_sequence:
                self.phase_sequence_index += 1
            self.phase_recent_obs = list(obs_payload)[
                -self.phys_phase_buffer_frames:]
            transition_encode_t0 = time.perf_counter()
            transition_encoded = self.online_phys_encoder.encode_phase(
                self.phase_recent_obs)
            timing['encode_s'] += (
                time.perf_counter() - transition_encode_t0)
            if transition_encoded is not None:
                feature = transition_encoded[0]
            self.phase_phys_tokens.append(feature)
            transition = True
            update_action = 'transition'
            update_reason = 'confirmed_next_phase'
        elif confident_same_phase:
            # Refresh the active phase from the latest real observations. Older
            # phase tokens stay frozen once a transition has been confirmed.
            self.phase_phys_tokens[-1] = feature
            update_action = 'refresh'
            update_reason = 'confident_same_phase'
        elif not predicted_label or predicted_label == 'no_event':
            update_reason = 'no_valid_phase_prediction'
        elif confidence < self.online_phys_encoder.event_threshold:
            update_reason = 'below_confidence_threshold'
        else:
            update_reason = 'illegal_or_unconfirmed_transition'

        if len(self.phase_phys_tokens) > self.phys_memory_max_tokens:
            self.phase_phys_tokens = self.phase_phys_tokens[
                -self.phys_memory_max_tokens:]
        if update_action in ('initialize', 'refresh', 'transition'):
            self.active_phys_confidence = float(np.clip(confidence, 0.0, 1.0))
        self._sync_active_phase_memory()

        timing['built'] = True
        timing['triggered'] = transition
        timing['num_blocks'] = 1
        timing['selected_blocks'] = 1
        timing['phys_tokens'] = len(self.phase_phys_tokens)
        timing['tokens_after'] = len(self.phase_phys_tokens)
        timing['condition_applied'] = bool(self.phase_phys_tokens)
        timing['token_updated'] = update_action in (
            'initialize', 'refresh', 'transition')
        timing['token_appended'] = update_action in ('initialize', 'transition')
        timing['token_replaced'] = update_action == 'refresh'
        timing['phase_label'] = self.phase_current_label
        timing['phase_predicted_label'] = predicted_label or ''
        timing['phase_confidence'] = confidence
        timing['phase_update_action'] = update_action
        timing['phase_update_reason'] = update_reason
        self._annotate_phys_condition_scale(timing)
        logger.info(
            "[PhyWAM Phase] "
            f"current={self.phase_current_label} predicted={predicted_label} "
            f"confidence={confidence:.3f} action={update_action} "
            f"reason={update_reason} updated={timing['token_updated']} "
            f"scale={timing['phys_condition_scale']:.3f} "
            f"frames=input:{timing['input_obs_frames']},"
            f"buffer:{timing['buffer_frames']},"
            f"mope:{timing['mope_input_frames']} "
            f"tokens={timing['tokens_before']}->{timing['tokens_after']}"
        )
        event_summary = {
            'label': predicted_label,
            'confidence': confidence,
            'transition': transition,
            'current_label': self.phase_current_label,
        }
        return dict(
            events=[event_summary],
            selected=[len(self.phase_phys_tokens) - 1],
            timing=self._finish_phys_timing(timing, start_t),
        )

    def _empty_phys_timing(self, real_update=False):
        return {
            'enabled': bool(self.use_phys_memory),
            'mode': str(self.phys_memory_infer_mode),
            'detector': str(self.phys_event_detector),
            'real_update': bool(real_update),
            'update_attempted': False,
            'condition_applied': False,
            'token_updated': False,
            'token_appended': False,
            'token_replaced': False,
            'triggered': False,
            'built': False,
            'skipped': False,
            'skip_reason': '',
            'num_blocks': 0,
            'selected_blocks': 0,
            'phys_tokens': 0,
            'tokens_before': 0,
            'tokens_after': 0,
            'input_obs_frames': 0,
            'buffer_frames_before': 0,
            'buffer_frames': 0,
            'mope_input_frames': 0,
            'phase_label': '',
            'phase_predicted_label': '',
            'phase_confidence': 0.0,
            'phase_confidence_active': 0.0,
            'phys_task_gate': 1.0,
            'phys_condition_scale': 0.0,
            'phase_update_action': '',
            'phase_update_reason': '',
            'detector_s': 0.0,
            'encode_s': 0.0,
            'select_s': 0.0,
            'total_s': 0.0,
        }

    def _finish_phys_timing(self, timing, start_t, skip_reason=''):
        timing['total_s'] = time.perf_counter() - start_t
        if skip_reason:
            timing['skipped'] = True
            timing['skip_reason'] = skip_reason
        self._last_phys_timing = timing
        logger.info(
            "[PhyWAM Timing] "
            f"mode={timing['mode']} enabled={timing['enabled']} "
            f"detector={timing['detector']} real_update={timing['real_update']} "
            f"attempted={timing['update_attempted']} "
            f"condition_applied={timing['condition_applied']} "
            f"token_updated={timing['token_updated']} "
            f"action={timing['phase_update_action']} "
            f"reason={timing['phase_update_reason']} "
            f"triggered={timing['triggered']} built={timing['built']} "
            f"skipped={timing['skipped']} skip_reason={timing['skip_reason']} "
            f"blocks={timing['num_blocks']} selected={timing['selected_blocks']} "
            f"frames=input:{timing['input_obs_frames']},"
            f"buffer:{timing['buffer_frames']},"
            f"mope:{timing['mope_input_frames']} "
            f"tokens={timing['tokens_before']}->{timing['tokens_after']} "
            f"scale={float(timing.get('phys_condition_scale', 0.0) or 0.0):.3f} "
            f"detector_s={timing['detector_s']:.4f} "
            f"encode_s={timing['encode_s']:.4f} "
            f"select_s={timing['select_s']:.4f} "
            f"total_s={timing['total_s']:.4f}"
        )
        return timing

    def _phase_phys_reuse_timing(self):
        timing = self._empty_phys_timing(real_update=False)
        token_count = len(self.phase_phys_tokens)
        timing.update({
            'condition_applied': token_count > 0,
            'phys_tokens': token_count,
            'tokens_before': token_count,
            'tokens_after': token_count,
            'buffer_frames_before': len(self.phase_recent_obs),
            'buffer_frames': len(self.phase_recent_obs),
            'mope_input_frames': (
                int(self.online_phys_encoder.num_frames)
                if self.online_phys_encoder is not None else 0
            ),
            'phase_label': self.phase_current_label or '',
            'phase_confidence': float(
                np.clip(self.active_phys_confidence, 0.0, 1.0)),
            'phase_update_action': 'reuse',
            'phase_update_reason': 'action_infer_reuses_existing_tokens',
        })
        self._annotate_phys_condition_scale(timing)
        return timing

    @staticmethod
    def _phys_timing_log_suffix(timing):
        if not isinstance(timing, dict):
            return " phys=none"
        return (
            f" phys_action={timing.get('phase_update_action') or 'none'}"
            f" attempted={int(bool(timing.get('update_attempted', False)))}"
            f" encoded={int(bool(timing.get('built', False)))}"
            f" updated={int(bool(timing.get('token_updated', False)))}"
            f" frames={int(timing.get('input_obs_frames', 0) or 0)}/"
            f"{int(timing.get('buffer_frames', 0) or 0)}/"
            f"{int(timing.get('mope_input_frames', 0) or 0)}"
            f" tokens={int(timing.get('tokens_before', 0) or 0)}"
            f"->{int(timing.get('tokens_after', 0) or 0)}"
            f" phase={timing.get('phase_predicted_label') or '-'}"
            f"->{timing.get('phase_label') or '-'}"
            f" confidence={float(timing.get('phase_confidence', 0.0) or 0.0):.3f}"
            f" scale={float(timing.get('phys_condition_scale', 0.0) or 0.0):.3f}"
            f" encode_s={float(timing.get('encode_s', 0.0) or 0.0):.4f}"
            f" phys_total_s={float(timing.get('total_s', 0.0) or 0.0):.4f}"
            f" reason={timing.get('phase_update_reason') or '-'}"
        )

    def _reset_manual_phys_event_state(self):
        self._manual_gripper_state = {}

    def _detect_gripper_close_event(self, gid, series):
        finite = series[np.isfinite(series)]
        if finite.size < 1:
            return None

        start_value = float(finite[0])
        end_value = float(finite[-1])
        min_value = float(np.min(finite))
        max_value = float(np.max(finite))
        delta = max_value - min_value

        state = self._manual_gripper_state.setdefault(
            gid, {'armed': False, 'last_value': None})
        if min_value <= self.phys_gripper_open_threshold:
            state['armed'] = True

        closed_now = end_value >= self.phys_gripper_closed_threshold
        if state['armed'] and closed_now:
            state['armed'] = False
            state['last_value'] = end_value
            return (
                f'gripper_{gid}_close_event={start_value:.3f}->{end_value:.3f}'
                f'_delta={delta:.3f}'
            )

        state['last_value'] = end_value
        return None

    def _detect_manual_phys_event(self, obs):
        reasons = []
        action = obs.get('state')
        if action is not None:
            action = np.asarray(action)
            if action.ndim >= 1:
                if action.shape[0] >= 16:
                    gripper_ids = [7, 15]
                elif action.shape[0] >= 14:
                    gripper_ids = [6, 13]
                else:
                    gripper_ids = []
                for gid in gripper_ids:
                    series = action[gid].reshape(-1)
                    reason = self._detect_gripper_close_event(gid, series)
                    if reason is not None:
                        reasons.append(reason)

        obs_frames = obs.get('obs')
        if self.phys_enable_image_delta_event and obs_frames is not None:
            if not isinstance(obs_frames, list):
                obs_frames = [obs_frames]
            cam_key = getattr(
                self.online_phys_encoder,
                'camera_key',
                self.job_config.obs_cam_keys[0],
            )
            if len(obs_frames) >= 2 and cam_key in obs_frames[0] and cam_key in obs_frames[-1]:
                first = np.asarray(obs_frames[0][cam_key], dtype=np.float32)
                last = np.asarray(obs_frames[-1][cam_key], dtype=np.float32)
                if first.shape == last.shape:
                    image_delta = float(np.mean(np.abs(last - first)) / 255.0)
                    if image_delta >= self.phys_image_event_threshold:
                        reasons.append(f'image_delta={image_delta:.3f}')

        return bool(reasons), reasons

    def _prepare_online_phys_memory(self, obs, real_update=False):
        if self.phys_memory_infer_mode == 'phase':
            return self._update_phase_phys_memory(
                obs, real_update=real_update)
        start_t = time.perf_counter()
        timing = self._empty_phys_timing(real_update=real_update)
        self._clear_active_phys_memory()
        if (not self.use_phys_memory) or self.phys_memory_infer_mode not in ('always', 'event'):
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='disabled_or_mode_none'))
        if self.online_phys_encoder is None:
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='missing_online_encoder'))

        use_mope_event_detector = (
            self.phys_memory_infer_mode == 'event'
            and real_update
            and self.phys_event_detector == 'mope'
        )
        if self.phys_memory_infer_mode == 'event' and real_update:
            if self.phys_event_detector == 'manual':
                detector_t0 = time.perf_counter()
                triggered, reasons = self._detect_manual_phys_event(obs)
                timing['detector_s'] += time.perf_counter() - detector_t0
                timing['triggered'] = bool(triggered)
                if not triggered:
                    logger.info(
                        f"PhyWAM event memory skipped by manual detector: reasons={reasons}"
                    )
                    return dict(
                        events=[],
                        selected=[],
                        timing=self._finish_phys_timing(
                            timing, start_t, skip_reason='manual_event_not_triggered'))
                logger.info(
                    f"PhyWAM event memory triggered by manual detector: reasons={reasons}"
                )

        encode_t0 = time.perf_counter()
        encoded = self.online_phys_encoder.encode(
            obs.get('obs'), with_event=use_mope_event_detector)
        timing['encode_s'] += time.perf_counter() - encode_t0
        if encoded is None:
            return dict(
                events=[],
                selected=[],
                timing=self._finish_phys_timing(
                    timing, start_t, skip_reason='empty_obs_for_phys_memory'))
        feat, events = encoded
        timing['num_blocks'] = int(feat.shape[0])
        selected_indices = None
        event_summaries = []

        if use_mope_event_detector:
            select_t0 = time.perf_counter()
            selected_indices, event_summaries = self.online_phys_encoder.select_event_blocks(events)
            timing['select_s'] += time.perf_counter() - select_t0
            timing['triggered'] = bool(selected_indices)
            if not selected_indices:
                logger.info(
                    f"PhyWAM event memory skipped: events={event_summaries}"
                )
                return dict(
                    events=event_summaries,
                    selected=[],
                    timing=self._finish_phys_timing(
                        timing, start_t, skip_reason='mope_event_not_triggered'))
            feat = feat[selected_indices]
            self.active_phys_mem_frame_indices = torch.tensor(
                selected_indices, dtype=torch.long, device=self.device)
            logger.info(
                "PhyWAM event memory triggered: "
                f"selected={selected_indices}, events={event_summaries}"
            )
        else:
            if use_mope_event_detector:
                select_t0 = time.perf_counter()
                _, event_summaries = self.online_phys_encoder.select_event_blocks(events)
                timing['select_s'] += time.perf_counter() - select_t0
            selected_indices = list(range(feat.shape[0]))

        self.active_phys_mem_feat = feat[None].to(self.device)
        timing['built'] = True
        timing['selected_blocks'] = int(len(selected_indices or []))
        timing['phys_tokens'] = int(self.active_phys_mem_feat.shape[1])
        timing['condition_applied'] = True
        self._annotate_phys_condition_scale(timing)
        return dict(
            events=event_summaries,
            selected=selected_indices,
            timing=self._finish_phys_timing(timing, start_t))

    def _get_t5_prompt_embeds(
        self,
        prompt=None,
        num_videos_per_prompt=1,
        max_sequence_length=512,
        device=None,
        dtype=None,
    ):
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        text_encoder_device = next(self.text_encoder.parameters()).device
        prompt_embeds = self.text_encoder(text_input_ids.to(text_encoder_device),
                                          mask.to(text_encoder_device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack([
            torch.cat(
                [u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))])
            for u in prompt_embeds
        ],
                                    dim=0)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt,
                                           seq_len, -1)

        return prompt_embeds.to(device)

    def encode_prompt(
        self,
        prompt,
        negative_prompt=None,
        do_classifier_free_guidance=True,
        num_videos_per_prompt=1,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        max_sequence_length=226,
        device=None,
        dtype=None,
    ):
        r"""
        TODO
        """
        device = device or self.device
        dtype = dtype or self.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(
                negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(
                    negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}.")
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`.")

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )
        return prompt_embeds, negative_prompt_embeds

    def normalize_latents(
        self,
        latents: torch.Tensor,
        latents_mean: torch.Tensor,
        latents_std: torch.Tensor,
    ) -> torch.Tensor:
        latents_mean = latents_mean.view(1, -1, 1, 1,
                                         1).to(device=latents.device)
        latents_std = latents_std.view(1, -1, 1, 1,
                                       1).to(device=latents.device)
        latents = ((latents.float() - latents_mean) * latents_std).to(latents)
        return latents

    def preprocess_action(self, action):
        action_model_input = torch.from_numpy(action)
        CA, FA, HA = action_model_input.shape  # C, F, H
        action_model_input_paded = F.pad(action_model_input,
                                         [0, 0, 0, 0, 0, 1],
                                         mode='constant',
                                         value=0)

        action_model_input = action_model_input_paded[
            self.job_config.inverse_used_action_channel_ids]

        if self.action_norm_method == 'quantiles':
            action_model_input = (action_model_input - self.actions_q01) / (
                self.actions_q99 - self.actions_q01 + 1e-6) * 2. - 1.
        else:
            raise NotImplementedError
        return action_model_input.unsqueeze(0).unsqueeze(-1)  # B, C, F, H, W

    def postprocess_action(self, action):
        action = action.cpu()  # B, C, F, H, W

        action = action[0, ..., 0]  #C, F, H
        if self.action_norm_method == 'quantiles':
            action = (action + 1) / 2 * (self.actions_q99 - self.actions_q01 +
                                         1e-6) + self.actions_q01
        else:
            raise NotImplementedError
        action = action.squeeze(0).detach().cpu().numpy()
        return action[self.job_config.used_action_channel_ids]
    
    def _repeat_input_for_cfg(self, input_dict):
        if self.use_cfg:
            input_dict['noisy_latents'] = input_dict['noisy_latents'].repeat(2, 1, 1, 1, 1)
            input_dict['text_emb'] = torch.cat([self.prompt_embeds.to(self.dtype).clone(), self.negative_prompt_embeds.to(self.dtype).clone()], dim=0)
            input_dict['grid_id'] = input_dict['grid_id'][None].repeat(2, 1, 1)
            input_dict['timesteps'] = input_dict['timesteps'][None].repeat(2, 1)
        else:
            input_dict['grid_id'] = input_dict['grid_id'][None]
            input_dict['timesteps'] = input_dict['timesteps'][None]
        return self._attach_infer_phys_memory(input_dict)

    def _prepare_latent_input(self,
                              latent_model_input,
                              action_model_input,
                              latent_t=0,
                              action_t=0,
                              latent_cond=None,
                              action_cond=None,
                              frame_st_id=0,
                              patch_size=(1, 2, 2)):
        if frame_st_id not in self._logged_frame_start_ids:
            logger.info(f"FRAME START ID: {frame_st_id}")
            self._logged_frame_start_ids.add(frame_st_id)
        input_dict = dict()
        if latent_model_input is not None:
            input_dict['latent_res_lst'] = {
                'noisy_latents':
                latent_model_input,
                'timesteps':
                torch.ones([latent_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * latent_t,
                'grid_id':
                get_mesh_id(latent_model_input.shape[-3] // patch_size[0],
                            latent_model_input.shape[-2] // patch_size[1],
                            latent_model_input.shape[-1] // patch_size[2], 0,
                            1, frame_st_id).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }
            if latent_cond is not None:
                input_dict['latent_res_lst'][
                    'noisy_latents'][:, :, 0:1] = latent_cond[:, :, 0:1]
                input_dict['latent_res_lst']['timesteps'][0:1] *= 0

        if action_model_input is not None:
            input_dict['action_res_lst'] = {
                'noisy_latents':
                action_model_input,
                'timesteps':
                torch.ones([action_model_input.shape[2]],
                           dtype=torch.float32,
                           device=self.device) * action_t,
                'grid_id':
                get_mesh_id(action_model_input.shape[-3],
                            action_model_input.shape[-2],
                            action_model_input.shape[-1],
                            1,
                            1,
                            frame_st_id,
                            action=True).to(self.device),
                'text_emb':
                self.prompt_embeds.to(self.dtype).clone(),
            }

            if action_cond is not None:
                input_dict['action_res_lst'][
                    'noisy_latents'][:, :, 0:1] = action_cond[:, :, 0:1]
                input_dict['action_res_lst']['timesteps'][0:1] *= 0
            input_dict['action_res_lst']['noisy_latents'][:, ~self.
                                                          action_mask] *= 0
        return input_dict

    def _encode_obs(self, obs):
        images = obs['obs']
        if not isinstance(images, list):
            images = [images]
        if len(images) < 1:
            return None
        videos = []
        for k_i, k in enumerate(self.job_config.obs_cam_keys):
            if self.env_type == 'robotwin_tshape':
                if k_i == 0:  # camera high
                    height_i, width_i = self.height, self.width
                else:
                    height_i, width_i = self.height // 2, self.width // 2
            else:
                height_i, width_i = self.height, self.width

            history_video_k = torch.from_numpy(
                np.stack([each[k]
                          for each in images])).float().permute(3, 0, 1, 2)
            history_video_k = F.interpolate(history_video_k,
                                            size=(height_i, width_i),
                                            mode='bilinear',
                                            align_corners=False).unsqueeze(0)
            videos.append(history_video_k)

        if self.env_type == 'robotwin_tshape':
            videos_high = videos[0] / 255.0 * 2.0 - 1.0
            videos_left_and_right = torch.cat(videos[1:],
                                              dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            enc_out_high = self.streaming_vae.encode_chunk(
                videos_high.to(vae_device).to(self.dtype))
            enc_out_left_and_right = self.streaming_vae_half.encode_chunk(
                videos_left_and_right.to(vae_device).to(self.dtype))
            enc_out = torch.cat([
                torch.cat(enc_out_left_and_right.split(1, dim=0), dim=-1),
                enc_out_high
            ],
                                dim=-2)
        else:
            videos = torch.cat(videos, dim=0) / 255.0 * 2.0 - 1.0
            vae_device = next(self.streaming_vae.vae.parameters()).device
            videos_chunk = videos.to(vae_device).to(self.dtype)
            enc_out = self.streaming_vae.encode_chunk(videos_chunk)

        mu, logvar = torch.chunk(enc_out, 2, dim=1)
        latents_mean = torch.tensor(self.vae.config.latents_mean).to(mu.device)
        latents_std = torch.tensor(self.vae.config.latents_std).to(mu.device)
        mu_norm = self.normalize_latents(mu, latents_mean, 1.0 / latents_std)
        video_latent = torch.cat(mu_norm.split(1, dim=0), dim=-1)
        return video_latent.to(self.device)

    @staticmethod
    def _sanitize_name(value, default):
        if value is None:
            return default
        safe = ''.join(ch if ch.isalnum() or ch in ('-', '_') else '_' for ch in str(value))
        return safe.strip('_') or default

    def _reset(self, prompt=None, task_name=None):
        logger.info('Reset.')
        self.current_task_name = task_name
        self.use_cfg = (self.job_config.guidance_scale > 1) or (self.job_config.action_guidance_scale > 1)
        #### Reset all parameters
        self.frame_st_id = 0
        self.init_latent = None
        self._last_phys_timing = None
        self._reset_manual_phys_event_state()
        self._reset_phase_phys_state(task_name=task_name)
        #### clean vae and transformer cache
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()

        self.action_per_frame = self.job_config.action_per_frame
        self.height, self.width = self.job_config.height, self.job_config.width

        if self.env_type == 'robotwin_tshape':
            self.latent_height, self.latent_width = (
                (self.height // 16) * 3) // 2, self.width // 16
            self.streaming_vae_half.clear_cache()
        else:
            self.latent_height, self.latent_width = self.height // 16, self.width // 16 * len(
                self.job_config.obs_cam_keys)

        patch_size = self.job_config.patch_size
        latent_token_per_chunk = (self.job_config.frame_chunk_size *
                                  self.latent_height * self.latent_width) // (
                                      patch_size[0] * patch_size[1] *
                                      patch_size[2])
        action_token_per_chunk = self.job_config.frame_chunk_size * self.action_per_frame
        self.transformer.create_empty_cache(self.cache_name,
                                            self.job_config.attn_window,
                                            latent_token_per_chunk,
                                            action_token_per_chunk,
                                            dtype=self.dtype,
                                            device=self.device,
                                            batch_size = 2 if self.use_cfg else 1,
                                            phys_token_per_chunk=0,
                                            )

        self.action_mask = torch.zeros([self.job_config.action_dim]).bool()
        self.action_mask[self.job_config.used_action_channel_ids] = True

        self.actions_q01 = torch.tensor(self.job_config.norm_stat['q01'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.tensor(self.job_config.norm_stat['q99'],
                                        dtype=torch.float32).reshape(-1, 1, 1)
        self.action_norm_method = self.job_config.action_norm_method

        ##### get prompt
        if prompt is None:
            self.prompt_embeds = self.negative_prompt_embeds = None
        else:
            self.prompt_embeds, self.negative_prompt_embeds = self.encode_prompt(
                prompt=prompt,
                negative_prompt=None,
                do_classifier_free_guidance=self.job_config.guidance_scale > 1,
                num_videos_per_prompt=1,
                prompt_embeds=None,
                negative_prompt_embeds=None,
                max_sequence_length=512,
                device=self.device,
                dtype=self.dtype,
            )

        task_dir_name = self._sanitize_name(task_name, "unknown_task")
        prompt_name = self._sanitize_name(prompt, "default")
        self.exp_name = f"{prompt_name}_{time.strftime('%Y%m%d_%H%M%S')}"
        self.exp_save_root = os.path.join(
            self.save_root, 'real', task_dir_name, self.exp_name)
        os.makedirs(self.exp_save_root, exist_ok=True)
        torch.cuda.empty_cache()

    def _infer(self, obs, frame_st_id=0):
        frame_chunk_size = self.job_config.frame_chunk_size
        if frame_st_id == 0:
            init_latent = self._encode_obs(obs)
            self.init_latent = init_latent

        latents = torch.randn(1,
                              48,
                              frame_chunk_size,
                              self.latent_height,
                              self.latent_width,
                              device=self.device,
                              dtype=self.dtype)
        actions = torch.randn(1,
                              self.job_config.action_dim,
                              frame_chunk_size,
                              self.action_per_frame,
                              1,
                              device=self.device,
                              dtype=self.dtype)

        video_inference_step = self.job_config.num_inference_steps
        action_inference_step = self.job_config.action_num_inference_steps
        video_step = self.job_config.video_exec_step

        self.scheduler.set_timesteps(video_inference_step)
        self.action_scheduler.set_timesteps(action_inference_step)
        timesteps = self.scheduler.timesteps
        action_timesteps = self.action_scheduler.timesteps

        timesteps = F.pad(timesteps, (0, 1), mode='constant', value=0)

        if video_step != -1:
            timesteps = timesteps[:video_step]

        action_timesteps = F.pad(
            action_timesteps,
            (0,
             1),  # pad 1 element at the end (right side) of the last dimension
            mode='constant',
            value=0)

        with (
                torch.no_grad(),
        ):
            # 1. Video Generation Loop
            for i, t in enumerate(tqdm(timesteps)):
                last_step = i == len(timesteps) - 1
                latent_cond = init_latent[:, :, 0:1].to(
                    self.dtype) if frame_st_id == 0 else None
                input_dict = self._prepare_latent_input(
                    latents,
                    None,
                    t,
                    t,
                    latent_cond,
                    None,
                    frame_st_id=frame_st_id)

                video_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=False)

                if not last_step or video_step != -1:
                    video_noise_pred = data_seq_to_patch(
                        self.job_config.patch_size, video_noise_pred,
                        frame_chunk_size, self.latent_height,
                        self.latent_width, batch_size=2 if self.use_cfg else 1)
                    if self.job_config.guidance_scale > 1:
                        video_noise_pred = video_noise_pred[1:] + self.job_config.guidance_scale * (video_noise_pred[:1] - video_noise_pred[1:])
                    else:
                        video_noise_pred = video_noise_pred[:1]
                    latents = self.scheduler.step(video_noise_pred,
                                                  t,
                                                  latents,
                                                  return_dict=False)

                latents[:, :, 0:1] = latent_cond if frame_st_id == 0 else latents[:, :, 0:1]

            for i, t in enumerate(tqdm(action_timesteps)):
                last_step = i == len(action_timesteps) - 1
                action_cond = torch.zeros(
                    [
                        1, self.job_config.action_dim, 1,
                        self.action_per_frame, 1
                    ],
                    device=self.device,
                    dtype=self.dtype) if frame_st_id == 0 else None

                input_dict = self._prepare_latent_input(
                    None,
                    actions,
                    t,
                    t,
                    None,
                    action_cond,
                    frame_st_id=frame_st_id)
                action_noise_pred = self.transformer(
                    self._repeat_input_for_cfg(input_dict['action_res_lst']),
                    update_cache=1 if last_step else 0,
                    cache_name=self.cache_name,
                    action_mode=True)

                if not last_step:
                    action_noise_pred = rearrange(action_noise_pred,
                                                  'b (f n) c -> b c f n 1',
                                                  f=frame_chunk_size)
                    if self.job_config.action_guidance_scale > 1:
                        action_noise_pred = action_noise_pred[1:] + self.job_config.action_guidance_scale * (action_noise_pred[:1] - action_noise_pred[1:])
                    else:
                        action_noise_pred = action_noise_pred[:1]
                    actions = self.action_scheduler.step(action_noise_pred,
                                                         t,
                                                         actions,
                                                         return_dict=False)

                actions[:, :, 0:1] = action_cond if frame_st_id == 0 else actions[:, :, 0:1]

        actions[:, ~self.action_mask] *= 0

        save_async(latents, os.path.join(self.exp_save_root, f'latents_{frame_st_id}.pt'))
        save_async(actions, os.path.join(self.exp_save_root, f'actions_{frame_st_id}.pt'))

        actions = self.postprocess_action(actions)
        torch.cuda.empty_cache()
        self._clear_active_phys_memory()
        return actions, latents

    def _compute_kv_cache(self, obs):
        request_t0 = time.perf_counter()
        ### optional async save obs for debug
        self.transformer.clear_pred_cache(self.cache_name)
        phys_info = self._prepare_online_phys_memory(obs, real_update=True)
        save_async(obs['obs'], os.path.join(self.exp_save_root, f'obs_data_{self.frame_st_id}.pt'))
        latent_model_input = self._encode_obs(obs)
        if self.frame_st_id == 0:
            latent_model_input = torch.cat(
                [self.init_latent, latent_model_input],
                dim=2) if latent_model_input is not None else self.init_latent

        action_model_input = self.preprocess_action(obs['state'])
        action_model_input = action_model_input.to(latent_model_input)
        logger.info(
            f"get KV cache obs: {latent_model_input.shape} {action_model_input.shape}"
        )
        input_dict = self._prepare_latent_input(latent_model_input,
                                                action_model_input,
                                                frame_st_id=self.frame_st_id)

        with (
                torch.no_grad(),
        ):
            self.transformer(self._repeat_input_for_cfg(input_dict['latent_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=False)

            self.transformer(self._repeat_input_for_cfg(input_dict['action_res_lst']),
                             update_cache=2,
                             cache_name=self.cache_name,
                             action_mode=True)
        torch.cuda.empty_cache()
        self._clear_active_phys_memory()
        self.frame_st_id += latent_model_input.shape[2]
        timing = {
            'request_type': 'kv_cache',
            'mode': str(self.phys_memory_infer_mode),
            'total_s': time.perf_counter() - request_t0,
            'phys': (phys_info or {}).get('timing', self._last_phys_timing),
        }
        logger.info(
            "[Server Timing] request=kv_cache "
            f"mode={timing['mode']} total_s={timing['total_s']:.4f}"
            f"{self._phys_timing_log_suffix(timing['phys'])}"
        )
        return timing

    @torch.no_grad()
    def infer(self, obs):
        request_t0 = time.perf_counter()
        reset = obs.get('reset', False)
        prompt = obs.get('prompt', None)
        task_name = obs.get('task_name', None)
        compute_kv_cache = obs.get('compute_kv_cache', False)

        if reset:
            logger.info(f"******************* Reset server ******************")
            self._reset(prompt=prompt, task_name=task_name)
            timing = {
                'request_type': 'reset',
                'mode': str(self.phys_memory_infer_mode),
                'total_s': time.perf_counter() - request_t0,
                'phys': None,
            }
            logger.info(
                "[Server Timing] request=reset "
                f"mode={timing['mode']} total_s={timing['total_s']:.4f}"
            )
            return dict(timing=timing)
        elif compute_kv_cache:
            logger.info(
                f"################# Compute KV Cache #################")
            timing = self._compute_kv_cache(obs)
            return dict(timing=timing)
        else:
            logger.info(f"################# Infer One Chunk #################")
            phys_info = None
            if (
                self.use_phys_memory
                and self.phys_memory_infer_mode == 'phase'
                and not self.phase_phys_tokens
            ):
                phys_info = self._prepare_online_phys_memory(
                    obs, real_update=True)
            action, _ = self._infer(obs, frame_st_id=self.frame_st_id)
            timing = {
                'request_type': 'action_infer',
                'mode': str(self.phys_memory_infer_mode),
                'total_s': time.perf_counter() - request_t0,
                'phys': (
                    (phys_info or {}).get('timing')
                    if phys_info is not None
                    else (
                        self._phase_phys_reuse_timing()
                        if self.phys_memory_infer_mode == 'phase'
                        else self._last_phys_timing
                    )
                ),
            }
            logger.info(
                "[Server Timing] request=action_infer "
                f"mode={timing['mode']} total_s={timing['total_s']:.4f}"
                f"{self._phys_timing_log_suffix(timing['phys'])}"
            )
            return dict(action=action, timing=timing)
    
    def decode_one_video(self, latents, output_type):
        latents = latents.to(self.vae.dtype)
        latents_mean = (
            torch.tensor(self.vae.config.latents_mean)
            .view(1, self.vae.config.z_dim, 1, 1, 1)
            .to(latents.device, latents.dtype)
        )
        latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(1, self.vae.config.z_dim, 1, 1, 1).to(
            latents.device, latents.dtype
        )
        latents = latents / latents_std + latents_mean
        video = self.vae.decode(latents, return_dict=False)[0]
        video = self.video_processor.postprocess_video(video, output_type=output_type)
        return video
    
    def load_init_obs(self):
        imf_dict = {v: np.array(Image.open(os.path.join(self.job_config.input_img_path, f"{v}.png")).convert("RGB")) for v in self.job_config.obs_cam_keys}
        init_obs = {}
        init_obs['obs'] = [imf_dict]
        return init_obs
    
    @torch.no_grad()
    def generate(self):
        self.video_processor = VideoProcessor(vae_scale_factor=1)
        self._reset(self.job_config.prompt)
        init_obs = self.load_init_obs()
        pred_latent_lst = []
        pred_action_lst = []
        for chunk_id in range(self.job_config.num_chunks_to_infer):
            actions, latents = self._infer(init_obs, frame_st_id=(chunk_id * self.job_config.frame_chunk_size))
            actions = torch.from_numpy(actions)
            pred_latent_lst.append(latents)
            pred_action_lst.append(actions)
        pred_latent = torch.cat(pred_latent_lst, dim=2)
        pred_action = torch.cat(pred_action_lst, dim=1).flatten(1)
        self.transformer.clear_cache(self.cache_name)
        self.streaming_vae.clear_cache()
        if self.streaming_vae_half:
            self.streaming_vae_half.clear_cache()
        del self.transformer
        del self.streaming_vae_half
        del self.text_encoder
        torch.cuda.empty_cache()
        
        # Move VAE to GPU for decoding
        if self.enable_offload:
            self.vae = self.vae.to(self.device).to(self.dtype)
        
        decoded_video = self.decode_one_video(pred_latent, 'np')[0]
        export_to_video(decoded_video, os.path.join(self.save_root, "demo.mp4"), fps=10)

def run(args):    
    
    config = VA_CONFIGS[args.config_name]
    port = config.port if args.port is None else args.port
    if args.save_root is not None:
        config.save_root = args.save_root
    if args.use_phys_memory:
        config.use_phys_memory = True
    if args.infer_phys_memory_npy is not None:
        config.infer_phys_memory_npy = args.infer_phys_memory_npy
    if args.phys_memory_dim is not None:
        config.phys_memory_dim = args.phys_memory_dim
    if args.phys_memory_block_size is not None:
        config.phys_memory_block_size = args.phys_memory_block_size
    if args.phys_memory_infer_mode is not None:
        config.phys_memory_infer_mode = args.phys_memory_infer_mode
    if args.mope_repo is not None:
        config.mope_repo = args.mope_repo
    if args.mope_ckpt is not None:
        config.mope_ckpt = args.mope_ckpt
    if args.phys_event_threshold is not None:
        config.phys_event_threshold = args.phys_event_threshold
    if args.phys_event_window_blocks is not None:
        config.phys_event_window_blocks = args.phys_event_window_blocks
    if args.phys_memory_camera_key is not None:
        config.phys_memory_camera_key = args.phys_memory_camera_key
    if args.phys_event_detector is not None:
        config.phys_event_detector = args.phys_event_detector
    if args.phys_event_label_path is not None:
        config.phys_event_label_path = args.phys_event_label_path
    if args.phys_phase_switch_patience is not None:
        config.phys_phase_switch_patience = args.phys_phase_switch_patience
    if args.phys_phase_buffer_frames is not None:
        config.phys_phase_buffer_frames = args.phys_phase_buffer_frames
    if args.phys_gripper_event_threshold is not None:
        config.phys_gripper_event_threshold = args.phys_gripper_event_threshold
    if args.phys_image_event_threshold is not None:
        config.phys_image_event_threshold = args.phys_image_event_threshold
    if args.enable_phys_image_delta_event:
        config.phys_enable_image_delta_event = True
    if args.phys_gripper_open_threshold is not None:
        config.phys_gripper_open_threshold = args.phys_gripper_open_threshold
    if args.phys_gripper_closed_threshold is not None:
        config.phys_gripper_closed_threshold = args.phys_gripper_closed_threshold
    if args.phys_phase_confidence_gate is not None:
        config.phys_use_phase_confidence_gate = args.phys_phase_confidence_gate
    if args.phys_default_task_gate is not None:
        config.phys_default_task_gate = args.phys_default_task_gate
    if args.phys_task_gates_json is not None:
        config.phys_task_gates = json.loads(args.phys_task_gates_json)

    model_root = getattr(config, 'wan22_pretrained_model_name_or_path', None)
    text_encoder_path = os.path.join(model_root, 'text_encoder') if model_root else None
    vae_path = os.path.join(model_root, 'vae') if model_root else None
    transformer_path = os.path.join(model_root, 'transformer') if model_root else None
    tokenizer_path = os.path.join(model_root, 'tokenizer') if model_root else None
    logger.info(f"[Startup] config_name={args.config_name}")
    logger.info(f"[Startup] wan22_pretrained_model_name_or_path={model_root}")
    logger.info(f"[Startup] text_encoder_path={text_encoder_path}")
    logger.info(f"[Startup] vae_path={vae_path}")
    logger.info(f"[Startup] transformer_path={transformer_path}")
    logger.info(f"[Startup] tokenizer_path={tokenizer_path}")
    logger.info(f"[Startup] use_phys_memory={getattr(config, 'use_phys_memory', False)}")
    logger.info(f"[Startup] infer_phys_memory_npy={getattr(config, 'infer_phys_memory_npy', None)}")
    logger.info(f"[Startup] phys_memory_infer_mode={getattr(config, 'phys_memory_infer_mode', None)}")
    logger.info(f"[Startup] mope_ckpt={getattr(config, 'mope_ckpt', None)}")
    logger.info(
        f"[Startup] phys_event_label_path="
        f"{getattr(config, 'phys_event_label_path', None)}")
    logger.info(
        "[Startup] manual_phys_event: "
        f"image_delta_enabled={getattr(config, 'phys_enable_image_delta_event', False)} "
        f"gripper_open_threshold={getattr(config, 'phys_gripper_open_threshold', 0.2)} "
        f"gripper_closed_threshold={getattr(config, 'phys_gripper_closed_threshold', 0.8)}"
    )
    logger.info(
        "[Startup] phys_scale_gate: "
        f"phase_confidence={getattr(config, 'phys_use_phase_confidence_gate', True)} "
        f"default_task_gate={getattr(config, 'phys_default_task_gate', 1.0)} "
        f"task_gates={getattr(config, 'phys_task_gates', {})}"
    )
    rank = int(os.getenv("RANK", 0))
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    init_distributed(world_size, local_rank, rank)
    config.rank = rank
    config.local_rank = local_rank
    config.world_size = world_size
    model = VA_Server(config)
    if config.infer_mode == 'i2va':
        logger.info(f"******************************USE I2AV mode******************************")
        model.generate()
    elif config.infer_mode == 'server':
        logger.info(f"******************************USE Server mode******************************")
        run_async_server_mode(model, local_rank, config.host, port)
    else:
        raise ValueError(f"Unknown infer mode: {config.infer_mode}")

def main():
    """
    TODO
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-name",
        type=str,
        required=False,
        default='robotwin',
        help="config name.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=None,
        help='(start) port'
    )
    parser.add_argument(
        "--save_root",
        type=str,
        default=None,
        help='save root'
    )
    parser.add_argument(
        "--use-phys-memory",
        action="store_true",
        help="Enable PhyWAM physics conditioning",
    )
    parser.add_argument(
        "--infer-phys-memory-npy",
        type=str,
        default=None,
        help="Physics memory npy/npz file for inference",
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
        help="Raw video frames per physics memory block",
    )
    parser.add_argument(
        "--phys-memory-infer-mode",
        type=str,
        default=None,
        choices=['none', 'static', 'always', 'event', 'phase'],
        help="How to update PhyWAM physics memory during inference",
    )
    parser.add_argument("--mope-repo", type=str, default=None)
    parser.add_argument("--mope-ckpt", type=str, default=None)
    parser.add_argument("--phys-event-threshold", type=float, default=None)
    parser.add_argument("--phys-event-window-blocks", type=int, default=None)
    parser.add_argument("--phys-memory-camera-key", type=str, default=None)
    parser.add_argument(
        "--phys-event-detector",
        type=str,
        default=None,
        choices=['manual', 'mope'],
    )
    parser.add_argument("--phys-event-label-path", type=str, default=None)
    parser.add_argument(
        "--phys-phase-switch-patience", type=int, default=None)
    parser.add_argument(
        "--phys-phase-buffer-frames", type=int, default=None)
    parser.add_argument("--phys-gripper-event-threshold", type=float, default=None)
    parser.add_argument("--phys-image-event-threshold", type=float, default=None)
    parser.add_argument(
        "--enable-phys-image-delta-event",
        action="store_true",
        help="Enable image-delta as a manual physics-memory event trigger",
    )
    parser.add_argument("--phys-gripper-open-threshold", type=float, default=None)
    parser.add_argument("--phys-gripper-closed-threshold", type=float, default=None)
    parser.add_argument(
        "--phys-phase-confidence-gate",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Multiply physics residuals by online phase confidence in phase mode.",
    )
    parser.add_argument("--phys-default-task-gate", type=float, default=None)
    parser.add_argument(
        "--phys-task-gates-json",
        type=str,
        default=None,
        help='JSON dict, for example {"hanging_mug": 0.5}.',
    )
    args = parser.parse_args()
    run(args)
    logger.info("Finish all process!!!!!!!!!!!!")


if __name__ == "__main__":
    init_logger()
    main()
