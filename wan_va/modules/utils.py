# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    use_phys_memory=None,
    phys_memory_dim=None,
    phys_zero_init=None,
    phys_gate=None,
    phys_cross_start_layer=None,
    attn_mode=None,
):
    load_kwargs = {}

    # Base LingBot-VA checkpoints do not contain PhyWAM modules. Passing
    # use_phys_memory=True into from_pretrained can leave those newly-added
    # parameters as meta tensors under diffusers' low-memory loading path.
    config_path = Path(transformer_path) / "config.json"
    checkpoint_has_phys_memory = False
    if config_path.exists():
        with config_path.open("r", encoding="utf-8") as f:
            checkpoint_config = json.load(f)
        checkpoint_has_phys_memory = bool(
            checkpoint_config.get("use_phys_memory", False))

    if use_phys_memory is not None and (
        checkpoint_has_phys_memory or not bool(use_phys_memory)
    ):
        load_kwargs["use_phys_memory"] = bool(use_phys_memory)
    if phys_memory_dim is not None and (
        checkpoint_has_phys_memory or not bool(use_phys_memory)
    ):
        load_kwargs["phys_memory_dim"] = int(phys_memory_dim)
    if phys_zero_init is not None and checkpoint_has_phys_memory:
        load_kwargs["phys_zero_init"] = bool(phys_zero_init)
    if phys_gate is not None and checkpoint_has_phys_memory:
        load_kwargs["phys_gate"] = bool(phys_gate)
    if phys_cross_start_layer is not None:
        load_kwargs["phys_cross_start_layer"] = int(phys_cross_start_layer)
    if attn_mode is not None:
        load_kwargs["attn_mode"] = str(attn_mode)
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **load_kwargs,
    )
    return model.to(torch_device)


def sync_transformer_phywam_config(
    model,
    use_phys_memory,
    phys_memory_dim,
    phys_memory_align_mode=None,
    phys_memory_max_tokens=None,
    phys_zero_init=None,
    phys_gate=None,
    phys_cross_start_layer=None,
):
    """Attach or remove the physics-memory projector after loading a base checkpoint."""
    updated = False
    target_use_phys_memory = bool(use_phys_memory)
    target_phys_memory_dim = int(phys_memory_dim)
    target_phys_zero_init = bool(
        getattr(model, "phys_zero_init", False)
        if phys_zero_init is None else phys_zero_init
    )
    target_phys_gate = bool(
        getattr(model, "phys_gate", False)
        if phys_gate is None else phys_gate
    )
    target_phys_cross_start_layer = int(
        getattr(model, "phys_cross_start_layer", 0)
        if phys_cross_start_layer is None else phys_cross_start_layer
    )
    if hasattr(model, "blocks"):
        target_phys_cross_start_layer = max(
            0, min(target_phys_cross_start_layer, len(model.blocks)))

    if getattr(model, "use_phys_memory", False) != target_use_phys_memory:
        model.use_phys_memory = target_use_phys_memory
        updated = True
    if getattr(model, "phys_memory_dim", None) != target_phys_memory_dim:
        model.phys_memory_dim = target_phys_memory_dim
        updated = True
    if getattr(model, "phys_zero_init", False) != target_phys_zero_init:
        model.phys_zero_init = target_phys_zero_init
        updated = True
    if getattr(model, "phys_gate", False) != target_phys_gate:
        model.phys_gate = target_phys_gate
        updated = True
    if getattr(model, "phys_cross_start_layer", 0) != target_phys_cross_start_layer:
        model.phys_cross_start_layer = target_phys_cross_start_layer
        updated = True

    if target_use_phys_memory:
        for block in model.blocks:
            block.phys_zero_init = target_phys_zero_init
            block.phys_gate_enabled = target_phys_gate
        inner_dim = int(model.num_attention_heads * model.attention_head_dim)
        projector = getattr(model, "phys_memory_projector", None)
        need_rebuild = (
            projector is None
            or not isinstance(projector, nn.Sequential)
            or len(projector) != 2
            or not isinstance(projector[0], nn.LayerNorm)
            or not isinstance(projector[1], nn.Linear)
            or projector[0].normalized_shape != (target_phys_memory_dim,)
            or projector[1].in_features != target_phys_memory_dim
            or projector[1].out_features != inner_dim
        )
        if need_rebuild:
            try:
                first_param = next(model.parameters())
                param_device = first_param.device
                param_dtype = first_param.dtype
            except StopIteration:
                param_device = torch.device("cpu")
                param_dtype = torch.float32
            model.phys_memory_projector = nn.Sequential(
                nn.LayerNorm(target_phys_memory_dim),
                nn.Linear(target_phys_memory_dim, inner_dim),
            ).to(device=param_device, dtype=param_dtype)
            model.phys_memory_type_embed = nn.Parameter(
                torch.randn(1, 1, inner_dim, device=param_device, dtype=param_dtype)
                / inner_dim**0.5
            )
            updated = True
        if hasattr(model, "set_phys_cross_start_layer"):
            model.set_phys_cross_start_layer(target_phys_cross_start_layer)
            updated = True
        elif hasattr(model, "enable_phys_cross_attention"):
            model.enable_phys_cross_attention()
            updated = True
    else:
        if getattr(model, "phys_memory_projector", None) is not None:
            model.phys_memory_projector = None
            model.phys_memory_type_embed = None
            updated = True
        if hasattr(model, "set_phys_cross_start_layer"):
            model.set_phys_cross_start_layer(target_phys_cross_start_layer)

    if hasattr(model, "register_to_config"):
        config_kwargs = dict(
            use_phys_memory=target_use_phys_memory,
            phys_memory_dim=target_phys_memory_dim,
            phys_zero_init=target_phys_zero_init,
            phys_gate=target_phys_gate,
            phys_cross_start_layer=target_phys_cross_start_layer,
        )
        if phys_memory_align_mode is not None:
            config_kwargs['phys_memory_align_mode'] = phys_memory_align_mode
        if phys_memory_max_tokens is not None:
            config_kwargs['phys_memory_max_tokens'] = int(phys_memory_max_tokens)
        model.register_to_config(**config_kwargs)
    return updated


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
