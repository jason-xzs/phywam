# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from typing import Callable, ClassVar
from torch.nn.attention.flex_attention import (
    _mask_mod_signature,
    BlockMask,
    create_block_mask,
    flex_attention,
    and_masks,
    or_masks
)
from functools import partial

try:
    from flash_attn_interface import flash_attn_func
except:
    try:
        from flash_attn import flash_attn_func
    except:
        flash_attn_func = None

__all__ = ['WanTransformer3DModel']


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2),
                                         v.transpose(1, 2))
    return out.transpose(1, 2)

class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention, dynamic=True, 
    )
    compiled_create_block_mask: ClassVar[Callable] = torch.compile(create_block_mask)
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None
    phys_cross_attention_mask: ClassVar[BlockMask] = None

    def __init__(
        self, 
        is_cross=False,
        cross_mask_name='text',
    ) -> None:
        super().__init__()
        self.is_cross = is_cross
        self.cross_mask_name = cross_mask_name
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes
        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)
        
        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        if self.is_cross:
            if self.cross_mask_name == 'phys':
                block_mask = FlexAttnFunc.phys_cross_attention_mask
            else:
                block_mask = FlexAttnFunc.cross_attention_mask
        else:
            block_mask = FlexAttnFunc.attention_mask

        x_out = FlexAttnFunc.flex_attn(q_varlen, k_varlen, v_varlen, block_mask=block_mask, kernel_options = {
                                                    "BLOCK_M": 64,
                                                    "BLOCK_N": 64,
                                                    "BLOCK_M1": 32,
                                                    "BLOCK_N1": 64,
                                                    "BLOCK_M2": 64,
                                                    "BLOCK_N2": 32,
                                                })

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape, 
        action_shape, 
        padded_length, 
        chunk_size,
        window_size,
        patch_size,
        text_seq_len,
        device,
        phys_shape=None,
        phys_cross_shape=None,
        phys_cross_spans=None,
        phys_cross_mask=None,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        B, _, L_F, L_H, L_W = latent_shape
        _, _, A_F, A_H, A_W = action_shape

        latent_seq_id = torch.arange(B)[:, None, None, None].\
            expand(-1, L_F // patch_size[0], L_H // patch_size[1], L_W // patch_size[2]).flatten()
        action_seq_id = torch.arange(B)[:, None, None, None].expand(-1, A_F, A_H, A_W).flatten()
        seq_ids = torch.cat([latent_seq_id] * 2 + [action_seq_id] * 2)

        latent_frame_id = torch.arange(L_F)[None, :, None, None].expand(B, -1, L_H // patch_size[1], L_W // patch_size[2])[None].flatten()
        action_frame_id = torch.arange(A_F)[None, :, None, None].expand(B, -1, A_H, A_W)[None].flatten()
        frame_ids = torch.cat([latent_frame_id // chunk_size * 2] * 2 + [action_frame_id // chunk_size * 2 + 1] * 2)

        noise_ids = torch.cat(
            [
                torch.zeros_like(latent_frame_id),
                torch.ones_like(latent_frame_id),
                torch.zeros_like(action_frame_id),
                torch.ones_like(action_frame_id),
            ]
        )

        if phys_shape is not None:
            _, P_F, _ = phys_shape
            phys_seq_id = torch.arange(B)[:, None].expand(-1, P_F).flatten()
            phys_frame_id = (
                torch.arange(P_F)[None, :].expand(B, -1).flatten()
                // chunk_size * 2
            )
            phys_noise_id = torch.full_like(phys_frame_id, 2)
            seq_ids = torch.cat([seq_ids, phys_seq_id])
            frame_ids = torch.cat([frame_ids, phys_frame_id])
            noise_ids = torch.cat([noise_ids, phys_noise_id])

        seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
        frame_ids = F.pad(frame_ids, (0, padded_length), value=-1)
        noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

        mask_mod = FlexAttnFunc._get_mask_mod(seq_ids.long().to(device), frame_ids.long().to(device), noise_ids.long().to(device), window_size)
        block_mask = FlexAttnFunc.compiled_create_block_mask(
                mask_mod, 1, 1, len(seq_ids), len(seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids = torch.arange(B)[:, None].expand(-1, text_seq_len).flatten()
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(seq_ids.long().to(device), text_seq_ids.long().to(device))
        block_mask_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_cross, 1, 1, len(seq_ids), len(text_seq_ids), device=device, _compile=True
            )
        FlexAttnFunc.cross_attention_mask = block_mask_cross

        if phys_cross_shape is not None:
            _, P_F, _ = phys_cross_shape
            phys_seq_ids = torch.arange(B)[:, None].expand(-1, P_F).flatten()
            if phys_cross_mask is None:
                phys_valid = torch.ones((B, P_F), dtype=torch.bool, device=device)
            else:
                phys_valid = phys_cross_mask.to(device=device, dtype=torch.bool)
            if phys_cross_spans is None:
                phys_start_ids = torch.zeros((B, P_F), dtype=torch.long, device=device)
                phys_end_ids = torch.full(
                    (B, P_F), fill_value=2**30, dtype=torch.long, device=device)
            else:
                spans = phys_cross_spans.to(device=device, dtype=torch.long)
                phys_start_ids = spans[..., 0] // chunk_size * 2
                phys_end_ids = ((spans[..., 1] - 1).clamp(min=0) // chunk_size * 2)
            phys_start_ids = phys_start_ids.flatten()
            phys_end_ids = phys_end_ids.flatten()
            phys_valid = phys_valid.flatten()
            mask_mod_phys_cross = FlexAttnFunc._get_phys_cross_mask_mod(
                seq_ids.long().to(device),
                frame_ids.long().to(device),
                phys_seq_ids.long().to(device),
                phys_start_ids.long().to(device),
                phys_end_ids.long().to(device),
                phys_valid.to(device),
            )
            block_mask_phys_cross = FlexAttnFunc.compiled_create_block_mask(
                mask_mod_phys_cross, 1, 1, len(seq_ids), len(phys_seq_ids),
                device=device, _compile=True
            )
            FlexAttnFunc.phys_cross_attention_mask = block_mask_phys_cross
        else:
            FlexAttnFunc.phys_cross_attention_mask = None
    
    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == text_seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (text_seq_ids[kv_idx] >= 0)
        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_phys_cross_mask_mod(
        seq_ids, frame_ids, phys_seq_ids, phys_start_ids, phys_end_ids, phys_valid,
    ):
        def phys_seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            query_frame_id = frame_ids[q_idx] - (frame_ids[q_idx] % 2)
            same_seq = (
                (seq_ids[q_idx] == phys_seq_ids[kv_idx])
                & (seq_ids[q_idx] >= 0)
                & (phys_seq_ids[kv_idx] >= 0)
            )
            in_time = query_frame_id >= phys_start_ids[kv_idx]
            return same_seq & in_time & phys_valid[kv_idx]
        return phys_seq_mask
    
    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(seq_ids, frame_ids, noise_ids, window_size):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (seq_ids[q_idx] == seq_ids[kv_idx]) & (seq_ids[q_idx] >=0 ) & (seq_ids[kv_idx] >= 0)
        
        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] <= frame_ids[q_idx])
        
        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] < frame_ids[q_idx])
        
        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (frame_ids[kv_idx] == frame_ids[q_idx])
        
        def clean2clean_mask(
                b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)
        
        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)
        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        def phys_memory_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[kv_idx] == 2) & (frame_ids[kv_idx] <= frame_ids[q_idx])

        def phys_query_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (
                (noise_ids[q_idx] == 2)
                & (noise_ids[kv_idx] == 2)
                & (frame_ids[kv_idx] <= frame_ids[q_idx])
            )
        
        def block_window_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor, window_size: int
        ):
            return ((frame_ids[q_idx] - frame_ids[kv_idx]).abs() <= window_size)

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask_list.append(phys_memory_mask)
        mask_list.append(phys_query_mask)
        mask = or_masks(*mask_list)
        mask = and_masks(mask, seq_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask
       
class WanTimeTextImageEmbedding(nn.Module):

    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim,
                                        flip_sin_to_cos=True,
                                        downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim,
                                               time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim,
                                                       dim,
                                                       act_fn="gelu_tanh")

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.f_dim, 2)[:(self.f_dim // 2)].double() / self.f_dim))
        h_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.h_dim, 2)[:(self.h_dim // 2)].double() / self.h_dim))
        w_freqs_base = 1.0 / (self.theta**(torch.arange(
            0, self.w_dim, 2)[:(self.w_dim // 2)].double() / self.w_dim))
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base.to(grid_ids.device)
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base.to(grid_ids.device)
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base.to(grid_ids.device)
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()
            freqs_cis = torch.polar(torch.ones_like(freqs), freqs)

        return freqs_cis


class WanAttention(torch.nn.Module):

    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        cross_mask_name='text',
        attn_mode='torch',
    ):
        super().__init__()
        if attn_mode == 'torch':
            self.attn_op = custom_sdpa
        elif attn_mode == 'flashattn':
            self.attn_op = flash_attn_func if flash_attn_func is not None else custom_sdpa
        elif attn_mode == 'flex':
            self.attn_op = FlexAttnFunc(
                cross_attention_dim_head is not None,
                cross_mask_name=cross_mask_name,
            )
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList([
            torch.nn.Linear(self.inner_dim, dim, bias=True),
            torch.nn.Dropout(dropout),
        ])
        self.norm_q = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads,
                                       eps=eps,
                                       elementwise_affine=True)
        self.attn_caches = {} if cross_attention_dim_head is None else None

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches[cache_name]
        is_pred = cache['is_pred']
        cache['mask'][is_pred] = False

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    def init_kv_cache(self, cache_name, total_tolen, num_head, head_dim,
                      device, dtype, batch_size):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = {
            'k':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'v':
            torch.empty([batch_size, total_tolen, num_head, head_dim],
                        device=device,
                        dtype=dtype),
            'id':
            torch.full((total_tolen, ), -1, device=device),
            "mask":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
            "is_pred":
            torch.zeros((total_tolen, ), dtype=torch.bool, device=device),
        }

    def allocate_slots(self, cache_name, key_size):
        cache = self.attn_caches[cache_name]
        mask = cache["mask"]
        ids = cache["id"]
        free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        if free.numel() < key_size:
            used = mask.nonzero(as_tuple=False).squeeze(-1)

            used_ids = ids[used]
            order = torch.argsort(used_ids)
            need = key_size - free.numel()
            to_free = used[order[:need]]

            mask[to_free] = False
            ids[to_free] = -1
            free = (~mask).nonzero(as_tuple=False).squeeze(-1)

        assert free.numel() >= key_size
        return free[:key_size]

    def _next_cache_id(self, cache_name):
        ids = self.attn_caches[cache_name]['id']
        mask = self.attn_caches[cache_name]['mask']

        if mask.any():
            return ids[mask].max() + 1
        else:
            return torch.tensor(0, device=ids.device, dtype=ids.dtype)

    def update_cache(self, cache_name, key, value, is_pred):
        cache = self.attn_caches[cache_name]

        key_size = key.shape[1]
        slots = self.allocate_slots(cache_name, key_size)

        new_id = self._next_cache_id(cache_name)

        cache['k'][:, slots] = key
        cache['v'][:, slots] = value
        cache['mask'][slots] = True
        cache['id'][slots] = new_id
        cache['is_pred'][slots] = is_pred
        return slots

    def restore_cache(self, cache_name, slots):
        self.attn_caches[cache_name]['mask'][slots] = False

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name='pos',
    ):
        kv_cache = self.attn_caches[
            cache_name] if (self.attn_caches is not None) and (cache_name in self.attn_caches) else None

        query, key, value = self.to_q(q), self.to_k(k), self.to_v(v)
        query = self.norm_q(query)
        query = query.unflatten(2, (self.heads, -1))
        key = self.norm_k(key)
        key = key.unflatten(2, (self.heads, -1))
        value = value.unflatten(2, (self.heads, -1))
        if rotary_emb is not None:

            def apply_rotary_emb(x, freqs):
                x_out = torch.view_as_complex(
                    x.to(torch.float64).reshape(x.shape[0], x.shape[1],
                                                x.shape[2], -1, 2))
                x_out = torch.view_as_real(x_out * freqs).flatten(3)
                return x_out.to(x.dtype)
            query = apply_rotary_emb(query, rotary_emb)
            key = apply_rotary_emb(key, rotary_emb)
        slots = None
        if kv_cache is not None and kv_cache['k'] is not None:
            slots = self.update_cache(cache_name,
                                      key,
                                      value,
                                      is_pred=(update_cache == 1))
            key_pool = self.attn_caches[cache_name]['k']
            value_pool = self.attn_caches[cache_name]['v']
            mask = self.attn_caches[cache_name]['mask']
            valid = mask.nonzero(as_tuple=False).squeeze(-1)
            key = key_pool[:, valid]
            value = value_pool[:, valid]

        hidden_states = self.attn_op(query, key, value)

        if update_cache == 0:
            if kv_cache is not None and kv_cache['k'] is not None:
                self.restore_cache(cache_name, slots)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):

    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
        phys_zero_init=False,
        phys_gate=False,
    ):
        super().__init__()
        self.attn_mode = attn_mode
        self.phys_zero_init = bool(phys_zero_init)
        self.phys_gate_enabled = bool(phys_gate)

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = FP32LayerNorm(
            dim, eps,
            elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim,
                               inner_dim=ffn_dim,
                               activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn_phys = None
        self.norm_phys = None
        self.phys_gate_logit = None
        self.phys_monitor_enabled = False
        self.last_phys_stats = None

        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 6, dim) / dim**0.5)

    def enable_phys_cross_attention(self):
        if self.attn_phys is not None:
            return
        dim = self.attn2.to_q.in_features
        num_heads = self.attn2.heads
        dim_head = dim // num_heads
        eps = self.attn2.norm_q.eps
        self.attn_phys = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim_head,
            eps=eps,
            cross_attention_dim_head=dim_head,
            cross_mask_name='phys',
            attn_mode=self.attn_mode,
        )
        self.norm_phys = FP32LayerNorm(
            dim, eps, elementwise_affine=True
        ) if isinstance(self.norm2, FP32LayerNorm) else nn.Identity()
        if self.phys_zero_init:
            nn.init.zeros_(self.attn_phys.to_out[0].weight)
            if self.attn_phys.to_out[0].bias is not None:
                nn.init.zeros_(self.attn_phys.to_out[0].bias)
        if self.phys_gate_enabled:
            initial_gate = 0.1
            self.phys_gate_logit = nn.Parameter(torch.tensor(
                [math.log(initial_gate / (1.0 - initial_gate))],
                dtype=torch.float32,
            ))
        ref_param = self.attn2.to_q.weight
        self.attn_phys.to(device=ref_param.device, dtype=ref_param.dtype)
        self.norm_phys.to(device=ref_param.device)
        if self.phys_gate_logit is not None:
            self.phys_gate_logit.data = self.phys_gate_logit.data.to(
                device=ref_param.device)

    def disable_phys_cross_attention(self):
        self.attn_phys = None
        self.norm_phys = None
        self.phys_gate_logit = None
        self.last_phys_stats = None

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        phys_encoder_hidden_states=None,
        phys_condition_scale=1.0,
        update_cache=0,
        cache_name='pos',
    ) -> torch.Tensor:
        temb_scale_shift_table = self.scale_shift_table[None] + temb.float()
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = \
            rearrange(temb_scale_shift_table, 'b l n c -> b n l c').chunk(6, dim=1)
        shift_msa = shift_msa.squeeze(1)
        scale_msa = scale_msa.squeeze(1)
        gate_msa = gate_msa.squeeze(1)
        c_shift_msa = c_shift_msa.squeeze(1)
        c_scale_msa = c_scale_msa.squeeze(1)
        c_gate_msa = c_gate_msa.squeeze(1)
        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) *
                              (1. + scale_msa) +
                              shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states,
                                 norm_hidden_states,
                                 norm_hidden_states,
                                 rotary_emb,
                                 update_cache=update_cache,
                                 cache_name=cache_name)
        hidden_states = (hidden_states.float() +
                         attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(
            hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states,
                                 encoder_hidden_states,
                                 encoder_hidden_states,
                                 None,
                                 update_cache=0,
                                 cache_name=cache_name)
        hidden_states = hidden_states + attn_output

        # 3. Phase physics cross-attention
        if phys_encoder_hidden_states is not None:
            if self.attn_phys is None:
                self.enable_phys_cross_attention()
                self.attn_phys.to(
                    device=hidden_states.device, dtype=hidden_states.dtype)
                self.norm_phys.to(device=hidden_states.device)
            norm_hidden_states = self.norm_phys(
                hidden_states.float()).type_as(hidden_states)
            attn_output = self.attn_phys(norm_hidden_states,
                                         phys_encoder_hidden_states,
                                         phys_encoder_hidden_states,
                                         None,
                                         update_cache=0,
                                         cache_name=cache_name)
            phys_scale = phys_condition_scale
            if self.phys_gate_logit is not None:
                phys_scale = phys_scale * torch.sigmoid(
                    self.phys_gate_logit).type_as(attn_output)
            phys_residual = attn_output * phys_scale
            if self.phys_monitor_enabled:
                with torch.no_grad():
                    hidden_norm = hidden_states.detach().float().norm()
                    attn_norm = attn_output.detach().float().norm()
                    residual_norm = phys_residual.detach().float().norm()
                    if self.phys_gate_logit is None:
                        gate_value = torch.ones(
                            (), device=attn_output.device,
                            dtype=torch.float32)
                    else:
                        gate_value = torch.sigmoid(
                            self.phys_gate_logit.detach().float()).mean()
                    scale_value = torch.as_tensor(
                        phys_condition_scale,
                        device=attn_output.device,
                        dtype=torch.float32,
                    ) * gate_value
                    self.last_phys_stats = {
                        'hidden_norm': hidden_norm,
                        'attn_out_norm': attn_norm,
                        'residual_norm': residual_norm,
                        'residual_ratio': (
                            residual_norm / hidden_norm.clamp_min(1e-6)
                        ),
                        'gate': gate_value,
                        'scale': scale_value,
                    }
            hidden_states = hidden_states + phys_residual

        # 4. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) *
                              (1. + c_scale_msa) +
                              c_shift_msa).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (hidden_states.float() +
                         ff_output.float() * c_gate_msa).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """
    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
                                        # "patch_embedding", 
                                        "patch_embedding_mlp",
                                        "condition_embedder", 
                                        'condition_embedder_action',
                                        "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", 
                             "scale_shift_table", 
                             "scale_shift_table_action",
                             "norm1", 
                             'action_norm1',
                             'text_norm1',
                             "norm2", 
                             'action_norm2',
                             'text_norm2',
                             "norm3",
                             'action_norm3',
                             'text_norm3'
                             ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(self,
                 patch_size=[1, 2, 2],
                 num_attention_heads=24,
                 attention_head_dim=128,
                 in_channels=48,
                 out_channels=48,
                 action_dim=30,
                 text_dim=4096,
                 use_phys_memory=False,
                 phys_memory_dim=768,
                 phys_zero_init=False,
                 phys_gate=False,
                 phys_cross_start_layer=0,
                 freq_dim=256,
                 ffn_dim=14336,
                 num_layers=30,
                 cross_attn_norm=True,
                 eps=1e-06,
                 rope_max_seq_len=1024,
                 pos_embed_seq_len=None,
                 attn_mode="torch"):
        r"""
        TODO
        """
        super().__init__()
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        self.text_dim = text_dim
        self.use_phys_memory = use_phys_memory
        self.phys_memory_dim = phys_memory_dim
        self.phys_zero_init = bool(phys_zero_init)
        self.phys_gate = bool(phys_gate)
        self.phys_cross_start_layer = max(0, int(phys_cross_start_layer or 0))
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size,
                                      rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2],
            inner_dim)
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)
        if self.use_phys_memory:
            self.phys_memory_projector = nn.Sequential(
                nn.LayerNorm(self.phys_memory_dim),
                nn.Linear(self.phys_memory_dim, inner_dim),
            )
            self.phys_memory_type_embed = nn.Parameter(
                torch.randn(1, 1, inner_dim) / inner_dim**0.5
            )
        else:
            self.phys_memory_projector = None
            self.phys_memory_type_embed = None

        self.blocks = nn.ModuleList([
            WanTransformerBlock(inner_dim,
                                ffn_dim,
                                num_attention_heads,
                                cross_attn_norm,
                                eps,
                                attn_mode=attn_mode,
                                phys_zero_init=self.phys_zero_init,
                                phys_gate=self.phys_gate)
            for _ in range(num_layers)
        ])

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim,
                                  out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5)
        self.phys_cross_start_layer = min(
            self.phys_cross_start_layer, len(self.blocks))
        if self.use_phys_memory:
            self.enable_phys_cross_attention()

    def enable_phys_cross_attention(self):
        for block_idx, block in enumerate(self.blocks):
            block = self._unwrap_block(block)
            if self._phys_cross_layer_enabled(block_idx):
                block.enable_phys_cross_attention()
            else:
                block.disable_phys_cross_attention()

    def set_phys_cross_start_layer(self, start_layer):
        start_layer = max(0, min(int(start_layer or 0), len(self.blocks)))
        self.phys_cross_start_layer = start_layer
        if self.use_phys_memory:
            self.enable_phys_cross_attention()
        else:
            for block in self.blocks:
                self._unwrap_block(block).disable_phys_cross_attention()

    def _phys_cross_layer_enabled(self, block_idx):
        return (
            bool(getattr(self, 'use_phys_memory', False))
            and int(block_idx) >= int(getattr(self, 'phys_cross_start_layer', 0))
        )

    @staticmethod
    def _unwrap_block(block):
        return getattr(block, '_checkpoint_wrapped_module', block)

    def set_phys_monitor_enabled(self, enabled: bool):
        for block in self.blocks:
            block = self._unwrap_block(block)
            block.phys_monitor_enabled = bool(enabled)
            if enabled:
                block.last_phys_stats = None

    def pop_phys_monitor_stats(self):
        stats = []
        for block in self.blocks:
            block = self._unwrap_block(block)
            if block.last_phys_stats is not None:
                stats.append(block.last_phys_stats)
                block.last_phys_stats = None
        if not stats:
            return {}

        out = {}
        for key in stats[0].keys():
            values = torch.stack([item[key] for item in stats])
            out[f'phys/{key}_mean'] = values.mean()
            out[f'phys/{key}_max'] = values.max()
        out['phys/active_layers'] = torch.as_tensor(
            len(stats), device=stats[0]['hidden_norm'].device,
            dtype=torch.float32)
        return out

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(self, cache_name, attn_window,
                           latent_token_per_chunk, action_token_per_chunk,
                           device, dtype, batch_size,
                           phys_token_per_chunk=0):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2) * action_token_per_chunk
        if phys_token_per_chunk:
            total_tolen += (attn_window // 2) * phys_token_per_chunk
        for block in self.blocks:
            block.attn1.init_kv_cache(cache_name, total_tolen,
                                      self.num_attention_heads,
                                      self.attention_head_dim, device, dtype, batch_size)
    
    def _input_embed(self, latents, input_type='latent'):
        if input_type == 'latent':
            hidden_states = rearrange(
                latents,
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            hidden_states = self.patch_embedding_mlp(hidden_states)
        elif input_type == 'action':
            hidden_states = rearrange(latents, 'b c f h w -> b (f h w) c')
            hidden_states = self.action_embedder(hidden_states)
        elif input_type == 'text':
            hidden_states = self.condition_embedder.text_embedder(latents)
        else:
            raise ValueError(f"Unsupported input type: {input_type}")
        return hidden_states

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])
        latent_time_steps = torch.repeat_interleave(
            timesteps,
            (H // pach_scale_h) *
            (W // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C
        return temb, timestep_proj

    def _prepare_phys_memory_tokens(self, phys_feat, latent_dict, chunk_size, dtype, device):
        if (not self.use_phys_memory) or phys_feat is None:
            return None
        if phys_feat.dim() != 3:
            raise ValueError(
                f"phys_mem_feat should have shape [B, F, D], got {tuple(phys_feat.shape)}"
            )
        if phys_feat.shape[-1] != self.phys_memory_dim:
            raise ValueError(
                f"phys_mem_feat dim mismatch: expected {self.phys_memory_dim}, got {phys_feat.shape[-1]}"
            )

        batch_size, num_phys_frames, _ = phys_feat.shape
        phys_mask = latent_dict.get('phys_mem_mask')
        phys_spans = latent_dict.get('phys_mem_spans')
        if phys_spans is not None:
            if self.blocks[0].attn_mode != 'flex':
                raise ValueError(
                    "phys_mem_spans requires attn_mode='flex' for masked "
                    "phase physics cross-attention"
                )
            phys_feat = phys_feat.to(device=device, dtype=dtype)
            phys_hidden = self.phys_memory_projector(phys_feat)
            phys_hidden = phys_hidden + self.phys_memory_type_embed.to(
                device=device, dtype=phys_hidden.dtype
            )
            phys_hidden = phys_hidden.flatten(0, 1)[None]
            if phys_mask is None:
                phys_mask = torch.ones(
                    batch_size, num_phys_frames, dtype=torch.bool, device=device)
            else:
                phys_mask = phys_mask.to(device=device, dtype=torch.bool)
            phys_spans = phys_spans.to(device=device, dtype=torch.long)
            return {
                'hidden': phys_hidden,
                'shape': (batch_size, num_phys_frames, self.phys_memory_dim),
                'spans': phys_spans,
                'mask': phys_mask,
                'mode': 'phase_tokens',
            }

        latent_frames = latent_dict['noisy_latents'].shape[2]
        if num_phys_frames != latent_frames:
            phys_feat = F.interpolate(
                phys_feat.transpose(1, 2),
                size=latent_frames,
                mode='nearest',
            ).transpose(1, 2)
            num_phys_frames = latent_frames

        phys_feat = phys_feat.to(device=device, dtype=dtype)
        phys_hidden = self.phys_memory_projector(phys_feat)
        phys_hidden = phys_hidden + self.phys_memory_type_embed.to(
            device=device, dtype=phys_hidden.dtype
        )
        phys_hidden = phys_hidden.flatten(0, 1)[None]

        frame_ids = (
            torch.arange(num_phys_frames, device=device)[None, :]
            .expand(batch_size, -1)
            // chunk_size * 2
        )
        zero_ids = torch.zeros_like(frame_ids)
        type_ids = torch.full_like(frame_ids, 2)
        phys_grid_id = torch.stack([frame_ids, zero_ids, zero_ids, type_ids], dim=1)
        phys_grid_id = phys_grid_id.permute(1, 0, 2).flatten(1)[None]
        phys_rotary_emb = self.rope(phys_grid_id)[:, :, None]

        zero_steps = torch.zeros(
            batch_size, num_phys_frames, device=device,
            dtype=latent_dict['timesteps'].dtype,
        )
        phys_temb, phys_timestep_proj = self.condition_embedder(
            zero_steps, dtype=phys_hidden.dtype)
        phys_temb = phys_temb.flatten(0, 1)[None]
        phys_timestep_proj = phys_timestep_proj.unflatten(2, (6, -1))
        phys_timestep_proj = phys_timestep_proj.flatten(0, 1)[None]
        return {
            'hidden': phys_hidden,
            'rotary_emb': phys_rotary_emb,
            'temb': phys_temb,
            'timestep_proj': phys_timestep_proj,
            'shape': (batch_size, num_phys_frames, self.phys_memory_dim),
            'mode': 'self_tokens',
        }

    def _prepare_phys_memory_tokens_infer(self, phys_feat, input_dict, dtype, device):
        if (not self.use_phys_memory) or phys_feat is None:
            return None
        if phys_feat.dim() != 3:
            raise ValueError(
                f"phys_mem_feat should have shape [B, F, D], got {tuple(phys_feat.shape)}"
            )
        if phys_feat.shape[-1] != self.phys_memory_dim:
            raise ValueError(
                f"phys_mem_feat dim mismatch: expected {self.phys_memory_dim}, got {phys_feat.shape[-1]}"
            )

        batch_size, num_phys_frames, _ = phys_feat.shape
        align_mode = getattr(self.config, 'phys_memory_align_mode', None)
        if align_mode == 'phase_tokens':
            phys_feat = phys_feat.to(device=device, dtype=dtype)
            phys_hidden = self.phys_memory_projector(phys_feat)
            phys_hidden = phys_hidden + self.phys_memory_type_embed.to(
                device=device, dtype=phys_hidden.dtype
            )
            return {
                'hidden': phys_hidden,
                'mode': 'phase_tokens',
            }

        token_frames = input_dict['noisy_latents'].shape[2]
        grid_id = input_dict['grid_id']
        token_count = grid_id.shape[-1]
        tokens_per_frame = max(token_count // max(token_frames, 1), 1)
        frame_ids_full = grid_id[:, 0, ::tokens_per_frame]
        if frame_ids_full.shape[1] > token_frames:
            frame_ids_full = frame_ids_full[:, :token_frames]

        phys_frame_indices = input_dict.get('phys_mem_frame_indices')
        if phys_frame_indices is not None:
            phys_frame_indices = phys_frame_indices.to(
                device=device, dtype=torch.long)
            if phys_frame_indices.dim() == 1:
                phys_frame_indices = phys_frame_indices[None].expand(batch_size, -1)
            if phys_frame_indices.shape[0] != batch_size:
                if phys_frame_indices.shape[0] != 1:
                    raise ValueError(
                        "phys_mem_frame_indices batch mismatch: "
                        f"got {phys_frame_indices.shape[0]}, expected {batch_size}"
                    )
                phys_frame_indices = phys_frame_indices.expand(batch_size, -1)
            if phys_feat.shape[1] != phys_frame_indices.shape[1]:
                raise ValueError(
                    "phys_mem_feat and phys_mem_frame_indices length mismatch: "
                    f"{phys_feat.shape[1]} vs {phys_frame_indices.shape[1]}"
                )
            safe_idx = phys_frame_indices.clamp(
                min=0, max=max(frame_ids_full.shape[1] - 1, 0))
            frame_ids = torch.gather(frame_ids_full, 1, safe_idx)
        else:
            if num_phys_frames != token_frames:
                phys_feat = F.interpolate(
                    phys_feat.transpose(1, 2),
                    size=token_frames,
                    mode='nearest',
                ).transpose(1, 2)
                num_phys_frames = token_frames
            frame_ids = frame_ids_full[:, :num_phys_frames]

        phys_feat = phys_feat.to(device=device, dtype=dtype)
        phys_hidden = self.phys_memory_projector(phys_feat)
        phys_hidden = phys_hidden + self.phys_memory_type_embed.to(
            device=device, dtype=phys_hidden.dtype
        )

        zero_ids = torch.zeros_like(frame_ids)
        type_ids = torch.full_like(frame_ids, 2)
        phys_grid_id = torch.stack([frame_ids, zero_ids, zero_ids, type_ids], dim=1)
        phys_rotary_emb = self.rope(phys_grid_id)[:, :, None]

        zero_steps = torch.zeros(
            batch_size, phys_hidden.shape[1], device=device,
            dtype=input_dict['timesteps'].dtype,
        )
        phys_temb, phys_timestep_proj = self.condition_embedder(
            zero_steps, dtype=phys_hidden.dtype)
        phys_timestep_proj = phys_timestep_proj.unflatten(2, (6, -1))
        return {
            'hidden': phys_hidden,
            'rotary_emb': phys_rotary_emb,
            'temb': phys_temb,
            'timestep_proj': phys_timestep_proj,
            'mode': 'self_tokens',
        }

    def forward_train(self, input_dict):
        input_dict['latent_dict']['noisy_latents'] = input_dict['latent_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['latent_dict']['latent'] = input_dict['latent_dict']['latent'].to(torch.bfloat16)
        input_dict['action_dict']['noisy_latents'] = input_dict['action_dict']['noisy_latents'].to(torch.bfloat16)
        input_dict['action_dict']['latent'] = input_dict['action_dict']['latent'].to(torch.bfloat16)

        latent_dict = input_dict['latent_dict']
        action_dict = input_dict['action_dict']
        batch_size = latent_dict['noisy_latents'].shape[0]

        latent_hidden_states = self._input_embed(latent_dict['noisy_latents'], input_type='latent').flatten(0, 1)[None]
        action_hidden_states = self._input_embed(action_dict['noisy_latents'], input_type='action').flatten(0, 1)[None]
        text_hidden_states = self._input_embed(latent_dict["text_emb"], input_type='text')

        text_hidden_states = text_hidden_states.flatten(0, 1)[None]

        condition_latent_hidden_states = self._input_embed(latent_dict['latent'], input_type='latent').flatten(0, 1)[None]
        condition_action_hidden_states = self._input_embed(action_dict['latent'], input_type='action').flatten(0, 1)[None]

        hidden_states = torch.cat([latent_hidden_states, 
                                   condition_latent_hidden_states,
                                   action_hidden_states, 
                                   condition_action_hidden_states], dim=1)


        latent_grid_id = latent_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_dict['grid_id'].permute(1, 0, 2).flatten(1)[None]
        full_grid_id = torch.cat([latent_grid_id] * 2 + [action_grid_id] * 2, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None] 

        latent_time_steps = torch.cat(
            [latent_dict['timesteps'].flatten(0, 1), latent_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        action_time_steps = torch.cat(
            [action_dict['timesteps'].flatten(0, 1), action_dict['cond_timesteps'].flatten(0, 1)]
        )[None]
        latent_temb, latent_timestep_proj =self._time_embed(latent_time_steps, 
                        latent_dict['noisy_latents'].shape[-2], 
                        latent_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=False)
        action_temb, action_timestep_proj = self._time_embed(action_time_steps,
                        action_dict['noisy_latents'].shape[-2], 
                        action_dict['noisy_latents'].shape[-1], 
                        dtype=hidden_states.dtype, 
                        action_mode=True)
        temb = torch.cat([latent_temb, action_temb], dim=1)
        timestep_proj = torch.cat([latent_timestep_proj, action_timestep_proj], dim=1)

        phys_pack = self._prepare_phys_memory_tokens(
            latent_dict.get('phys_mem_feat'),
            latent_dict,
            chunk_size=input_dict["chunk_size"],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        phys_hidden_states = None
        phys_shape = None
        phys_cross_hidden_states = None
        phys_cross_shape = None
        phys_cross_spans = None
        phys_cross_mask = None
        if phys_pack is not None:
            if phys_pack.get('mode') == 'phase_tokens':
                phys_cross_hidden_states = phys_pack['hidden']
                phys_cross_shape = phys_pack['shape']
                phys_cross_spans = phys_pack['spans']
                phys_cross_mask = phys_pack['mask']
            else:
                phys_hidden_states = phys_pack['hidden']
                hidden_states = torch.cat([hidden_states, phys_hidden_states], dim=1)
                rotary_emb = torch.cat([rotary_emb, phys_pack['rotary_emb']], dim=1)
                temb = torch.cat([temb, phys_pack['temb']], dim=1)
                timestep_proj = torch.cat([timestep_proj, phys_pack['timestep_proj']], dim=1)
                phys_shape = phys_pack['shape']
        phys_condition_scale = float(
            input_dict.get('phys_condition_scale', 1.0)
        )

        total_length = hidden_states.shape[1]
        padded_length = (128 - total_length % 128) % 128
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))

        split_list = [latent_hidden_states.shape[1], 
                      condition_latent_hidden_states.shape[1], 
                      action_hidden_states.shape[1], 
                      condition_action_hidden_states.shape[1]]
        if phys_hidden_states is not None:
            split_list.append(phys_hidden_states.shape[1])
        split_list.append(padded_length)

        FlexAttnFunc.init_mask(latent_dict['noisy_latents'].shape, 
                               action_dict['noisy_latents'].shape, 
                               padded_length, 
                               input_dict["chunk_size"],
                               window_size=input_dict['window_size'],
                               patch_size=self.patch_size,
                               text_seq_len=text_hidden_states.shape[1],
                               device=hidden_states.device,
                               phys_shape=phys_shape,
                               phys_cross_shape=phys_cross_shape,
                               phys_cross_spans=phys_cross_spans,
                               phys_cross_mask=phys_cross_mask,
                               )

        for block_idx, block in enumerate(self.blocks):
            block_phys_cross_hidden_states = (
                phys_cross_hidden_states
                if self._phys_cross_layer_enabled(block_idx) else None
            )
            hidden_states = block(hidden_states,
                                  text_hidden_states,
                                  timestep_proj,
                                  rotary_emb,
                                  phys_encoder_hidden_states=block_phys_cross_hidden_states,
                                  phys_condition_scale=phys_condition_scale,
                                  update_cache=False)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(hidden_states.device).squeeze(1)
        scale = scale.to(hidden_states.device).squeeze(1)
        hidden_states = (self.norm_out(hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(hidden_states)
        split_hidden = torch.split(hidden_states, split_list, dim=1)
        latent_hidden_states = split_hidden[0]
        action_hidden_states = split_hidden[2]
        latent_hidden_states = self.proj_out(latent_hidden_states)
        latent_hidden_states = rearrange(latent_hidden_states,
                                             '1 (b l) (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size), b=batch_size)  #
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(action_hidden_states,
                                             '1 (b l) c -> b l c',
                                             b=batch_size)  #

        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)
        if action_mode:  # action input emb
            latent_hidden_states = rearrange(input_dict['noisy_latents'],
                                             'b c f h w -> b (f h w) c')
            latent_hidden_states = self.action_embedder(
                latent_hidden_states)  # B L1 C
        else:  # latent input emb
            latent_hidden_states = rearrange(
                input_dict['noisy_latents'],
                'b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)',
                p1=self.patch_size[0],
                p2=self.patch_size[1],
                p3=self.patch_size[2])
            latent_hidden_states = self.patch_embedding_mlp(
                latent_hidden_states)
        text_hidden_states = self.condition_embedder.text_embedder(
            input_dict["text_emb"])  # B L2 C

        latent_grid_id = input_dict['grid_id']
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        pach_scale_h, pach_scale_w = (1, 1) if action_mode else (
            self.patch_size[1], self.patch_size[2])

        latent_time_steps = torch.repeat_interleave(
            input_dict['timesteps'],
            (input_dict['noisy_latents'].shape[-2] // pach_scale_h) *
            (input_dict['noisy_latents'].shape[-1] // pach_scale_w), dim=1)  # L
        current_condition_embedder = self.condition_embedder_action if action_mode else self.condition_embedder
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        main_token_len = latent_hidden_states.shape[1]
        phys_pack = self._prepare_phys_memory_tokens_infer(
            input_dict.get('phys_mem_feat'),
            input_dict,
            dtype=latent_hidden_states.dtype,
            device=latent_hidden_states.device,
        )
        phys_cross_hidden_states = None
        if phys_pack is not None:
            if phys_pack['mode'] == 'phase_tokens':
                phys_cross_hidden_states = phys_pack['hidden']
            else:
                latent_hidden_states = torch.cat(
                    [latent_hidden_states, phys_pack['hidden']], dim=1)
                rotary_emb = torch.cat(
                    [rotary_emb, phys_pack['rotary_emb']], dim=1)
                temb = torch.cat([temb, phys_pack['temb']], dim=1)
                timestep_proj = torch.cat(
                    [timestep_proj, phys_pack['timestep_proj']], dim=1)
        phys_condition_scale = float(
            input_dict.get('phys_condition_scale', 1.0)
        )

        for block_idx, block in enumerate(self.blocks):
            block_phys_cross_hidden_states = (
                phys_cross_hidden_states
                if self._phys_cross_layer_enabled(block_idx) else None
            )
            latent_hidden_states = block(latent_hidden_states,
                                         text_hidden_states,
                                         timestep_proj,
                                         rotary_emb,
                                         phys_encoder_hidden_states=block_phys_cross_hidden_states,
                                         phys_condition_scale=phys_condition_scale,
                                         update_cache=update_cache,
                                         cache_name=cache_name)
        temb_scale_shift_table = self.scale_shift_table[None] + temb[:, :, None, ...]
        shift, scale = rearrange(temb_scale_shift_table,
                                 'b l n c -> b n l c').chunk(2, dim=1)
        shift = shift.to(latent_hidden_states.device).squeeze(1)
        scale = scale.to(latent_hidden_states.device).squeeze(1)
        latent_hidden_states = (self.norm_out(latent_hidden_states.float()) *
                                (1. + scale) +
                                shift).type_as(latent_hidden_states)

        if action_mode:
            latent_hidden_states = self.action_proj_out(
                latent_hidden_states[:, :main_token_len])
        else:
            latent_hidden_states = self.proj_out(
                latent_hidden_states[:, :main_token_len])
            latent_hidden_states = rearrange(latent_hidden_states,
                                             'b l (n c) -> b (l n) c',
                                             n=math.prod(self.patch_size))  #

        return latent_hidden_states


if __name__ == '__main__':
    model = WanTransformer3DModel(patch_size=[1, 2, 2],
                                  num_attention_heads=24,
                                  attention_head_dim=128,
                                  in_channels=48,
                                  out_channels=48,
                                  action_dim=30,
                                  text_dim=4096,
                                  freq_dim=256,
                                  ffn_dim=14336,
                                  num_layers=30,
                                  cross_attn_norm=True,
                                  eps=1e-6,
                                  rope_max_seq_len=1024,
                                  pos_embed_seq_len=None,
                                  attn_mode="torch")
    print(model)
