# --------------------------------------------------------
# models/modeling_pretrain.py  ── MoPE-JEPA 版本
# 删除 MAE decoder，替换为 JEPA predictor
# --------------------------------------------------------
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model

from .modeling_finetune import (
    Block,
    PatchEmbed,
    _cfg,
    get_sinusoid_encoding_table,
)
from .jepa_predictor import MoPEJEPAPredictor
from .moe_ffn import GlobalRouter


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


class PretrainVisionTransformerEncoder(nn.Module):
    # ── 重构：路由一次贯穿。block7 后用 GlobalRouter 算决策，block8-11 复用 ──
    def __init__(self,
                 img_size=224, patch_size=16, in_chans=3,
                 num_classes=0, embed_dim=768, depth=12, num_heads=12,
                 mlp_ratio=4., qkv_bias=False, qk_scale=None,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None,
                 tubelet_size=2, use_learnable_pos_emb=False,
                 with_cp=False, all_frames=16, cos_attn=False,
                 moe_layer_indices=None,
                 num_physics_experts=17, num_general_experts=10,
                 num_shared_experts=4,
                 candidate_k=5, gate_threshold=0.0,
                 key_top_m=3, key_alpha=0.5,
                 gate_hidden=0, gate_layers=2, gate_dims=None,
                 # 兼容旧签名
                 num_routable_experts=None, top_k=5):
        super().__init__()
        if num_routable_experts is not None:
            num_physics_experts = num_routable_experts

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_physics_experts = num_physics_experts
        self.num_general_experts = num_general_experts
        self.candidate_k = candidate_k
        self.gate_threshold = gate_threshold
        # 运行时开关：阶段一 False（通用组不参与），阶段二 True
        self.enable_general = False

        self.patch_embed = PatchEmbed(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans,
            embed_dim=embed_dim, num_frames=all_frames,
            tubelet_size=tubelet_size)
        num_patches = self.patch_embed.num_patches
        self.with_cp = with_cp

        if use_learnable_pos_emb:
            self.pos_embed = nn.Parameter(
                torch.zeros(1, num_patches + 1, embed_dim))
        else:
            self.pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)

        if moe_layer_indices is None:
            moe_layer_indices = list(range(depth * 2 // 3, depth))
        self.moe_layer_indices = moe_layer_indices
        # 路由器插入点：第一个 MoE 层之前（即 block7 之后）
        self.route_after_block = min(moe_layer_indices) - 1

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        _first_moe = min(moe_layer_indices)
        _router_kwargs = dict(
            candidate_k=candidate_k,
            key_top_m=key_top_m, key_alpha=key_alpha,
            gate_hidden=gate_hidden, gate_layers=gate_layers, gate_dims=gate_dims)
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i],
                norm_layer=norm_layer, init_values=init_values,
                cos_attn=cos_attn,
                use_moe=(i in moe_layer_indices),
                num_physics_experts=num_physics_experts,
                num_general_experts=num_general_experts,
                num_shared_experts=num_shared_experts,
                # 只有第一个 MoE 层持有 router（在层内算一次决策，后续层复用）
                with_router=(i == _first_moe),
                router_kwargs=(_router_kwargs if i == _first_moe else None))
            for i in range(depth)
        ])

        # router 现在在第一个 MoE block 内部（self.blocks[_first_moe].mlp.router），
        # encoder 不再持有独立的 global_router。
        # 这样 JEPA target path 的 no_grad 完整覆盖 router → 避免 DDP "ready twice"。
        self._first_moe_idx = _first_moe

        self.norm = norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) \
            if num_classes > 0 else nn.Identity()

        if use_learnable_pos_emb:
            trunc_normal_(self.pos_embed, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward_features(self, x, mask, physics_label=None,
                         physics_label_soft=None):
        num_time_bins = x.shape[2] // self.patch_embed.tubelet_size
        x = self.patch_embed(x)
        x = x + self.pos_embed.type_as(x).to(x.device).clone().detach()
        B, _, C = x.shape
        x_vis = x[~mask].reshape(B, -1, C)

        decision = None
        self._decision = None
        self._balance_loss = None
        self._physics_loss = None
        self._general_loss = None
        self._last_token_scores = None
        self._cls_scores = None

        # 第一个 MoE 层用的 router 输入（标签/阈值等）
        router_inputs = dict(
            physics_label=physics_label,
            physics_label_soft=physics_label_soft,
            gate_threshold=self.gate_threshold,
            candidate_k=self.candidate_k,
        )

        for i, blk in enumerate(self.blocks):
            if getattr(blk, 'use_moe', False):
                is_first_moe = (i == self._first_moe_idx)
                if self.with_cp:
                    if is_first_moe:
                        # 第一个 MoE 层：传 router_inputs，层内算 decision
                        x_vis = cp.checkpoint(
                            lambda t, _blk=blk: _blk(
                                t, None, enable_general=self.enable_general,
                                router_inputs=router_inputs),
                            x_vis, use_reentrant=False)
                        decision = blk._decision
                    else:
                        x_vis = cp.checkpoint(
                            lambda t, _blk=blk, _d=decision: _blk(
                                t, _d, enable_general=self.enable_general),
                            x_vis, use_reentrant=False)
                else:
                    if is_first_moe:
                        x_vis = blk(x_vis, None,
                                    enable_general=self.enable_general,
                                    router_inputs=router_inputs)
                        decision = blk._decision
                    else:
                        x_vis = blk(x_vis, decision,
                                    enable_general=self.enable_general)

                # 第一个 MoE 层算完 decision 后，记录 loss/分类分数到 encoder
                if is_first_moe and decision is not None:
                    self._decision     = decision
                    self._balance_loss = decision["balance_loss"]
                    self._physics_loss = decision["physics_loss"]
                    self._general_loss = decision["general_loss"]
                    self._cls_scores   = decision["cls_scores"]
                    self._last_token_scores = decision["cls_scores"]
                    # 供后续 block 复用的 decision：可导权重 detach，
                    # 避免 phys_w/gen_w（含 router gate 梯度）跨 block 传递导致
                    # DDP 对 gate 参数重复挂 hook（ready twice）。
                    # gate 的 JEPA 梯度已由 block8 这一次（用可导权重）提供，足够。
                    decision = dict(decision)
                    decision["phys_w"] = decision["phys_w"].detach()
                    decision["gen_w"]  = decision["gen_w"].detach()
            else:
                # 普通层：不传 decision
                if self.with_cp:
                    x_vis = cp.checkpoint(
                        lambda t, _blk=blk: _blk(t), x_vis, use_reentrant=False)
                else:
                    x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        self._last_x_vis = x_vis
        return x_vis

    def forward(self, x, mask, physics_label=None, physics_label_soft=None):
        x = self.forward_features(x, mask, physics_label, physics_label_soft)
        x = self.head(x)
        return x


class PretrainVisionTransformer(nn.Module):
    """MoPE-JEPA：用 JEPA predictor 替换 MAE decoder"""

    def __init__(
        self,
        img_size=224, patch_size=16,
        encoder_in_chans=3, encoder_num_classes=0,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        # JEPA predictor 参数（替换原来的 decoder 参数）
        predictor_dim=384, predictor_depth=6, predictor_num_heads=6,
        mlp_ratio=4., qkv_bias=False, qk_scale=None,
        drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
        norm_layer=nn.LayerNorm, init_values=0.,
        use_learnable_pos_emb=False, tubelet_size=2,
        num_classes=0, in_chans=0,  # timm 兼容
        with_cp=False, all_frames=16, cos_attn=False,
        moe_layer_indices=None,
        num_physics_experts=17, num_general_experts=10,
        num_shared_experts=4,
        candidate_k=5, gate_threshold=0.0,
        key_top_m=3, key_alpha=0.5,
        gate_hidden=0, gate_layers=2, gate_dims=None,
        num_event_classes=0,
        # 兼容旧签名
        num_routable_experts=None, top_k=5,
    ):
        super().__init__()
        if num_routable_experts is not None:
            num_physics_experts = num_routable_experts
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=img_size, patch_size=patch_size,
            in_chans=encoder_in_chans, num_classes=encoder_num_classes,
            embed_dim=encoder_embed_dim, depth=encoder_depth,
            num_heads=encoder_num_heads, mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop_rate=drop_rate, attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, norm_layer=norm_layer,
            init_values=init_values, tubelet_size=tubelet_size,
            use_learnable_pos_emb=use_learnable_pos_emb,
            with_cp=with_cp, all_frames=all_frames, cos_attn=cos_attn,
            moe_layer_indices=moe_layer_indices,
            num_physics_experts=num_physics_experts,
            num_general_experts=num_general_experts,
            num_shared_experts=num_shared_experts,
            candidate_k=candidate_k, gate_threshold=gate_threshold,
            key_top_m=key_top_m, key_alpha=key_alpha,
            gate_hidden=gate_hidden, gate_layers=gate_layers, gate_dims=gate_dims)

        num_patches = self.encoder.patch_embed.num_patches

        # JEPA predictor（替换 MAE decoder）
        self.predictor = MoPEJEPAPredictor(
            num_patches=num_patches,
            encoder_dim=encoder_embed_dim,
            predictor_dim=predictor_dim,
            depth=predictor_depth,
            num_heads=predictor_num_heads,
            mlp_ratio=mlp_ratio,
        )
        self.event_head = (
            nn.Linear(encoder_embed_dim, num_event_classes)
            if num_event_classes and num_event_classes > 0 else None
        )

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def forward(self, x, mask, physics_label=None, physics_label_soft=None,
                event_only_full_visible=False):
        """
        x:    [B, 3, T, H, W]
        mask: [B, N_all]  bool，True=被mask

        return:
            pred:   [B, N_mask, encoder_dim]  ← predictor 预测
            target: [B, N_mask, encoder_dim]  ← target encoder 输出（detach）
        """
        if event_only_full_visible:
            if self.event_head is None:
                raise RuntimeError(
                    'event_only_full_visible requires a configured event_head')
            x_event = self.encoder(
                x, mask, physics_label=None, physics_label_soft=None)
            return self.event_head(x_event.mean(dim=1))

        B = x.shape[0]
        N_all = self.encoder.patch_embed.num_patches

        # ── context encoder：只处理 visible tokens ──────────────────────
        x_vis = self.encoder(x, mask, physics_label, physics_label_soft)
        # x_vis: [B, N_vis, encoder_dim]
        # 立刻保存 context path 的 aux loss，target path 会覆盖这些属性
        _balance_loss  = self.encoder._balance_loss
        _physics_loss  = self.encoder._physics_loss
        _general_loss  = self.encoder._general_loss
        _token_scores  = self.encoder._last_token_scores
        _last_x_vis    = self.encoder._last_x_vis    # ← 新增

        # ── target encoder：处理全部 token，取 mask 位置，stop_gradient ──
        # 方案B：同一个 encoder，mask 全为 False（全部可见）
        full_mask = torch.zeros(B, N_all, dtype=torch.bool, device=x.device)
        with torch.no_grad():
            x_all = self.encoder(x, full_mask)  # [B, N_all, encoder_dim]
        # 取 mask 位置的 latent 作为预测目标
        target = x_all[mask].reshape(B, -1, x_all.shape[-1])  # [B, N_mask, encoder_dim]
        target = target.detach()
        # 恢复 context path 的 aux loss
        self.encoder._balance_loss      = _balance_loss
        self.encoder._physics_loss      = _physics_loss
        self.encoder._general_loss      = _general_loss
        self.encoder._last_token_scores = _token_scores
        self.encoder._last_x_vis        = _last_x_vis  # ← 新增

        # ── predictor：预测 mask 位置的 latent ──────────────────────────
        pred = self.predictor(x_vis, mask)  # [B, N_mask, encoder_dim]

        if self.event_head is not None:
            event_feat = self.encoder._last_x_vis.mean(dim=1)
            event_logits = self.event_head(event_feat)
            return pred, target, event_logits

        return pred, target


@register_model
def pretrain_mope_jepa_base_patch16_224(pretrained=False, **kwargs):
    model = PretrainVisionTransformer(
        img_size=224, patch_size=16,
        encoder_embed_dim=768, encoder_depth=12, encoder_num_heads=12,
        encoder_num_classes=0,
        predictor_dim=384, predictor_depth=6, predictor_num_heads=6,
        mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(kwargs["init_ckpt"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
    return model
