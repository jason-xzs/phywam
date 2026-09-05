# --------------------------------------------------------
# models/jepa_predictor.py  ── MoPE-JEPA Predictor
#
# 输入：context tokens 的 latent + mask tokens 的位置编码
# 输出：对 mask token 位置的 latent 预测
#
# 架构：6层标准 ViT（无 MoE，比 encoder 更轻量）
# embed_dim: 384（encoder 是 768，predictor 用更小的维度）
# --------------------------------------------------------

import torch
import torch.nn as nn
from functools import partial
from timm.models.layers import trunc_normal_


class PredictorAttention(nn.Module):
    def __init__(self, dim, num_heads=6, qkv_bias=True, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PredictorBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=True,
                 drop=0., attn_drop=0., norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = PredictorAttention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                                       attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(drop),
            nn.Linear(mlp_hidden_dim, dim),
            nn.Dropout(drop),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MoPEJEPAPredictor(nn.Module):
    """
    JEPA Predictor for MoPE.

    接收 context encoder 的输出（visible token latents）
    和 mask token 的位置编码，预测 mask 位置的 latent。

    输入：
        x_vis:  [B, N_vis, encoder_dim]   ← context encoder 输出
        mask:   [B, N_all]  bool，True 表示被 mask 的位置

    输出：
        pred:   [B, N_mask, encoder_dim]  ← 对 mask 位置的预测
    """

    def __init__(
        self,
        num_patches: int = 1568,      # 总 patch 数（训练时 mask 后是 157）
        encoder_dim: int = 768,       # encoder 输出维度
        predictor_dim: int = 384,     # predictor 内部维度（更轻量）
        depth: int = 6,
        num_heads: int = 6,
        mlp_ratio: float = 4.0,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.predictor_dim = predictor_dim
        self.encoder_dim = encoder_dim

        # 输入投影：encoder_dim → predictor_dim
        self.input_proj = nn.Linear(encoder_dim, predictor_dim)

        # mask token：可学习向量，代表被 mask 掉的 token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, predictor_dim))

        # 位置编码（覆盖全部 N_all 个位置）
        self.predictor_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches, predictor_dim))

        # 6层标准 ViT block
        self.blocks = nn.ModuleList([
            PredictorBlock(
                dim=predictor_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                norm_layer=norm_layer,
            )
            for _ in range(depth)
        ])

        self.norm = norm_layer(predictor_dim)

        # 输出投影：predictor_dim → encoder_dim（对齐 target latent 维度）
        self.output_proj = nn.Linear(predictor_dim, encoder_dim)

        self._init_weights()

    def _init_weights(self):
        trunc_normal_(self.mask_token, std=0.02)
        trunc_normal_(self.predictor_pos_embed, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x_vis: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        x_vis: [B, N_vis, encoder_dim]  ← visible token 的 latent
        mask:  [B, N_all]  bool，True 表示被 mask（N_vis + N_mask = N_all）

        return: [B, N_mask, encoder_dim]
        """
        B, N_vis, _ = x_vis.shape
        N_all = mask.shape[1]

        # 1. 投影到 predictor 维度
        x_vis = self.input_proj(x_vis)  # [B, N_vis, predictor_dim]

        # 2. 加位置编码（visible 部分）
        pos_embed_all = self.predictor_pos_embed.expand(B, -1, -1)  # [B, N_all, predictor_dim]
        pos_vis = pos_embed_all[~mask].reshape(B, N_vis, self.predictor_dim)
        x_vis = x_vis + pos_vis

        # 3. 构造 mask tokens，加上对应位置编码
        N_mask = N_all - N_vis
        mask_tokens = self.mask_token.expand(B, N_mask, -1)  # [B, N_mask, predictor_dim]
        pos_mask = pos_embed_all[mask].reshape(B, N_mask, self.predictor_dim)
        mask_tokens = mask_tokens + pos_mask

        # 4. 拼接：visible tokens + mask tokens
        x_full = torch.cat([x_vis, mask_tokens], dim=1)  # [B, N_all, predictor_dim]

        # 5. 过 6 层 ViT block
        for blk in self.blocks:
            x_full = blk(x_full)
        x_full = self.norm(x_full)

        # 6. 只取 mask token 位置的输出
        x_mask_out = x_full[:, N_vis:]  # [B, N_mask, predictor_dim]

        # 7. 投影回 encoder_dim
        pred = self.output_proj(x_mask_out)  # [B, N_mask, encoder_dim]

        return pred