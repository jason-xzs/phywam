# --------------------------------------------------------
# models/sigreg.py  ── Sketched Isotropic Gaussian Regularizer
# 移植自 LeWorldModel (https://github.com/lcswillems/le-wm)
# 用于防止 JEPA 训练中的 representation collapse
# --------------------------------------------------------

import torch
import torch.nn as nn


class SIGReg(nn.Module):
    """
    Sketched Isotropic Gaussian Regularizer (SIGReg)

    通过将 latent embeddings 投影到随机方向，
    约束其分布接近各向同性高斯，防止 collapse。

    基于 Cramér-Wold 定理：匹配所有一维边缘分布等价于匹配联合分布。

    输入格式：(T, B, D)
        T: 时间步数（token 数或帧数）
        B: batch size
        D: embedding 维度

    参考：
        LeWorldModel: https://arxiv.org/abs/2603.19312
        LeJEPA: https://arxiv.org/abs/2511.08544
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024):
        super().__init__()
        self.num_proj = num_proj

        # 积分节点：均匀分布在 [0, 3]
        t = torch.linspace(0, 3, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)

        # 梯形积分权重
        weights = torch.full((knots,), 2 * dt, dtype=torch.float32)
        weights[[0, -1]] = dt

        # 高斯窗函数 w(t) = exp(-t^2 / 2)
        window = torch.exp(-t.square() / 2.0)

        self.register_buffer("t", t)
        self.register_buffer("phi", window)           # 标准高斯的特征函数值
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """
        z: (T, B, D)  ← 注意是 T 在前，调用时需要 emb.transpose(0, 1)
        return: scalar loss
        """
        # 随机单位方向向量 [D, num_proj]
        A = torch.randn(z.size(-1), self.num_proj, device=z.device, dtype=z.dtype)
        A = A / A.norm(p=2, dim=0, keepdim=True).clamp_min(1e-8)

        # 投影到随机方向后乘积分节点：[T, B, num_proj, knots]
        x_t = (z @ A).unsqueeze(-1) * self.t  # broadcasting

        # Epps-Pulley 统计量
        # 经验特征函数与标准高斯特征函数的差的平方
        err = (x_t.cos().mean(-3) - self.phi).square() \
            + x_t.sin().mean(-3).square()   # [T, num_proj, knots]

        # 梯形积分 + 平均
        statistic = (err @ self.weights) * z.size(-2)  # [T, num_proj]
        return statistic.mean()