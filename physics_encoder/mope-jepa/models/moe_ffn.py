# --------------------------------------------------------
# models/moe_ffn.py  ── MoPE 物理感知 MoE（路线B + 解法一：统一 router）
#
# 架构（路线B，与旧版统一 router 思想一致，最准且省）：
#   block7 后【一次】路由，block8-11 复用同一身份。
#
#   物理 router（一个 gate，阶段一训好、阶段二冻结）：
#     - 所有 token 共享一个 Linear(C→17)，并行算出 token_scores [B,N,17]（softmax）
#     - .mean(dim=1) → [B,17]：① 门控(max置信度判物理/通用组) ② 分类(推理用)
#                                ③ 阶段一 physics_loss 监督此聚合输出
#     - .argmax(dim=-1) → [B,N]：token 选物理专家(全17类，无候选限制)
#       专家 i ↔ 物理类 i，有物理语义
#
#   通用 router（一个 gate，阶段一不参与、阶段二从随机学）：
#     - Linear(C→10) → token 选通用专家(全10类自由)，无监督涌现，仅 balance loss
#
#   LayerExperts（每层独立）：17 物理 + 10 通用 + 4 共享，按身份 dispatch。
#
# 物理标签对齐 WISA 17 类（expert i ↔ 物理类 i）。
# --------------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional

LABEL_TO_EXPERT: Dict[int, int] = {i: i for i in range(17)}
NUM_PHYSICS_CLASSES = 17

PHYSICS_SOFT_LABEL_KEYS = (
    "collision", "rigid_body_motion", "elastic_motion", "liquid_motion",
    "gas_motion", "deformation", "melting", "solidification", "vaporization",
    "liquefaction", "explosion", "combustion", "reflection", "refraction",
    "scattering", "interference_diffraction", "unnatural_light_sources",
)


def distribution_dict_to_tensor(d: Dict, device=None, dtype=torch.float32) -> torch.Tensor:
    vec = [float(d.get(k, 0.0)) for k in PHYSICS_SOFT_LABEL_KEYS]
    t = torch.tensor(vec, device=device, dtype=dtype)
    s = t.sum()
    if s > 0:
        t = t / s
    return t


# ── 1. 单个 Expert FFN ────────────────────────────────────────────────────────

class ExpertFFN(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, drop: float = 0.0):
        super().__init__()
        self.fc1  = nn.Linear(dim, hidden_dim)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(drop)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(self.fc2(self.act(self.fc1(x))))


# ── 2. 物理 router（统一 router：分类/门控/选专家）─────────────────────────────

class PhysicsRouter(nn.Module):
    """
    物理组 token 路由 + 物理分类。gate 输出纯 17 类（和原版150一致，无通用维）。
      token_scores = softmax(gate(x))   [B,N,17]
      video_scores = token_scores.mean(1)  [B,17]  → 物理分类(物理软标签监督)
      token 选物理专家：argmax → [B,N]

    gate 可配置：
      gate_dims=[512,256,128] → 768→512→256→128→17
      gate_hidden>0           → MLP
      默认                    → 单层 Linear(C→17)
    """
    def __init__(self, dim: int, num_physics_experts: int = 17,
                 gate_hidden: int = 0, gate_layers: int = 2,
                 gate_dims=None):
        super().__init__()
        self.num_physics_experts = num_physics_experts
        out_dim = num_physics_experts   # 纯17类，无通用维
        if gate_dims:
            dims = [dim] + list(gate_dims)
            layers = []
            for i in range(len(dims) - 1):
                layers += [nn.Linear(dims[i], dims[i + 1]), nn.GELU()]
            layers += [nn.Linear(dims[-1], out_dim)]
            self.gate = nn.Sequential(*layers)
        elif gate_hidden and gate_hidden > 0:
            layers = [nn.Linear(dim, gate_hidden), nn.GELU()]
            for _ in range(max(0, gate_layers - 2)):
                layers += [nn.Linear(gate_hidden, gate_hidden), nn.GELU()]
            layers += [nn.Linear(gate_hidden, out_dim)]
            self.gate = nn.Sequential(*layers)
        else:
            self.gate = nn.Linear(dim, out_dim, bias=False)

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        logits = self.gate(x)                          # [B,N,17]
        token_scores = F.softmax(logits, dim=-1)       # [B,N,17]
        video_scores = token_scores.mean(dim=1)        # [B,17]
        phys_expert_id = token_scores.argmax(dim=-1)   # [B,N]
        return token_scores, video_scores, phys_expert_id


# ── 2.5 二分类门控（video级，判物理组 vs 通用组）──────────────────────────────

class BinaryGate(nn.Module):
    """
    视频级二分类门控：判整条视频是物理组(0) 还是通用组(1)。
      输入完整 token（不平均）：logits = gate(x)  [B,N,2]
      video 级聚合：token logits mean → [B,2]
      硬路由：argmax → is_general [B] bool（整条视频统一走物理组或通用组）
    标签：WISA=物理组(0)，OpenVid=通用组(1)，CE 监督。
    """
    def __init__(self, dim: int):
        super().__init__()
        self.gate = nn.Linear(dim, 2, bias=False)

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        token_logits = self.gate(x)                    # [B,N,2]
        video_logits = token_logits.mean(dim=1)        # [B,2]
        video_probs  = F.softmax(video_logits, dim=-1) # [B,2]  [:,1]=通用概率（可导，进general_loss）
        # is_general 是硬标定（决定走哪路），彻底 detach，不带任何梯度。
        # binary_gate.gate 的梯度只经 video_probs → general_loss 这一条路径。
        is_general   = (video_logits.argmax(dim=-1) == 1).detach()  # [B] bool
        return video_logits.detach(), video_probs, is_general


# ── 3. 通用 router（无监督，自己学）──────────────────────────────────────────

class GeneralTokenRouter(nn.Module):
    def __init__(self, dim: int, num_general_experts: int = 10):
        super().__init__()
        self.num_experts = num_general_experts
        self.gate = nn.Linear(dim, num_general_experts, bias=False)

    def forward(self, x: torch.Tensor):
        B, N, _ = x.shape
        logits = self.gate(x)                          # [B,N,10]
        probs  = F.softmax(logits, dim=-1)             # [B,N,10]
        soft_weight, expert_id = probs.max(dim=-1)     # [B,N]
        return expert_id, soft_weight, probs


# ── 4. 全局路由器（封装物理 + 通用 router，forward_features 调用一次）──────────

class GlobalRouter(nn.Module):
    """
    block7 后调用一次。门控由物理 router 的聚合置信度决定。
    阶段一 enable_general=False（通用组不参与）；阶段二 True。
    """
    def __init__(self, dim: int, num_physics_experts: int = 17,
                 num_general_experts: int = 10, **kwargs):
        super().__init__()
        self.num_physics_experts = num_physics_experts
        self.num_general_experts = num_general_experts
        gate_hidden = kwargs.get('gate_hidden', 0)
        gate_layers = kwargs.get('gate_layers', 2)
        gate_dims   = kwargs.get('gate_dims', None)
        # 二分类门控：判物理组 vs 通用组
        self.binary_gate = BinaryGate(dim)
        # 物理 router：纯17类（同原版150）
        self.phys_router = PhysicsRouter(dim, num_physics_experts,
                                         gate_hidden=gate_hidden,
                                         gate_layers=gate_layers,
                                         gate_dims=gate_dims)
        # 通用 router：10类无监督
        self.gen_router  = GeneralTokenRouter(dim, num_general_experts)

    def forward(self, x: torch.Tensor, time_ids: Optional[torch.Tensor] = None,
                num_time_bins: Optional[int] = None,
                physics_label: Optional[torch.Tensor] = None,
                physics_label_soft: Optional[torch.Tensor] = None,
                gate_threshold: float = 0.0,
                candidate_k: Optional[int] = None,
                enable_general: bool = False,
                is_general_label: Optional[torch.Tensor] = None) -> Dict:
        """
        三段式路由（梯度路径分离，避免 DDP ready twice）：
          1. 二分类门控 binary_gate：判物理组/通用组（is_general 标签监督）
          2. 物理 router：17类物理专家分数（物理软标签监督，同原版150）
          3. 通用 router：10类通用专家分数（无监督，balance 涌现）
        硬路由：物理组视频 token 全走物理专家，通用组全走通用专家。
        """
        B, N, C = x.shape

        # ── 1. 二分类门控（video 级，硬路由）────────────────────────────
        if enable_general:
            bin_logits, bin_probs, is_general = self.binary_gate(x)  # [B,2],[B,2],[B]bool
            use_physics = ~is_general                                # [B] bool
        else:
            bin_probs = x.new_zeros(B, 2)
            bin_probs[:, 0] = 1.0
            is_general = torch.zeros(B, device=x.device, dtype=torch.bool)
            use_physics = torch.ones(B, device=x.device, dtype=torch.bool)

        # ── 2. 物理 router（17类）────────────────────────────────────────
        phys_token_scores, phys_video_scores, phys_expert_id = self.phys_router(x)
        # token 选中物理专家的 softmax 权重（可导）→ 乘进专家输出，让 gate 从 JEPA 学路由
        phys_w = phys_token_scores.gather(-1, phys_expert_id.unsqueeze(-1)).squeeze(-1)  # [B,N]

        # ── 3. 通用 router（10类，启用时）──────────────────────────────
        if False:
            gen_expert_id, gen_w, gen_probs = self.gen_router(x)
            # gen_w 已是 probs.max（token 选中通用专家的 softmax 权重，可导）
        else:
            gen_expert_id = x.new_zeros(B, N, dtype=torch.long)
            gen_w = x.new_zeros(B, N)
            gen_probs = x.new_zeros(B, N, self.num_general_experts)

        # ── router_loss：两部分独立 ──────────────────────────────────────
        # ── loss：拆成两个独立部分 ────────────────────────────────────────
        #   general_loss：二分类门控（WISA=物理0 / OpenVid=通用1），监督 binary_gate
        #                 用 video_probs（softmax 可导）算 CE，argmax 只决定去哪
        #   physics_loss：物理软标签（仅物理组样本，前17类），监督 phys_router，同原版150
        # 两者在 engine 里各 0.5 权重进总 loss（梯度路径完全分离）。
        #
        # 标签来源：physics_label_soft 是 18 维（前17物理 + 第18通用）。
        general_loss = x.new_zeros(())
        physics_loss = x.new_zeros(())

        soft18 = None
        if physics_label_soft is not None:
            soft18 = physics_label_soft.to(dtype=phys_video_scores.dtype, device=x.device)

        # general_loss：二分类（is_general 从第18维推出）
        #   用 bin_probs（softmax 概率，可导）算 NLL，等价 CE，梯度走 softmax
        if enable_general and soft18 is not None:
            gl = (soft18[:, self.num_physics_experts] > 0.5).long()  # [B] 0物理/1通用
            bp = bin_probs.clamp_min(1e-8)                            # [B,2] softmax概率
            general_loss = F.nll_loss(bp.log(), gl)
        elif enable_general and is_general_label is not None:
            gl = is_general_label.to(device=x.device).long()
            bp = bin_probs.clamp_min(1e-8)
            general_loss = F.nll_loss(bp.log(), gl)

        # physics_loss：物理软标签 CE（只监督物理组样本：第18维=0，用前17维）
        if soft18 is not None:
            y = soft18[:, :self.num_physics_experts].clamp_min(0.0)   # [B,17]
            ysum = y.sum(dim=-1, keepdim=True)                        # [B,1]
            valid = (ysum.squeeze(-1) > 1e-6)                         # 物理组样本
            if valid.any():
                yv  = y[valid] / (ysum[valid] + 1e-8)
                vsv = phys_video_scores[valid].clamp_min(1e-8)
                physics_loss = -(yv * vsv.log()).sum(dim=-1).mean()
        elif physics_label is not None:
            y = F.one_hot(physics_label, num_classes=self.num_physics_experts).to(
                dtype=phys_video_scores.dtype)
            vs = phys_video_scores.clamp_min(1e-8)
            physics_loss = -(y * vs.log()).sum(dim=-1).mean()

        # ── balance loss：物理/通用分开统计 ──────────────────────────────
        E_p = self.num_physics_experts
        f_p = torch.zeros(E_p, device=x.device, dtype=x.dtype)
        f_p.scatter_add_(0, phys_expert_id.reshape(-1),
                         torch.ones(B * N, device=x.device, dtype=x.dtype))
        f_p = f_p / (B * N + 1e-6)
        p_p = phys_token_scores.reshape(B * N, E_p).mean(0)
        balance_phys = E_p * (f_p * p_p).sum()

        if False:
            E_g = self.num_general_experts
            f_g = torch.zeros(E_g, device=x.device, dtype=x.dtype)
            f_g.scatter_add_(0, gen_expert_id.reshape(-1),
                             torch.ones(B * N, device=x.device, dtype=x.dtype))
            f_g = f_g / (B * N + 1e-6)
            p_g = gen_probs.reshape(B * N, E_g).mean(0)
            balance_loss = balance_phys + E_g * (f_g * p_g).sum()
        else:
            balance_loss = balance_phys

        # ── 推理分类展示：[物理17类, 通用概率]拼接 → 18维 ───────────────
        #   仅用于推理/日志，不参与反传。必须用 detach 输入构建，否则
        #   general_prob(bin_probs) 会给 binary_gate 建第二条 backward 路径
        #   → 与 general_loss 的路径冲突 → DDP "ready twice"。
        general_prob = bin_probs[:, 1:2].detach()                # [B,1] detach
        cls_scores = torch.cat([phys_video_scores.detach(), general_prob], dim=-1)  # [B,18]

        # decision 分两类，关键避免 DDP "ready twice"：
        #   路由字段（给 block8/9/10/11 共享，用于选专家/分流）→ 全部 detach，
        #     因为 dispatch 只需 expert_id / use_physics 做索引，本就不需要梯度。
        #     若带梯度的张量（如 gen_probs）随 decision 跨多个 block 传递，
        #     DDP 会对其关联参数（gen_router.gate）重复触发 hook → ready twice。
        #   loss 字段（router_loss/balance_loss）→ 保留梯度，但只在 block8 那次
        #     被 encoder 取走用于反传，不随 decision 进入后续 block 的计算。
        return {
            # —— 路由索引：无梯度，可安全跨 block 共享 ——
            "use_physics":     use_physics.detach(),       # [B] bool
            "is_general":      is_general.detach(),         # [B] bool
            "phys_expert_id":  phys_expert_id.detach(),     # [B,N] 17类argmax
            "gen_expert_id":   gen_expert_id.detach(),      # [B,N]
            # —— 选中专家的 softmax 权重：可导，乘进专家输出 → gate 从 JEPA 学路由 ——
            #    只在第一个 MoE block（持有 router 的层）用可导版；
            #    后续 block 复用时由 forward 内部 detach（见 Block.forward 传参）。
            "phys_w":          phys_w,                      # [B,N] 可导
            "gen_w":           gen_w,                       # [B,N] 可导
            # —— 展示/门控字段：detach ——
            "bin_probs":       bin_probs.detach(),          # [B,2]
            "general_score":   bin_probs[:, 1].detach(),    # [B]
            "cls_scores":      cls_scores.detach(),          # [B,18]
            # —— loss 字段：保留梯度，只在 block8 被 encoder 取走反传 ——
            "physics_loss":    physics_loss,                # 标量（物理软标签，监督物理路由）
            "general_loss":    general_loss,                # 标量（二分类，监督门控）
            "balance_loss":    balance_loss,                # 标量
        }


# ── 5. 每层专家容器（不路由，按身份 dispatch + 共享专家）──────────────────────

class LayerExperts(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0,
                 num_physics_experts: int = 17, num_general_experts: int = 10,
                 num_shared_experts: int = 4, drop: float = 0.0,
                 with_router: bool = False, router_kwargs: Optional[Dict] = None):
        super().__init__()
        hidden_dim = int(dim * mlp_ratio)
        self.dim = dim
        self.num_physics_experts = num_physics_experts
        self.num_general_experts = num_general_experts
        self.num_shared_experts  = num_shared_experts

        self.physics_experts = nn.ModuleList([
            ExpertFFN(dim, hidden_dim, drop=drop) for _ in range(num_physics_experts)])
        self.general_experts = nn.ModuleList([
            ExpertFFN(dim, hidden_dim, drop=drop) for _ in range(num_general_experts)])
        self.general_dense = ExpertFFN(dim, hidden_dim, drop=drop)
        self.shared_experts = nn.ModuleList([
            ExpertFFN(dim, hidden_dim, drop=drop) for _ in range(num_shared_experts)])
        self.shared_weight = nn.Parameter(torch.ones(num_shared_experts) / num_shared_experts)

        # 只有【第一个 MoE 层】持有 router：在自己的 forward 内算一次决策，
        # 后续 MoE 层复用（通过主循环传入 decision）。
        # router 在 block 内部 → JEPA target path 的 no_grad 完整覆盖它 →
        # DDP 不会对 router 参数重复挂 hook（避免 "marked ready twice"）。
        self.with_router = with_router
        if with_router:
            rk = router_kwargs or {}
            self.router = GlobalRouter(
                dim,
                num_physics_experts=num_physics_experts,
                num_general_experts=num_general_experts,
                **rk)
        else:
            self.router = None

    def _dispatch(self, x_flat, expert_id_flat, experts, num_experts, active, weight_flat=None):
        # 用 index_add 累加专家输出（反向稳定，避免 out[sel]=y 的 AMP NaN）。
        # weight_flat: [B*N] token 选中专家的 softmax 权重（可导），乘进专家输出，
        #   让 router gate 通过 JEPA loss 学路由。argmax 选索引(无梯度)，softmax权重传梯度。
        #
        # static_graph 要求每步图固定：每个专家每步都必须出现在计算图中。
        # 但某 batch 某专家可能 0 token → 跳过 → 图变 → static_graph 报错。
        # 解决：0 token 的专家也过一个 dummy 输入（x_flat[:1]）但乘 0 权重，
        #   输出实际为 0（不影响前向数值），梯度也为 0（不影响参数），
        #   仅使专家参数恒在计算图中 → 图结构每步固定。对训练结果零影响。
        out = torch.zeros_like(x_flat)
        for e in range(num_experts):
            sel = active & (expert_id_flat == e)
            idx = sel.nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                # dummy：过第一个 token，乘 0，输出加 0（专家恒参与图，梯度=0）
                y_dummy = experts[e](x_flat[:1]) * 0.0
                out = out.index_add(
                    0, torch.zeros(1, dtype=torch.long, device=x_flat.device),
                    y_dummy.to(out.dtype))
                continue
            y_e = experts[e](x_flat.index_select(0, idx))
            if weight_flat is not None:
                y_e = y_e * weight_flat.index_select(0, idx).unsqueeze(-1).to(y_e.dtype)
            out = out.index_add(0, idx, y_e.to(out.dtype))
        return out

    def compute_decision(self, x_normed, physics_label=None,
                         physics_label_soft=None, gate_threshold=0.0,
                         candidate_k=None, enable_general=False):
        """
        第一个 MoE 层用自己的 router 算路由决策。
        x_normed: 已过 norm2 的特征 [B,N,C]（与专家输入一致）。
        在 block.forward 内部调用 → no_grad 可覆盖。
        """
        assert self.router is not None, "compute_decision 只能在持有 router 的层调用"
        return self.router(
            x_normed,
            physics_label=physics_label,
            physics_label_soft=physics_label_soft,
            gate_threshold=gate_threshold,
            candidate_k=candidate_k,
            enable_general=enable_general)

    def forward(self, x: torch.Tensor, decision: Dict, enable_general: bool = False) -> torch.Tensor:
        B, N, C = x.shape
        x_flat = x.reshape(B * N, C)

        w = F.softmax(self.shared_weight, dim=0)
        shared_out = None
        for wi, exp in zip(w, self.shared_experts):
            y = wi * exp(x_flat)
            shared_out = y if shared_out is None else shared_out + y

        use_physics = decision["use_physics"]
        use_phys_tok = use_physics.unsqueeze(1).expand(B, N).reshape(-1)

        # 选中专家的 softmax 权重（可导）→ 乘进专家输出，gate 从 JEPA 学路由
        phys_w_flat = decision["phys_w"].reshape(-1)
        phys_id_flat = decision["phys_expert_id"].reshape(-1)
        routed_out = self._dispatch(x_flat, phys_id_flat, self.physics_experts,
                                    self.num_physics_experts, active=use_phys_tok,
                                    weight_flat=phys_w_flat)

        if enable_general:
            gen_active = (~use_phys_tok).to(x_flat.dtype).unsqueeze(-1)
            routed_out = routed_out + self.general_dense(x_flat) * gen_active

        out = shared_out + routed_out.to(shared_out.dtype)
        return out.reshape(B, N, C)