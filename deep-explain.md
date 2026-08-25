# PhyWAM v3.6 Adapter：面向研究员与 Agent 的技术报告

> 本文是对 `/data/worldmodel_xzs/phywam_v3_adapter_exp` 当前实现的代码级说明。它区分“论文/同学描述的设计目标”和“仓库里已经存在、且能从代码与产物核验的行为”。文中的路径、张量形状、默认值和训练状态均以当前 checkout 为准；若后续切换分支、配置或 checkpoint，应重新执行文末的 preflight。

## 1. 执行摘要

PhyWAM v3.6 adapter 的核心思想是：把一段机器人视频中与物理和任务阶段有关的观测压缩成少量 768 维物理 token，再通过额外的 cross-attention 注入一个原本已经训练好的 WAM/VA 主干。训练时冻结主干，仅学习物理记忆投影、类型嵌入和最后 8 个 Transformer block 中的物理 cross-attention；推理时由 MoPE-JEPA 从滚动图像窗口预测当前 event/phase，在阶段变化或同阶段需要刷新时更新外部物理记忆。

当前可核验的实现链路如下：

```text
头部 episode 视频 + event 标注
        │
        ├─ 离线：MoPE-JEPA v3.6 → 每个 event segment 一个 768-D 特征
        │                         → physics_features3.6/*.npz
        │
        ├─ 训练：LeRobot latent dataset 读取 NPZ
        │       → 按 phase_tokens 对齐到 WAM latent 时间轴
        │       → [B, M<=8, 768] 外部物理 token
        │       → LayerNorm+Linear → [B, M, 3072]
        │       → WAM 第 22--29 层的物理 cross-attention
        │       → 只更新 adapter 参数
        │
        └─ 在线：最近 16 帧图像 → MoPE event head
                → 标签/置信度/任务合法转移判定
                → append / replace / hold 物理记忆
                → 在真实 action/video 更新边界计算一次 WAM 输出
```

最重要的工程判断有四个：

1. “物理 token 以 cross-attention 注入”是已实现的；它不是把物理 token 拼进 WAM 的 self-attention 序列，而是给后 8 层额外提供一套外部 K/V。
2. “phase 划分”在当前实现中由离线 event segment 和在线 event 分类共同承担，二者不是同一个模块：离线边界来自标注/规则，在线边界来自 MoPE 的分类、置信度、任务序列和 patience 逻辑。
3. 当前 online 更新协议是 `append on transition + replace on confident same-phase refresh + hold otherwise`。它不等同于严格的 phase-entry（只在新阶段第一次出现时写入）协议，也不等同于后续 phase-free `kv_refresh` 协议。
4. 训练日志只证明模型至少训练到 step 16928，并保存了 step 16000 checkpoint；目标配置是 50000 steps，因此不能把当前目录描述成“完整收敛的最终模型”。

## 2. 术语和边界

### 2.1 Phase、event segment 和 physics feature

- **event segment**：连续帧区间，例如 `object_pick_phase` 或 `object_place_phase`。离线构建器把它当作一个可变长度区间，再均匀采样固定数量帧。
- **phase**：供 WAM 记忆更新使用的离散任务阶段。当前代码用 event vocabulary 中的标签充当 phase label，并用 canonical task sequence 约束转移。
- **physics feature/token**：MoPE-JEPA 编码一段 segment 后输出的一个 768 维向量。它携带的是视觉表征和物理/event 监督塑造出的信息，并不等于显式的质量、摩擦、位姿或接触力传感器。
- **physical memory**：WAM 在一次更新时可见的 token 集合，形状通常为 `[B, M, 768]`，其中 `M<=8`。它是外部条件，不是 WAM 的视频 latent，也不是标准 self-KV cache。

### 2.2 当前协议与相邻协议的区别

报告中必须明确以下区别，否则容易把不同实验分支混写：

| 协议 | 何时写 token | 同一阶段是否更新 | 当前仓库状态 |
|---|---|---:|---|
| 当前 v3.6 phase 模式 | 首次确定阶段、合法阶段转移 | 高置信同阶段可 replace | 已实现并由 `wan_va_server.py` 使用 |
| phase-entry | 新阶段第一次进入时写一次 | 通常不写 | 设计讨论/其他分支概念，不能当作当前事实 |
| phase-free `kv_refresh` | 由刷新策略决定，不依赖离散 phase | 可定期刷新 | 其他后续变体，不是本目录默认协议 |
| `targetprob` 转移分数 | 预测 canonical 序列的下一阶段概率 | 取决于阈值 | 其他变体；当前代码使用 label/confidence + sequence gate |

因此，当前报告使用“phase token”时，含义是“由 event/phase 分类驱动的外部记忆 token”，不是宣称系统已经实现了所有 phase-entry 或 phase-free 机制。

## 3. 从目录到运行时：代码地图

| 功能 | 真实入口/文件 | 关键责任 |
|---|---|---|
| WAM 配置 | `wan_va/configs/va_robotwin_train_phywam_event_cfg.py` | 数据集、物理记忆维度、对齐模式、cross 起始层 |
| 冻结训练配置 | `wan_va/configs/va_robotwin_train_phywam_freeze_phys_only_cfg.py` | base checkpoint、冻结策略、优化器和 batch |
| 推理配置 | `wan_va/configs/va_robotwin_infer_cfg.py` | phase 模式、MoPE 默认路径、buffer 与 gate |
| WAM block/物理 CA | `wan_va/modules/model.py` | 3072 维主干、外部 K/V、层选择、门控 |
| Flex cross mask | `wan_va/modules/model.py` 中的 `FlexAttnFunc` | phase memory 的 query/key 可见性 |
| dataset 对齐 | `wan_va/dataset/lerobot_latent_dataset.py` | NPZ 读取、padding、latent frame span |
| 训练冻结与 loss | `wan_va/train.py` | adapter 参数白名单、CFG dropout、MSE |
| checkpoint/config 加载 | `wan_va/modules/utils.py` | 恢复 transformer 配置并同步 PhyWAM 参数 |
| 在线 WAM 服务 | `wan_va/wan_va_server.py` | MoPE 编码、phase 状态机、memory 更新和 action 推理 |
| v3.6 特征构建 | `/data/worldmodel_xzs/phywam_v3/mope-jepa_0706/tools/build_phywam_phys_tokens_v36.py` | 读取 event segment，生成 NPZ |
| 在线启动脚本 | `evaluation/robotwin/launch_server_multigpus.sh` | 显式覆盖 MoPE、phase、gate、label 路径 |

### 3.1 当前核心配置快照

训练配置 `va_robotwin_train_phywam_event_cfg.py` 的关键值：

```python
use_phys_memory = True
phys_memory_dirname = "physics_features3.6"
phys_memory_dim = 768
phys_memory_block_size = 16
phys_memory_align_mode = "phase_tokens"
phys_memory_max_tokens = 8
phys_cross_start_layer = 22
```

冻结训练配置还指定：

```python
base_checkpoint = "/data/worldmodel_xzs/lingbot-va/train_out/lingbot_va_s2_base_8gpu/checkpoints/checkpoint_step_16000"
freeze_backbone = True
attn_mode = "flex"
batch_size = 1
gradient_accumulation_steps = 2
learning_rate = 1e-5
```

配置中的保存名和 W&B 名称仍含 `v3.5` 字样，但物理特征目录和 checkpoint config 已经是 `physics_features3.6`/v3.6 路径。对 Agent 来说，**运行时字段和 checkpoint config 优先于实验名字符串**。

## 4. MoPE-JEPA：物理 token 的来源

### 4.1 输入与编码器输出

MoPE-JEPA 是一个带 mixture-of-physics-experts（MoPE）路由的视觉表征模型。当前 v3.6 builder 的输入约束是：

- 每个 event segment 均匀采样 16 帧；
- 使用 `cam_high`；
- 图像输入大小为 224；
- 使用微调后的 `.pth` checkpoint，而不是一个仅存在于目录中的任意文件；
- 默认关闭 general branch，物理路由和 event head 才是目标路径；
- `infer_extract` 得到最终 encoder token 后，对 token 维做均值池化，输出一个 768 维向量。

单个 block 的抽象计算可写成：

\[
X\in\mathbb{R}^{16\times 3\times224\times224},\qquad
H=E_\theta(X)\in\mathbb{R}^{N\times768},\qquad
z=\frac1N\sum_{n=1}^{N}H_n\in\mathbb{R}^{768}.
\]

这里的 `N` 是最终 encoder token 数，不是 16 帧数；帧维已在视觉编码器内部融合。`feature_pooling=mean_final_encoder_tokens` 说明最终落盘的是 `z`，而不是每帧 token。

### 4.2 MoPE 路由和两个监督头

可以把 MoPE 的抽象结构写成：

\[
h=E_\theta(x),\quad
r=R_\phi(h),\quad
u=\sum_{k=1}^{K}\pi_k(h)E_k(h),
\]

其中 `E_k` 是物理专家，`R_φ` 是 router，`π_k` 是专家权重，`u` 是用于下游的物理表征。当前事件训练还接一个 event classifier：

\[
p_\text{event}=\operatorname{softmax}(W_e u),
\qquad
\hat y=\arg\max_c p_\text{event}(c).
\]

在在线服务中，`u` 经 `x_vis.mean(dim=0)` 得到候选的 768 维输入；event head 输出 label/conf/probs，供 phase 状态机判断是否推进。

### 4.3 v3.6 两阶段微调语义

`README_PHYSICS_FEATURES3_6.md` 所描述的训练分两阶段：

1. **Stage A head warmup**：主要训练 event head，使视觉表征能区分事件/阶段。
2. **Stage B joint adaptation**：解冻规定的 MoE、predictor 等部分，冻结 physics router，general branch 关闭，同时优化 JEPA、SIGReg、balance、physics 和 event loss。

抽象联合目标为：

\[
\mathcal L=\lambda_j\mathcal L_\text{JEPA}
 +\lambda_s\mathcal L_\text{SIGReg}
 +\lambda_b\mathcal L_\text{balance}
 +\lambda_p\mathcal L_\text{physics}
 +\lambda_e\mathcal L_\text{event}.
\]

本文不把上式的每个权重当成 WAM 训练 loss；它们属于 MoPE feature extractor 的预训练/适配阶段。WAM 阶段只读取其输出 token，并通过下游动作/视频目标学习“如何使用”它。

## 5. v3.6 离线特征构建

### 5.1 segment 到 NPZ 的过程

`build_phywam_phys_tokens_v36.py` 的 episode 级流程为：

```text
episode 视频/帧索引
  → event_segments_for_episode()
      ├─ 优先读取 event JSON segment
      ├─ 必要时调用 build_event_segments()
      └─ 按明确 fallback policy 处理缺失标注
  → 每个 segment 均匀采样 16 帧
  → MoPE infer_extract
  → 对最终 encoder token 求均值
  → 按 chunk 写入 physics_features3.6/episode_*.npz
```

在样例 `episode_000112.npz` 中，观测到：

```text
features: (2, 768), float32
starts:   [0, 100]
ends:     [100, 167]
episode_len: 167
camera: cam_high
segment_mode: event
event_labels: [object_pick_phase, object_place_phase]
```

这说明该 episode 的两个 feature 分别代表两个变长阶段，而不是每 16 帧写一个 token。`starts/ends` 是原始 episode 帧坐标，后续 dataset 再把它们投影到 WAM latent 时间轴。

### 5.2 关键安全检查

builder 在加载 MoPE checkpoint 时有几个不能省略的 guard：

- 路径必须解析到 `.pth` 文件；不能只检查 checkpoint 目录存在；
- state dict 的关键 shape 必须与当前模型一致；
- 精确形状覆盖率至少达到 95% 左右，否则拒绝静默加载；
- `enable_general=False` 等构建参数必须和训练得到的特征版本一致；
- 输出 NPZ 必须保存 provenance，便于追溯 encoder、checkpoint、采样和 pooling。

样例 provenance 表明当前特征由 `mope_jepa_0706_global_router`、`mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth`、`uniform_full_event_segment` 和 `mean_final_encoder_tokens` 产生。Agent 修改 builder 时，应把这些字段视为数据契约，而不是普通注释。

### 5.3 为什么不能把 768 维 feature 叫成物理量

`z∈R^768` 是学习到的 latent。它可能编码接触、运动、遮挡、物体状态等与物理相关的线索，但没有单位，也没有保证某一维对应摩擦系数或质量。更准确的论文措辞是：

> physics-aware visual latent / physical context token

除非另有可解释性实验，不应写成“恢复了真实物理参数”。

## 6. 数据集契约与时间对齐

### 6.1 NPZ 读取

`wan_va/dataset/lerobot_latent_dataset.py` 对每个 episode 读取：

- `features` 或兼容键 `phys_feat`：`[S,768]`；
- `starts`、`ends`：原始帧边界；
- `episode_len`/元数据：原始视频长度；
- 可选 `event_labels`、`feature_version`、`origin` 等 provenance。

读取失败、shape 不匹配或版本不一致时，正确行为是显式报错或按配置的禁用策略处理；不应把一个错误维度的数组 reshape 成看似合法的 token。

### 6.2 phase_tokens 打包

在 `phys_memory_align_mode="phase_tokens"` 下，dataset 不把每帧物理特征复制到整个时间序列，而是构造一个最多 `M=8` 个元素的外部 memory：

\[
P=[z_1,\ldots,z_S,0,\ldots,0]\in\mathbb{R}^{M\times768},
\]

并返回 `P` 与 `valid_mask`。`S>M` 时必须有明确截断策略；当前配置上限为 8，因此长任务的早期 token 会被窗口化/截断，不能假设所有历史阶段永久保留。

### 6.3 从原始帧到 WAM latent span

WAM 视频 latent 并不一定是一帧一个 token。dataset 使用 latent 时间轴的 key frame id（代码中通过 `raw_frame_ids[::4]` 等采样关系）计算每个 phase 的 `phys_mem_spans`：

\[
\text{span}_i=(s_i^{latent},e_i^{latent}),\qquad
s_i^{latent}=\operatorname{map}(s_i^{raw}),
\quad e_i^{latent}=\operatorname{map}(e_i^{raw}).
\]

因此必须同时检查：

1. NPZ 的 `starts/ends` 是否是原始帧坐标；
2. `raw_frame_ids` 的抽样步长是否和当前 VAE/latent 编码器一致；
3. `phys_mem_spans` 是否落在当前 batch 的 latent 序列范围内；
4. padding token 的 span 是否被 mask 掉。

一个常见错误是把 `starts=[0,100]` 直接当作 latent index；这会让 phase boundary 偏移数倍，训练仍能跑，但 cross-attention 看见的时间关系是错的。

### 6.4 数据输出抽象

送入 WAM 的核心字段可概括为：

```python
batch["phys_memory"]      # [B, M, 768]
batch["phys_memory_mask"] # [B, M]，padding=false
batch["phys_mem_spans"]   # [B, M, 2]，phase 对应的 latent start/end
batch["phys_memory_origin"]
```

`origin` 不是网络输入的数值特征，但对离线/在线一致性、实验复现和错误定位非常重要。

## 7. WAM 主干中的物理 cross-attention

### 7.1 主干规模和 adapter 插入位置

`wan_va/modules/model.py` 中的 Wan Transformer 有 30 个 block，主隐藏维度 `d=3072`，24 个 attention head，每个 head 的维度为 128。配置 `phys_cross_start_layer=22` 时，按 0-based layer index 计算，只有 block 22--29 八层启用物理 cross-attention。

每个普通 block 的顺序是：

```text
hidden
  → self-attention
  → text cross-attention
  → physics cross-attention（仅 active layer）
  → FFN
```

因此该 adapter 是“late-layer conditioning”，不是从输入层开始重写视频表征。前 22 层先形成通用视频/文本表示，最后 8 层再用物理上下文调整动作/视频生成所需的局部决策。

### 7.2 物理 token 投影

数据侧 token 是 768 维，WAM block 的 hidden 维度是 3072。模型中有独立的 projector 和类型嵌入：

\[
K_p=V_p=\operatorname{TypeEmbed}
       \left(\operatorname{Linear}(\operatorname{LayerNorm}(P))\right)
       \in\mathbb{R}^{B\times M\times3072}.
\]

代码中 `phys_memory_projector` 负责 `LayerNorm(768) → Linear(768,3072)`，`phys_memory_type_embed` 区分物理 memory 与其他条件。它不改变 token 数 `M`，只改变通道维度。

### 7.3 数学形式

对一个 active block，令主干 hidden 为 `H∈R^{B×T×3072}`，投影后的物理记忆为 `P∈R^{B×M×3072}`。物理 cross-attention 为：

\[
Q=HW_Q,\quad K=PW_K,\quad V=PW_V,
\]

\[
A=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}+\mathcal M\right),
\quad
O=AW_V',
\]

\[
H' = H + \alpha_\text{phys}\,\sigma(g_\ell)\,O.
\]

其中：

- `d_h=128`；
- `𝓜` 是 padding/span/causal 可见性 mask；
- `g_ℓ` 是第 `ℓ` 个 block 的可学习 gate logit；
- `σ(g_ℓ)` 初始约为 0.1；
- `α_phys` 是运行时传入的物理条件尺度，推理时还会乘 phase confidence gate 和 task gate。

### 7.4 零初始化和渐进接入

配置 `phys_zero_init=True` 时，物理 attention 输出投影 `to_out[0]` 以零值初始化，同时 gate 初始为较小的正值。于是训练初期近似有：

\[
H'\approx H,
\]

adapter 先从 base WAM 的行为出发，再逐步学习物理残差。这种设计有三个工程效果：

1. 避免随机物理支路一开始破坏 base policy；
2. 允许只训练新增参数而保持 base 表征稳定；
3. 可直接用 residual ratio/gate 监控物理支路是否真正被使用。

“token 有信息”和“控制器会利用 token”是两个不同命题。零初始化只保证安全接入，不保证最终动作一定改善。

### 7.5 phase 模式不是 self-KV 拼接

`_prepare_phys_memory_tokens` 在 phase 模式下返回独立的 `phys_cross_hidden_states`、mask 和 spans；它不会把物理 token 添加到 `hidden_states` 的视频序列，也不会为它们追加视频 rotary embedding、时间 embedding 或 diffusion timestep。因而物理 token 是外部 K/V：

```text
Q 来自主干视频/action hidden
K,V 来自物理 memory projector
主干视频 self-attention 的 T 不变
```

legacy `self_tokens` 模式才会把 token 拼到 self-attention 序列中；当前配置是 `phase_tokens`，不能用 self-token 的序列长度直觉来解释它。

### 7.6 Flex mask 的当前语义

phase memory 使用 `attn_mode="flex"`。`FlexAttnFunc._get_phys_cross_mask_mod` 会根据 query frame id、每个 memory token 的 start/end id 和 valid mask 生成 cross mask。代码当前的 predicate 以 `query_frame_id >= phys_start_ids` 为主要生效条件；`end_ids` 在接口中存在，但应在修改或论文描述前再次确认它是否真正参与了最终 predicate。

这意味着当前实现更接近“从 token 生效的阶段开始，后续 query 可以累积看见该 token”的 start-triggered memory，而不是严格的 `[start,end)` 窗口 mask。Agent 若要改变它，必须同时更新：dataset span、attention mask、单元测试、离线/在线对齐和论文公式。

## 8. 冻结训练：究竟训练了什么

### 8.1 加载和配置同步

训练脚本先从 base checkpoint 加载 WAM transformer，再通过 `load_transformer` 和 `sync_transformer_phywam_config` 同步：

- `use_phys_memory`；
- `phys_memory_dim=768`；
- `phys_zero_init`；
- `phys_gate`；
- `phys_cross_start_layer=22`；
- `phys_memory_align_mode`；
- `attn_mode=flex`。

如果 checkpoint 自带 config 与 Python config 不一致，不能只看 launcher 参数。实际运行时的最终结构应以 checkpoint transformer config、加载日志和模型参数名三者交叉确认。

### 8.2 冻结白名单

`_is_phys_trainable_name` 只允许以下类型的参数训练：

```text
phys_memory_projector.*
phys_memory_type_embed.*
blocks.22--29.*.attn_phys.*
blocks.22--29.*.norm_phys.*
blocks.22--29.*.phys_gate_logit
```

base WAM 的 self-attention、text cross-attention、FFN、VAE、文本编码器等被冻结。当前日志报告约 304.6M trainable、5.09B frozen；这个比例说明实验是在“巨大冻结模型上训练条件支路”，而不是全量微调。

### 8.3 训练目标

`compute_loss` 仍然使用 WAM 原有的视频/动作预测目标，通常是 diffusion/flow 风格的加权 MSE。抽象写成：

\[
\mathcal L_\text{WAM}
 =\frac{1}{|\Omega|}\sum_{(b,t,c)\in\Omega}
 w_{b,t,c}\left(\hat\epsilon_{b,t,c}-\epsilon_{b,t,c}\right)^2,
\]

其中 `Ω` 由视频和 action mask 决定，权重可区分 action/video 区域。物理 token 没有单独的 token reconstruction loss；它通过下游生成目标获得梯度：

\[
\frac{\partial\mathcal L}{\partial P}
 =\frac{\partial\mathcal L}{\partial H'}
  \frac{\partial H'}{\partial P}.
\]

这正是“MoPE 负责产生物理表征，WAM 负责学习使用它”的分工。

### 8.4 CFG dropout 和物理条件依赖

配置开启 `phys_cfg_dropout`、概率约为 0.1。`_sample_phys_cfg_dropout` 在 DDP rank 间同步，使一个 batch 对物理条件的丢弃决定一致。被 drop 时，训练步骤把 `phys_condition_scale=0`；否则为 1。

它的目的类似 classifier-free guidance 的条件 dropout：

- 让 base 分支在缺少物理条件时仍可工作；
- 防止模型把物理 token 当作唯一信息源；
- 推理时可以显式调节条件强度。

但它也会降低物理支路收到的有效梯度比例。研究员比较不同实验时，应记录 `phys_cfg_dropout_prob`，不能只比较最终 loss。

### 8.5 监控量如何解释

训练日志有 `phys_memory_monitor` 一类统计，例如：

- `residual_ratio_mean/max`：物理 residual 相对主干 hidden 的幅度；
- `phys_gate_mean`：各 active layer gate 的平均 sigmoid 值；
- 物理 token 范数、mask 有效比例和 span 合法率。

在已保存日志的 step 16920 附近，residual mean 约 0.036、max 约 0.099，gate mean 约 0.107。它们能证明支路有非零活动，但不能单独证明控制性能提升；必须配合 action success、phase transition accuracy、条件/无条件 ablation。

## 9. 训练产物与当前实验状态

当前可见 checkpoint：

```text
train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/
└── checkpoints/checkpoint_step_16000/
    ├── transformer/config.json
    ├── diffusion_pytorch_model.safetensors
    ├── vae/
    ├── text_encoder/
    ├── tokenizer/
    └── training_state.pt
```

`transformer/config.json` 已核验包含：

```json
{
  "use_phys_memory": true,
  "phys_memory_dim": 768,
  "phys_zero_init": true,
  "phys_gate": true,
  "phys_cross_start_layer": 22,
  "num_layers": 30,
  "attn_mode": "flex",
  "phys_memory_align_mode": "phase_tokens",
  "phys_memory_max_tokens": 8
}
```

训练日志从 step 16000 继续到约 16928，随后收到 SIGTERM。目标是 50000 steps。因此推荐论文/报告使用以下 provenance 表述：

> We analyze the step-16000 saved adapter checkpoint from a run that had progressed to step 16928 before termination; this directory does not establish completion of the nominal 50000-step schedule.

不要写“训练完成”或“最终 checkpoint”，除非另有完整运行日志和最后一步产物。

## 10. 在线推理：MoPE、phase 状态机和 WAM 更新边界

### 10.1 启动时的实际覆盖关系

`evaluation/robotwin/launch_server_multigpus.sh` 会显式传给 server：

```text
PHYS_MEMORY_INFER_MODE=phase
MOPE_REPO=/data/worldmodel_xzs/phywam_v3/mope-jepa_0706
MOPE_CKPT=.../mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth
PHYS_PHASE_BUFFER_SIZE=16
PHYS_PHASE_SWITCH_PATIENCE=1
PHYS_PHASE_CONF_THRESHOLD=0.5
PHYS_USE_PHASE_CONFIDENCE_GATE=1
PHYS_USE_TASK_PHASE_GATE=1
```

因此 `va_robotwin_infer_cfg.py` 中的旧默认 MoPE 路径不能单独作为真实运行路径。Agent 排障时按优先级读取：launcher 显式参数 → server argparse → Python config → checkpoint config。

### 10.2 在线 MoPE 编码器

`OnlinePhysMemoryEncoder` 启动时优先寻找：

```text
${mope_repo}/tools/build_phywam_phys_tokens_v36.py
```

找到后导入 v3.6 builder；只有在该路径不可用时才 fallback 到旧本地 builder。启动日志会记录 builder path、MoPE repo、checkpoint、event label path 和构建参数。当前在线参数包括：

```text
num_frames=16
image_size=224
candidate_k=5
threshold=0
enable_general=False
```

每次 `encode_phase`：

1. 从滚动 buffer 取最近 16 帧；
2. 对每帧做相同的 `cam_high` 图像预处理；
3. 调用 MoPE encoder 得到视觉 token；
4. 对最终 encoder token 求均值得到 `x_vis∈R^768`；
5. 调用 event head 得到 `label`、`confidence` 和 `probs`；
6. 返回供状态机使用的 candidate，而不是立即修改 WAM memory。

### 10.3 event vocabulary 与任务序列

在线服务加载 event metadata：

- 事件词表当前为 28 类；
- 从 event JSON 读取 episode-level canonical phase sequence；
- 预计算最常见/合法的任务阶段序列；
- 对 `no_event`、低置信预测和不符合任务顺序的预测进行过滤。

这一步把视觉分类从“局部 argmax”变成“带任务先验的有限状态更新”。它可以减少阶段抖动，但也可能把真实的异常动作误判成非法转移；所以应记录 raw prediction 与 accepted transition 两条日志。

### 10.4 状态变量

可以将 `VA_Server` 的 phase memory 状态抽象成：

```python
phase_tokens:   list[TokenRecord]  # 最多 8 个
current_phase:  Optional[str]
pending_phase:  Optional[str]
pending_count:  int
phase_buffer:   deque(maxlen=16)
active_conf:    float
task_gate:      float
```

每个 `TokenRecord` 至少包含：token `[768]`、phase label、confidence、start frame/step、来源和有效标志。reset episode 时应清空这些状态，不能把上一条任务的 memory 带进下一条任务。

### 10.5 phase transition 判定

`_phase_transition_target` 的语义可以写成：

```python
if label == "no_event" or confidence < threshold:
    return INVALID

if current_phase is None:
    return FIRST(label)

if label == current_phase:
    return SAME_PHASE_REFRESH(label)

if task_sequence is not None:
    expected = next_phase_in_task(current_phase)
    if label != expected:
        return INVALID

if patience_count(label) < switch_patience:
    return PENDING

return TRANSITION(label)
```

当前 `switch_patience=1` 意味着一帧/一次窗口的合法新标签即可触发切换；如果改大，可以抑制抖动，但会延迟 token 写入。`threshold=0.5` 和 `candidate_k=5` 需要在实验记录中固定，否则 phase transition 指标不可比。

### 10.6 memory 更新协议

`_update_phase_phys_memory` 的当前行为是：

| 条件 | 行为 |
|---|---|
| 首次得到合法 phase | append 一个 token，设为 current |
| 得到合法的 expected next phase | append 新 token，更新 current，必要时重新编码当前 payload |
| 高置信度且 label 与 current 相同 | replace 当前 token（refresh） |
| 低置信度、`no_event` 或非法跳转 | hold，保留旧 memory |
| token 数超过 8 | 按实现的 max-token 策略截断 |

这里的 replace 不是把整个历史 memory 重建一遍；它只更新当前 phase 对应的 payload。与 phase-entry 论文叙述相比，这是更积极的同阶段刷新策略。

### 10.7 物理条件尺度

送入 WAM 前，server 计算：

\[
\alpha_\text{phys}
 =g_\text{conf}(c_\text{phase})
  \cdot g_\text{task}(s_\text{task}),
\]

其中 `g_conf` 是 phase confidence gate，`g_task` 是任务序列 gate。两者都可能小于 1；低置信或任务状态不确定时，物理支路被软关闭，而不是强行替换 base policy。

### 10.8 WAM 的真实更新边界

`_compute_kv_cache` 在真实 observation/action update boundary 调用在线 phase update。一次 WAM action 推理内部的 denoising steps 不会每步重新跑 MoPE，也不会每步刷新 phase memory：

```text
new observation
  → phase update（必要时一次）
  → prepare external physical memory
  → action/video denoising loop 重用该 memory
```

如果初始调用已经有 phase tokens，action inference 会复用它们；只有没有 token 时才会触发首次更新。这是 latency 和稳定性设计，也是一个重要的 causal assumption。

## 11. 训练与推理的一致性审计

### 11.1 一致的部分

当前实现已经保持以下关键一致性：

1. 离线 NPZ 和在线 builder 都使用 v3.6 feature directory/encoder 语义；
2. 离线和在线都使用 16-frame、224 输入、`cam_high`、768-D token；
3. WAM 训练 checkpoint 和 inference server 都启用 `phase_tokens`、`phys_cross_start_layer=22`、`max_tokens=8`；
4. checkpoint config 采用 `attn_mode=flex`，与 phase span mask 需求匹配；
5. 在线 launcher 显式把 MoPE checkpoint 和 event label path 传给 server，避免完全依赖旧 Python 默认值。

### 11.2 仍然需要警惕的差异

#### 差异 A：离线 feature 是标注 segment，在线 feature 是滚动预测窗口

离线 builder 对真实 event segment 采样 16 帧；在线 server 对最近 16 帧进行推理。这两者的分布不同：

```text
offline: segment boundary 已知 → 均匀覆盖完整 event
online: 当前位置未知 → 最近窗口可能跨越两个 event 或只覆盖前缀
```

如果 MoPE 对完整 segment 的表征很好，但对前缀/跨边界窗口不稳定，WAM 仍可能得不到可用条件。应单独评估 prefix、boundary-crossing、short-window 三类输入。

#### 差异 B：offline spans 与 online append 时刻

训练时 span 来自原始 event `starts/ends`，推理时 token 的起点来自 server 接受 transition 的时刻。若 transition 被延迟一个 update boundary，phase token 的 start id 会向后偏移；若 server 将 token 视作“当前永久有效”，又可能比训练 mask 更早生效。

#### 差异 C：mask 的 end 语义

dataset 保存 start/end，但当前 Flex mask 代码需要复查 end 是否实际限制 query。若只使用 start 条件，训练和推理共享同一个“累积可见”语义；若未来改成有限窗口，必须重算离线 spans 和在线 records。

#### 差异 D：confidence 与 transition score

当前 server 是 event label + confidence + task sequence gate。不要把它改写成“预测下一阶段概率 `targetprob`”而不改变代码；后者是不同协议。实验日志应同时保存 `raw_label/raw_conf/accepted_phase/gate`。

#### 差异 E：phase refresh 的定义

训练样本的一个 phase feature 是完整 segment 的均值；在线 same-phase refresh 是最新 16 帧的均值。它们并不严格同分布。可以将 refresh 作为研究变量，而不是默认认为它一定有益。

### 11.3 需要做的最小一致性测试

建议在改动模型前先实现三个 CPU/小 batch 测试：

1. **NPZ replay test**：把离线 episode 的每个 segment 切成 16 帧，送入 online builder，比较 cosine similarity 与 NPZ `features`；
2. **boundary test**：构造 query frame 在 start 前、start 上、start 后、end 后的 mask，核对可见 token；
3. **state-machine replay test**：给定固定 label/conf 序列，检查 append/replace/hold、patience、task gate 与预期 trace 完全一致。

没有这三个测试，不应仅凭 loss 下降判断 offline/online pipeline 对齐。

## 12. 可证伪的研究假设和实验设计

下面把“看起来合理”的设计拆成能被实验否定的假设。每个实验都应同时报告控制性能和机制指标。

### H1：物理 token 真的含有比通用视觉 latent 更多的阶段/物理信息

**实验**：冻结 MoPE，训练一个轻量 probe 预测 event label、阶段边界和下一阶段；与未做 physics/event adaptation 的视觉特征、随机 feature 和 WAM hidden baseline 比较。

**指标**：macro-F1、boundary F1、phase transition accuracy、ECE、跨任务泛化。

**判据**：如果 token probe 不超过通用视觉 baseline，问题在 MoPE/标注/采样，而不在 WAM cross-attention。

### H2：WAM 后 8 层足以把 token 转化为动作收益

**消融**：

```text
base WAM（无物理）
late-8 cross-attn（当前）
early-8 / middle-8 / all-layer cross-attn
randomized token / shuffled phase token
zero token / same-phase only / transition-only
```

**指标**：task success、action MSE、contact/placement failure、residual ratio、gate value。

**判据**：只有真实 token 显著优于 shuffled/zero，且提升随 layer placement 稳定出现，才支持“物理条件被控制器利用”。

### H3：phase 更新策略是收益来源，而不只是额外观测量

**消融**：

1. transition-only（严格 phase-entry）；
2. same-phase replace（当前）；
3. 固定周期 refresh；
4. 每次 update 都重编码；
5. 永不更新 token。

比较 phase transition precision/recall、平均 action latency 和任务成功率。

**关键风险**：更频繁的 refresh 可能提高新鲜度，却引入 observation jitter；严格 phase-entry 更稳定，却可能错过阶段内部的接触变化。

### H4：task sequence gate 的收益来自抑制错误转移，而不是把异常吞掉

**实验**：关闭 task gate、只保留 confidence gate、两者都开；额外构造故意跳阶段/异常操作 episode。

**指标**：非法转移率、恢复率、误拒绝率、异常 episode 的安全动作比例。

如果 task gate 让正常任务成功率提高，却在异常任务中一直 hold 旧 token，应在论文中明确这是安全-适应性 trade-off，而不是简单称为更强的 phase detector。

### H5：低置信度时软缩放优于硬替换

比较：

```text
alpha = confidence * task_gate
alpha = 1 whenever label is accepted
alpha = 0 below threshold, else 1
```

绘制 confidence bucket 与成功率/残差的关系。若 confidence 未校准，乘法 gate 可能把有用但低置信的 token 过度压低；需要 calibration 或 learned gate。

## 13. 面向 Agent 的操作契约

### 13.1 启动前 preflight

任何 Agent 在训练或推理前都应按以下顺序检查，不要先启动 GPU 进程：

```bash
cd /data/worldmodel_xzs/phywam_v3_adapter_exp

# 1) checkpoint 与物理配置
python - <<'PY'
import json
p = "train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/checkpoints/checkpoint_step_16000/transformer/config.json"
cfg = json.load(open(p))
for k in ("use_phys_memory", "phys_memory_dim", "phys_cross_start_layer",
          "num_layers", "attn_mode", "phys_memory_align_mode",
          "phys_memory_max_tokens"):
    print(k, cfg.get(k))
PY

# 2) NPZ schema 和版本
python - <<'PY'
import glob, numpy as np
p = glob.glob("/data/public_data/xzs_data/lingbotva-post-training-dataset_s2/lerobot_robotwin_eef_aug_500_s/*/physics_features3.6/chunk-000/*.npz")
print("count", len(p))
if p:
    x = np.load(p[0], allow_pickle=True)
    print({k: x[k].shape for k in x.files if hasattr(x[k], "shape")})
    print("feature_version", x["feature_version"] if "feature_version" in x.files else "<missing>")
PY

# 3) launcher 的显式覆盖
rg -n "PHYS_MEMORY_INFER_MODE|MOPE_REPO|MOPE_CKPT|PHYS_PHASE|EVENT_LABEL" \
  evaluation/robotwin/launch_server_multigpus.sh

# 4) 只读检查脚本语法
bash -n evaluation/robotwin/launch_server_multigpus.sh
```

上面第二段的 glob 指向当前实际数据根 `/data/public_data/xzs_data/lingbotva-post-training-dataset_s2`；正式执行前仍应先用 `test -d` 验证数据挂载状态。

### 13.2 关键不变量

Agent 修改代码后应保持：

```text
feature_dim == phys_memory_dim == 768
projector.out_features == WAM hidden dim == 3072
phys_cross_start_layer < num_layers == 30
phase memory token count <= phys_memory_max_tokens == 8
attn_mode == flex whenever phys_mem_spans is supplied
offline feature version == online builder version == checkpoint provenance
```

如果改变 `phys_memory_dim`，必须同时修改 projector、NPZ validator、online builder output 和 checkpoint compatibility；如果改变 `phys_cross_start_layer`，必须重新初始化/加载对应 block 的 adapter 参数。

### 13.3 分层排障

把失败定位到四层，而不是笼统说“模型没学到”：

1. **数据层**：NPZ 缺失、shape、版本、边界和 padding；
2. **编码层**：MoPE checkpoint、输入预处理、event label 和在线/离线分布；
3. **注入层**：projector、mask、late layer、gate、CFG dropout、参数冻结；
4. **控制层**：phase 状态机、task gate、动作更新边界和机器人执行。

推荐的最短证据链：

```text
NPZ feature 非零且 shape 正确
 → online builder replay 相似
 → WAM batch phys mask 有效
 → active block gate/residual 非零
 → shuffled token 让性能下降
 → 真实任务指标改善
```

缺少中间任何一环，都只能说“链路运行了”，不能说“物理 token 改善了控制”。

## 14. 配置、命名和可复现性风险

### 14.1 `v3.5` 名称残留

训练输出目录、save name、W&B name 仍包含 `v3.5`，而物理目录和 transformer config 是 v3.6。研究员读日志时容易误以为使用了旧 builder。建议后续做一次非破坏性重命名或在 manifest 中明确：

```yaml
experiment_family: phywam_v3_adapter
feature_version: physics_features3.6
run_label_legacy: v3.5
mope_builder: build_phywam_phys_tokens_v36.py
```

不要通过移动旧 checkpoint 来“修正”名字；应保留原路径并写 manifest/alias。

### 14.2 默认值和 launcher 覆盖冲突

infer config 默认 MoPE repo/checkpoint 仍指向旧 `mope-jepa` 路径，但 launcher 覆盖到 `mope-jepa_0706` v3.6。离线脚本、server、训练 config 三处必须记录最终 resolved values。推荐启动时打印一份 JSON manifest：

```json
{
  "wam_checkpoint": ".../checkpoint_step_16000",
  "mope_repo": ".../mope-jepa_0706",
  "mope_checkpoint": ".../checkpoint-100.pth",
  "feature_version": "physics_features3.6",
  "phase_mode": "phase",
  "max_tokens": 8,
  "cross_start_layer": 22
}
```

### 14.3 训练进度的表述风险

日志有最后的 SIGTERM，不等于 loss 崩溃，也不等于正常完成。应区分：

- **saved checkpoint**：确实写盘；
- **last observed step**：日志最后出现的 step；
- **nominal target**：配置中的 50000；
- **evaluation complete**：有完整评测结果。

当前只能确认前两项，不能自动推导后两项。

## 15. 论文/技术报告中可以怎样准确表述

### 15.1 方法段落模板

可以使用以下英文/中文含义（按实际实验数据补充数字）：

> We first adapt a MoPE-JEPA visual encoder with event/phase supervision. For each annotated event segment, 16 uniformly sampled high-camera frames are encoded and mean-pooled over the final encoder tokens to obtain a 768-dimensional physics-aware latent. During WAM adaptation, these latents are packed as at most eight phase memory tokens and projected to the 3072-dimensional WAM hidden space. The base WAM is frozen, while a gated physics cross-attention branch is trained only in the last eight of 30 transformer blocks. At inference time, a rolling 16-frame MoPE window predicts an event label and confidence; a task-constrained state machine appends tokens on accepted transitions, refreshes the current token for confident same-phase observations, and otherwise holds the memory. The resulting external K/V is reused throughout the denoising loop of one action update.

对应中文可写为：

> 我们首先使用事件/阶段监督适配 MoPE-JEPA 视觉编码器。对每个标注事件段均匀采样 16 帧高位相机图像，对最终编码器 token 求均值得到 768 维物理感知 latent。WAM 适配阶段将这些 latent 打包为最多 8 个阶段记忆 token，并投影到 3072 维主干空间。冻结 base WAM，仅在 30 个 Transformer block 的最后 8 层训练带门控的物理 cross-attention。推理时，滚动 16 帧窗口输出事件标签和置信度；任务序列约束的状态机在合法转移时追加 token，在高置信同阶段时刷新当前 token，否则保持已有记忆。一次动作更新的去噪循环复用同一份外部 K/V。

### 15.2 不应直接写的句子

以下说法需要额外证据，当前目录不能直接支持：

- “每个 token 对应真实的物理参数”；
- “阶段边界由模型精确检测”；
- “物理 token 已证明提升了机器人成功率”；
- “训练已完成 50000 steps”；
- “cross-attention 让物理 token 与视频 token 在 self-attention 中直接拼接”；
- “phase-entry、targetprob 和 kv_refresh 都已在当前版本实现”。

## 16. 研究诊断：如果性能没有提升，先问哪一个问题

按因果顺序排查：

```text
Q1. MoPE token 能否被独立 probe 预测 phase/边界？
    └─ 否：先修 MoPE、标注或采样。

Q2. 离线 NPZ 与在线 16-frame replay 是否一致？
    └─ 否：修 preprocessing、窗口、checkpoint 或版本。

Q3. WAM 是否实际看到了有效 token？
    └─ 看 phys mask、span、projector 输出、gate/residual。

Q4. shuffled/zero token 是否破坏性能？
    └─ 否：cross-attention 可能没被使用，或任务不需要它。

Q5. 使用真实 token 后是 phase 错误还是 action 映射错误？
    └─ 分离 detector、memory policy、controller 三层指标。

Q6. update 频率/置信度 gate 是否造成延迟或抖动？
    └─ 对比 transition-only、refresh、always-update。
```

这套顺序避免把所有失败都归因于“模型容量不足”。在当前架构中，MoPE 表征质量、offline/online causal mismatch、memory mask 和 phase 更新策略都可能比 adapter 容量更先成为瓶颈。

## 17. 结论

当前 PhyWAM v3.6 adapter 是一个冻结 WAM + MoPE-JEPA 外部物理记忆的 late-layer conditioning 系统：

1. MoPE-JEPA 将 16 帧 event 窗口压缩成 768 维物理感知 latent；
2. 离线 builder 将每个变长 event segment 写成带边界/provenance 的 NPZ；
3. dataset 把它们按 phase span 对齐为最多 8 个外部 token；
4. WAM 通过 `LayerNorm(768) → Linear(768,3072)` 后的独立 K/V，在第 22--29 层进行 gated physics cross-attention；
5. 训练只更新 adapter 参数，原有 WAM 主干保持冻结；
6. 在线服务用 rolling MoPE event 分类、confidence gate、task sequence gate 和 patience 维护 append/replace/hold 的 phase memory；
7. 一次 action update 内复用 memory，避免每个 denoising step 重跑 MoPE。

因此，该方案的科学问题不是“有没有加一个 token”，而是：

> 学到的物理/阶段 latent 是否在离线到在线的时间语义转换后仍然可靠，并且 late-layer cross-attention 是否能把它转换成可测量的控制收益。

下一步最有价值的工作不是继续堆叠模块，而是完成 NPZ replay、span mask、state-machine trace 和 shuffled-token 四类验证，再用 transition-only / refresh / confidence-gate 消融回答“何时更新物理记忆”这一核心问题。

## 附录 A：当前文件与产物索引

```text
/data/worldmodel_xzs/phywam_v3_adapter_exp/
├── wan_va/configs/va_robotwin_train_phywam_event_cfg.py
├── wan_va/configs/va_robotwin_train_phywam_freeze_phys_only_cfg.py
├── wan_va/configs/va_robotwin_infer_cfg.py
├── wan_va/modules/model.py
├── wan_va/modules/utils.py
├── wan_va/dataset/lerobot_latent_dataset.py
├── wan_va/train.py
├── wan_va/wan_va_server.py
├── evaluation/robotwin/launch_server_multigpus.sh
└── train_out/phywam_v3_s2_phase_cross_phys_v3.5_late8_conf_taskgate_from_base16000_phys36/
    ├── checkpoints/checkpoint_step_16000/
    └── logs/.../train_20260716_105316.log
```

v3.6 MoPE 侧：

```text
/data/worldmodel_xzs/phywam_v3/mope-jepa_0706/
├── README_PHYSICS_FEATURES3_6.md
└── tools/build_phywam_phys_tokens_v36.py
```

数据侧：

```text
/data/public_data/xzs_data/lingbotva-post-training-dataset_s2/
└── lerobot_robotwin_eef_aug_500_s/<task>/physics_features3.6/chunk-*/episode_*.npz
```

## 附录 B：审阅清单

- [ ] report 中的 checkpoint、MoPE `.pth`、feature version 与运行 manifest 一致；
- [ ] `use_phys_memory=True` 且 `phys_memory_dim=768`；
- [ ] `phys_cross_start_layer=22`、`num_layers=30`，确为后 8 层；
- [ ] phase 模式使用 `attn_mode=flex`；
- [ ] NPZ 的 starts/ends 被映射到 latent span，而非直接当 latent index；
- [ ] padding token 不参与 cross-attention；
- [ ] 离线 feature 和 online 16-frame builder 可 replay 对齐；
- [ ] raw MoPE prediction 与 accepted phase transition 都被记录；
- [ ] append/replace/hold trace 可由固定输入序列复现；
- [ ] shuffled/zero token ablation 已完成；
- [ ] 训练进度、checkpoint step 和评测完成状态分开表述。
