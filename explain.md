# PhyWAM v3.6 原理与代码导读：从视频阶段到机器人动作

> 面向具身智能初学者。本文解释的是当前目录
> `/data/worldmodel_xzs/phywam_v3_adapter_exp` 的实际实现，并沿用它在推理时引用的
> `/data/worldmodel_xzs/phywam_v3/mope-jepa_0706` v3.6 代码。
>
> 最后核对日期：2026-08-03。

## 0. 先用一句话说清楚

这个版本的 PhyWAM 可以理解成：

**先用一个视频理解器 MoPE-JEPA，把“当前操作处于什么阶段、画面体现了什么物理交互”压缩成少量 768 维 physics token；再让冻结的 LingBot-VA 视频—动作世界模型在最后 8 个 Transformer 层中，通过新增的 cross-attention 读取这些 token，从而更好地预测后续视频和机器人动作。**

它不是让 MoPE 直接输出动作，也不是把“阶段类别编号”直接塞进 WAM。真正传入 WAM 的是 MoPE 从阶段视频中提取的连续向量；阶段分类主要负责确定该在什么时候创建、刷新或追加这个向量。

可以先记住下面这条主链：

```text
俯视相机视频 cam_high
        │
        ├── 离线：已有阶段标注给出每段的起止帧
        │
        ▼
   MoPE-JEPA 视频编码器
        │  对每个阶段输出一个 768 维连续特征
        ▼
  physics_features3.6/*.npz
        │  最多取 8 个 phase token
        ▼
  768 → WAM hidden size 的 projector
        │
        ▼
WAM 第 22～29 层新增 physics cross-attention
        │  原有 WAM 主干冻结，只训练物理分支
        ▼
联合预测未来视频 latent + 机器人动作
```

在线推理时没有未来视频可用，所以改为：

```text
截至当前时刻的真实观测 → MoPE 预测当前 phase →
根据合法阶段顺序判断是否发生切换 → 刷新/追加 physics token → WAM 预测动作
```

---

## 1. 初学者必须先理解的几个概念

### 1.1 什么是具身智能

具身智能不是只看一张图片回答问题，而是智能体处在环境中，循环执行：

```text
观察环境 → 理解当前状态 → 预测行动后果 → 输出动作 → 获得新观察
```

机器人操作尤其难，因为视觉上很小的变化可能对应完全不同的控制含义。例如：

- 手爪离杯子还有 2 cm：应继续接近；
- 手爪刚接触杯子：应开始闭合；
- 杯子已抓稳：应抬起并运输；
- 杯子已到目标位置：应松开，而不是继续夹紧。

这些时刻在任务语义上属于不同的 **phase（操作阶段）**。

### 1.2 什么是世界模型（World Model）

世界模型学习“如果处在当前状态并采取动作，接下来会发生什么”。本仓库的 LingBot-VA 不是一个纯动作回归器，而是统一建模：

- 视频/视觉 latent 如何随时间变化；
- 机器人动作如何随时间变化。

训练时，两者都被加噪，模型学习 flow-matching/diffusion 的去噪目标。代码中的最终损失是：

```text
L_WAM = L_video + L_action
```

对应实现位于 `wan_va/train.py:457-496`。这里没有再额外训练一个“WAM 内部 phase 分类损失”；phase 和物理类别的分类监督发生在上游 MoPE 微调阶段。

### 1.3 什么是 token

token 不一定是文字。它只是供 Transformer 处理的一个向量单位。

在本项目中至少有三类 token：

- 视频 token：表示局部时空视觉内容；
- 动作 token：表示机器人动作序列；
- physics/phase token：一个 768 维连续向量，概括某个操作阶段的视频物理语义。

“phase token”不是 `object_pick_phase` 这个字符串，也不是类别 ID 1。类别告诉系统“现在是什么阶段”；真正传入 WAM 的 token 是 MoPE 编码器对该阶段视频特征做池化后得到的连续向量。

### 1.4 self-attention 与 cross-attention

self-attention 是同一序列内部互相读取信息；cross-attention 是一组 token 去读取另一组 token。

本项目的 physics cross-attention 可以写成：

```text
Q = W_Q · h_WAM
K = W_K · z_phys
V = W_V · z_phys

A_phys = softmax(QKᵀ / √d) V
h' = h + scale · gate · W_O(A_phys)
```

其中：

- `h_WAM` 是 WAM 当前层的隐藏状态；
- `z_phys` 是投影后的 physics token；
- `scale` 是推理时的 phase 置信度和 task gate；
- `gate` 是每一层可学习的物理分支门控；
- 最后以残差形式加回主干，因此物理信息是“附加建议”，不是替换主干状态。

对应代码是 `wan_va/modules/model.py:700-758`。

---

## 2. 对同学五点描述的逐条校正

你同学的描述抓住了主线，但若写进论文，需要更精确：

| 同学的说法 | 当前代码中更准确的含义 |
|---|---|
| 1. 对头部 episode 视频划分 phase | 使用 `observation.images.cam_high` 俯视相机视频；phase 边界主要来自已有的 task-canonical event JSON，而不是构建 token 时让 MoPE 自己无监督切段。每段确实可变长。 |
| 2. 微调 MoPE（phase 分类 + 物理类别分类） | v3.6 推荐流程是两阶段：Stage A 只热身线性 `event_head`；Stage B 联合 JEPA、物理路由监督和 event 分类。17 类物理监督主要作用在 MoE physics router；28 类 phase/event 由独立线性 `event_head` 分类。 |
| 3. 微调后的 MoPE 构建物理 token | 正确。每个完整 phase 内均匀采样 16 帧，MoPE 输出最终 encoder token，再对 token 维做 mean pooling，得到一个 768 维向量。 |
| 4. token 以 cross-attention 注入冻结 WAM 后八层训练 | 基本正确。但不是训练“冻结后八层本身”，而是在总共 30 层的第 22～29 层挂上新的 `attn_phys`、`norm_phys`、gate，并训练这些新增模块和共享 projector；原 WAM 参数冻结。 |
| 5. MoPE 预测下一阶段来判断是否更新 token | 不完全准确。`event_head` 对最近真实观测预测的是**当前观测所体现的 phase**；服务器再将它与当前 phase 及任务的预期下一 phase 比较。并且 token 不只在切换时更新：同一 phase 且置信度足够时会刷新最后一个 token，发生合法切换时才追加新 token。 |

论文中建议把第 5 点写成：

> 推理时，MoPE 从最新历史观测估计当前阶段及置信度；阶段状态机结合任务的典型阶段顺序确认合法转移。同阶段时刷新当前 physics token，确认转移时追加新阶段 token，随后将累计的阶段记忆供冻结 WAM 的后八层物理 cross-attention 使用。

---

## 3. 第一部分：phase 是怎么来的

### 3.1 当前实际使用的是 task-canonical phase

推理配置使用 28 类词表，其中包括：

```text
no_event
object_pick_phase
object_place_phase
object_transfer_phase
cabinet_open_phase
cabinet_place_phase
approach_grasp_phase
release_settle_phase
...
```

词表位于：

```text
/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/
robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical/
robotwin_s2_8tasks_c50_a500_event_segments_task_canonical_mope_jepa.json
```

这里的 canonical 意思是：同一类任务尽量使用稳定、有限、具有顺序的阶段模板。例如某个任务的典型序列可能是：

```text
object_pick_phase → cabinet_open_phase → cabinet_place_phase
```

它比非常细碎的瞬时事件（接触、闭爪、抬起）更适合当 WAM 的阶段记忆单位，因为 token 数量更少、阶段顺序也更稳定。

### 3.2 为什么是不定长

phase 使用 `[start_frame, end_frame)` 表示。同一个阶段在不同 episode 中可能持续 40 帧、80 帧或 120 帧，所以它不是每 16 帧硬切一块。

仓库中的一个真实 v3.6 样本为：

```text
episode 长度：167 帧

phase 0: [0, 100)   object_pick_phase
phase 1: [100, 167) object_place_phase

输出 features shape = [2, 768]
```

也就是说，这个 episode 最终只有两个 physics token，每个 token 对应一个可变长度阶段。

如果输入标注只有若干“事件中心帧”，`event_language.py:86-180` 会用相邻事件中心的中点构造边界；但当前 v3.6 构建器也能直接读取 JSON 中已经给好的完整 `event_segments`。因此论文不要笼统写成“MoPE 自动完成 phase segmentation”，除非以后真的训练了一个在线边界检测器并单独评估它。

### 3.3 为什么只用 cam_high

默认 camera key 是：

```text
observation.images.cam_high
```

它是俯视/高位相机，不是“episode 的头部几帧”。俯视相机通常更容易看到：

- 双臂与目标物体的整体关系；
- 抓取、运输、放置的全局阶段；
- 物体是否已进入目标容器或区域。

腕部相机细节更丰富，但更容易被遮挡，且视野随手臂剧烈变化。当前实现选择 cam_high 是一种稳定性取舍，不代表多视角一定无效。

---

## 4. 第二部分：MoPE-JEPA 到底学了什么

### 4.1 MoPE-JEPA 的角色

MoPE-JEPA 是上游视频表征模型。它负责从短视频中提炼两种相关但不同的信息：

1. **连续表征**：视频中出现了怎样的运动、接触、物体状态和交互；
2. **离散判断**：属于哪个物理类别、哪个 phase/event 类别。

PhyWAM 真正消费的是第 1 种连续表征；第 2 种监督帮助连续表征更有结构，并在在线推理时承担阶段状态机的观测器。

### 4.2 JEPA 是什么

JEPA 的核心思想不是重建每个 RGB 像素，而是在表征空间中预测被遮挡区域的特征：

```text
可见视频块 ──context encoder──> 表征
                                  │
                                  ▼
                              predictor
                                  │
                                  ▼
                         预测被遮挡块的 latent

完整视频 ──target path（停止梯度）──> 目标 latent
```

主要损失是：

```text
L_JEPA = MSE(predicted_latent, target_latent)
```

这种目标鼓励模型理解可预测的运动和状态变化，而不是把容量花在纹理级像素复原上。实现见
`mope-jepa_0706/models/modeling_pretrain.py:290-344`。

### 4.3 MoPE 的专家结构

当前新架构在 encoder 的 MoE 层中声明：

- 17 个 physics experts；
- 10 个 general experts；
- 4 个 shared experts。

这些数量是**每一个 MoE 层**的数量，不是整个网络总共只有这一组。当前 MoE 位于 encoder 后部的多个 block。

它们的直观分工是：

- physics expert：偏向特定物理现象/类别；
- shared expert：所有样本都可复用的通用运动与视觉能力；
- general expert：为非明确物理类别的视频提供更通用路径。

但 v3.6 RobotWin 路径明确设置 `enable_general=false`，所以当前特征构建主要是：

```text
shared experts + 被 physics router 选中的 physics expert
```

10 个 general experts 虽然在结构中存在，但这条路径没有实际启用。原因是 RobotWin 微调数据缺少可靠的 general/physics 负样本，贸然训练 general gate 容易学习数据集偏差。

### 4.4 “物理类别分类”具体是什么

17 类物理标签不是 WAM 的动作类别。它们用于约束 MoE 的 physics router：给定一个视频，router 应把 token 送到适当的物理专家。

它可以接收 hard label，也可以接收 Qwen 产生的 soft distribution。soft label 能表达：某段视频可能同时具有接触、摩擦、刚体运动等不确定或混合属性，而不是强制认为只有一个绝对真类。

### 4.5 phase/event head 是什么网络

当前 event head 很简单：

```text
最终 encoder visible tokens
        │ mean pooling
        ▼
     768 维向量
        │ Linear(768, 28)
        ▼
  28 类 phase logits
```

代码是：

```python
self.event_head = nn.Linear(encoder_embed_dim, num_event_classes)
```

见 `mope-jepa_0706/models/modeling_pretrain.py:272-275`。推理时对 logits 做 softmax，得到 `label_id`、`label` 和 `confidence`。

注意：**event head 只是一层线性分类器；真正的识别能力主要来自前面的 MoPE encoder 表征。**

### 4.6 v3.6 的两阶段微调

#### Stage A：event head warmup

从 WISA physics-only checkpoint 初始化，冻结 encoder、专家、router 和 predictor，只训练随机新增的 event head。

```text
L_A = confidence_weighted_cross_entropy(event_logits, phase_label)
```

它使用 full-visible 视频特征，不做 JEPA mask 预测。这样先让分类头进入合理尺度，避免随机 event head 在联合训练初期扰乱共享表征。

#### Stage B：joint adaptation

从 Stage A 继续：

- encoder 普通主干冻结；
- MoE physics/shared experts 可训练；
- predictor 可训练；
- physics router 冻结，保留 WISA 学到的路由；
- event head 继续训练；
- general 分支关闭。

联合目标为：

```text
L_B = L_JEPA
    + λ_sigreg  · L_SIGReg
    + λ_balance · L_MoE_balance
    + λ_phys    · L_physics_router
    + λ_event   · L_phase_event
    + L_general
```

当前 launcher 的主要默认权重为：

```text
λ_sigreg  = 0.3
λ_balance = 0.01
λ_phys    = 0.5
λ_event   = 1.0
```

`L_general` 在关闭 general 的当前路径中不是主要信号。

---

## 5. 第三部分：怎样从一段 phase 得到 physics token

v3.6 构建器位于：

```text
/data/worldmodel_xzs/phywam_v3/mope-jepa_0706/
tools/build_phywam_phys_tokens_v36.py
```

对每个 episode，它按以下过程工作。

### 5.1 找到阶段区间

从 event JSON 得到：

```text
phase_i = [start_i, end_i), label_i
```

若指定 event 模式但缺少标注，默认策略是报错；代码也支持 skip 或退化成 fixed block，但正式 v3.6 数据应使用完整 event 标注。

### 5.2 在完整阶段中均匀采样 16 帧

无论该阶段有 30 帧还是 120 帧，都会在整个区间均匀选 16 帧，再缩放/变换为 MoPE 所需的 224×224 输入。

这一步的优点是计算量固定；缺点是极短接触事件可能被稀释，极长阶段的细节也可能丢失。

### 5.3 经过 MoPE encoder

MoPE 最终给出形如：

```text
[N_video_tokens, 768]
```

的最终 encoder token。构建器在 token 维做平均：

```text
z_phase = mean(final_encoder_tokens, dim=token)
```

得到：

```text
z_phase ∈ R^768
```

见 `build_phywam_phys_tokens_v36.py:460-467`。

因此“物理 token”的准确含义是：

> 经过 physics-routing MoPE-JEPA 编码后，对该阶段 16 帧视频的最终时空 token 做全局平均得到的 768 维阶段级视频表征。

它不是显式的质量、摩擦系数、力或接触图。称为 physics token，是因为模型训练和专家路由受物理类别监督，而不是因为每个维度都有可解释的物理单位。

### 5.4 写入 NPZ

每个 episode 的文件主要包含：

```text
features       [num_phases, 768]
block_starts   [num_phases]
block_ends     [num_phases]
event_labels   [num_phases]
event_scores   [num_phases]
camera_key
feature_checkpoint
feature_model_repo
frame_sampling = uniform_full_event_segment
feature_pooling = mean_final_encoder_tokens
```

当前数据根下实际检查到：

- 16 个 `physics_features3.6` 目录；
- 4400 个 episode NPZ；
- 特征 checkpoint provenance 指向 `mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth`。

这些字段很重要，因为它们让我们能追溯“某个 WAM checkpoint 到底吃的是哪一版 MoPE 特征”。

---

## 6. 第四部分：数据加载时怎样把 token 对齐到 WAM

入口在 `wan_va/dataset/lerobot_latent_dataset.py`。

训练配置为：

```text
phys_memory_dirname   = physics_features3.6
phys_memory_dim       = 768
phys_memory_align_mode = phase_tokens
phys_memory_max_tokens = 8
phys_cross_start_layer = 22
```

### 6.1 padding 与 mask

一个 episode 的 phase 数量不固定，但 batch 中张量必须同形，因此 loader 创建：

```text
feat  : [8, 768]   不足 8 个补零
mask  : [8]        哪些 token 有效
spans : [8, 2]     每个 token 对应的时间区间
```

若 phase 多于 8 个，只保留前 8 个特征，并让最后一个 span 覆盖后续时段。对当前 S2 canonical 任务通常少于 8 个，但这仍是一个需要关注的截断策略。

### 6.2 因果可见性

训练使用 `attn_mode='flex'`，根据 phase 的起点建立 physics cross-attention mask。核心约束是：

```text
query_time >= phase_start
```

也就是说，WAM 在阶段开始以前不能读取该阶段 token；阶段开始以后可以继续读取它。因此这里实现的是**累积阶段记忆**：过去阶段 token 不会在 `end_frame` 后立刻消失。

这是一个非常重要的代码细节：虽然 NPZ 同时保存 `block_starts` 和 `block_ends`，当前 `_get_phys_cross_mask_mod` 实际检查了 start，没有用 end 限制可见性。论文若写“每个 token 只作用于自己的阶段区间”会与当前代码不符；更准确的写法是“token 从其阶段起点开始进入可读记忆”。

---

## 7. 第五部分：physics token 如何注入 WAM 后八层

### 7.1 WAM 主干结构

当前 `WanTransformer3DModel` 有 30 层 Transformer block。每一层原本按顺序执行：

```text
1. causal self-attention
2. text cross-attention
3. feed-forward network
```

PhyWAM 在 text cross-attention 后、FFN 前加入：

```text
3. physics cross-attention
```

所以实际顺序变为：

```text
self-attention
    ↓
text cross-attention
    ↓
physics cross-attention  ← 新增
    ↓
FFN
```

### 7.2 为什么只放在后八层

配置 `phys_cross_start_layer=22`，30 层按 0 开始编号，因此启用的是：

```text
22, 23, 24, 25, 26, 27, 28, 29
```

恰好 8 层。

直觉上：

- 前层更多处理局部视觉/动作结构；
- 后层表征更语义化，更适合融入“当前处于抓取还是放置”这种高层条件；
- 少改层数能减少新条件破坏原预训练世界模型的风险；
- 计算和可训练参数也低于 30 层全部注入。

“后八层最好”仍是实验假设，严谨论文需要和全层、后四层、无物理分支等消融比较，而不能只靠直觉宣称。

### 7.3 维度怎样匹配

MoPE token 是 768 维，WAM hidden size 是：

```text
24 heads × 128 dim/head = 3072
```

所以先经过共享 projector：

```text
LayerNorm(768) → Linear(768, 3072) → 加 physics type embedding
```

然后每个后八层都有自己独立的 `attn_phys`，用当前 WAM hidden state 作 query，用 physics token 作 key/value。

### 7.4 为什么不是把 token 拼进文字

较早的 `lingbot-mope` 方案把物理特征投影成“软文字 token”，再与 text embedding 拼接，复用原 text cross-attention。

当前 v3 则保留文字通道不变，单独增加 physics cross-attention。优点是：

- 文字语义和物理记忆职责更清楚；
- 可以单独设置因果 mask、门控、层范围和消融；
- physics token 不需要伪装成文字 token。

代价是新增分支参数很多，也更容易干扰强大的预训练主干，所以必须配合冻结、零初始化和 gate。

---

## 8. 第六部分：冻结训练到底训练了什么

### 8.1 冻结范围

`freeze_backbone=True` 时，训练代码先冻结整个 transformer，再只按参数名开放：

```text
phys_memory_projector
phys_memory_type_embed
attn_phys
norm_phys
phys_gate_logit
```

所以以下原有部分不更新：

- WAM self-attention；
- 原 text cross-attention；
- FFN；
- 视频/动作 embedding 与输出头；
- 原有时间条件模块。

2026-07-16 的真实训练日志显示：

```text
trainable =   304,553,480
frozen    = 5,088,872,670
active physics layers = 8
```

因此“冻结主干”不等于“小模型训练”；8 套 3072 维 cross-attention 仍有约 3.0 亿可训练参数。

### 8.2 三个稳定化设计

#### 设计 A：输出零初始化

每个 `attn_phys` 的输出投影 `W_O` 初始为零，因此刚开始训练时：

```text
h' = h + 0
```

模型一开始严格退化为原 LingBot-VA，而不是被随机物理分支突然扰乱。代码见
`wan_va/modules/model.py:644-647`。

#### 设计 B：可学习 gate

每层 gate 初始为 0.1：

```text
gate = sigmoid(gate_logit) ≈ 0.1
```

网络可以逐层学习物理分支应当多强。注意即使 gate 是 0.1，零初始化输出仍使第一步的实际残差为零。

#### 设计 C：physics CFG dropout

训练中以 `cfg_prob=0.1` 的概率把整个 microbatch 的：

```text
phys_condition_scale = 0
```

即暂时关闭物理条件。所有 FSDP rank 同步执行同一个 drop 决策。这让模型同时见到“有物理 token”和“无物理 token”的情况，降低对上游 MoPE 错误的脆弱性。

### 8.3 训练目标如何反向训练物理分支

WAM 阶段没有 phase CE 或 physics CE，只有原来的：

```text
L = L_video + L_action
```

由于主干冻结，梯度只能改变 projector、physics cross-attention、norm 和 gate。若某种 physics token 有助于降低视频/动作去噪误差，cross-attention 就会学会读取它；若无用，gate 可以保持较小。

这体现了两个不同层次：

```text
MoPE 阶段：让 token 含有物理/阶段信息
WAM 阶段 ：让控制世界模型学会有效使用这些信息
```

“token 可分类”不自动等于“token 能提高机器人成功率”，后者必须用下游动作与任务成功率验证。

---

## 9. 第七部分：在线推理的完整状态机

当前默认 launcher 使用：

```text
PHYS_MEMORY_INFER_MODE=phase
PHYS_EVENT_DETECTOR=mope
PHYS_EVENT_THRESHOLD=0.5
PHYS_PHASE_SWITCH_PATIENCE=1
PHYS_PHASE_BUFFER_FRAMES=16
PHYS_PHASE_CONFIDENCE_GATE=1
PHYS_DEFAULT_TASK_GATE=1.0
```

### 9.1 reset 时做什么

服务器接到新任务后清空：

```text
phase_phys_tokens
current_phase
candidate_phase
recent_observations
active_confidence
```

同时从训练 event metadata 中统计该任务最常见的 phase 序列。例如：

```text
expected = [object_pick_phase, object_place_phase]
```

这是一个**任务先验状态机**，不是纯粹让分类器随意在 28 类之间跳转。

### 9.2 何时运行 MoPE

真实新观测进入 `_compute_kv_cache` 时，会调用 `_prepare_online_phys_memory(..., real_update=True)`，phase 模式再进入 `_update_phase_phys_memory`。

服务器保留最近最多 16 个真实关键观测，并重采样成 MoPE 的 16 帧输入。动作去噪的多步迭代不会每一步重复跑 MoPE；它复用已经构建的 token。

### 9.3 event head 预测什么

MoPE 对最近历史输出：

```text
predicted_label
confidence
feature ∈ R^768
```

这里的 label 是从已观察视频判断出的当前 phase。服务器的 `_phase_transition_target` 再做以下检查：

1. label 不能是 `no_event`；
2. confidence 必须 ≥ 0.5；
3. 如果和当前 phase 相同，不发生切换；
4. 如果任务有已知顺序，新 label 必须正好等于 expected next phase；
5. 连续命中次数达到 patience 才确认。当前 patience=1，即一次高置信命中就切换。

### 9.4 token 的三种更新动作

#### initialize

还没有 token 时：

```text
tokens = [feature_current]
```

#### refresh

预测仍是当前 phase 且 confidence 足够：

```text
tokens[-1] = newest_feature
```

也就是说，当前 phase token 会随新观测更新，不是阶段开始后永远固定。

#### transition

确认进入合法下一阶段时：

```text
tokens.append(feature_new_phase)
```

旧 phase token 保留，形成累积记忆。超过 8 个时只保留最近 8 个。

### 9.5 注入强度

推理时服务器计算：

```text
condition_scale = active_phase_confidence × task_gate
```

进入每个 WAM 层后还会乘该层学到的 `sigmoid(phys_gate_logit)`：

```text
effective_scale_layer
= confidence × task_gate × learned_layer_gate
```

因此低置信度 MoPE 预测不会以满强度干预 WAM。默认 `task_gate=1.0`，也可以针对不适合 physics token 的任务设为 0 或较小值。

### 9.6 用一个抓取—放置例子串起来

假设任务是“把面包放进平底锅”：

```text
t0: 看到手臂接近并抓取
    MoPE: object_pick_phase, conf=0.91
    → initialize pick token
    → WAM 通过物理 CA 读取它并预测抓取动作

t1: 仍在抓取阶段，画面更新
    MoPE: object_pick_phase, conf=0.94
    → refresh pick token
    → 不增加 token 数

t2: 面包已到锅上方
    MoPE: object_place_phase, conf=0.88
    → 与任务 expected next phase 一致
    → append place token
    → memory = [pick token, place token]

t3: 动作 diffusion 多步去噪
    → 不重复运行 MoPE
    → 复用当前 memory 生成本 chunk 动作
```

---

## 10. 训练与推理并不完全相同

这是当前方案最值得改进的地方。

### 10.1 离线 token 看到了完整 phase，在线 token 只看到历史

训练特征对一个完整 `[start, end)` 阶段均匀采样 16 帧。若 token 从 `start` 起就允许 WAM 读取，那么其向量实际上包含该阶段后半段的信息。这会产生潜在的 future leakage：训练时 token 可能“知道本阶段后来发生了什么”，在线推理却不可能知道。

在线阶段用最近真实观测编码，因而是因果的，但它与离线完整阶段特征存在分布差异。

更严格的改进方向是：

- 离线也只用截至 query 时刻的前缀视频构建 causal token；或
- 在每个阶段内构建多个递增前缀 token；或
- 训练一个流式 MoPE/缓存式编码器，使训练和推理使用相同观测窗口。

### 10.2 训练有 span mask，在线 phase token 没有显式 span

训练时 flex mask 控制 token 从 phase start 后可见。在线时 token 只在观测到对应阶段后才创建，所以时间因果性由状态机保证；创建以后，当前推理 query 可以读取所有保留 token。

两者语义接近“累积记忆”，但实现并不完全同构。应通过对齐测试验证同一历史输入在 train-style 和 infer-style 路径中的可见 token 一致。

### 10.3 状态机强依赖任务模板

若真实执行出现：

- 抓取失败后重试；
- 阶段回退；
- 跳过某阶段；
- 同一阶段重复多次；

严格的 “只能去 expected next” 规则可能拒绝正确观测。当前 `patience=1` 又可能对一次误分类过于敏感。

建议评估：

- `patience=1/2/3`；
- 是否允许 self-loop、retry edge 和有限回退；
- 使用 soft posterior/HMM，而不是 hard argmax；
- 把 gripper、action history、图像变化与 MoPE posterior 融合。

### 10.4 refresh 与 transition 需要分别消融

同学的描述只强调切换时更新，但代码中同阶段 refresh 也很重要。应至少比较：

```text
A. 只在 phase 切换时追加
B. 同 phase 持续 refresh
C. 固定频率 refresh
D. 不使用 phase，固定窗口实时编码
```

否则无法判断收益来自“阶段边界”，还是仅仅来自“不断重编码最新视觉”。

---

## 11. 当前目录的真实状态与容易踩坑的默认值

### 11.1 已确认存在的产物

当前能看到：

- `physics_features3.6`：4400 个 episode NPZ；
- WAM checkpoint：`...phys36/checkpoints/checkpoint_step_16000`；
- checkpoint config：`use_phys_memory=true`、`phase_tokens`、`max_tokens=8`、`start_layer=22`、`num_layers=30`；
- 训练日志确认 8 个 active physics layers；
- 在线 server launcher 默认指向 `mope-jepa_0706` joint `checkpoint-100.pth`。

### 11.2 训练没有跑完配置中的 50k steps

v3.6 日志从头训练到约 step 16928 后收到 SIGTERM。目录中保存了 step 16000 checkpoint，而配置目标是 50000 steps。

所以可以说“已有可评估的 step-16000 checkpoint”，不能仅根据目录存在就说“50k 训练完整收敛”。论文报告最终模型时，应明确使用哪个 step，并用验证/RobotWin 成功率选择，而不是默认最后保存点最好。

### 11.3 名称中仍残留 v3.5

若干 `save_root`、实验名仍带 `v3.5`，但有效配置和数据实际是 `physics_features3.6`。例如训练配置继承链里有旧命名，而 launcher 又把保存路径覆盖为带 `phys36` 的目录。

判断实际版本时优先看：

1. checkpoint `transformer/config.json`；
2. 训练启动日志中的 effective config；
3. NPZ 的 `feature_version` 与 `feature_checkpoint`；
4. server 启动日志打印的 builder 和 MoPE checkpoint。

不要只看文件夹名字。

### 11.4 直接用 infer config 与用 launcher 可能不同

`va_robotwin_infer_cfg.py` 中仍有旧 MoPE repo/checkpoint 默认值，而
`evaluation/robotwin/launch_server_multigpus.sh` 会显式覆盖为：

```text
MOPE_REPO = /data/worldmodel_xzs/phywam_v3/mope-jepa_0706
MOPE_CKPT = .../mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth
```

因此复现实验时应记录完整启动命令和 server startup log，不能只写“使用 robotwin_infer config”。

---

## 12. 如何科学验证这套方法真的有效

### 12.1 最基本的受控对照

至少固定同一个 WAM base、数据、训练步数和 seed，比较：

```text
1. Base LingBot-VA，无 physics token
2. PhyWAM，随机/打乱 physics token
3. PhyWAM，正确 token，但 phase 更新关闭
4. PhyWAM，正确 token + phase 状态机
5. PhyWAM，oracle phase 边界（性能上界）
```

如果 4 优于 1，却不优于 2，说明模型可能只是从新增参数获益，而不是使用了有意义的物理表征。

### 12.2 分模块指标

上游 MoPE：

- phase macro-F1，而不只是 accuracy；
- phase transition detection delay；
- confusion matrix；
- physics router soft-label KL/CE；
- 失败重试、遮挡、域外场景下的校准误差。

WAM：

- video flow loss 与 action flow loss；
- physics residual / hidden norm ratio；
- 每层 gate；
- token 打乱后的性能下降；
- 有/无 token 的动作差异。

机器人任务：

- overall success rate；
- 分阶段成功率；
- 抓取成功但放置失败等 failure taxonomy；
- 长程任务完成时间和重试次数；
- 多 seed 置信区间。

### 12.3 最关键的因果性检查

为了排除离线 future leakage，建议做：

```text
完整 phase token vs causal-prefix token
```

如果完整 token 在离线指标很好，但 causal-prefix token 和在线运行显著下降，就说明模型利用了在线不可获得的信息，需要重构 token 构建流程。

---

## 13. 推荐的代码阅读顺序

按下面顺序读，最不容易迷路：

1. `wan_va/configs/va_robotwin_train_phywam_event_cfg.py`
   - 看 v3.6 数据目录、phase token、8 token、后 8 层。

2. `wan_va/dataset/lerobot_latent_dataset.py:148-298`
   - 看 NPZ 如何加载、padding、mask、span 对齐。

3. `wan_va/modules/model.py:800-906,991-1030`
   - 看 projector、物理分支启用层和 phase token 准备。

4. `wan_va/modules/model.py:667-758,1217-1287`
   - 看 physics cross-attention 真正加在哪里。

5. `wan_va/train.py:233-255,333-357,457-524`
   - 看冻结范围、CFG dropout 和 WAM 损失。

6. `mope-jepa_0706/models/modeling_pretrain.py:220-344`
   - 看 JEPA encoder/predictor 和线性 event head。

7. `mope-jepa_0706/engine_for_pretraining.py:31-51,161-245`
   - 看 event、physics、JEPA 等损失如何组合。

8. `mope-jepa_0706/tools/build_phywam_phys_tokens_v36.py:231-269,460-467,664-746`
   - 看 phase 区间、16 帧采样、mean pooling 和 NPZ。

9. `wan_va/wan_va_server.py:45-316,640-864`
   - 看在线 MoPE、phase 顺序、initialize/refresh/transition。

10. `wan_va/wan_va_server.py:556-631,1630-1736`
    - 看置信度/task gate、token 怎样进入动作推理以及何时复用。

---

## 14. 可以用于论文方法部分的简洁表述

下面这段比最开始的五点更贴合当前实现：

> 我们首先依据任务规范化事件标注，将每条机器人操作轨迹划分为若干可变长度的语义阶段。随后在带有物理类别软监督和阶段类别监督的视频数据上适配 MoPE-JEPA：物理监督约束其专家路由，阶段监督训练视频级事件分类头。对每个阶段，我们在完整区间内均匀采样 16 帧，经 MoPE-JEPA 编码并对最终时空 token 全局平均，得到一个 768 维阶段物理表征。下游以冻结的 LingBot-VA 为视频—动作世界模型，将阶段表征投影至主干维度，并在其 30 层 Transformer 的后 8 层增加独立的物理 cross-attention 残差分支。训练仅更新物理 projector、cross-attention、归一化与门控参数，并保留原视频和动作 flow-matching 目标。在线推理时，MoPE 根据最新历史观测估计当前阶段及置信度；任务阶段状态机确认合法转移，同阶段时刷新当前 token，阶段切换时追加新 token，累积阶段记忆再以置信度加权方式条件化动作生成。

这段话仍应在论文限制部分明确：当前离线 token 使用完整阶段，而在线 token 只能使用历史观测，二者存在因果与分布对齐问题。

---

## 15. 最后再检查你是否真正理解了

如果能回答下面问题，就已经掌握了这套系统的核心：

1. phase label 和 physics token 为什么不是同一个东西？
2. 17 类 physics router 与 28 类 event head 分别监督什么？
3. 为什么一个 100 帧阶段最后只产生一个 768 维 token？
4. 为什么说第 22～29 层“主干冻结”，但仍有约 3 亿参数在训练？
5. zero-init 和 0.1 gate 各自解决什么问题？
6. 为什么 WAM 训练没有 phase CE，仍能学会使用 phase token？
7. 在线同一 phase 时为什么也会更新 token？
8. 为什么 event head 更准确地说是在预测当前 phase，而不是直接预测下一 phase？
9. 训练时完整 phase token 可能造成什么 future leakage？
10. 为什么必须用 token shuffle、oracle phase 和 causal-prefix 做消融？

最重要的一句话是：

> **MoPE 负责把历史视频变成“有物理结构的阶段记忆”，WAM 负责判断这份记忆怎样改善未来视频与动作预测；phase 状态机负责决定记忆何时刷新和增长。**
