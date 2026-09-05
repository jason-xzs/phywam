# MoPE-JEPA 0706 / physics_features3.6

这个目录是 `/data2/mope-jepa` 的源码级分支，位置为：

```text
/data/worldmodel_xzs/phywam_v3/mope-jepa_0706
```

这里不保存大文件。`/data2/mope-jepa` 下的 checkpoint 继续用绝对路径引用；`output/`、`pretrained/`、`datasets/`、日志、视频、缓存等大文件没有复制进来。

默认运行环境是：

```text
/data1/miniconda3/envs/worldmodel_mopeva
```

`/data2/envs/mope-jepa` 只作为旧环境参考，不作为当前 0706 训练、评估、构建脚本的默认运行环境。

## 当前微调逻辑

当前推荐路径不是直接沿用旧的 `physics_features3.5` 微调脚本，而是在新架构上做两阶段 RobotWin 适配：

1. 先从 `/data2/mope-jepa/output/stage1_wisa7k_physics_only/checkpoint-100.pth` 初始化。
2. Stage A 只训练新加的 v39 event head。
3. Stage B 再做 JEPA + v35 physics + v39 event 的联合适配。
4. 评估 c1/a2 held-out 后，再构建 `physics_features3.6`。

整体目标是：保留 `/data2` 新 MoPE-JEPA 已学到的物理专家和物理路由能力，只把它适配到 RobotWin 的 v35 物理标签和 v39 事件标签上，最后产出和 PhyWAM phase-token 数据侧契约兼容的新物理特征。

## 为什么默认关闭 enable_general

新 MoPE-JEPA 架构里有一个视频级 general/physics 路由开关：

- `enable_general=true`：启用 general-video 分支，模型可以把视频送到通用专家分支。
- `enable_general=false`：关闭 general 分支，RobotWin 样本稳定走 physics 专家路径。

RobotWin 微调阶段只有 v35 physics 标签和 v39 event 标签，没有可靠的“通用视频负样本”。如果在这个阶段启用并训练 general 分支，二分类/general gate 容易学到 RobotWin 数据集偏差，而不是“这个视频是否应该走物理专家”的真实边界。

因此当前 3.6 路径明确采用：

```text
enable_general=false
```

对应代码行为是：

- 训练参数默认 `--enable_general` 关闭。
- launcher 不传 `--enable_general`。
- `run_jepa_pretraining.py` 会把 `model.encoder.enable_general` 设为 `False`。
- 未使用的 `binary_gate`、`gen_router`、`general_experts`、`general_dense` 参数会被冻结。
- 3.6 构建器默认也关闭 general 分支。

这不是单纯为了省显存，而是为了避免 RobotWin 微调污染 WISA 阶段已经学到的 physics routing。

## Stage A：event head warmup

启动命令：

```bash
cd /data/worldmodel_xzs/phywam_v3/mope-jepa_0706
bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh head 0,1,2,3 fg
```

这一阶段的默认输入 checkpoint 是：

```text
/data2/mope-jepa/output/stage1_wisa7k_physics_only/checkpoint-100.pth
```

训练策略：

- 冻结整个 encoder、MoE 专家、physics router、predictor。
- 只训练新增的 `event_head`。
- 使用 full-visible segment feature，不做 JEPA mask 重建训练。
- event 类别数从 v39 event label JSON 自动推断。
- loss 只有 v39 event 分类 loss，带 event score 置信度加权。

这一阶段的作用是先让随机初始化的 event head 对齐 RobotWin v39 事件空间，避免一开始 joint 训练时 event loss 对共享表征产生过强扰动。

默认输出：

```text
output/mope_jepa_0706_robotwin_v39_head_warmup/checkpoint-10.pth
```

默认只保存最终的 `checkpoint-10.pth`。这一阶段只训练 event head，主要是 warmup，不建议用它作为最终物理特征构建 checkpoint。

## Stage B：joint adaptation

启动命令：

```bash
cd /data/worldmodel_xzs/phywam_v3/mope-jepa_0706
bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh joint 0,1,2,3 bg
```

这一阶段默认从 Stage A 输出继续训练：

```text
output/mope_jepa_0706_robotwin_v39_head_warmup/checkpoint-10.pth
```

训练策略：

- `--freeze_encoder_except_moe`：冻结 encoder 主干，只开放 MoE 相关参数。
- `--train_predictor`：继续训练 JEPA predictor。
- `--freeze_phys_router`：冻结 WISA 阶段学到的 physics router。
- 不启用 `--enable_general`，所以 general/binary 路由相关参数也会被冻结。
- 继续训练 physics experts、shared experts、最终 norm、predictor 和 event head。

联合 loss 结构为：

```text
JEPA
+ sigreg_weight * SIGReg
+ moe_balance_weight * balance
+ physics_cls_weight * v35_physics_loss
+ event_loss_weight * v39_event_loss
+ general_loss
```

其中：

- `JEPA`：latent-space MSE，用于保持预测式视频表征学习。
- `SIGReg`：防止 context 表征 collapse。
- `balance`：MoE 专家负载均衡。
- `v35_physics_loss`：来自 RobotWin v35 物理软标签，用来约束当前样本的物理路由输出保持和 v35 标签一致；Stage B 中 physics router 权重本身是冻结的。
- `v39_event_loss`：来自 RobotWin v39 canonical event 标签，对 event head 提供监督。
- `general_loss`：保留代码兼容项；当前路径关闭 general 分支，因此不作为主要训练信号使用。

默认输出：

```text
output/mope_jepa_0706_robotwin_v39_joint/checkpoint-10.pth
output/mope_jepa_0706_robotwin_v39_joint/checkpoint-20.pth
...
output/mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth
```

Stage B 默认每 10 epoch 保存一次 checkpoint，而不是只保留最终点。这样可以用 held-out c1/a2 评估或训练曲线选择更合适的 checkpoint，避免最终点过拟合或还没收敛时没有回退空间。若存储压力较大，可以用 `JOINT_SAVE_CKPT_FREQ=20` 或 `SAVE_CKPT_FREQ=20` 调大保存间隔。

## W&B 在线日志

`scripts/robotwin_v39/run_robotwin_v39_multitask.sh` 默认启用 W&B online，除 project name 外，W&B 环境配置和 `/data/worldmodel_xzs/phywam_v3/script/run_va_posttrain.sh` 对齐：

```text
WANDB_BASE_URL=https://api.wandb.ai
WANDB_TEAM_NAME=1661825351-beijing-institute-of-technology
WANDB_ENTITY=1661825351-beijing-institute-of-technology
WANDB_MODE=online
```

当前默认 project name 是：

```text
mope_jepa_0706_robotwin_v39
```

每个 epoch 会记录一次训练统计，包括 loss、event_acc、各 loss 分量、学习率、是否保存 checkpoint 等。只在 rank0 初始化和写入 W&B。

如果网络或 W&B 服务异常，可以临时关闭：

```bash
WANDB_MODE=disabled bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh joint 4,5,6,7 bg
```

也可以改成离线：

```bash
WANDB_MODE=offline bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh joint 4,5,6,7 bg
```

## 快速 smoke 检查

建议正式训练前先跑 smoke：

```bash
cd /data/worldmodel_xzs/phywam_v3/mope-jepa_0706

bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh smoke_head 0 fg

FINETUNE_CKPT=/data2/mope-jepa/output/stage1_wisa7k_physics_only/checkpoint-100.pth \
  bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh smoke_joint 0 fg
```

`smoke_head` 只跑很少 step，用来确认 dataset、event labels、event head、checkpoint load 正常。

`smoke_joint` 用 stage1 checkpoint 直接做最小 joint 路径检查，不依赖已经跑完的 head warmup。

## 评估

正式构建特征前，建议先在 held-out c1/a2 上评估：

```bash
cd /data/worldmodel_xzs/phywam_v3/mope-jepa_0706
GPU=0 bash scripts/robotwin_v39/run_eval_robotwin_v39_c1a2.sh
```

默认评估 checkpoint：

```text
output/mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth
```

评估结果会写到：

```text
validation/robotwin_v39_c1a2_metrics.json
```

## 构建 physics_features3.6

联合微调 checkpoint 准备好后，构建新版本物理特征：

```bash
cd /data/worldmodel_xzs/phywam_v3/mope-jepa_0706

GPUS=0,1,2,3,4,5,6,7 \
  bash scripts/physics_features3_6/run_build_physics_features3_6.sh
```

默认输入 checkpoint：

```text
output/mope_jepa_0706_robotwin_v39_joint/checkpoint-100.pth
```

默认输出目录名：

```text
physics_features3.6
```

构建器会做这些检查和处理：

- 要求输入是微调后的 `.pth` MoPE checkpoint，不接受原始 VideoMAEv2 safetensors 直接构建。
- 使用 `mmap=True` 加载大 checkpoint。
- 检查关键 router/expert 权重是否存在。
- 要求 exact-shape tensor coverage 至少达到 95%。
- 对每个完整 v39 event segment 均匀采样 16 帧。
- 提取最终 encoder tokens 的 mean pooled feature。
- 写入 `features`、`block_starts`、`block_ends`、`block_size`、`start_frame`、`episode_length` 等下游 phase-token 对齐需要的字段。
- 额外写入 checkpoint、模型仓库、采样方式、pooling 方式等 provenance 信息。
- 只写 `physics_features3.6`，不覆盖旧的 `physics_features3.5`。

## 常用覆盖参数

训练脚本里的主要路径都可以通过环境变量覆盖：

```bash
DATASETS_ROOT=... \
EVENT_LABEL_PATH=... \
PHYSICS_SOFT_PATH=... \
BASE_CKPT=... \
HEAD_OUTPUT_DIR=... \
JOINT_OUTPUT_DIR=... \
bash scripts/robotwin_v39/run_robotwin_v39_multitask.sh joint 0,1,2,3 bg
```

构建脚本也可以覆盖 checkpoint 和数据路径：

```bash
CKPT=... \
DATASET_ROOT=... \
EVENT_LABEL_ROOT=... \
GPUS=0,1,2,3 \
bash scripts/physics_features3_6/run_build_physics_features3_6.sh
```

## 当前路径和旧 physics_features3.5 的区别

- 旧 `physics_features3.5` 主要沿用旧 MoPE/v39 物理特征构建链路。
- 新 `physics_features3.6` 针对 `/data2/mope-jepa` 的新 MoPE-JEPA 架构、stage1 WISA physics-only checkpoint 和 RobotWin v39 event head 做了适配。
- 新路径不会修改 `/data2/mope-jepa`。
- 新路径不会覆盖旧 `physics_features3.5`。
- 新路径默认关闭 general 分支，保持 RobotWin 物理特征构建走 physics expert route。
