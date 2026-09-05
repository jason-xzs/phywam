# --------------------------------------------------------
# run_jepa_pretraining.py  ── MoPE-JEPA 版本
#
# 相比 MoPE MAE 版改动：
#   1. 删掉 decoder 相关参数，换成 predictor 参数
#   2. 新增 sigreg_weight 参数
#   3. train_one_epoch 透传 sigreg_weight
#   4. model forward 不再需要 decode_masked_pos
#   5. 支持从 VideoMAEv2 预训练权重初始化，冻结 encoder 除 MoE 外的层
# --------------------------------------------------------

import argparse
import datetime
import json
import os
import random
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
try:
    from torch.serialization import safe_globals
except ImportError:
    # torch < 2.5 没有 safe_globals；配合 weights_only=False 时本就无需它，
    # 用一个接受参数的空上下文管理器替代。
    from contextlib import contextmanager

    @contextmanager
    def safe_globals(_globals=None):
        yield
from packaging import version
from timm.models import create_model

import models  # noqa: F401
import utils
from dataset.wisa_dataset import build_wisa_pretraining_dataset
from engine_for_pretraining import train_one_epoch
from optim_factory import create_optimizer
from utils import NativeScalerWithGradNormCount as NativeScaler
from utils import multiple_pretrain_samples_collate


def maybe_init_wandb(args, total_batch_size, steps_per_epoch, n_parameters):
    """Initialize W&B on rank0 when WANDB_MODE is enabled via environment."""
    if not utils.is_main_process():
        return None

    mode = os.environ.get('WANDB_MODE', 'disabled').strip().lower()
    if mode in ('', 'disabled', 'off', 'false', '0'):
        print('W&B disabled (set WANDB_MODE=online/offline to enable).')
        return None

    project = os.environ.get('WANDB_PROJECT', '').strip()
    if not project:
        print('W&B disabled because WANDB_PROJECT is empty.')
        return None

    try:
        import wandb
    except ImportError as exc:
        if mode == 'online':
            raise RuntimeError(
                'WANDB_MODE=online but the wandb package is not installed.'
            ) from exc
        print(f'W&B disabled: wandb import failed: {exc}')
        return None

    wandb_dir = os.environ.get('WANDB_DIR', '').strip()
    if wandb_dir:
        os.makedirs(wandb_dir, exist_ok=True)

    config = vars(args).copy()
    config.update({
        'total_batch_size': total_batch_size,
        'steps_per_epoch': steps_per_epoch,
        'n_parameters_trainable': n_parameters,
    })
    run = wandb.init(
        project=project,
        entity=os.environ.get('WANDB_ENTITY') or None,
        name=os.environ.get('WANDB_NAME') or None,
        group=os.environ.get('WANDB_GROUP') or None,
        tags=[
            tag.strip() for tag in os.environ.get('WANDB_TAGS', '').split(',')
            if tag.strip()
        ] or None,
        mode=mode,
        dir=wandb_dir or None,
        config=config,
    )
    print(
        f"W&B initialized: project={project}, "
        f"name={os.environ.get('WANDB_NAME', '')}, mode={mode}"
    )
    return run


def get_args():
    parser = argparse.ArgumentParser(
        'MoPE-JEPA pre-training script', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int)
    parser.add_argument('--epochs', default=400, type=int)
    parser.add_argument('--save_ckpt_freq', default=50, type=int)

    # Model parameters
    parser.add_argument('--model',
                        default='pretrain_mope_jepa_base_patch16_224',
                        type=str, metavar='MODEL')
    parser.add_argument('--tubelet_size', type=int, default=2)
    parser.add_argument('--with_checkpoint', action='store_true', default=False)
    parser.add_argument('--mask_type', default='tube',
                        choices=['random', 'tube'], type=str)
    parser.add_argument('--mask_ratio', default=0.9, type=float)
    parser.add_argument('--input_size', default=224, type=int)
    parser.add_argument('--drop_path', type=float, default=0.0, metavar='PCT')

    # ── JEPA predictor 参数 ───────────────────────────────────────────────
    parser.add_argument('--predictor_dim',       type=int,   default=384)
    parser.add_argument('--predictor_depth',     type=int,   default=6)
    parser.add_argument('--predictor_num_heads', type=int,   default=6)

    # ── SIGReg 参数 ───────────────────────────────────────────────────────
    parser.add_argument('--sigreg_weight', type=float, default=0.3)

    # ── MoPE 模型参数 ──────────────────────────────────────────────────────
    parser.add_argument('--use_mope', action='store_true', default=False)
    parser.add_argument('--num_physics_experts', type=int, default=17)
    parser.add_argument('--num_general_experts', type=int, default=10)
    parser.add_argument('--num_shared_experts',  type=int, default=4)
    parser.add_argument('--candidate_k',         type=int, default=5,
                        help='物理组 token router 的 top-k 候选数')
    parser.add_argument('--gate_threshold',      type=float, default=0.0,
                        help='门控阈值；阶段一用0.0(全走物理)，阶段二设>0')
    parser.add_argument('--enable_general', action='store_true', default=False,
                        help='启用10个通用专家组(阶段二)，阶段一不开')
    parser.add_argument('--moe_balance_weight',   type=float, default=0.01)
    parser.add_argument('--physics_cls_weight',   type=float, default=1.0)
    parser.add_argument('--has_physics_label', action='store_true', default=True)
    # ── 阶段一：单独训练全局分类器 ──────────────────────────────────────
    parser.add_argument('--stage1_train_classifier', action='store_true', default=False,
                        help='阶段一：冻结全部，只训 global_router.phys_router，仅用 router_loss')
    parser.add_argument('--stage2_train_experts', action='store_true', default=False,
                        help='阶段二：冻结 backbone+物理router，训通用router+所有专家(jepa+balance，无physics_loss)')
    # ── 分类器 gate 结构（实验：加激活/加深看分类是否提升）──────────────
    parser.add_argument('--gate_hidden', type=int, default=0,
                        help='物理router gate隐藏维度；0=单层Linear(原始)，>0=MLP带GELU')
    parser.add_argument('--gate_layers', type=int, default=2,
                        help='gate_hidden>0时的层数(含输出层)，2=单隐藏层，3=两隐藏层')
    parser.add_argument('--gate_dims', type=str, default='',
                        help='自定义gate隐藏层维度，逗号分隔，如 512,256,128 → 768→512→256→128→17')

    # ── MoPE 数据集参数 ────────────────────────────────────────────────────
    parser.add_argument('--datasets_root',
                        default='/home/nvme04/mope-jepa/datasets', type=str)
    parser.add_argument('--anno_path',
                        default='/home/nvme04/mope-jepa/datasets/wisa_7k.json',
                        type=str)
    parser.add_argument('--physics_soft_path', default='', type=str)
    parser.add_argument('--event_label_path', default='', type=str)
    parser.add_argument('--event_loss_weight', type=float, default=1.0)
    parser.add_argument('--num_event_classes', type=int, default=0)
    parser.add_argument('--openvid_dir', default='', type=str,
                        help='OpenVid 平铺mp4目录（通用数据）；为空则只用WISA')
    parser.add_argument('--openvid_max', default=0, type=int,
                        help='限制OpenVid用多少条，0=全部')

    # Optimizer parameters
    parser.add_argument('--opt',         default='adamw', type=str)
    parser.add_argument('--opt_eps',     default=1e-8,    type=float)
    parser.add_argument('--opt_betas',   default=None,    type=float, nargs='+')
    parser.add_argument('--clip_grad',   type=float,      default=1.0)
    parser.add_argument('--momentum',    type=float,      default=0.9)
    parser.add_argument('--weight_decay',     type=float, default=0.05)
    parser.add_argument('--weight_decay_end', type=float, default=None)
    parser.add_argument('--lr',          type=float, default=1.5e-4)
    parser.add_argument('--warmup_lr',   type=float, default=1e-6)
    parser.add_argument('--min_lr',      type=float, default=1e-5)
    parser.add_argument('--warmup_epochs', type=int, default=20)
    parser.add_argument('--warmup_steps',  type=int, default=-1)

    # Augmentation
    parser.add_argument('--color_jitter',       type=float, default=0.0)
    parser.add_argument('--train_interpolation', type=str,  default='bicubic',
                        choices=['random', 'bilinear', 'bicubic'])

    # Finetuning / 预训练权重初始化
    parser.add_argument('--finetune', default='',
                        help='从预训练权重初始化，例如 VideoMAEv2 预训练权重路径')
    parser.add_argument('--freeze_encoder_except_moe', action='store_true', default=False,
                        help='冻结 encoder 中除 MoE 层以外的所有参数')
    parser.add_argument('--train_predictor', action='store_true', default=False,
                        help='freeze_encoder_except_moe 时同时训练 JEPA predictor')
    parser.add_argument('--unfreeze_moe_attn_norm1', action='store_true', default=False,
                        help='freeze_encoder_except_moe 时额外解冻 MoE blocks 的 attn 和 norm1')
    parser.add_argument('--freeze_binary_gate', action='store_true', default=False,
                        help='冻住binary_gate, 用第一阶段学好的物理/通用路由(第二阶段避免数据不平衡污染)')
    parser.add_argument(
        '--event_head_only',
        action='store_true',
        default=False,
        help='RobotWin warmup: freeze everything except event_head and train it on full-visible features')
    parser.add_argument(
        '--freeze_phys_router',
        action='store_true',
        default=False,
        help='Keep the WISA-trained physics router frozen during RobotWin joint adaptation')

    # 保留兼容性
    parser.add_argument('--data_path',   default='', type=str)
    parser.add_argument('--data_root',   default='', type=str)
    parser.add_argument('--fname_tmpl',  default='img_{:05}.jpg', type=str)
    parser.add_argument('--imagenet_default_mean_and_std',
                        default=True, action='store_true')
    parser.add_argument('--num_frames',    type=int, default=16)
    parser.add_argument('--sampling_rate', type=int, default=4)
    parser.add_argument('--num_sample',    type=int, default=1)
    parser.add_argument('--output_dir',  default='')
    parser.add_argument('--log_dir',     default=None)
    parser.add_argument('--device',      default='cuda')
    parser.add_argument('--seed',        default=0, type=int)
    parser.add_argument('--resume',      default='')
    parser.add_argument('--auto_resume', action='store_true')
    parser.add_argument('--no_auto_resume', action='store_false',
                        dest='auto_resume')
    parser.set_defaults(auto_resume=True)
    parser.add_argument('--start_epoch', default=0, type=int)
    parser.add_argument('--max_train_steps_per_epoch', default=-1, type=int)
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem',    action='store_true')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed
    parser.add_argument('--world_size',  default=1,    type=int)
    parser.add_argument('--local_rank',  default=-1,   type=int)
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url',    default='env://')
    parser.add_argument('--decoder_mask_type', default='run_cell', type=str)
    parser.add_argument('--decoder_mask_ratio', default=0.0, type=float)

    return parser.parse_args()


def get_model(args):
    print(f"Creating model: {args.model}")
    extra_kwargs = {}
    if args.use_mope:
        extra_kwargs.update(dict(
            num_physics_experts=args.num_physics_experts,
            num_general_experts=args.num_general_experts,
            num_shared_experts=args.num_shared_experts,
            candidate_k=args.candidate_k,
            gate_threshold=args.gate_threshold,
            gate_hidden=args.gate_hidden,
            gate_layers=args.gate_layers,
            gate_dims=[int(x) for x in args.gate_dims.split(',') if x.strip()] if args.gate_dims else None,
        ))
    if args.num_event_classes > 0:
        extra_kwargs['num_event_classes'] = args.num_event_classes

    model = create_model(
        args.model,
        pretrained=False,
        drop_path_rate=args.drop_path,
        drop_block_rate=None,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        with_cp=args.with_checkpoint,
        **extra_kwargs,
    )

    if version.parse(torch.__version__) > version.parse('1.13.1'):
        torch.set_float32_matmul_precision('high')
        if args.with_checkpoint:
            print('torch.compile disabled because --with_checkpoint is set')
        elif getattr(args, 'event_label_path', ''):
            print('torch.compile disabled for RobotWin event multi-task training')
        elif getattr(args, 'stage1_train_classifier', False):
            print('torch.compile disabled in stage1 (classifier-only training)')
        elif getattr(args, 'stage2_train_experts', False):
            print('torch.compile disabled in stage2 (data-dependent MoE routing)')
        elif getattr(args, 'freeze_encoder_except_moe', False):
            # 动态 MoE 路由 + 各 MoE 层结构不同（block8 有 router，其余没有）
            # 会导致 torch.compile 反复重编译撞 cache_size_limit；直接禁用。
            print('torch.compile disabled in freeze_encoder_except_moe (dynamic MoE routing)')
        else:
            model = torch.compile(model)

    return model


def load_pretrained_checkpoint(model, checkpoint_path):
    """
    从 VideoMAEv2 等预训练权重初始化模型。

    key 处理规则：
      - backbone.xxx  → xxx      （部分老格式）
      - encoder.xxx   → encoder.xxx  （保留前缀，与模型 key 匹配）
      - predictor.xxx → predictor.xxx
      - event_head.xxx → event_head.xxx
      - 其余裸 backbone key → encoder.xxx

    注意：VideoMAEv2 预训练权重的 key 格式是 encoder.blocks.x.xxx，
    与本模型 encoder.blocks.x.xxx 完全对齐，不能剥掉 encoder. 前缀。
    Block 8-11 的 MoE 层 key 不匹配会自动被 strict=False 忽略，随机初始化。
    """
    print(f"  Loading pretrained checkpoint from: {checkpoint_path}")
    if str(checkpoint_path).endswith('.safetensors'):
        from safetensors.torch import load_file
        checkpoint = load_file(checkpoint_path)
        print("  Loaded safetensors format (official VideoMAE V2)")
    else:
        with safe_globals([np._core.multiarray.scalar]):
            checkpoint = torch.load(
                checkpoint_path,
                map_location='cpu',
                weights_only=False,
                mmap=True)

    checkpoint_model = None
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint and 'model' not in checkpoint:
            checkpoint_model = checkpoint['state_dict']
        else:
            for model_key in ['model', 'module']:
                if model_key in checkpoint:
                    checkpoint_model = checkpoint[model_key]
                    print(f"  Loaded state_dict via key='{model_key}'")
                    break
    if checkpoint_model is None:
        checkpoint_model = checkpoint
        print("  Loaded state_dict directly")

    if isinstance(checkpoint_model, dict):
        new_dict = OrderedDict()
        for key, value in checkpoint_model.items():
            if key.startswith('backbone.'):
                new_dict['encoder.' + key[len('backbone.'):]] = value
            elif key.startswith('model.'):
                new_dict['encoder.' + key[len('model.'):]] = value
            elif key.startswith('encoder.predictor.'):
                new_dict[key.replace('encoder.predictor.', 'predictor.', 1)] = value
            elif key.startswith('predictor.'):
                new_dict[key] = value
            elif key.startswith('event_head.'):
                # Checkpoints saved by this training script keep event_head at
                # the top level. Do not rewrite it to encoder.event_head.*.
                new_dict[key] = value
            elif key.startswith('encoder.'):
                new_dict[key] = value
            else:
                new_dict['encoder.' + key] = value
        checkpoint_model = new_dict

        model_keys = set(model.state_dict().keys())
        extra = OrderedDict()
        num_physics = getattr(model.encoder, 'num_physics_experts', 17)
        num_shared = 4

        for key, value in checkpoint_model.items():
            if '.mlp.fc' not in key:
                continue

            dense_key = key.replace('.mlp.fc', '.mlp.general_dense.fc')
            if dense_key in model_keys:
                extra[dense_key] = value

            for e in range(num_physics):
                expert_key = key.replace('.mlp.fc', f'.mlp.physics_experts.{e}.fc')
                if expert_key in model_keys:
                    extra[expert_key] = value

            for e in range(num_shared):
                shared_key = key.replace('.mlp.fc', f'.mlp.shared_experts.{e}.fc')
                if shared_key in model_keys:
                    extra[shared_key] = value

        if extra:
            checkpoint_model.update(extra)
            print(f"  Copied official VideoMAE MLP weights to MoPE experts/general_dense: {len(extra)} tensors")

        # 去掉分类头（VideoMAEv2 finetune 权重才有，pretrain 权重通常没有）
        for key in ['head.weight', 'head.bias']:
            if key in checkpoint_model and key not in dict(model.named_parameters()):
                checkpoint_model.pop(key, None)
                print(f"  Removed classification head key: {key}")

    msg = utils.load_state_dict(model, checkpoint_model)
    # 打印加载情况，方便确认哪些 key 被加载、哪些 missing
    if hasattr(msg, 'missing_keys'):
        missing = [k for k in msg.missing_keys if 'moe' not in k.lower() and 'predictor' not in k.lower()]
        moe_missing = [k for k in msg.missing_keys if 'moe' in k.lower() or 'sparse' in k.lower()]
        print(f"  Missing keys (non-MoE, non-predictor): {len(missing)}")
        if missing:
            print(f"    First 5: {missing[:5]}")
        print(f"  Missing MoE keys (expected, random init): {len(moe_missing)}")
        print(f"  Unexpected keys: {len(msg.unexpected_keys)}")


def _find_block_router(encoder):
    """找到持有 router 的那个 MoE block 的 router（新架构：router 在第一个 MoE block.mlp 里）。"""
    for blk in encoder.blocks:
        if getattr(blk, 'use_moe', False):
            r = getattr(blk.mlp, 'router', None)
            if r is not None:
                return r
    return None


def freeze_encoder_except_moe(model, train_predictor=False, unfreeze_moe_attn_norm1=False):
    """
    冻结 encoder 中除 MoE FFN 层以外的所有参数。
    MoE 块（Block 8-11）的 Self-Attention 也会被冻结，只训练 mlp（MoE部分）。
    """
    # 1. 冻结整个 encoder
    for param in model.encoder.parameters():
        param.requires_grad = False

    # 2. 冻结 predictor（默认）
    for param in model.predictor.parameters():
        param.requires_grad = False

    # 3. 解冻 MoE 块的 mlp 参数（Block 8-11 的 SharedExpertMoE）
    unfrozen_count = 0
    for blk in model.encoder.blocks:
        if getattr(blk, 'use_moe', False):
            for param in blk.mlp.parameters():
                param.requires_grad = True
                unfrozen_count += param.numel()
            # 解冻该 block 的 LayerNorm（norm2，在 MoE 之前）
            if hasattr(blk, 'norm2'):
                for param in blk.norm2.parameters():
                    param.requires_grad = True
                    unfrozen_count += param.numel()

            if unfreeze_moe_attn_norm1:
                if hasattr(blk, 'norm1'):
                    for param in blk.norm1.parameters():
                        param.requires_grad = True
                        unfrozen_count += param.numel()
                if hasattr(blk, 'attn'):
                    for param in blk.attn.parameters():
                        param.requires_grad = True
                        unfrozen_count += param.numel()
            # 解冻 gamma scaling（如果存在）
            if hasattr(blk, 'gamma_1') and blk.gamma_1 is not None:
                blk.gamma_1.requires_grad = True
            if hasattr(blk, 'gamma_2') and blk.gamma_2 is not None:
                blk.gamma_2.requires_grad = True

    # 解冻 encoder 最终 LayerNorm
    if hasattr(model.encoder, 'norm'):
        for param in model.encoder.norm.parameters():
            param.requires_grad = True

    # router 现在在第一个 MoE block 的 mlp 内部（LayerExperts.router），
    # 已被上面 blk.mlp.parameters() 一并解冻。这里统计确认。
    router_count = 0
    for blk in model.encoder.blocks:
        if getattr(blk, 'use_moe', False) and getattr(blk.mlp, 'router', None) is not None:
            for param in blk.mlp.router.parameters():
                router_count += param.numel()  # 已在 mlp.parameters() 中解冻
    print(f"  Router parameters (in first MoE block, trainable): {router_count / 1e6:.4f} M")

    print(f"  Unfrozen MoE parameters (incl. router): {unfrozen_count / 1e6:.2f} M")

    # 4. 可选：同时训练 predictor
    if train_predictor:
        for param in model.predictor.parameters():
            param.requires_grad = True
        predictor_count = sum(p.numel() for p in model.predictor.parameters())
        print(f"  Predictor also unfrozen: {predictor_count / 1e6:.2f} M")

    if getattr(model, 'event_head', None) is not None:
        for param in model.event_head.parameters():
            param.requires_grad = True
        event_count = sum(p.numel() for p in model.event_head.parameters())
        print(f"  Event head unfrozen: {event_count / 1e6:.4f} M")


def freeze_all_except_event_head(model):
    """Warm up the new RobotWin event classifier without changing the encoder."""
    for param in model.parameters():
        param.requires_grad = False
    if getattr(model, 'event_head', None) is None:
        raise ValueError('--event_head_only requires --num_event_classes > 0')
    for param in model.event_head.parameters():
        param.requires_grad = True
    event_count = sum(p.numel() for p in model.event_head.parameters())
    print(f"  Event-head warmup: trainable={event_count / 1e6:.4f} M")


def infer_num_event_classes(event_label_path):
    if not event_label_path:
        return 0
    with open(event_label_path, 'r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if isinstance(payload, dict):
        if payload.get('num_event_classes'):
            return int(payload['num_event_classes'])
        vocabulary = payload.get('event_vocabulary') or []
        if vocabulary:
            return len(vocabulary)
        samples = payload.get('samples') or []
    elif isinstance(payload, list):
        samples = payload
    else:
        raise ValueError(f'Unsupported event-label JSON: {event_label_path}')
    return len({str(row.get('event_label', 'no_event')) for row in samples})


def freeze_all_except_classifier(model):
    """
    阶段一：冻结【全部】，只解冻物理 router（global_router.phys_router）。
    物理 router 的 gate 用 wisa-7k 物理标签 + router_loss 训：
      其聚合输出 token_scores.mean 做分类/门控（physics_loss/router_loss 监督）。
    阶段一训好后，阶段二冻结它。
    """
    for p in model.parameters():
        p.requires_grad = False

    enc = model.encoder
    cnt = 0
    # router 现在在第一个 MoE block 的 mlp.router 里
    _router = _find_block_router(enc)
    if _router is not None:
        for p in _router.phys_router.parameters():
            p.requires_grad = True
            cnt += p.numel()
    print(f"  Stage1: only physics router trainable: {cnt / 1e6:.4f} M")
    if cnt == 0:
        print("  [WARN] 未找到 block 内的 router，检查模型是否为新架构！")


def freeze_for_stage2(model):
    """
    阶段二：冻结 backbone + 物理 router（门控/分类已训好，保持稳定）；
    训练：通用 router + 17物理专家 + 10通用专家 + 共享专家。
    （backbone 冻结以保证物理 router 看到的 block7 特征不变、门控稳定。）
    """
    # 1. 全部冻结
    for p in model.parameters():
        p.requires_grad = False

    enc = model.encoder
    train_cnt = 0

    # 2. 解冻每个 MoE block 的专家（物理+通用+共享）
    for blk in enc.blocks:
        if getattr(blk, 'use_moe', False):
            for p in blk.mlp.parameters():   # LayerExperts: 全部专家 + shared_weight
                p.requires_grad = True
                train_cnt += p.numel()

    # 3. 解冻通用 router（从随机学，无监督）；物理 router 保持冻结
    _router = _find_block_router(enc)
    if _router is not None:
        for p in _router.gen_router.parameters():
            p.requires_grad = True
            train_cnt += p.numel()
        for p in _router.phys_router.parameters():
            p.requires_grad = False

    # 4. 解冻 encoder 最终 norm + predictor（jepa 需要）
    if hasattr(enc, 'norm'):
        for p in enc.norm.parameters():
            p.requires_grad = True
            train_cnt += p.numel()
    if hasattr(model, 'predictor'):
        for p in model.predictor.parameters():
            p.requires_grad = True
            train_cnt += p.numel()

    print(f"  Stage2: trainable (experts + gen_router + predictor): "
          f"{train_cnt / 1e6:.2f} M")
    print(f"  Stage2: backbone & physics_router FROZEN")


def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    if args.event_label_path:
        if not os.path.isfile(args.event_label_path):
            raise FileNotFoundError(args.event_label_path)
        if args.has_physics_label and not os.path.isfile(args.physics_soft_path):
            raise FileNotFoundError(
                'RobotWin event+physics training requires --physics_soft_path: '
                f'{args.physics_soft_path}')
        if args.num_event_classes <= 0:
            args.num_event_classes = infer_num_event_classes(args.event_label_path)
        print(
            f"RobotWin event training: classes={args.num_event_classes}, "
            f"event_weight={args.event_loss_weight}, "
            f"physics_weight={args.physics_cls_weight}")

    model = get_model(args)
    patch_size = model.encoder.patch_embed.patch_size
    print("Patch size = %s" % str(patch_size))
    args.window_size = (
        args.num_frames // args.tubelet_size,
        args.input_size // patch_size[0],
        args.input_size // patch_size[1])
    args.patch_size = patch_size

    dataset_train = build_wisa_pretraining_dataset(args)

    num_tasks    = utils.get_world_size()
    global_rank  = utils.get_rank()
    total_batch_size = args.batch_size * num_tasks
    num_training_steps_per_epoch = len(dataset_train) // total_batch_size
    if args.max_train_steps_per_epoch > 0:
        num_training_steps_per_epoch = min(
            num_training_steps_per_epoch, args.max_train_steps_per_epoch)

    sampler_train = torch.utils.data.DistributedSampler(
        dataset_train, num_replicas=num_tasks,
        rank=global_rank, shuffle=True)
    print("Sampler_train = %s" % str(sampler_train))

    if global_rank == 0 and args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = utils.TensorboardLogger(log_dir=args.log_dir)
    else:
        log_writer = None

    collate_func = None
    if args.num_sample > 1:
        collate_func = partial(multiple_pretrain_samples_collate, fold=False)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=True,
        collate_fn=collate_func,
        worker_init_fn=utils.seed_worker,
        persistent_workers=args.num_workers > 0)

    # ── 加载预训练权重（VideoMAEv2）──────────────────────────────────────
    if args.finetune:
        print(f"Load pretrained ckpt from {args.finetune}")
        load_pretrained_checkpoint(model, args.finetune)

    # ── 冻结策略 ─────────────────────────────────────────────────────────
    if args.event_head_only:
        freeze_all_except_event_head(model)
    elif args.stage1_train_classifier:
        print('Stage1: freezing all except physics router')
        freeze_all_except_classifier(model)
    elif args.stage2_train_experts:
        print('Stage2: freezing backbone & physics router; training experts + gen_router')
        freeze_for_stage2(model)
    elif args.freeze_encoder_except_moe:
        if args.train_predictor:
            print('Freezing encoder except MoE; predictor remains trainable')
        else:
            print('Freezing encoder except MoE; predictor is frozen')
        freeze_encoder_except_moe(model, train_predictor=args.train_predictor, unfreeze_moe_attn_norm1=args.unfreeze_moe_attn_norm1)
        if getattr(args, 'freeze_binary_gate', False):
            _bg = 0
            for _name, _p in model.named_parameters():
                if 'binary_gate' in _name:
                    _p.requires_grad_(False); _bg += 1
            print(f"[freeze_binary_gate] 冻住 {_bg} 个binary_gate参数(用第一阶段学好的物理/通用路由)")

    if args.freeze_phys_router:
        router = _find_block_router(model.encoder)
        if router is None:
            raise RuntimeError('Cannot freeze physics router: block router not found')
        for param in router.phys_router.parameters():
            param.requires_grad = False
        print('  Physics router frozen for RobotWin adaptation')

    # ── encoder 运行开关：通用专家组 / 门控阈值 ──────────────────────────
    if args.use_mope:
        model.encoder.enable_general = args.enable_general
        model.encoder.gate_threshold = args.gate_threshold
        print(f"  enable_general={args.enable_general}, "
              f"gate_threshold={args.gate_threshold}")
        if not args.enable_general:
            frozen_general = 0
            for name, param in model.named_parameters():
                if any(k in name for k in (
                    'binary_gate', 'gen_router', 'general_experts', 'general_dense'
                )):
                    if param.requires_grad:
                        frozen_general += param.numel()
                    param.requires_grad = False
            print(f"  Frozen unused general/binary params: {frozen_general / 1e6:.2f} M")

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of trainable params: {:.2f} M'.format(n_parameters / 1e6))

    args.lr        = args.lr        * total_batch_size / 256
    args.min_lr    = args.min_lr    * total_batch_size / 256
    args.warmup_lr = args.warmup_lr * total_batch_size / 256
    print("LR = %.8f" % args.lr)
    print("Batch size = %d" % total_batch_size)
    wandb_run = maybe_init_wandb(
        args=args,
        total_batch_size=total_batch_size,
        steps_per_epoch=num_training_steps_per_epoch,
        n_parameters=n_parameters,
    )

    if args.distributed:
        # 专家已改为恒参与计算图（_dispatch 中 0 token 专家过 dummy×0），
        # 图结构每步固定，满足 static_graph 前提。static_graph 正确处理
        # "参数每步被使用固定次数"，解决 binary_gate 等的 reducer 重复标记。
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu],
            find_unused_parameters=False,
            static_graph=True)
        model_without_ddp = model.module

    optimizer   = create_optimizer(args, model_without_ddp)
    loss_scaler = NativeScaler()

    lr_schedule_values = utils.cosine_scheduler(
        args.lr, args.min_lr, args.epochs,
        num_training_steps_per_epoch,
        warmup_epochs=args.warmup_epochs,
        warmup_steps=args.warmup_steps)
    if args.weight_decay_end is None:
        args.weight_decay_end = args.weight_decay
    wd_schedule_values = utils.cosine_scheduler(
        args.weight_decay, args.weight_decay_end,
        args.epochs, num_training_steps_per_epoch)

    utils.auto_load_model(
        args=args, model=model,
        model_without_ddp=model_without_ddp,
        optimizer=optimizer, loss_scaler=loss_scaler)

    torch.cuda.empty_cache()
    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)
        if log_writer is not None:
            log_writer.set_step(epoch * num_training_steps_per_epoch)

        train_stats = train_one_epoch(
            model,
            data_loader_train,
            optimizer,
            device,
            epoch,
            loss_scaler,
            args.clip_grad,
            log_writer=log_writer,
            start_steps=epoch * num_training_steps_per_epoch,
            lr_schedule_values=lr_schedule_values,
            wd_schedule_values=wd_schedule_values,
            patch_size=patch_size[0],
            num_training_steps_per_epoch=num_training_steps_per_epoch,
            moe_balance_weight=args.moe_balance_weight,
            has_physics_label=args.has_physics_label,
            has_event_label=bool(args.event_label_path),
            physics_cls_weight=args.physics_cls_weight,
            event_loss_weight=args.event_loss_weight,
            event_head_only=args.event_head_only,
            sigreg_weight=args.sigreg_weight,
            stage1=args.stage1_train_classifier,
            stage2=args.stage2_train_experts,
        )

        if args.output_dir:
            _epoch = epoch + 1
            checkpoint_saved = (
                _epoch % args.save_ckpt_freq == 0 or _epoch == args.epochs)
            if checkpoint_saved:
                utils.save_model(
                    args=args, model=model,
                    model_without_ddp=model_without_ddp,
                    optimizer=optimizer, loss_scaler=loss_scaler,
                    epoch=_epoch)
        else:
            checkpoint_saved = False

        log_stats = {
            **{f'train_{k}': v for k, v in train_stats.items()},
            'epoch': epoch,
            'epoch_1based': epoch + 1,
            'n_parameters': n_parameters,
            'checkpoint_saved': int(checkpoint_saved),
        }

        if args.output_dir and utils.is_main_process():
            if log_writer is not None:
                log_writer.flush()
            with open(os.path.join(args.output_dir, "log.txt"),
                      mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        if wandb_run is not None:
            wandb_run.log(log_stats, step=epoch + 1)

    total_time = time.time() - start_time
    print('Training time {}'.format(
        str(datetime.timedelta(seconds=int(total_time)))))
    if wandb_run is not None:
        wandb_run.finish()


if __name__ == '__main__':
    # decord/OpenCV 在 fork 出的 DataLoader worker 中易与主进程线程库冲突而死锁，
    # 用 spawn 启动 worker 可根治（代价是 worker 启动略慢，仅一次性开销）。
    import torch.multiprocessing as _mp
    try:
        _mp.set_start_method('spawn', force=True)
    except RuntimeError:
        pass
    opts = get_args()
    if opts.output_dir:
        Path(opts.output_dir).mkdir(parents=True, exist_ok=True)
    main(opts)
