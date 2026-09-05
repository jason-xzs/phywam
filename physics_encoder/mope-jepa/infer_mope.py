"""
infer_mope.py  ── MoPE_jepa推理脚本（适配 time-aware router 版本）
==================================

使用示例：
    # 提取 npy 特征
    CUDA_VISIBLE_DEVICES=7 python infer_mope.py \
        --ckpt /data2/mope-jepa/output/stage_18class/checkpoint-100.pth \
        --video_dir /data2/mope-jepa/datasets/test/scattering \
        --save_dir /data2/mope-jepa/features \
        --use_mope --enable_general 

    # 分类调试（打印 Top-5）
    CUDA_VISIBLE_DEVICES=7 python infer_mope.py \
        --ckpt /data2/mope-jepa/output/stage1_official_vmae/checkpoint-50.pth \
        --video_dir /data2/mope-jepa/datasets/test/reflection \
        --save_dir /data2/mope-jepa/features \
        --use_mope --show_cls --enable_general

    # 单视频
    CUDA_VISIBLE_DEVICES=7 python infer_mope.py \
        --ckpt /data2/mope-jepa/output/stage1_official_vmae/checkpoint-50.pth \
        --video /data2/openvid/videos/3/pexels_landscape_landscape_4763085_000.mp4 \
        --save_dir /data2/mope-jepa/features \
        --use_mope --show_cls --enable_general 
"""

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from timm.models import create_model
from torchvision import transforms

import sys
sys.path.insert(0, str(Path(__file__).parent))
import models  # noqa: F401

from dataset.loader import get_video_loader

IDX2LABEL = {
    0:  "Collision",
    1:  "Rigid Body Motion",
    2:  "Elastic Motion",
    3:  "Liquid Motion",
    4:  "Gas Motion",
    5:  "Deformation",
    6:  "Melting",
    7:  "Solidification",
    8:  "Vaporization",
    9:  "Liquefaction",
    10: "Explosion",
    11: "Combustion",
    12: "Reflection",
    13: "Refraction",
    14: "Scattering",
    15: "Interference and Diffraction",
    16: "Unnatural Light Sources",
    17: "General (通用/非物理)",
}
VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}


def build_inference_transform(input_size=224):
    mean = [0.485, 0.456, 0.406]
    std  = [0.229, 0.224, 0.225]
    return transforms.Compose([
        transforms.Resize(input_size,
                          interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def load_video_frames(video_path, num_frames=16, sampling_rate=4):
    video_loader = get_video_loader()
    vr = video_loader(video_path)
    total = len(vr)
    skip_length = num_frames * sampling_rate
    if total <= skip_length:
        indices = np.linspace(0, total - 1, num_frames).astype(int)
    else:
        center = total // 2
        start  = max(0, center - skip_length // 2)
        start  = min(start, total - skip_length)
        indices = np.arange(start, start + skip_length, sampling_rate)[:num_frames]
    indices = np.clip(indices, 0, total - 1).tolist()
    video_data = vr.get_batch(indices).asnumpy()
    return [Image.fromarray(video_data[i]).convert("RGB") for i in range(len(indices))]


def frames_to_tensor(images, transform, num_frames=16):
    frames = [transform(img) for img in images]
    while len(frames) < num_frames:
        frames.append(frames[-1])
    frames = frames[:num_frames]
    return torch.stack(frames, dim=1).unsqueeze(0)  # [1, C, T, H, W]


def make_full_visible_mask(batch_size, num_patches, device):
    return torch.zeros(batch_size, num_patches, dtype=torch.bool, device=device)


def load_pretrain_model(ckpt_path, args, device):
    print(f"Loading pretrain model: {args.model}")
    extra = {}
    if args.use_mope:
        extra = dict(
            num_physics_experts=args.num_physics_experts,
            num_general_experts=args.num_general_experts,
            num_shared_experts=args.num_shared_experts,
            candidate_k=args.candidate_k,
            gate_threshold=args.gate_threshold,
            gate_hidden=args.gate_hidden,
            gate_layers=args.gate_layers,
            gate_dims=[int(x) for x in args.gate_dims.split(',') if x.strip()] if args.gate_dims else None,
        )
    model = create_model(
        args.model, pretrained=False, drop_path_rate=0.0,
        all_frames=args.num_frames, tubelet_size=args.tubelet_size,
        with_cp=False, **extra,
    )
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    state_dict = None
    for key in ['model', 'module', 'state_dict']:
        if key in ckpt:
            state_dict = ckpt[key]
            print(f"  Loaded state_dict via key='{key}'")
            break
    if state_dict is None:
        state_dict = ckpt
        print("  Loaded state_dict directly")
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
    msg = model.load_state_dict(state_dict, strict=False)
    print(f"  Missing keys: {len(msg.missing_keys)}, Unexpected keys: {len(msg.unexpected_keys)}")
    if msg.missing_keys:
        print(f"  Missing (first 5): {msg.missing_keys[:5]}")

    # 推理期运行开关：是否启用通用专家组 + 门控阈值
    if args.use_mope:
        model.encoder.enable_general = args.enable_general
        model.encoder.gate_threshold = args.gate_threshold
        print(f"  enable_general={args.enable_general}, "
              f"gate_threshold={args.gate_threshold}, "
              f"candidate_k={args.candidate_k}")

    model.eval()
    model.to(device)
    return model


@torch.no_grad()
def _encode(encoder, video_tensor, device):
    """
    适配新版 forward_features 签名：
        forward_features(x, mask, physics_label=None, physics_label_soft=None)
    推理时标签参数传 None，router 自主路由无需标签。
    """
    video_tensor = video_tensor.to(device)
    B = video_tensor.size(0)
    mask = make_full_visible_mask(B, encoder.patch_embed.num_patches, device)
    return encoder.forward_features(
        video_tensor, mask,
        physics_label=None,
        physics_label_soft=None,
    )  # [B, N_vis, C]


@torch.no_grad()
def diagnose_mask(encoder, video_tensor, device, ratios=(0.0, 0.5, 0.9), seed=0):
    """
    诊断：同一视频在不同 mask 比例下，gate 输出的通用维分数。
      ratio=0.0  → 全可见（推理默认）
      ratio=0.9  → 训练时的mask比例
    如果 0.9 时通用维明显高于 0.0，说明训练-推理 mask 不一致是问题根源。
    """
    video_tensor = video_tensor.to(device)
    B = video_tensor.size(0)
    N = encoder.patch_embed.num_patches
    print(f"  [Mask诊断] 同一视频在不同mask比例下的通用维(第18维)分数：")
    for r in ratios:
        # 构造 mask：r 比例的 token 被遮（True=masked）
        torch.manual_seed(seed)
        mask = torch.zeros(B, N, dtype=torch.bool, device=device)
        if r > 0:
            n_mask = int(N * r)
            for b in range(B):
                idx = torch.randperm(N, device=device)[:n_mask]
                mask[b, idx] = True
        _ = encoder.forward_features(video_tensor, mask,
                                     physics_label=None, physics_label_soft=None)
        cls = encoder._cls_scores            # [B,18]
        gen = cls[0, 17].item()              # 通用维
        phys_top_p, phys_top_i = cls[0, :17].max(dim=-1)
        n_vis = N - int(N * r)
        print(f"    mask={r:.0%} (可见{n_vis:4d} token): "
              f"通用维={gen:.4f} | 物理最高={IDX2LABEL[phys_top_i.item()]}({phys_top_p.item():.3f})")


@torch.no_grad()
def infer_classify(model, video_tensor, device):
    encoder = model.encoder
    x_vis = _encode(encoder, video_tensor, device)

    cls_scores = getattr(encoder, '_cls_scores', None)
    if cls_scores is None:
        cls_scores = x_vis.new_full((x_vis.size(0), 17), 1.0 / 17)

    # 当前第一阶段是纯物理 17 类：不使用 binary/general gate。
    video_scores = cls_scores
    enable_general = bool(getattr(encoder, "enable_general", False))

    if enable_general and video_scores.size(-1) > 17:
        general_score = video_scores[0, 17].item()
        gate_threshold = getattr(encoder, 'gate_threshold', 0.5)
        use_physics = bool(general_score < gate_threshold)
    else:
        general_score = 0.0
        use_physics = True

    phys_scores = video_scores[0, :17]
    phys_top_p, phys_top_i = phys_scores.max(dim=-1)
    phys_sorted_p, phys_sorted_i = phys_scores.sort(descending=True)
    phys_rank = [
        (IDX2LABEL[i.item()], p.item())
        for i, p in zip(phys_sorted_i, phys_sorted_p)
    ]

    if use_physics:
        final_label = IDX2LABEL[phys_top_i.item()]
        final_conf = phys_top_p.item()
    else:
        final_label = IDX2LABEL[17]
        final_conf = general_score

    return {
        'label': final_label,
        'confidence': final_conf,
        'general_score': general_score,
        'phys_top': (IDX2LABEL[phys_top_i.item()], phys_top_p.item()),
        'phys_top5': phys_rank[:5],
        'use_physics': use_physics,
    }


@torch.no_grad()
def infer_extract(model, video_tensor, device):
    x_vis = _encode(model.encoder, video_tensor, device)

    # ── 特征聚合方式（二选一）────────────────────────────────────────
    # 方式1：等权GAP，与VideoMAEv2一致，输出[768]，下游兼容性好
    # feat = x_vis.mean(dim=1)

    # 方式2：直接保存完整latent，输出[1568, 768]，下游可自行聚合
    feat = x_vis

    return feat[0].cpu().numpy()


def get_video_list(args):
    if args.video:
        return [args.video]
    elif args.video_dir:
        d = Path(args.video_dir)
        videos = [str(p) for p in sorted(d.iterdir())
                  if p.suffix.lower() in VIDEO_EXTENSIONS]
        print(f"找到视频 {len(videos)} 个")
        return videos
    return []


def run_classify(model, videos, transform, args, device):
    for i, vpath in enumerate(videos):
        try:
            images = load_video_frames(vpath, args.num_frames, args.sampling_rate)
            video_tensor = frames_to_tensor(images, transform, args.num_frames)
            r = infer_classify(model, video_tensor, device)
            group = "物理组" if r['use_physics'] else "通用组"
            print(f"[{i+1}/{len(videos)}] {Path(vpath).name}")
            print(f"  判定：{group}（通用概率 {r['general_score']:.3f}，阈值 {getattr(model.encoder,'gate_threshold',0.5)}）")
            if r['use_physics']:
                print(f"  → 物理预测：{r['label']}（{r['confidence']:.3f}）")
            else:
                print(f"  → 通用组（非物理）")
            print(f"  物理17类Top-5：" + " | ".join(f"{l}({c:.3f})" for l, c in r['phys_top5']))
            if getattr(args, 'diagnose_mask', False):
                diagnose_mask(model.encoder, video_tensor, device)
        except Exception as e:
            print(f"[{i+1}/{len(videos)}] 处理失败 {vpath}: {e}")


def run_extract(model, videos, transform, args, device):
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for i, vpath in enumerate(videos):
        save_path = save_dir / f"{Path(vpath).stem}.npy"
        if save_path.exists() and not args.overwrite:
            print(f"[{i+1}/{len(videos)}] 跳过（已存在）: {save_path}")
            continue
        try:
            images = load_video_frames(vpath, args.num_frames, args.sampling_rate)
            video_tensor = frames_to_tensor(images, transform, args.num_frames)
            feat = infer_extract(model, video_tensor, device)
            np.save(str(save_path), feat)
            print(f"[{i+1}/{len(videos)}] 特征已保存: {save_path} shape={feat.shape}")
        except Exception as e:
            print(f"[{i+1}/{len(videos)}] 处理失败 {vpath}: {e}")


def get_args():
    parser = argparse.ArgumentParser('MoPE 推理脚本', add_help=True)
    parser.add_argument('--show_cls',             action='store_true', default=False)
    parser.add_argument('--diagnose_mask',        action='store_true', default=False,
                        help='诊断：同一视频在 mask 0%/50%/90% 下的通用维分数对比')
    parser.add_argument('--video',                default='', type=str)
    parser.add_argument('--video_dir',            default='', type=str)
    parser.add_argument('--save_dir',             default='./features', type=str)
    parser.add_argument('--overwrite',            action='store_true', default=False)
    parser.add_argument('--ckpt',                 required=True, type=str)
    parser.add_argument('--model',                default='pretrain_mope_jepa_base_patch16_224', type=str)
    parser.add_argument('--num_frames',           type=int, default=16)
    parser.add_argument('--sampling_rate',        type=int, default=4)
    parser.add_argument('--input_size',           type=int, default=224)
    parser.add_argument('--tubelet_size',         type=int, default=2)
    parser.add_argument('--use_mope',             action='store_true', default=False)
    parser.add_argument('--num_physics_experts',  type=int, default=17)
    parser.add_argument('--num_general_experts',  type=int, default=10)
    parser.add_argument('--num_shared_experts',   type=int, default=4)
    parser.add_argument('--candidate_k',          type=int, default=5,
                        help='物理组 token router 的 top-k 候选数（分类器给出）')
    parser.add_argument('--gate_threshold',       type=float, default=0.5,
                        help='门控阈值：max置信度<此值→通用组。阶段一/纯物理推理用0.0')
    parser.add_argument('--gate_hidden',          type=int, default=0,
                        help='物理router gate隐藏层维度，须与训练时一致(0=单层线性)')
    parser.add_argument('--gate_layers',          type=int, default=2,
                        help='gate_hidden>0时的层数，须与训练时一致')
    parser.add_argument('--gate_dims',            type=str, default='',
                        help='自定义gate维度(逗号分隔)，须与训练时一致，如 512,256,128')
    parser.add_argument('--enable_general',       action='store_true', default=False,
                        help='是否启用10个通用专家组（阶段二训练后的模型推理时打开）')
    parser.add_argument('--device',               default='cuda', type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备：{device}")
    videos = get_video_list(args)
    if not videos:
        raise ValueError("请通过 --video 或 --video_dir 指定输入视频")
    model     = load_pretrain_model(args.ckpt, args, device)
    transform = build_inference_transform(args.input_size)
    if args.show_cls:
        run_classify(model, videos, transform, args, device)
    run_extract(model, videos, transform, args, device)


# # 提取特征
# python infer_mope.py \
#     --ckpt /home/nvme04/mope-jepa/output/mope_wisa7k_vitb/checkpoint-199.pth \
#     --video_dir /home/nvme04/mope-jepa/datasets/test/collision/ \
#     --save_dir /home/nvme04/mope-jepa/features/collision/ \
#     --use_mope

# # 分类调试
# python infer_mope.py \
#     --ckpt /home/nvme04/mope-jepa/output/mope_wisa7k_vitb_1/checkpoint-299.pth \
#     --video_dir /home/nvme04/mope-jepa/datasets/test/rigid_body_motion/ \
#     --save_dir /home/nvme04/mope-jepa/features/rigid_body_motion/ \
#     --use_mope --show_cls
# python infer_mope.py \
#     --ckpt /home/nvme04/mope-jepa/output/mope_jepa_wisa7k_vitb_1/checkpoint-199.pth \
#     --video_dir /home/nvme04/mope-jepa/datasets/test/gas_motion/ \
#     --save_dir /home/nvme04/mope-jepa/features/gas_motion/ \
#     --use_mope --show_cls
# python infer_mope.py \
#     --ckpt /home/nvme04/mope-jepa/output/mope_jepa_wisa7k_vitb_freeze/checkpoint-150.pth \
#     --video_dir /home/nvme04/mope-jepa/datasets/test/gas_motion/ \
#     --save_dir /home/nvme04/mope-jepa/features/gas_motion/ \
#     --use_mope --show_cls\
#     --device cuda:7






