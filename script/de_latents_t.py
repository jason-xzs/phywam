"""
离线解码脚本：读取 wan_va_server 保存的 latents_*.pt 文件，解码成 mp4 视频。

用法：
  # 解码某个 episode 目录
  python decode_latents.py --episode_dir visualization/real/adjust_bottle_20260308_120000

  # 批量解码 visualization/real/ 下所有 episode
  python decode_latents.py --all
"""

import argparse
import os
import re
import torch
from pathlib import Path
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

VAE_PATH = "/data/worldmodel_xzs/.cache/lingbot-va-posttrain-robotwin/vae"
REAL_ROOT = "visualization/real"
OUT_ROOT  = "/data/worldmodel_xzs/phywam_v3/visualization"
DTYPE     = torch.bfloat16
DEVICE    = "cuda"


def load_vae(vae_path, dtype, device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype)
    vae = vae.to(device).eval()
    return vae


def decode_episode(episode_dir: Path, vae, processor: VideoProcessor, out_path: Path):
    # 找到所有 latents_*.pt，按 frame_st_id 排序
    pt_files = sorted(
        episode_dir.glob("latents_*.pt"),
        key=lambda p: int(re.search(r"latents_(\d+)\.pt", p.name).group(1))
    )
    if not pt_files:
        print(f"  [skip] no latents_*.pt found in {episode_dir}")
        return

    print(f"  loading {len(pt_files)} chunk(s) from {episode_dir.name}")
    chunks = []
    for pt in pt_files:
        latent = torch.load(pt, map_location="cpu")  # (1, 48, chunk_F, H, W)
        chunks.append(latent)

    latents = torch.cat(chunks, dim=2).to(DEVICE).to(DTYPE)  # (1, 48, T, H, W)

    vae_dtype = next(vae.parameters()).dtype
    latents = latents.to(vae_dtype)

    latents_mean = (
        torch.tensor(vae.config.latents_mean)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents_std = (
        1.0 / torch.tensor(vae.config.latents_std)
        .view(1, vae.config.z_dim, 1, 1, 1)
        .to(latents.device, latents.dtype)
    )
    latents = latents / latents_std + latents_mean

    with torch.no_grad():
        video = vae.decode(latents, return_dict=False)[0]

    video_np = processor.postprocess_video(video, output_type="np")[0]  # list of (H,W,3) uint8

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(video_np, str(out_path), fps=10)
    print(f"  saved -> {out_path}")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode_dir", type=str, help="单个 episode 目录路径")
    group.add_argument("--all", action="store_true", help="批量处理 visualization/real/ 下所有 episode")
    args = parser.parse_args()

    print("Loading VAE...")
    vae = load_vae(VAE_PATH, DTYPE, DEVICE)
    processor = VideoProcessor(vae_scale_factor=1)

    if args.all:
        real_root = Path(REAL_ROOT)
        episode_dirs = sorted([d for d in real_root.iterdir() if d.is_dir()])
        print(f"Found {len(episode_dirs)} episode(s) under {real_root}")
        for ep_dir in episode_dirs:
            out_path = Path(OUT_ROOT) / f"{ep_dir.name}.mp4"
            decode_episode(ep_dir, vae, processor, out_path)
    else:
        ep_dir = Path(args.episode_dir)
        out_path = Path(OUT_ROOT) / f"{ep_dir.name}.mp4"
        decode_episode(ep_dir, vae, processor, out_path)


if __name__ == "__main__":
    main()
