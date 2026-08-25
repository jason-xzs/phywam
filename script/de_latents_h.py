"""
离线解码脚本：读取 wan_va_server 保存的 latents_*.pt，解码成横排 [high|left|right] mp4。

策略：先整体解码全 latent（保证 VAE 卷积上下文完整），再在像素空间按比例切割三路视图。

用法：
  python decode_latents.py --episode_dir visualization/real/adjust_bottle_20260308_120000
  python decode_latents.py --all
"""

import argparse
import re
import cv2
import numpy as np
import torch
from pathlib import Path
from diffusers.video_processor import VideoProcessor
from diffusers.utils import export_to_video

VAE_PATH  = "/data/worldmodel_xzs/.cache/lingbot-va-posttrain-robotwin/vae"
REAL_ROOT = "visualization/real"
OUT_ROOT  = "/data/worldmodel_xzs/phywam_v3/visualization"
DTYPE     = torch.bfloat16
DEVICE    = "cuda"


def load_vae(vae_path, dtype, device):
    from diffusers import AutoencoderKLWan
    vae = AutoencoderKLWan.from_pretrained(vae_path, torch_dtype=dtype)
    return vae.to(device).eval()


def decode_full_latent(latents: torch.Tensor, vae, processor: VideoProcessor):
    """整体解码，返回 list[np.ndarray(H,W,3)] uint8(0~255) 或 float(0~1)。"""
    lat = latents.to(vae.dtype)
    latents_mean = (torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(lat.device, lat.dtype))
    latents_std = (1.0 / torch.tensor(vae.config.latents_std)
                   .view(1, vae.config.z_dim, 1, 1, 1)
                   .to(lat.device, lat.dtype))
    lat = lat / latents_std + latents_mean   # 与原始 decode_latents.py 完全一致
    with torch.no_grad():
        video_tensor = vae.decode(lat, return_dict=False)[0]
    frames = processor.postprocess_video(video_tensor, output_type="np")[0]
    # 统一转 uint8 [0,255]
    frames_u8 = []
    for f in frames:
        if f.max() <= 1.001:
            f = (f * 255).astype(np.uint8)
        else:
            f = f.astype(np.uint8)
        frames_u8.append(f)
    return frames_u8


def split_tshape_frames(frames):
    """
    在像素空间把 T 型拼接帧拆成三路，横排输出 [high | left | right]。

    T 型 latent 结构（服务端 env_type='robotwin_tshape'）：
      latent shape : (1, C, F, H_lat, W_lat)
      H_lat = (H//16)*3//2，W_lat = W//16
      latent 行 [0        : H_lat//3] → wrist 两摄：左半=left，右半=right
      latent 行 [H_lat//3 : H_lat  ] → cam_high

    对应像素行（H_lat//3 : H_lat = 1:2，即 wrist 占 1/3，high 占 2/3）：
      像素总高 = H_full
      wrist 高 = H_full // 3
      high  高 = H_full * 2 // 3  （= wrist_h * 2）
    """
    H_full = frames[0].shape[0]
    W_full = frames[0].shape[1]
    wrist_h = H_full // 3          # 上 1/3 行 → wrist
    # high_h  = H_full - wrist_h   # 下 2/3 行 → cam_high（直接用切片）

    result = []
    for f in frames:
        wrist_row = f[:wrist_h, :, :]           # (wrist_h, W_full, 3)
        high_row  = f[wrist_h:, :, :]           # (high_h,  W_full, 3)

        left_wrist  = wrist_row[:, :W_full // 2, :]   # 左半
        right_wrist = wrist_row[:, W_full // 2:, :]   # 右半

        # 统一高度（以 high 为基准）resize left/right
        tgt_h = high_row.shape[0]
        def rsh(img):
            if img.shape[0] == tgt_h:
                return img
            w = int(img.shape[1] * tgt_h / img.shape[0])
            return cv2.resize(img, (w, tgt_h), interpolation=cv2.INTER_LINEAR)

        row = np.concatenate([high_row, rsh(left_wrist), rsh(right_wrist)], axis=1)
        result.append(row)
    return result


def decode_episode(episode_dir: Path, vae, processor: VideoProcessor, out_path: Path):
    pt_files = sorted(
        episode_dir.glob("latents_*.pt"),
        key=lambda p: int(re.search(r"latents_(\d+)\.pt", p.name).group(1))
    )
    if not pt_files:
        print(f"  [skip] no latents_*.pt found in {episode_dir}")
        return

    print(f"  loading {len(pt_files)} chunk(s) …")
    chunks = [torch.load(p, map_location="cpu") for p in pt_files]
    latents = torch.cat(chunks, dim=2).to(DEVICE).to(DTYPE)
    print(f"  latent shape: {latents.shape}")

    print("  decoding full latent …")
    frames = decode_full_latent(latents, vae, processor)
    print(f"  decoded {len(frames)} frames, pixel size: {frames[0].shape[1]}×{frames[0].shape[0]}")

    print("  splitting T-shape → [high | left | right] …")
    frames = split_tshape_frames(frames)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    export_to_video(frames, str(out_path), fps=10)
    print(f"  saved → {out_path}  ({frames[0].shape[1]}×{frames[0].shape[0]})")


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--episode_dir", type=str)
    group.add_argument("--all", action="store_true")
    args = parser.parse_args()

    print("Loading VAE …")
    vae = load_vae(VAE_PATH, DTYPE, DEVICE)
    processor = VideoProcessor(vae_scale_factor=1)

    if args.all:
        real_root = Path(REAL_ROOT)
        ep_dirs = sorted(d for d in real_root.iterdir() if d.is_dir())
        print(f"Found {len(ep_dirs)} episode(s)")
        for ep_dir in ep_dirs:
            decode_episode(ep_dir, vae, processor, Path(OUT_ROOT) / f"{ep_dir.name}.mp4")
    else:
        ep_dir = Path(args.episode_dir)
        decode_episode(ep_dir, vae, processor, Path(OUT_ROOT) / f"{ep_dir.name}.mp4")


if __name__ == "__main__":
    main()
