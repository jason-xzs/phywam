# --------------------------------------------------------
# dataset/wisa_dataset.py
#
# 混合数据集：WISA-7K(物理，18维软标签前17维有值+第18维=0)
#            + OpenVid(通用，18维软标签=[0]*17+[1])
#
# 18 维软标签布局：
#   index 0..16 : 17 个物理类（Qwen 软分布 或 one-hot）
#   index 17    : 通用类（WISA=0，OpenVid=1）
#
# OpenVid 为空时自动只用 WISA，照常可训。
#
# __getitem__ 返回：
#   process_data [C,T,H,W], enc_mask, dec_mask,
#   physics_label (scalar long), physics_label_soft (float [18])
# --------------------------------------------------------

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from .loader import get_video_loader
from .pretrain_datasets import DataAugmentationForVideoMAEv2
from decord import VideoReader, cpu as decord_cpu

# ── WISA 17 类：JSON label → int ─────────────────────────────────────────────
WISA_LABEL_STR2INT = {
    "collision": 0, "rigid body motion": 1, "elastic motion": 2,
    "liquid motion": 3, "gas motion": 4, "deformation": 5, "melting": 6,
    "solidification": 7, "vaporization": 8, "liquefaction": 9, "explosion": 10,
    "combustion": 11, "reflection": 12, "refraction": 13, "scattering": 14,
    "interference and diffraction": 15, "unnatural light source": 16,
    "unnatural light sources": 16, "interference & diffraction": 15,
}
NUM_PHYSICS_CLASSES = 17
NUM_CLASSES = 18                 # 17 物理 + 1 通用
GENERAL_IDX = 17                 # 第 18 维（通用类）的 index

QWEN_SOFT_LABEL_KEYS = (
    "collision", "rigid_body_motion", "elastic_motion", "liquid_motion",
    "gas_motion", "deformation", "melting", "solidification", "vaporization",
    "liquefaction", "explosion", "combustion", "reflection", "refraction",
    "scattering", "interference_diffraction", "unnatural_light_sources",
)

LABEL_TO_FOLDER = {
    "collision": "collision_qwen3vl_caption_frame_top10",
    "rigid body motion": "rigid_body_motion_qwen3vl_caption_frame_top10",
    "elastic motion": "elastic_motion_qwen3vl_caption_frame_top10",
    "liquid motion": "liquid_motion_qwen3vl_caption_frame_top10",
    "gas motion": "gas_motion_qwen3vl_caption_frame_top10",
    "deformation": "deformation_qwen3vl_caption_frame_top10",
    "melting": "melting_qwen3vl_caption_frame_top10",
    "solidification": "solidification_qwen3vl_caption_frame_top10",
    "vaporization": "vaporization_qwen3vl_caption_frame_top10",
    "liquefaction": "liquefaction_qwen3vl_caption_frame_top10",
    "explosion": "explosion_qwen3vl_caption_frame_top10",
    "combustion": "combustion_qwen3vl_caption_frame_top10",
    "reflection": "reflection_qwen3vl_caption_frame_top10",
    "refraction": "refraction_qwen3vl_caption_frame_top10",
    "scattering": "scattering_qwen3vl_caption_frame_top10",
    "interference and diffraction": "interference_and_diffraction_qwen3vl_caption_frame_top10",
    "unnatural light source": "unnatural_light_source_qwen3vl_caption_frame_top10",
    "unnatural light sources": "unnatural_light_source_qwen3vl_caption_frame_top10",
    "interference & diffraction": "interference_and_diffraction_qwen3vl_caption_frame_top10",
}

VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}


def _label_distribution_to_tensor18(dist: dict) -> torch.Tensor:
    """Qwen 17 维软分布 → 18 维（第18维通用=0）。"""
    vec = [float(dist.get(k, 0.0)) for k in QWEN_SOFT_LABEL_KEYS]
    t = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    t[:NUM_PHYSICS_CLASSES] = torch.tensor(vec, dtype=torch.float32)
    s = float(t[:NUM_PHYSICS_CLASSES].sum().item())
    if s > 0:
        t[:NUM_PHYSICS_CLASSES] /= s          # 物理部分归一，通用维保持 0
    return t


class WISAPretrainDataset(torch.utils.data.Dataset):
    """
    WISA(物理) + OpenVid(通用) 混合预训练数据集。
    每条 clip: (video_path, label_int, soft18)
      - WISA:    label_int = 物理类(0-16)，soft18 前17维有值、第18维=0
      - OpenVid: label_int = 17(通用)，    soft18 = [0]*17 + [1]
    """

    def __init__(
        self,
        datasets_root: str,
        anno_path: str,
        transform,
        num_frames: int = 16,
        sampling_rate: int = 4,
        num_sample: int = 1,
        physics_soft_path: str = "",
        openvid_dir: str = "",        # OpenVid 平铺 mp4 目录；为空则只用 WISA
        openvid_max: int = 0,         # 限制 OpenVid 用多少条；0=全部
    ):
        self.datasets_root = Path(datasets_root)
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.skip_length = num_frames * sampling_rate
        self.num_sample = num_sample

        # ── 软标签 JSON ──
        self._soft_by_video = {}
        if physics_soft_path:
            p = Path(physics_soft_path)
            if p.is_file():
                with open(p, "r", encoding="utf-8") as f:
                    soft_raw = json.load(f)
                for row in soft_raw:
                    vn = row.get("video_name")
                    dist = row.get("label_distribution")
                    if vn and isinstance(dist, dict):
                        t = _label_distribution_to_tensor18(dist)
                        if float(t[:NUM_PHYSICS_CLASSES].sum().item()) > 0:
                            self._soft_by_video[vn] = t
                print(f"[Dataset] Loaded soft labels for "
                      f"{len(self._soft_by_video)} videos from {physics_soft_path}")
            else:
                print(f"[Dataset] WARN: physics_soft_path not found: {physics_soft_path}")

        self.clips = []
        # 纯视频模式(理解1): 递归扫两个目录, 无JSON标注
        #   WISA   -> is_general=0, soft18全0(物理组, 前17维不监督=physics_loss=0)
        #   OpenVid-> is_general=1, soft18=[0]*17+[1](通用组)
        #   binary_gate靠"来自哪个目录"监督(general_loss), 物理router靠JEPA涌现
        wisa_n = 0
        wisa_root = self.datasets_root
        if wisa_root.is_dir():
            for vp in sorted(wisa_root.rglob("*")):
                if vp.suffix.lower() in VIDEO_EXTS:
                    self.clips.append((str(vp), 0))
                    wisa_n += 1
            print(f"[Dataset] WISA: {wisa_n} clips from {wisa_root}")
        else:
            print(f"[Dataset] WISA dir not found: {wisa_root}")
        openvid_n = 0
        if openvid_dir:
            od = Path(openvid_dir)
            if od.is_dir():
                vids = sorted(vp for vp in od.rglob("*")
                              if vp.suffix.lower() in VIDEO_EXTS)
                if openvid_max and openvid_max > 0:
                    vids = vids[:openvid_max]
                for vp in vids:
                    self.clips.append((str(vp), GENERAL_IDX))
                    openvid_n += 1
                print(f"[Dataset] OpenVid: {openvid_n} clips from {openvid_dir}")
            else:
                print(f"[Dataset] OpenVid dir not found (skip): {openvid_dir}")
        else:
            print(f"[Dataset] OpenVid not provided (WISA-only mode)")
        print(f"[Dataset] TOTAL {len(self.clips)} clips "
              f"(WISA={wisa_n}, OpenVid={openvid_n})")

        if len(self.clips) == 0:
            raise RuntimeError(
                f"No valid clips. datasets_root={datasets_root}, anno_path={anno_path}")

    def _sample_frame_ids(self, total: int):
        skip = self.skip_length
        if total <= skip:
            indices = np.arange(total)
            indices = np.tile(indices, skip // max(total, 1) + 1)[:skip]
        else:
            start = random.randint(0, total - skip)
            indices = np.arange(start, start + skip, self.sampling_rate)
        indices = np.clip(indices, 0, total - 1).astype(int)
        return indices[:self.num_frames].tolist()

    def _load_frames(self, video_path: str):
        vr = VideoReader(video_path, num_threads=1, ctx=decord_cpu(0))
        total = len(vr)
        frame_ids = self._sample_frame_ids(total)
        video_data = vr.get_batch(frame_ids).asnumpy()
        return [Image.fromarray(video_data[i]).convert("RGB")
                for i in range(len(frame_ids))]

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        video_path, label_int = self.clips[index]
        try:
            images = self._load_frames(video_path)
        except Exception as e:
            print(f"[Dataset] Load failed ({video_path}): {e}, retrying...")
            return self.__getitem__(random.randint(0, len(self.clips) - 1))

        physics_label = torch.tensor(label_int, dtype=torch.long)
        physics_label_soft = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        if label_int == GENERAL_IDX:
            physics_label_soft[GENERAL_IDX] = 1.0

        if self.num_sample > 1:
            pd_list, enc_list, dec_list, lbl_list, soft_list = [], [], [], [], []
            for _ in range(self.num_sample):
                pd, enc, dec = self.transform((images, None))
                pd = pd.view((self.num_frames, 3) + pd.size()[-2:]).transpose(0, 1)
                pd_list.append(pd); enc_list.append(enc); dec_list.append(dec)
                lbl_list.append(physics_label); soft_list.append(physics_label_soft.clone())
            return pd_list, enc_list, dec_list, lbl_list, soft_list
        else:
            process_data, enc_mask, dec_mask = self.transform((images, None))
            process_data = process_data.view(
                (self.num_frames, 3) + process_data.size()[-2:]).transpose(0, 1)
            return (process_data, enc_mask, dec_mask, physics_label,
                    physics_label_soft.clone())


def build_wisa_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAEv2(args)
    soft_path = getattr(args, "physics_soft_path", "") or ""
    openvid_dir = getattr(args, "openvid_dir", "") or ""
    openvid_max = getattr(args, "openvid_max", 0) or 0
    dataset = WISAPretrainDataset(
        datasets_root=args.datasets_root,
        anno_path=args.anno_path,
        transform=transform,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        num_sample=args.num_sample,
        physics_soft_path=soft_path,
        openvid_dir=openvid_dir,
        openvid_max=openvid_max,
    )
    print("Data Aug = %s" % str(transform))
    return dataset