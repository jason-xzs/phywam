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

        # ── 1. WISA 物理数据 / 无标签视频数据 ──
        wisa_n, skipped = 0, 0

        if not anno_path:
            # Stage-2 无标签模式：递归扫描 datasets_root 下所有 mp4。
            # label_int 仅作占位；soft_t 全 0，避免产生 physics 分类监督。
            for video_path in sorted(self.datasets_root.rglob("*.mp4")):
                self.clips.append((str(video_path), 0, None))
                wisa_n += 1
            print(f"[Dataset] Unlabeled scan mode: {wisa_n} mp4 clips from {self.datasets_root}")
        else:
            with open(anno_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            for item in raw:
                label_str = item["label"].strip().lower()
                label_int = WISA_LABEL_STR2INT.get(label_str)
                folder = LABEL_TO_FOLDER.get(label_str)
                if label_int is None or folder is None:
                    skipped += 1
                    continue
                video_path = self.datasets_root / folder / item["video_name"]
                if not video_path.is_file():
                    skipped += 1
                    continue
                vn = item["video_name"]
                soft_t = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                soft_t[label_int] = 1.0
                self.clips.append((str(video_path), label_int, soft_t.clone()))
                wisa_n += 1
            print(f"[Dataset] WISA: {wisa_n} clips, skipped {skipped}")

        # ── 2. OpenVid 通用数据（可选，为空则跳过）──
        openvid_n = 0
        if openvid_dir:
            od = Path(openvid_dir)
            if od.is_file() and od.suffix.lower() == '.txt':
                # 清单文件: 每行一个视频路径(已做去重+每类均匀采样)
                vids = [Path(line.strip()) for line in open(od)
                        if line.strip() and Path(line.strip()).exists()]
                if openvid_max and openvid_max > 0:
                    vids = vids[:openvid_max]
                gen_soft = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                gen_soft[GENERAL_IDX] = 1.0
                for vp in vids:
                    self.clips.append((str(vp), GENERAL_IDX, gen_soft.clone()))
                    openvid_n += 1
                print(f"[Dataset] OpenVid: {openvid_n} clips from list {openvid_dir}")
            elif od.is_dir():
                vids = sorted(p for p in od.rglob("*")
                              if p.suffix.lower() in VIDEO_EXTS)
                if openvid_max and openvid_max > 0:
                    vids = vids[:openvid_max]
                gen_soft = torch.zeros(NUM_CLASSES, dtype=torch.float32)
                gen_soft[GENERAL_IDX] = 1.0   # 通用 one-hot：[0]*17 + [1]
                for vp in vids:
                    self.clips.append((str(vp), GENERAL_IDX, gen_soft.clone()))
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
        if total >= skip:
            # >=64帧: 覆盖2.67秒(随机64帧窗口每4取1)
            start = random.randint(0, total - skip)
            indices = np.arange(start, start + skip, self.sampling_rate)
        else:
            # <64帧: 均匀取16帧覆盖全片(能覆盖多长覆盖多长, 不取前16)
            indices = np.linspace(0, total - 1, self.num_frames).astype(int)
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
        video_path, label_int, physics_label_soft = self.clips[index]
        if physics_label_soft is None:
            physics_label_soft = torch.zeros(NUM_CLASSES, dtype=torch.float32)
        try:
            images = self._load_frames(video_path)
        except Exception as e:
            print(f"[Dataset] Load failed ({video_path}): {e}, retrying...")
            return self.__getitem__(random.randint(0, len(self.clips) - 1))

        physics_label = torch.tensor(label_int, dtype=torch.long)

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
    event_label_path = getattr(args, "event_label_path", "") or ""
    if event_label_path:
        # Kept separate from the WISA/OpenVid dataset because RobotWin samples
        # are variable-duration event segments with episode-level metadata.
        from .robotwin_event_dataset import RobotWinEventBlockDataset

        dataset = RobotWinEventBlockDataset(
            datasets_root=args.datasets_root,
            event_label_path=event_label_path,
            transform=transform,
            num_frames=args.num_frames,
            physics_soft_path=getattr(args, "physics_soft_path", "") or "",
            has_physics_label=bool(getattr(args, "has_physics_label", False)),
        )
        print("Data Aug = %s" % str(transform))
        return dataset

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
