"""RobotWin event-segment dataset ported from the PhyWAM v3 MoPE pipeline.

Only ``RobotWinEventBlockDataset`` is consumed by the 0706 training path.  It
is kept in a separate module so the new WISA/OpenVid dataset implementation
remains unchanged.
"""

# --------------------------------------------------------
# dataset/wisa_dataset.py
#
# 基于 VideoMAEv2 原版 HybridVideoMAE 改造。
# 复用原版的视频加载（get_video_loader）、DataAugmentationForVideoMAEv2、
# TubeMaskingGenerator，只在 __getitem__ 末尾多返回一个 physics_label。
#
# 数据集结构：
#   /home/nvme04/mope/datasets/
#   ├── collision_qwen3vl_caption_frame_top10/  {hash}.mp4 ...
#   ├── rigid_body_motion_qwen3vl_caption_frame_top10/
#   ├── ...
#   └── wisa_7k.json
#
# wisa_7k.json 格式（每条）：
#   {"video_name": "xxx.mp4", "label": "collision", ...}
#
# __getitem__ 返回：
#   process_data         : [C, T, H, W]
#   encoder_mask_map     : [N_patches] bool
#   decoder_mask_map     : [N_patches] bool
#   physics_label        : scalar long（文件夹主类，供 PhysicsHead CE）
#   physics_label_soft   : float [17]（Qwen 分布或 one-hot；供 router soft CE）
# --------------------------------------------------------

import json
import os
import random
import subprocess
import hashlib
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image

from .loader import get_video_loader
from .pretrain_datasets import DataAugmentationForVideoMAEv2

# ── WISA 17 类：JSON label（小写+空格）→ int ──────────────────────────────────
WISA_LABEL_STR2INT = {
    "collision":                    0,
    "rigid body motion":            1,
    "elastic motion":               2,
    "liquid motion":                3,
    "gas motion":                   4,
    "deformation":                  5,
    "melting":                      6,
    "solidification":               7,
    "vaporization":                 8,
    "liquefaction":                 9,
    "explosion":                    10,
    "combustion":                   11,
    "reflection":                   12,
    "refraction":                   13,
    "scattering":                   14,
    "interference and diffraction": 15,
    "unnatural light source":       16,
    # 兼容变体
    "unnatural light sources":      16,
    "interference & diffraction":   15,
}
NUM_PHYSICS_CLASSES = 17

# 与 models/moe_ffn.py 中 PHYSICS_SOFT_LABEL_KEYS、Qwen JSON 字段一致（顺序=expert id）
QWEN_SOFT_LABEL_KEYS = (
    "collision",
    "rigid_body_motion",
    "elastic_motion",
    "liquid_motion",
    "gas_motion",
    "deformation",
    "melting",
    "solidification",
    "vaporization",
    "liquefaction",
    "explosion",
    "combustion",
    "reflection",
    "refraction",
    "scattering",
    "interference_diffraction",
    "unnatural_light_sources",
)


def _normalize_label_key(label: str) -> str:
    """Normalize label text so both underscore and space styles are accepted."""
    if not isinstance(label, str):
        return ""
    s = label.strip().lower().replace("_", " ")
    s = s.replace("&", " and ")
    s = " ".join(s.split())
    return s


def _label_distribution_to_tensor(dist: dict) -> "torch.Tensor":
    vec = [float(dist.get(k, 0.0)) for k in QWEN_SOFT_LABEL_KEYS]
    # The new router contract is [17 physics classes, 1 general class].
    # RobotWin clips are physics-domain data, so the last component stays zero.
    t = torch.zeros(NUM_PHYSICS_CLASSES + 1, dtype=torch.float32)
    t[:NUM_PHYSICS_CLASSES] = torch.tensor(vec, dtype=torch.float32)
    s = float(t[:NUM_PHYSICS_CLASSES].sum().item())
    if s > 0:
        t[:NUM_PHYSICS_CLASSES] /= s
    return t


def _event_distribution_to_tensor(dist: dict, event_vocabulary, label_id: int) -> "torch.Tensor":
    vec = []
    if isinstance(dist, dict):
        for label in event_vocabulary:
            try:
                vec.append(max(float(dist.get(label, 0.0) or 0.0), 0.0))
            except Exception:
                vec.append(0.0)
    else:
        vec = [0.0] * len(event_vocabulary)
    t = torch.tensor(vec, dtype=torch.float32)
    s = float(t.sum().item())
    if s > 0:
        return t / s
    if len(event_vocabulary) > 0:
        label_id = max(0, min(int(label_id), len(event_vocabulary) - 1))
        t[label_id] = 1.0
    return t


# ── label → 视频子文件夹名 ────────────────────────────────────────────────────
LABEL_TO_FOLDER = {
    "collision":                    "collision_qwen3vl_caption_frame_top10",
    "rigid body motion":            "rigid_body_motion_qwen3vl_caption_frame_top10",
    "elastic motion":               "elastic_motion_qwen3vl_caption_frame_top10",
    "liquid motion":                "liquid_motion_qwen3vl_caption_frame_top10",
    "gas motion":                   "gas_motion_qwen3vl_caption_frame_top10",
    "deformation":                  "deformation_qwen3vl_caption_frame_top10",
    "melting":                      "melting_qwen3vl_caption_frame_top10",
    "solidification":               "solidification_qwen3vl_caption_frame_top10",
    "vaporization":                 "vaporization_qwen3vl_caption_frame_top10",
    "liquefaction":                 "liquefaction_qwen3vl_caption_frame_top10",
    "explosion":                    "explosion_qwen3vl_caption_frame_top10",
    "combustion":                   "combustion_qwen3vl_caption_frame_top10",
    "reflection":                   "reflection_qwen3vl_caption_frame_top10",
    "refraction":                   "refraction_qwen3vl_caption_frame_top10",
    "scattering":                   "scattering_qwen3vl_caption_frame_top10",
    "interference and diffraction": "interference_and_diffraction_qwen3vl_caption_frame_top10",
    "unnatural light source":       "unnatural_light_source_qwen3vl_caption_frame_top10",
    "unnatural light sources":      "unnatural_light_source_qwen3vl_caption_frame_top10",
    "interference & diffraction":   "interference_and_diffraction_qwen3vl_caption_frame_top10",
}

WISA_LABEL_STR2INT_NORM = {
    _normalize_label_key(k): v for k, v in WISA_LABEL_STR2INT.items()
}
LABEL_TO_FOLDER_NORM = {
    _normalize_label_key(k): v for k, v in LABEL_TO_FOLDER.items()
}


class WISAPretrainDataset(torch.utils.data.Dataset):
    """
    WISA-7K 预训练数据集。
    复用 VideoMAEv2 的 DataAugmentationForVideoMAEv2。
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
    ):
        self.datasets_root = Path(datasets_root)
        self.transform = transform
        self.num_frames = num_frames
        self.sampling_rate = sampling_rate
        self.skip_length = num_frames * sampling_rate
        self.num_sample = num_sample
        # get_video_loader() returns a local closure, which cannot be pickled by
        # torch DataLoader workers.  Create it lazily inside each worker instead
        # of storing the closure in the parent-process dataset object.
        self.video_loader = None
        self._bad_indices = set()
        self._decode_fallback_map = {}
        self._decode_fallback_failed = set()
        self._ffmpeg_bin = os.environ.get(
            "MOPE_FFMPEG_BIN",
            "/data/worldmodel_xzs/ffmpeg-7.0.2-amd64-static/ffmpeg")
        cache_dir = os.environ.get("MOPE_DECODE_CACHE_DIR", "/tmp/mope_decode_cache")
        self._decode_cache_dir = Path(cache_dir)
        self._decode_cache_dir.mkdir(parents=True, exist_ok=True)

        print(
            "[WISAPretrainDataset] Decode fallback enabled: "
            f"ffmpeg={self._ffmpeg_bin}, cache={self._decode_cache_dir}"
        )

        # 可选：Qwen 软标签 JSON（wisa_7k_qwen3vl8b_scores_all.json）
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
                        t = _label_distribution_to_tensor(dist)
                        if float(t.sum().item()) > 0:
                            self._soft_by_video[vn] = t
                print(f"[WISAPretrainDataset] Loaded soft labels for "
                      f"{len(self._soft_by_video)} videos from {physics_soft_path}")
            else:
                print(f"[WISAPretrainDataset] WARN: physics_soft_path not found: "
                      f"{physics_soft_path}")

        # 加载 JSON 标注，构建 clips 列表
        with open(anno_path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        self.clips = []
        skipped = 0
        skipped_bad_label = 0
        skipped_missing_file = 0
        for item in raw:
            label_str = _normalize_label_key(item.get("label", ""))
            label_int = WISA_LABEL_STR2INT_NORM.get(label_str)
            folder = LABEL_TO_FOLDER_NORM.get(label_str)
            if label_int is None:
                skipped += 1
                skipped_bad_label += 1
                continue

            video_rel = str(item.get("video_name", "")).strip().lstrip("/")
            if not video_rel:
                skipped += 1
                skipped_missing_file += 1
                continue

            # RobotWin JSON usually stores full relative path from datasets_root.
            # Keep legacy fallback for old WISA-style folder-based layout.
            candidate_paths = [self.datasets_root / video_rel]
            if folder is not None:
                candidate_paths.append(self.datasets_root / folder / video_rel)

            video_path = None
            for p in candidate_paths:
                if p.is_file():
                    video_path = p
                    break

            if video_path is None:
                skipped += 1
                skipped_missing_file += 1
                continue

            vn = item["video_name"]
            soft_t = self._soft_by_video.get(vn)
            if soft_t is None:
                y = torch.zeros(NUM_PHYSICS_CLASSES, dtype=torch.float32)
                y[label_int] = 1.0
                soft_t = y
            self.clips.append((str(video_path), label_int, soft_t.clone()))

        print(f"[WISAPretrainDataset] Loaded {len(self.clips)} clips, "
              f"skipped {skipped} "
              f"(bad_label={skipped_bad_label}, missing_file={skipped_missing_file})")
        from collections import Counter
        dist = Counter(lbl for _, lbl, _ in self.clips)
        print(f"[WISAPretrainDataset] Distribution: {dict(sorted(dist.items()))}")

        if len(self.clips) == 0:
            raise RuntimeError(
                f"No valid clips found. "
                f"datasets_root={datasets_root}, anno_path={anno_path}")

    def _sample_frame_ids(self, total: int):
        """随机起始点 + 固定 sampling_rate 采样，与原版一致"""
        skip = self.skip_length
        if total <= skip:
            indices = np.arange(total)
            # 循环填充到 skip_length
            indices = np.tile(indices, skip // total + 1)[:skip]
        else:
            start = random.randint(0, total - skip)
            indices = np.arange(start, start + skip, self.sampling_rate)
        indices = np.clip(indices, 0, total - 1).astype(int)
        return indices[:self.num_frames].tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["video_loader"] = None
        return state

    def _get_video_loader(self):
        if self.video_loader is None:
            self.video_loader = get_video_loader()
        return self.video_loader

    def _load_frames(self, video_path: str):
        vr = self._get_video_loader()(video_path)
        total = len(vr)
        frame_ids = self._sample_frame_ids(total)
        video_data = vr.get_batch(frame_ids).asnumpy()   # [T, H, W, C]
        images = [
            Image.fromarray(video_data[i]).convert("RGB")
            for i in range(len(frame_ids))
        ]
        return images

    def _transcode_for_decode(self, video_path: str):
        src = str(video_path)
        if src in self._decode_fallback_map:
            return self._decode_fallback_map[src]
        if src in self._decode_fallback_failed:
            return None

        try:
            st = os.stat(src)
        except OSError:
            self._decode_fallback_failed.add(src)
            return None

        key_src = f"{src}|{st.st_size}|{int(st.st_mtime_ns)}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:24]
        dst = self._decode_cache_dir / f"{key}.mp4"

        if dst.is_file() and dst.stat().st_size > 0:
            p = str(dst)
            self._decode_fallback_map[src] = p
            return p

        tmp = self._decode_cache_dir / f"{key}.tmp.mp4"
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-v",
            "error",
            "-hwaccel",
            "none",
            "-i",
            src,
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(tmp),
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            print(f"[WISAPretrainDataset] Decode fallback spawn failed: {e}")
            self._decode_fallback_failed.add(src)
            return None

        if proc.returncode != 0 or (not tmp.is_file()) or tmp.stat().st_size == 0:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            msg = proc.stderr.strip() if proc.stderr else "unknown ffmpeg error"
            print(
                "[WISAPretrainDataset] Decode fallback transcode failed: "
                f"{src}; err={msg}"
            )
            self._decode_fallback_failed.add(src)
            return None

        tmp.replace(dst)
        p = str(dst)
        self._decode_fallback_map[src] = p
        print(f"[WISAPretrainDataset] Decode fallback succeeded: {src} -> {p}")
        return p

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, index):
        if len(self.clips) == 0:
            raise RuntimeError("WISAPretrainDataset has no clips")

        max_retry = min(64, len(self.clips))
        last_err = None

        for attempt in range(max_retry):
            if index in self._bad_indices:
                index = random.randint(0, len(self.clips) - 1)
                continue

            video_path, label_int, physics_label_soft = self.clips[index]
            try:
                images = self._load_frames(video_path)
                break
            except Exception as e:
                transcode_path = self._transcode_for_decode(video_path)
                if transcode_path is not None:
                    try:
                        images = self._load_frames(transcode_path)
                        break
                    except Exception as e2:
                        last_err = RuntimeError(
                            f"raw_decode_err={e}; fallback_decode_err={e2}"
                        )
                else:
                    last_err = e

                self._bad_indices.add(index)
                if attempt < 5 or (attempt + 1) % 10 == 0:
                    print(
                        "[WISAPretrainDataset] Load failed "
                        f"(attempt={attempt + 1}/{max_retry}, index={index}): "
                        f"{video_path}; err={e}"
                    )
                if len(self._bad_indices) >= len(self.clips):
                    raise RuntimeError(
                        "All clips failed to decode. "
                        f"last_error={last_err}"
                    )
                index = random.randint(0, len(self.clips) - 1)
        else:
            raise RuntimeError(
                "Exceeded max retries while decoding videos. "
                f"bad_indices={len(self._bad_indices)}/{len(self.clips)}, "
                f"last_error={last_err}"
            )

        physics_label = torch.tensor(label_int, dtype=torch.long)

        if self.num_sample > 1:
            process_data_list, enc_list, dec_list = [], [], []
            lbl_list, soft_list = [], []
            for _ in range(self.num_sample):
                pd, enc, dec = self.transform((images, None))
                pd = pd.view((self.num_frames, 3) + pd.size()[-2:]).transpose(0, 1)
                process_data_list.append(pd)
                enc_list.append(enc)
                dec_list.append(dec)
                lbl_list.append(physics_label)
                soft_list.append(physics_label_soft.clone())
            return process_data_list, enc_list, dec_list, lbl_list, soft_list
        else:
            process_data, enc_mask, dec_mask = self.transform((images, None))
            process_data = process_data.view(
                (self.num_frames, 3) + process_data.size()[-2:]
            ).transpose(0, 1)   # [C, T, H, W]
            return (process_data, enc_mask, dec_mask, physics_label,
                    physics_label_soft.clone())


class RobotWinEventBlockDataset(torch.utils.data.Dataset):
    """One RobotWin block maps to one categorical manipulation event label."""

    def __init__(
        self,
        datasets_root: str,
        event_label_path: str,
        transform,
        num_frames: int = 16,
        physics_soft_path: str = "",
        has_physics_label: bool = False,
    ):
        self.datasets_root = Path(datasets_root)
        self.transform = transform
        self.num_frames = num_frames
        self.has_physics_label = has_physics_label
        # get_video_loader() returns a local closure, which cannot be pickled by
        # torch DataLoader workers.  Keep it out of the pickled dataset state and
        # initialize it lazily per worker if the decord path is used.
        self.video_loader = None
        self._bad_indices = set()
        self._decode_fallback_map = {}
        self._decode_fallback_failed = set()
        self._ffmpeg_bin = os.environ.get(
            "MOPE_FFMPEG_BIN",
            "/data/worldmodel_xzs/ffmpeg-7.0.2-amd64-static/ffmpeg")
        cache_dir = os.environ.get(
            "MOPE_DECODE_CACHE_DIR",
            "/data/worldmodel_xzs/phywam_v3/tmp/mope_event_decode_cache_v3.6")
        self._decode_cache_dir = Path(cache_dir)
        self._decode_cache_dir.mkdir(parents=True, exist_ok=True)

        print(
            "[RobotWinEventBlockDataset] Decode fallback enabled: "
            f"ffmpeg={self._ffmpeg_bin}, cache={self._decode_cache_dir}"
        )

        self._physics_by_video = {}
        if physics_soft_path:
            p = Path(physics_soft_path)
            if not p.is_file():
                raise FileNotFoundError(
                    f"physics_soft_path not found for event+physics training: {physics_soft_path}")
            with open(p, "r", encoding="utf-8") as f:
                soft_raw = json.load(f)
            for row in soft_raw:
                vn = row.get("video_name")
                dist = row.get("label_distribution")
                if not vn or not isinstance(dist, dict):
                    continue
                soft_t = _label_distribution_to_tensor(dist)
                if float(soft_t.sum().item()) <= 0:
                    continue
                label = row.get("dominant_label") or row.get("original_label")
                label_int = WISA_LABEL_STR2INT_NORM.get(_normalize_label_key(label))
                if label_int is None:
                    label_int = int(torch.argmax(soft_t).item())
                self._physics_by_video[str(vn)] = (label_int, soft_t)
            print(
                "[RobotWinEventBlockDataset] Loaded physics soft labels for "
                f"{len(self._physics_by_video)} videos from {physics_soft_path}")
        elif self.has_physics_label:
            raise ValueError(
                "RobotWinEventBlockDataset requires --physics_soft_path when "
                "has_physics_label=True.")

        with open(event_label_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        if isinstance(raw, dict):
            self.samples = list(raw.get("samples", []))
            self.event_vocabulary = list(raw.get("event_vocabulary", []))
        elif isinstance(raw, list):
            self.samples = list(raw)
            self.event_vocabulary = sorted(
                {str(x.get("event_label", "no_event")) for x in self.samples})
        else:
            raise ValueError(f"Unsupported event label JSON: {event_label_path}")

        if not self.samples:
            raise RuntimeError(f"No event block samples found: {event_label_path}")

        self.event_to_id = {label: i for i, label in enumerate(self.event_vocabulary)}
        for sample in self.samples:
            label = str(sample.get("event_label", "no_event"))
            if label not in self.event_to_id:
                self.event_to_id[label] = len(self.event_vocabulary)
                self.event_vocabulary.append(label)
            sample["event_label_id"] = int(
                sample.get("event_label_id", self.event_to_id[label]))
            sample["_event_label_soft"] = _event_distribution_to_tensor(
                sample.get("event_label_distribution")
                or sample.get("label_distribution"),
                self.event_vocabulary,
                sample["event_label_id"],
            )
            if self.has_physics_label:
                video_rel = str(sample.get("video_name", "")).strip().lstrip("/")
                if video_rel not in self._physics_by_video:
                    raise KeyError(
                        "Missing physics soft label for event block video: "
                        f"{video_rel}")
        for sample in self.samples:
            sample["_event_label_soft"] = _event_distribution_to_tensor(
                sample.get("event_label_distribution")
                or sample.get("label_distribution"),
                self.event_vocabulary,
                sample["event_label_id"],
            )

        from collections import Counter
        dist = Counter(str(x.get("event_label", "no_event")) for x in self.samples)
        print(
            f"[RobotWinEventBlockDataset] Loaded {len(self.samples)} blocks "
            f"from {event_label_path}; classes={len(self.event_vocabulary)}")
        print(f"[RobotWinEventBlockDataset] Distribution: {dict(sorted(dist.items()))}")

    def _sample_block_frame_ids(self, start: int, end: int, total: int):
        start = max(0, int(start))
        end = min(max(start + 1, int(end)), int(total))
        indices = np.linspace(start, end - 1, self.num_frames).round().astype(int)
        indices = np.clip(indices, 0, max(int(total) - 1, 0))
        return indices.tolist()

    def __getstate__(self):
        state = self.__dict__.copy()
        state["video_loader"] = None
        return state

    def _get_video_loader(self):
        if self.video_loader is None:
            self.video_loader = get_video_loader()
        return self.video_loader

    def _load_block_frames_decord(self, video_path: str, start: int, end: int):
        vr = self._get_video_loader()(video_path)
        frame_ids = self._sample_block_frame_ids(start, end, len(vr))
        video_data = vr.get_batch(frame_ids).asnumpy()
        return [
            Image.fromarray(video_data[i]).convert("RGB")
            for i in range(len(frame_ids))
        ]

    def _load_block_frames_cv2(self, video_path: str, start: int, end: int):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise RuntimeError(f"cv2 failed to open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            cap.release()
            raise RuntimeError(f"cv2 got invalid frame count for video: {video_path}")

        frame_ids = self._sample_block_frame_ids(start, end, total)
        images = []
        last_rgb = None
        for fid in frame_ids:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fid))
            ok, frame = cap.read()
            if not ok or frame is None:
                if last_rgb is None:
                    cap.release()
                    raise RuntimeError(
                        f"cv2 failed to read frame {fid} from video: {video_path}")
                images.append(Image.fromarray(last_rgb.copy()).convert("RGB"))
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            last_rgb = rgb
            images.append(Image.fromarray(rgb).convert("RGB"))
        cap.release()
        return images

    def _transcode_for_decode(self, video_path: str):
        src = str(video_path)
        if src in self._decode_fallback_map:
            return self._decode_fallback_map[src]
        if src in self._decode_fallback_failed:
            return None

        try:
            st = os.stat(src)
        except OSError:
            self._decode_fallback_failed.add(src)
            return None

        key_src = f"{src}|{st.st_size}|{int(st.st_mtime_ns)}"
        key = hashlib.sha1(key_src.encode("utf-8")).hexdigest()[:24]
        dst = self._decode_cache_dir / f"{key}.mp4"
        lock = self._decode_cache_dir / f"{key}.lock"

        if dst.is_file() and dst.stat().st_size > 0:
            p = str(dst)
            self._decode_fallback_map[src] = p
            return p

        lock_fd = None
        for _ in range(600):
            try:
                lock_fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                os.write(lock_fd, str(os.getpid()).encode("utf-8"))
                break
            except FileExistsError:
                if dst.is_file() and dst.stat().st_size > 0:
                    p = str(dst)
                    self._decode_fallback_map[src] = p
                    return p
                time.sleep(0.1)

        if lock_fd is None:
            self._decode_fallback_failed.add(src)
            return None

        tmp = self._decode_cache_dir / f"{key}.{os.getpid()}.tmp.mp4"
        cmd = [
            self._ffmpeg_bin,
            "-y",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-hwaccel",
            "none",
            "-fflags",
            "+discardcorrupt",
            "-err_detect",
            "ignore_err",
            "-i",
            src,
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(tmp),
        ]

        try:
            proc = subprocess.run(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
        except Exception as e:
            print(f"[RobotWinEventBlockDataset] Decode fallback spawn failed: {e}")
            self._decode_fallback_failed.add(src)
            proc = None

        try:
            if proc is None or proc.returncode != 0 or (not tmp.is_file()) or tmp.stat().st_size == 0:
                if tmp.exists():
                    tmp.unlink(missing_ok=True)
                msg = proc.stderr.strip() if proc is not None and proc.stderr else "unknown ffmpeg error"
                if len(msg) > 500:
                    msg = msg[:500] + " ...[truncated]"
                print(
                    "[RobotWinEventBlockDataset] Decode fallback transcode failed: "
                    f"{src}; err={msg}"
                )
                self._decode_fallback_failed.add(src)
                return None

            tmp.replace(dst)
            p = str(dst)
            self._decode_fallback_map[src] = p
            print(f"[RobotWinEventBlockDataset] Decode fallback succeeded: {src} -> {p}")
            return p
        finally:
            if tmp.exists():
                tmp.unlink(missing_ok=True)
            if lock_fd is not None:
                os.close(lock_fd)
            lock.unlink(missing_ok=True)

    def _load_block_frames(self, video_path: str, start: int, end: int):
        transcode_path = self._transcode_for_decode(video_path)
        if transcode_path is None:
            raise RuntimeError(f"ffmpeg transcode failed for video: {video_path}")
        try:
            return self._load_block_frames_cv2(transcode_path, start, end)
        except Exception as e:
            self._decode_fallback_failed.add(str(video_path))
            raise RuntimeError(
                f"cv2 failed to read ffmpeg cache for video: {video_path}; err={e}"
            )

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        if len(self.samples) == 0:
            raise RuntimeError("RobotWinEventBlockDataset has no samples")

        max_retry = min(64, len(self.samples))
        last_err = None
        sample = None
        images = None

        for attempt in range(max_retry):
            if index in self._bad_indices:
                index = random.randint(0, len(self.samples) - 1)
                continue

            sample = self.samples[index]
            video_rel = str(sample.get("video_name", "")).strip().lstrip("/")
            video_path = self.datasets_root / video_rel
            if not video_path.is_file():
                last_err = FileNotFoundError(
                    f"missing video for event block: {video_path}")
                self._bad_indices.add(index)
                index = random.randint(0, len(self.samples) - 1)
                continue

            try:
                images = self._load_block_frames(
                    str(video_path),
                    int(sample["start_frame"]),
                    int(sample["end_frame"]))
                break
            except Exception as e:
                last_err = e
                self._bad_indices.add(index)
                if attempt < 5 or (attempt + 1) % 10 == 0:
                    print(
                        "[RobotWinEventBlockDataset] Load failed "
                        f"(attempt={attempt + 1}/{max_retry}, index={index}): "
                        f"{video_path}; err={e}"
                    )
                if len(self._bad_indices) >= len(self.samples):
                    raise RuntimeError(
                        "All event blocks failed to decode. "
                        f"last_error={last_err}"
                    )
                index = random.randint(0, len(self.samples) - 1)
        else:
            raise RuntimeError(
                "Exceeded max retries while decoding event blocks. "
                f"bad_indices={len(self._bad_indices)}/{len(self.samples)}, "
                f"last_error={last_err}"
            )

        process_data, enc_mask, dec_mask = self.transform((images, None))
        process_data = process_data.view(
            (self.num_frames, 3) + process_data.size()[-2:]
        ).transpose(0, 1)

        event_label = torch.tensor(int(sample["event_label_id"]), dtype=torch.long)
        event_score = torch.tensor(float(sample.get("event_score", 1.0)),
                                   dtype=torch.float32)
        event_label_soft = sample.get("_event_label_soft")
        if event_label_soft is None:
            event_label_soft = _event_distribution_to_tensor(
                sample.get("event_label_distribution")
                or sample.get("label_distribution"),
                self.event_vocabulary,
                int(sample["event_label_id"]),
            )
        else:
            event_label_soft = event_label_soft.clone()
        if self.has_physics_label:
            video_rel = str(sample.get("video_name", "")).strip().lstrip("/")
            physics_label_int, physics_label_soft = self._physics_by_video[video_rel]
            physics_label = torch.tensor(int(physics_label_int), dtype=torch.long)
            return (
                process_data,
                enc_mask,
                dec_mask,
                physics_label,
                physics_label_soft.clone(),
                event_label,
                event_score,
                event_label_soft,
            )
        return process_data, enc_mask, dec_mask, event_label, event_score, event_label_soft


def build_wisa_pretraining_dataset(args):
    """在 build.py 里调用的工厂函数"""
    transform = DataAugmentationForVideoMAEv2(args)
    event_label_path = getattr(args, "event_label_path", "") or ""
    if event_label_path:
        dataset = RobotWinEventBlockDataset(
            datasets_root=args.datasets_root,
            event_label_path=event_label_path,
            transform=transform,
            num_frames=args.num_frames,
            physics_soft_path=getattr(args, "physics_soft_path", "") or "",
            has_physics_label=bool(getattr(args, "has_physics_label", False)))
        print("Data Aug = %s" % str(transform))
        return dataset

    soft_path = getattr(args, "physics_soft_path", "") or ""
    dataset = WISAPretrainDataset(
        datasets_root=args.datasets_root,
        anno_path=args.anno_path,
        transform=transform,
        num_frames=args.num_frames,
        sampling_rate=args.sampling_rate,
        num_sample=args.num_sample,
        physics_soft_path=soft_path,
    )
    print("Data Aug = %s" % str(transform))
    return dataset
