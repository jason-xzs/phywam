import argparse
import importlib
import inspect
import json
import os
import re
import subprocess
import sys
import tempfile
import types
from pathlib import Path
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv'}


def read_jsonl(path: Path):
    records = []
    with path.open('r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def find_task_roots(dataset_root: Path):
    if (dataset_root / 'meta' / 'episodes.jsonl').is_file():
        return [dataset_root]
    roots = []
    for path in dataset_root.glob('**/meta/episodes.jsonl'):
        roots.append(path.parent.parent)
    return sorted(set(roots))


def find_video_path(task_root: Path, camera_key: str, episode_index: int):
    name = f'episode_{episode_index:06d}'
    for path in sorted((task_root / 'videos').glob(f'*/{camera_key}/{name}.*')):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            return path, path.parents[1].name
    return None, None


def infer_task_key(task_name: str) -> str:
    return str(task_name).split('-')[0]


def load_event_records(event_label_json: str, mope_repo: str, args):
    if not event_label_json:
        return None
    dataset_dir = Path(mope_repo) / 'dataset'
    if str(dataset_dir) not in sys.path:
        sys.path.insert(0, str(dataset_dir))
    from event_language import build_event_segments  # noqa: WPS433

    path = Path(event_label_json)
    with path.open('r', encoding='utf-8') as f:
        payload = json.load(f)
    payload_mode = 'list'
    if isinstance(payload, list):
        records = payload
    elif isinstance(payload, dict) and isinstance(payload.get('samples'), list):
        records = records_from_segment_samples(payload)
        payload_mode = str(payload.get('annotation_version', 'dict_samples'))
    else:
        raise ValueError(
            f'event label JSON should be a list or a dict with samples: {path}')

    primary: Dict[Tuple[str, str, int], dict] = {}
    loose: Dict[Tuple[str, int], List[dict]] = {}
    for record in records:
        episode_index = int(record.get('episode_index', -1))
        split = str(record.get('split', ''))
        task_name = str(record.get('task_name', ''))
        task_key = str(record.get('task_key', ''))
        if episode_index < 0:
            continue
        base_task_name = str(record.get('base_task_name', ''))
        for name in {task_name, task_key, base_task_name}:
            if not name:
                continue
            for split_alias in split_aliases(split):
                primary[(split_alias, name, episode_index)] = record
            loose.setdefault((name, episode_index), []).append(record)

    unique_loose = {
        key: values[0]
        for key, values in loose.items()
        if len({
            (str(v.get('split', '')), str(v.get('task_name', '')), int(v.get('episode_index', -1)))
            for v in values
        }) == 1
    }
    print(f'[Events] loaded records={len(records)} mode={payload_mode} from {path}')
    return {
        'primary': primary,
        'loose': unique_loose,
        'build_event_segments': build_event_segments,
        'min_segment_frames': int(args.event_min_segment_frames),
        'max_segment_frames': int(args.event_max_segment_frames),
    }


def split_aliases(split: str):
    split = str(split or '')
    aliases = {split}
    if split.endswith('_s'):
        aliases.add(split[:-2])
    elif split:
        aliases.add(f'{split}_s')
    return aliases


def records_from_segment_samples(payload: dict) -> List[dict]:
    """Convert v39 task-canonical segment samples into episode-level records."""
    grouped = OrderedDict()
    for sample in payload.get('samples', []):
        episode_index = int(sample.get('episode_index', -1))
        if episode_index < 0:
            continue
        split = str(sample.get('split', ''))
        task_name = str(sample.get('task_name') or sample.get('base_task_name') or sample.get('task_key') or '')
        task_key = str(sample.get('task_key') or infer_task_key(task_name))
        base_task_name = str(sample.get('base_task_name') or task_key)
        video_name = str(sample.get('video_name', ''))
        key = (split, task_name, episode_index, video_name)
        if key not in grouped:
            grouped[key] = {
                'split': split,
                'task_name': task_name,
                'task_key': task_key,
                'base_task_name': base_task_name,
                'episode_index': episode_index,
                'video_name': video_name,
                'event_segments': [],
            }
        grouped[key]['event_segments'].append(segment_from_sample(sample))

    records = []
    for record in grouped.values():
        record['event_segments'] = sorted(
            record['event_segments'],
            key=lambda item: (
                int(item.get('start_frame', 0)),
                int(item.get('end_frame', 0)),
            ),
        )
        records.append(record)
    return records


def segment_from_sample(sample: dict) -> dict:
    start = int(sample.get('start_frame', 0))
    end = int(sample.get('end_frame', start + 1))
    event_frame = sample.get('event_frame', None)
    if event_frame is None:
        event_frame = (start + end) // 2
    return {
        'start_frame': start,
        'end_frame': end,
        'event_label': str(sample.get('event_label', sample.get('event', 'no_event'))),
        'event_frame': int(event_frame),
        'event_score': float(sample.get('event_score', sample.get('score', 1.0)) or 0.0),
        'event_text': str(sample.get('event_text', '')),
        'action_text': str(sample.get('action_text', '')),
        'segment_source': str(sample.get('segment_source', '')),
    }


def find_event_record(event_index, task_root: Path, episode_index: int) -> Optional[dict]:
    if event_index is None:
        return None
    split = task_root.parent.name
    task_name = task_root.name
    task_key = infer_task_key(task_name)
    for key in (
        (split, task_name, episode_index),
        (split, task_key, episode_index),
    ):
        record = event_index['primary'].get(key)
        if record is not None:
            return record
    for key in (
        (task_name, episode_index),
        (task_key, episode_index),
    ):
        record = event_index['loose'].get(key)
        if record is not None:
            return record
    return None


def fixed_block_segments(length: int, block_size: int) -> List[dict]:
    return [
        {
            'start_frame': start,
            'end_frame': min(length, start + block_size),
            'event_label': 'fixed_block',
            'event_frame': -1,
            'event_score': 1.0,
            'event_text': '',
            'action_text': '',
            'segment_source': 'fixed_block',
        }
        for start in range(0, length, block_size)
    ]


def event_segments_for_episode(args, event_index, task_root: Path, episode: dict) -> Optional[List[dict]]:
    length = int(episode.get('length') or episode.get('episode_length') or 0)
    record = find_event_record(event_index, task_root, int(episode['episode_index']))
    if record is None:
        message = (
            f'missing event labels for split={task_root.parent.name} '
            f'task={task_root.name} episode={int(episode["episode_index"]):06d}'
        )
        if args.missing_event_policy == 'error':
            raise KeyError(message)
        if args.missing_event_policy == 'skip':
            print(f'[Events][SKIP] {message}')
            return None
        print(f'[Events][WARN] {message}; falling back to fixed blocks')
        return fixed_block_segments(length, args.block_size)

    segments = list(record.get('event_segments') or [])
    if not segments:
        segments = event_index['build_event_segments'](
            record,
            min_segment_frames=event_index['min_segment_frames'],
            max_segment_frames=event_index['max_segment_frames'],
            include_no_event_when_empty=True,
        )
    if args.drop_no_event_segments:
        segments = [
            segment for segment in segments
            if str(segment.get('event_label', segment.get('event', 'no_event'))) != 'no_event'
        ]

    out = []
    for segment in segments:
        start = max(0, min(length - 1, int(segment.get('start_frame', 0))))
        end = max(start + 1, min(length, int(segment.get('end_frame', start + 1))))
        item = dict(segment)
        item['start_frame'] = start
        item['end_frame'] = end
        out.append(item)
    return sorted(out, key=lambda item: (int(item['start_frame']), int(item['end_frame'])))


def clip_video_by_frame(src_video: Path, dst_video: Path, start_frame: int, end_frame: int, ffmpeg_bin: str):
    if end_frame <= start_frame:
        raise ValueError(f'invalid frame range: start={start_frame}, end={end_frame}')

    dst_video.parent.mkdir(parents=True, exist_ok=True)
    select_expr = f"select='between(n\\,{start_frame}\\,{end_frame - 1})',setpts=N/FRAME_RATE/TB"
    cmd = [
        str(ffmpeg_bin),
        '-y',
        '-v',
        'error',
        '-hwaccel',
        'none',
        '-i',
        str(src_video),
        '-vf',
        select_expr,
        '-an',
        '-c:v',
        'libx264',
        '-pix_fmt',
        'yuv420p',
        str(dst_video),
    ]
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f'ffmpeg clip failed for {src_video}, range=({start_frame},{end_frame}). '
            f'stderr: {proc.stderr.strip()}')


def _get_module_by_path(root_module: nn.Module, path: str):
    cur = root_module
    if not path:
        return cur
    for part in path.split('.'):
        cur = cur[int(part)] if part.isdigit() else getattr(cur, part)
    return cur


def _set_module_by_path(root_module: nn.Module, path: str, new_module: nn.Module):
    parts = path.split('.')
    parent_path = '.'.join(parts[:-1])
    leaf = parts[-1]
    parent = _get_module_by_path(root_module, parent_path) if parent_path else root_module
    if leaf.isdigit():
        parent[int(leaf)] = new_module
    else:
        setattr(parent, leaf, new_module)


def _legacy_time_summary(self, x, time_ids=None, num_time_bins=None):
    return x.mean(dim=1)


def patch_router_to_legacy_768_if_needed(model: nn.Module, state_dict: dict):
    model_state = model.state_dict()
    patched = []
    for key, value in state_dict.items():
        if not key.endswith('sparse_moe.router.gate.weight'):
            continue
        if key not in model_state:
            continue
        target = model_state[key]
        if value.ndim != 2 or target.ndim != 2:
            continue
        if value.shape[0] != target.shape[0]:
            continue
        if target.shape[1] != value.shape[1] * 2:
            continue

        gate_path = key[:-len('.weight')]
        gate_module = _get_module_by_path(model, gate_path)
        new_gate = nn.Linear(value.shape[1], value.shape[0], bias=False)
        new_gate = new_gate.to(device=gate_module.weight.device, dtype=gate_module.weight.dtype)
        _set_module_by_path(model, gate_path, new_gate)

        sparse_moe_path = gate_path.rsplit('.router.gate', 1)[0]
        sparse_moe_module = _get_module_by_path(model, sparse_moe_path)
        sparse_moe_module._build_time_aware_summary = types.MethodType(
            _legacy_time_summary, sparse_moe_module
        )
        patched.append(gate_path)
    if patched:
        print(f'[MoPE] patched {len(patched)} router gates to legacy 768-input mode')


def _build_mope_model(args, state_dict):
    modeling_pretrain = importlib.import_module('models.modeling_pretrain')
    if not hasattr(modeling_pretrain, args.model):
        raise ValueError(f"MoPE model constructor not found: {args.model}")
    create_fn = getattr(modeling_pretrain, args.model)

    model_kwargs = dict(
        pretrained=False,
        drop_path_rate=0.0,
        all_frames=args.num_frames,
        tubelet_size=args.tubelet_size,
        with_cp=False,
    )
    if args.use_mope:
        model_kwargs.update(
            num_routable_experts=args.num_routable_experts,
            num_shared_experts=args.num_shared_experts,
            top_k=args.top_k,
        )
    num_event_classes = int(getattr(args, 'num_event_classes', 0) or 0)
    if num_event_classes <= 0:
        event_weight = state_dict.get('event_head.weight')
        if event_weight is not None and getattr(event_weight, 'ndim', 0) == 2:
            num_event_classes = int(event_weight.shape[0])
    if num_event_classes > 0:
        model_kwargs.update(num_event_classes=num_event_classes)

    sig = inspect.signature(create_fn)
    has_var_kw = any(
        param.kind == inspect.Parameter.VAR_KEYWORD
        for param in sig.parameters.values()
    )
    if not has_var_kw:
        supported = set(sig.parameters.keys())
        model_kwargs = {
            key: value for key, value in model_kwargs.items()
            if key in supported
        }

    while True:
        try:
            model = create_fn(**model_kwargs)
            break
        except TypeError as exc:
            match = re.search(r"unexpected keyword argument '([^']+)'", str(exc))
            if match is None or match.group(1) not in model_kwargs:
                raise
            bad_kw = match.group(1)
            print(f'[MoPE][WARN] retry model build without unsupported kwarg: {bad_kw}')
            model_kwargs.pop(bad_kw)

    patch_router_to_legacy_768_if_needed(model, state_dict)
    return model


def build_mope(args):
    mope_repo = Path(args.mope_repo)
    if str(mope_repo) not in sys.path:
        sys.path.insert(0, str(mope_repo))

    import infer_mope  # noqa: WPS433
    import models  # noqa: F401, WPS433

    ckpt = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    state_dict = ckpt.get('model') or ckpt.get('module') or ckpt.get('state_dict') or ckpt
    state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}

    model = _build_mope_model(args, state_dict)
    model_state = model.state_dict()
    adapted_state = {}
    skipped = []
    for key, value in state_dict.items():
        if key not in model_state:
            continue
        if tuple(value.shape) == tuple(model_state[key].shape):
            adapted_state[key] = value
        else:
            skipped.append((key, tuple(value.shape), tuple(model_state[key].shape)))
    if skipped:
        print(f'[MoPE][WARN] skipped {len(skipped)} mismatched checkpoint keys')
        for key, src_shape, dst_shape in skipped[:8]:
            print(f'  - {key}: ckpt={src_shape}, model={dst_shape}')
    msg = model.load_state_dict(adapted_state, strict=False)
    print(f'[MoPE] loaded missing={len(msg.missing_keys)} unexpected={len(msg.unexpected_keys)}')

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model.eval().to(device)
    transform = infer_mope.build_inference_transform(args.input_size)
    return infer_mope, model, transform, device


@torch.no_grad()
def extract_block_feature(infer_mope, model, transform, device, images, num_frames: int):
    video_tensor = infer_mope.frames_to_tensor(images, transform, num_frames)
    feat = infer_mope.infer_extract(model, video_tensor, device)
    feat = np.asarray(feat, dtype=np.float32)
    if feat.ndim != 2:
        raise ValueError(f'expected MoPE tokens [N, C], got {feat.shape}')
    return feat.mean(axis=0).astype(np.float32, copy=False)


@torch.no_grad()
def extract_block_feature_and_event(infer_mope, model, transform, device, images, num_frames: int):
    video_tensor = infer_mope.frames_to_tensor(images, transform, num_frames)
    feat_tokens = infer_mope.infer_extract(model, video_tensor, device)
    feat_tokens = np.asarray(feat_tokens, dtype=np.float32)
    if feat_tokens.ndim != 2:
        raise ValueError(f'expected MoPE tokens [N, C], got {feat_tokens.shape}')

    event = None
    event_head = getattr(model, 'event_head', None)
    if event_head is not None:
        video_tensor = video_tensor.to(device)
        batch_size = video_tensor.size(0)
        mask = infer_mope.make_full_visible_mask(
            batch_size, model.encoder.patch_embed.num_patches, device)
        x_vis = model.encoder.forward_features(
            video_tensor,
            mask,
            physics_label=None,
            physics_label_soft=None,
        )
        logits = event_head(x_vis.mean(dim=1))
        probs = torch.softmax(logits.float(), dim=-1)
        confidence, label_id = probs.max(dim=-1)
        event = {
            'label_id': int(label_id.item()),
            'confidence': float(confidence.item()),
            'probs': probs[0].detach().cpu().numpy().astype(np.float32),
        }

    return feat_tokens.mean(axis=0).astype(np.float32, copy=False), event


def process_episode(args, infer_mope, model, transform, device, task_root: Path, episode: dict, event_index=None):
    episode_index = int(episode['episode_index'])
    length = int(episode.get('length') or episode.get('episode_length') or 0)
    if length <= 0:
        return False

    video_path, chunk_name = find_video_path(task_root, args.camera_key, episode_index)
    if video_path is None:
        print(f'[WARN] missing video task={task_root.name} episode={episode_index:06d}')
        return False

    out_dir = task_root / args.output_dirname / chunk_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f'episode_{episode_index:06d}.npz'
    if out_path.exists() and not args.overwrite:
        return True

    if args.segment_mode == 'event':
        segments = event_segments_for_episode(args, event_index, task_root, episode)
        if segments is None:
            return False
    else:
        segments = fixed_block_segments(length, args.block_size)
    if not segments:
        print(f'[WARN] no segments task={task_root.name} episode={episode_index:06d}')
        return False

    features = []
    starts = []
    ends = []
    event_labels = []
    event_frames = []
    event_scores = []
    event_texts = []
    action_texts = []
    segment_sources = []
    temp_parent = Path(args.temp_root) if args.temp_root else task_root
    temp_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix='.tmp_phywam_phys_videos_',
        dir=temp_parent,
    ) as temp_dir:
        temp_clip_dir = Path(temp_dir)
        for segment_idx, segment in enumerate(segments):
            if args.max_blocks_per_episode > 0 and segment_idx >= args.max_blocks_per_episode:
                break
            start = int(segment['start_frame'])
            end = int(segment['end_frame'])
            clip_path = temp_clip_dir / f'episode_{episode_index:06d}_{start}_{end}.mp4'
            clip_video_by_frame(video_path, clip_path, start, end, args.ffmpeg_bin)
            images = infer_mope.load_video_frames(
                str(clip_path), args.num_frames, args.sampling_rate)
            features.append(extract_block_feature(
                infer_mope, model, transform, device, images, args.num_frames))
            starts.append(start)
            ends.append(end)
            event_labels.append(str(segment.get('event_label', segment.get('event', 'fixed_block'))))
            event_frames.append(int(segment.get('event_frame', -1) if segment.get('event_frame') is not None else -1))
            event_scores.append(float(segment.get('event_score', segment.get('score', 1.0)) or 0.0))
            event_texts.append(str(segment.get('event_text', '')))
            action_texts.append(str(segment.get('action_text', '')))
            segment_sources.append(str(segment.get('segment_source', '')))

    np.savez_compressed(
        out_path,
        features=np.stack(features, axis=0).astype(np.float32),
        block_starts=np.asarray(starts, dtype=np.int32),
        block_ends=np.asarray(ends, dtype=np.int32),
        block_size=np.asarray(args.block_size if args.segment_mode == 'fixed' else -1, dtype=np.int32),
        start_frame=np.asarray(0, dtype=np.int32),
        episode_length=np.asarray(length, dtype=np.int32),
        camera_key=np.asarray(args.camera_key),
        segment_mode=np.asarray(args.segment_mode),
        event_labels=np.asarray(event_labels, dtype=np.str_),
        event_frames=np.asarray(event_frames, dtype=np.int32),
        event_scores=np.asarray(event_scores, dtype=np.float32),
        event_texts=np.asarray(event_texts, dtype=np.str_),
        action_texts=np.asarray(action_texts, dtype=np.str_),
        segment_sources=np.asarray(segment_sources, dtype=np.str_),
    )
    print(f'[OK] {out_path} mode={args.segment_mode} segments={len(features)}')
    return True


def build_parser():
    parser = argparse.ArgumentParser('Build block- or event-aligned PhyWAM physics tokens.')
    parser.add_argument('--dataset-root', required=True)
    parser.add_argument('--mope-repo', required=True)
    parser.add_argument('--ckpt', required=True)
    parser.add_argument('--camera-key', default='observation.images.cam_high')
    parser.add_argument('--output-dirname', default='physics_features3.4')
    parser.add_argument('--segment-mode', choices=['fixed', 'event'], default='fixed')
    parser.add_argument('--event-label-json', default='')
    parser.add_argument('--event-min-segment-frames', type=int, default=4)
    parser.add_argument('--event-max-segment-frames', type=int, default=0)
    parser.add_argument(
        '--missing-event-policy',
        choices=['error', 'fixed', 'skip'],
        default='error',
        help='Policy when --segment-mode event cannot find labels for an episode.',
    )
    parser.add_argument(
        '--drop-no-event-segments',
        action='store_true',
        default=False,
        help='Drop synthesized no_event segments before building event physics tokens.',
    )
    parser.add_argument('--block-size', type=int, default=16)
    parser.add_argument('--num-frames', type=int, default=16)
    parser.add_argument('--sampling-rate', type=int, default=4)
    parser.add_argument('--input-size', type=int, default=224)
    parser.add_argument('--device', default='cuda')
    parser.add_argument(
        '--ffmpeg-bin',
        default=os.environ.get('MOPE_FFMPEG_BIN', 'ffmpeg'),
    )
    parser.add_argument(
        '--temp-root',
        default=os.environ.get('PHYWAM_TEMP_ROOT', ''),
        help='Where to write temporary clipped videos; defaults to the task root.',
    )
    parser.add_argument('--overwrite', action='store_true')

    parser.add_argument('--model', default='pretrain_mope_jepa_base_patch16_224')
    parser.add_argument('--tubelet-size', type=int, default=2)
    parser.add_argument('--use-mope', action='store_true', default=True)
    parser.add_argument('--num-routable-experts', type=int, default=17)
    parser.add_argument('--num-shared-experts', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--num-event-classes', type=int, default=0)
    parser.add_argument('--max-episodes', type=int, default=0)
    parser.add_argument('--max-blocks-per-episode', type=int, default=0)
    parser.add_argument(
        '--episode-num-shards',
        type=int,
        default=1,
        help='Split episodes by episode_index modulo this value.',
    )
    parser.add_argument(
        '--episode-shard-index',
        type=int,
        default=0,
        help='Run only episodes whose episode_index modulo episode-num-shards equals this value.',
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.episode_num_shards < 1:
        raise ValueError('--episode-num-shards must be >= 1')
    if args.episode_shard_index < 0 or args.episode_shard_index >= args.episode_num_shards:
        raise ValueError('--episode-shard-index must be in [0, episode-num-shards)')
    if args.event_label_json and args.segment_mode == 'fixed':
        args.segment_mode = 'event'
    if args.segment_mode == 'event' and not args.event_label_json:
        raise ValueError('--segment-mode event requires --event-label-json')

    dataset_root = Path(args.dataset_root)
    event_index = load_event_records(args.event_label_json, args.mope_repo, args) if args.segment_mode == 'event' else None
    infer_mope, model, transform, device = build_mope(args)
    task_roots = find_task_roots(dataset_root)
    print(f'[INFO] task_roots={len(task_roots)}')
    print(f'[INFO] segment_mode={args.segment_mode}')
    if args.episode_num_shards > 1:
        print(
            f'[INFO] episode_shard={args.episode_shard_index}/'
            f'{args.episode_num_shards}'
        )

    total = 0
    ok = 0
    for task_root in task_roots:
        episodes_path = task_root / 'meta' / 'episodes.jsonl'
        episodes = read_jsonl(episodes_path)
        print(f'[TASK] {task_root.name} episodes={len(episodes)}')
        for episode in episodes:
            episode_index = int(episode['episode_index'])
            if episode_index % args.episode_num_shards != args.episode_shard_index:
                continue
            total += 1
            ok += int(process_episode(
                args, infer_mope, model, transform, device, task_root, episode, event_index=event_index))
            if args.max_episodes > 0 and total >= args.max_episodes:
                print(f'[SUMMARY] ok={ok} total={total}')
                return
    print(f'[SUMMARY] ok={ok} total={total}')


if __name__ == '__main__':
    main()
