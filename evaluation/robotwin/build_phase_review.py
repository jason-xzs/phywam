#!/usr/bin/env python3
import argparse
import csv
import html
import json
import re
import shutil
import subprocess
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any, Dict, Iterable, List, Sequence, Tuple


TASKS = (
    "hanging_mug",
    "place_can_basket",
    "handover_block",
    "pick_diverse_bottles",
)


def safe_name(value: Any, max_len: int = 80) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "unknown")).strip("_")
    return text[:max_len] or "unknown"


def read_jsonl(path: Path) -> List[dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(float(x) for x in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def find_binary(explicit: str, name: str) -> str:
    if explicit:
        return explicit
    found = shutil.which(name)
    if found:
        return found
    fallback = Path("/data1/miniconda3/envs/worldmodel_phyva/bin") / name
    if fallback.is_file():
        return str(fallback)
    raise FileNotFoundError(f"cannot find {name}")


def probe_video(ffprobe: str, path: Path) -> dict:
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,r_frame_rate,width,height,duration",
        "-of",
        "json",
        str(path),
    ]
    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    stream = json.loads(result.stdout)["streams"][0]
    rate = stream.get("r_frame_rate", "0/1").split("/")
    fps = float(rate[0]) / float(rate[1]) if float(rate[1]) else 0.0
    frame_count = int(stream.get("nb_frames") or 0)
    duration = float(stream.get("duration") or 0.0)
    if not frame_count and fps and duration:
        frame_count = int(round(fps * duration))
    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_seconds": duration,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
    }


def cut_video(
    ffmpeg: str,
    source: Path,
    output: Path,
    start_frame: int,
    end_frame: int,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    end_frame = max(start_frame + 1, end_frame)
    select = (
        f"select='between(n\\,{start_frame}\\,{end_frame - 1})',"
        "setpts=N/FRAME_RATE/TB"
    )
    command = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(source),
        "-vf",
        select,
        "-fps_mode",
        "vfr",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    subprocess.run(command, check=True)


def relative_link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        destination.unlink()
    destination.symlink_to(source)


def group_label_episodes(samples: Iterable[dict]) -> Dict[str, List[dict]]:
    grouped: Dict[Tuple[Any, ...], List[dict]] = defaultdict(list)
    for row in samples:
        key = (
            row.get("task_key"),
            row.get("split"),
            row.get("task_name"),
            int(row.get("episode_index", -1)),
            row.get("video_name"),
        )
        grouped[key].append(row)

    by_task: Dict[str, List[dict]] = defaultdict(list)
    for key, segments in grouped.items():
        task, split, task_name, episode_index, video_name = key
        ordered = sorted(segments, key=lambda x: int(x.get("segment_index", 0)))
        first = ordered[0]
        by_task[task].append(
            {
                "task_name": task,
                "split": split,
                "source_task_name": task_name,
                "episode_index": episode_index,
                "episode_length": int(first.get("episode_length", 0)),
                "task_text": first.get("task_text", ""),
                "video_name": video_name,
                "segments": [
                    {
                        "segment_index": int(row.get("segment_index", 0)),
                        "phase": row.get("event_label", ""),
                        "start_frame": int(row.get("start_frame", 0)),
                        "end_frame": int(row.get("end_frame", 0)),
                        "event_score": float(row.get("event_score", 0.0)),
                        "fine_event_labels": row.get("fine_event_labels", []),
                        "segment_source": row.get("segment_source", ""),
                    }
                    for row in ordered
                ],
            }
        )
    for episodes in by_task.values():
        episodes.sort(key=lambda x: (x["split"], x["episode_index"]))
    return by_task


def canonical_reference(task: str, episodes: Sequence[dict]) -> dict:
    if not episodes:
        return {"sequence": [], "boundary_statistics": [], "episode_count": 0}
    sequence = [row["phase"] for row in episodes[0]["segments"]]
    stats = []
    for segment_index, phase in enumerate(sequence):
        candidates = [
            episode["segments"][segment_index]
            for episode in episodes
            if len(episode["segments"]) > segment_index
            and episode["segments"][segment_index]["phase"] == phase
            and episode["episode_length"] > 0
        ]
        start_ratios = [
            row["start_frame"] / episode["episode_length"]
            for episode in episodes
            for row in (
                [episode["segments"][segment_index]]
                if len(episode["segments"]) > segment_index
                and episode["segments"][segment_index]["phase"] == phase
                else []
            )
            if episode["episode_length"] > 0
        ]
        end_ratios = [
            row["end_frame"] / episode["episode_length"]
            for episode in episodes
            for row in (
                [episode["segments"][segment_index]]
                if len(episode["segments"]) > segment_index
                and episode["segments"][segment_index]["phase"] == phase
                else []
            )
            if episode["episode_length"] > 0
        ]
        durations = [
            row["end_frame"] - row["start_frame"]
            for row in candidates
        ]
        stats.append(
            {
                "segment_index": segment_index,
                "phase": phase,
                "sample_count": len(candidates),
                "start_ratio": {
                    "p10": percentile(start_ratios, 0.1),
                    "median": median(start_ratios) if start_ratios else 0.0,
                    "p90": percentile(start_ratios, 0.9),
                },
                "end_ratio": {
                    "p10": percentile(end_ratios, 0.1),
                    "median": median(end_ratios) if end_ratios else 0.0,
                    "p90": percentile(end_ratios, 0.9),
                },
                "duration_frames": {
                    "p10": percentile(durations, 0.1),
                    "median": median(durations) if durations else 0.0,
                    "p90": percentile(durations, 0.9),
                },
            }
        )
    return {
        "task_name": task,
        "sequence": sequence,
        "episode_count": len(episodes),
        "boundary_statistics": stats,
    }


def build_phase_events(trace: Sequence[dict], fps: float, frame_count: int) -> List[dict]:
    events = []
    frame_cursor = 0
    real_update_index = 0
    for trace_index, item in enumerate(trace):
        real_update = bool(item.get("real_update", False))
        if real_update:
            if real_update_index:
                frame_cursor += int(item.get("input_obs_frames", 0) or 0)
            real_update_index += 1
        video_frame = min(max(frame_cursor, 0), max(frame_count - 1, 0))
        events.append(
            {
                "trace_index": trace_index,
                "detection_index": real_update_index - 1 if real_update else None,
                "request_kind": "phase_detection" if real_update else "token_reuse",
                "video_frame": video_frame,
                "video_time_seconds": video_frame / fps if fps else 0.0,
                "estimated_env_step": video_frame * 4,
                "input_window": {
                    "observation_frames_added": int(
                        item.get("input_obs_frames", 0) or 0
                    ),
                    "buffer_frames": int(item.get("buffer_frames", 0) or 0),
                    "mope_input_frames": int(
                        item.get("mope_input_frames", 0) or 0
                    ),
                },
                "predicted_phase": item.get("predicted_label", ""),
                "confidence": float(item.get("confidence", 0.0) or 0.0),
                "accepted_current_phase": item.get("current_label", ""),
                "update_action": item.get("update_action", ""),
                "update_reason": item.get("update_reason", ""),
                "encoded": bool(item.get("encoded", False)),
                "token_updated": bool(item.get("token_updated", False)),
                "token_appended": bool(item.get("token_appended", False)),
                "token_replaced": bool(item.get("token_replaced", False)),
                "tokens_before": int(item.get("tokens_before", 0) or 0),
                "tokens_after": int(item.get("tokens_after", 0) or 0),
                "manual_review": {
                    "observed_phase": None,
                    "prediction_correct": None,
                    "accepted_phase_correct": None,
                    "token_update_correct": None,
                    "update_timely": None,
                    "boundary_error_frames": None,
                    "notes": "",
                },
            }
        )
    return events


def build_accepted_segments(events: Sequence[dict], frame_count: int) -> List[dict]:
    boundaries = []
    for event in events:
        if event["update_action"] not in {"initialize", "transition"}:
            continue
        boundary = {
            "phase": event["accepted_current_phase"],
            "start_frame": event["video_frame"],
            "start_time_seconds": event["video_time_seconds"],
            "trigger_prediction": event["predicted_phase"],
            "trigger_confidence": event["confidence"],
            "trigger_action": event["update_action"],
            "tokens_after": event["tokens_after"],
        }
        if not boundaries or boundaries[-1]["start_frame"] != boundary["start_frame"]:
            boundaries.append(boundary)
    if not boundaries and events:
        boundaries.append(
            {
                "phase": events[0]["accepted_current_phase"] or "unknown_phase",
                "start_frame": 0,
                "start_time_seconds": 0.0,
                "trigger_prediction": events[0]["predicted_phase"],
                "trigger_confidence": events[0]["confidence"],
                "trigger_action": events[0]["update_action"],
                "tokens_after": events[0]["tokens_after"],
            }
        )

    segments = []
    for index, boundary in enumerate(boundaries):
        start = boundary["start_frame"]
        end = (
            boundaries[index + 1]["start_frame"]
            if index + 1 < len(boundaries)
            else frame_count
        )
        segment = dict(boundary)
        segment.update(
            {
                "segment_index": index,
                "end_frame": max(start + 1, end),
            }
        )
        segments.append(segment)
    return segments


def manual_review_template() -> dict:
    return {
        "episode_phase_sequence_correct": None,
        "all_transitions_timely": None,
        "token_construction_usable": None,
        "failure_cause_category": None,
        "notes": "",
    }


def render_episode_html(record: dict) -> str:
    event_rows = []
    for event in record["phase_events"]:
        if event["request_kind"] != "phase_detection":
            continue
        seek = f"seek({event['video_time_seconds']:.6f})"
        event_rows.append(
            "<tr>"
            f"<td>{event['detection_index']}</td>"
            f"<td><button onclick=\"{seek}\">{event['video_frame']}</button></td>"
            f"<td>{event['video_time_seconds']:.2f}</td>"
            f"<td>{html.escape(event['predicted_phase'])}</td>"
            f"<td>{event['confidence']:.3f}</td>"
            f"<td>{html.escape(event['accepted_current_phase'])}</td>"
            f"<td>{html.escape(event['update_action'])}</td>"
            f"<td>{html.escape(event['update_reason'])}</td>"
            f"<td>{event['tokens_before']} -&gt; {event['tokens_after']}</td>"
            "</tr>"
        )
    clip_links = "\n".join(
        (
            "<li>"
            f"<a href=\"{html.escape(segment['clip'])}\">"
            f"{segment['segment_index']:02d} {html.escape(segment['phase'])} "
            f"[{segment['start_frame']}, {segment['end_frame']})"
            "</a></li>"
        )
        for segment in record["accepted_phase_segments"]
    )
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>{html.escape(record['task_name'])} episode {record['video_id']}</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
video {{ width: min(100%, 1100px); }}
table {{ border-collapse: collapse; width: 100%; margin-top: 16px; }}
th, td {{ border: 1px solid #bbb; padding: 5px; text-align: left; }}
th {{ position: sticky; top: 0; background: white; }}
button {{ cursor: pointer; }}
code {{ white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>{html.escape(record['task_name'])} / episode {record['video_id']}</h1>
<p>seed={record['seed']} success={record['success']}</p>
<p>{html.escape(record['prompt'])}</p>
<video id="video" src="original.mp4" controls preload="metadata"></video>
<h2>Accepted phase clips</h2>
<ul>{clip_links}</ul>
<h2>Phase detections</h2>
<table>
<thead><tr><th>#</th><th>frame</th><th>time(s)</th><th>predicted</th>
<th>conf</th><th>accepted</th><th>action</th><th>reason</th><th>tokens</th></tr></thead>
<tbody>{''.join(event_rows)}</tbody>
</table>
<p><a href="audit.json">audit.json</a></p>
<script>
const video = document.getElementById("video");
function seek(seconds) {{
  video.currentTime = seconds;
  video.play();
}}
</script>
</body>
</html>
"""


def render_index_html(records: Sequence[dict]) -> str:
    rows = []
    for record in records:
        rel = record["review_html"]
        rows.append(
            "<tr>"
            f"<td>{html.escape(record['task_name'])}</td>"
            f"<td><a href=\"{html.escape(rel)}\">{record['video_id']}</a></td>"
            f"<td>{record['seed']}</td>"
            f"<td>{record['success']}</td>"
            f"<td>{record['phase_detection_count']}</td>"
            f"<td>{record['transition_count']}</td>"
            f"<td>{record['final_token_count']}</td>"
            f"<td>{html.escape(record['prompt'])}</td>"
            "</tr>"
        )
    return f"""<!doctype html>
<html lang="zh">
<head>
<meta charset="utf-8">
<title>PhyWAM phase review</title>
<style>
body {{ font-family: sans-serif; margin: 24px; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #bbb; padding: 5px; text-align: left; }}
th {{ position: sticky; top: 0; background: white; }}
</style>
</head>
<body>
<h1>PhyWAM phase manual review</h1>
<p>Dataset annotations are automatic canonical references, not frame-level
ground truth for the online evaluation episodes.</p>
<p><a href="manifest.json">manifest.json</a> |
<a href="manifest.csv">manifest.csv</a> |
<a href="reference_labeled_examples/index.html">reference labeled examples</a></p>
<table>
<thead><tr><th>task</th><th>episode</th><th>seed</th><th>success</th>
<th>detections</th><th>transitions</th><th>tokens</th><th>prompt</th></tr></thead>
<tbody>{''.join(rows)}</tbody>
</table>
</body>
</html>
"""


def select_reference_episodes(episodes: Sequence[dict]) -> List[dict]:
    selected = []
    for split_hint in ("clean", "aug"):
        match = next(
            (
                episode
                for episode in episodes
                if split_hint in episode["split"].lower()
            ),
            None,
        )
        if match:
            selected.append(match)
    return selected


def build_reference_examples(
    by_task: Dict[str, List[dict]],
    dataset_root: Path,
    output_root: Path,
    ffmpeg: str,
) -> List[dict]:
    manifest = []
    reference_root = output_root / "reference_labeled_examples"
    page_links = []
    for task in TASKS:
        for episode in select_reference_episodes(by_task.get(task, [])):
            source = dataset_root / episode["video_name"]
            if not source.is_file():
                raise FileNotFoundError(source)
            split_tag = "clean" if "clean" in episode["split"] else "aug"
            episode_dir = (
                reference_root
                / task
                / f"{split_tag}_episode_{episode['episode_index']:06d}"
            )
            episode_dir.mkdir(parents=True, exist_ok=True)
            relative_link(source, episode_dir / "source.mp4")
            clips = []
            for segment in episode["segments"]:
                clip_name = (
                    f"{segment['segment_index']:02d}_{safe_name(segment['phase'])}_"
                    f"{segment['start_frame']:04d}_{segment['end_frame']:04d}.mp4"
                )
                clip_path = episode_dir / clip_name
                cut_video(
                    ffmpeg,
                    source,
                    clip_path,
                    segment["start_frame"],
                    segment["end_frame"],
                )
                clips.append({**segment, "clip": clip_name})
            record = {
                **episode,
                "annotation_role": "automatic_canonical_reference",
                "is_ground_truth_for_online_evaluation": False,
                "source_video": str(source),
                "clips": clips,
            }
            write_json(episode_dir / "annotation.json", record)
            clip_list = "".join(
                f'<li><a href="{html.escape(x["clip"])}">'
                f'{html.escape(x["phase"])} [{x["start_frame"]}, {x["end_frame"]})'
                "</a></li>"
                for x in clips
            )
            (episode_dir / "review.html").write_text(
                f"""<!doctype html><html lang="zh"><meta charset="utf-8">
<title>{html.escape(task)} reference</title>
<body><h1>{html.escape(task)} / {split_tag} reference</h1>
<p>{html.escape(episode['task_text'])}</p>
<video src="source.mp4" controls style="width:min(100%,900px)"></video>
<ul>{clip_list}</ul><p><a href="annotation.json">annotation.json</a></p>
</body></html>""",
                encoding="utf-8",
            )
            relative_page = str(
                (episode_dir / "review.html").relative_to(reference_root)
            )
            page_links.append(
                f'<li><a href="{html.escape(relative_page)}">'
                f"{html.escape(task)} / {split_tag}</a></li>"
            )
            manifest.append(
                {
                    "task_name": task,
                    "split": episode["split"],
                    "episode_index": episode["episode_index"],
                    "review_html": str(
                        (episode_dir / "review.html").relative_to(output_root)
                    ),
                }
            )
    (reference_root / "index.html").write_text(
        "<!doctype html><html lang=\"zh\"><meta charset=\"utf-8\">"
        "<title>Reference labels</title><body><h1>Reference labeled examples</h1>"
        "<p>These are automatic v39 canonical annotations from training data, "
        "not ground truth for online evaluation episodes.</p><ul>"
        + "".join(page_links)
        + "</ul></body></html>",
        encoding="utf-8",
    )
    write_json(reference_root / "manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a manual review package for PhyWAM phase inference."
    )
    parser.add_argument(
        "--result-root",
        type=Path,
        default=Path(
            "/data/public_data/xzs_data/"
            "phywam_v3_s2_phase_cross_phys_v3.5/inference/"
            "checkpoint_step_16000/phase/client/stseed-10000"
        ),
    )
    parser.add_argument(
        "--label-json",
        type=Path,
        default=Path(
            "/data/worldmodel_xzs/phywam_v3/mope-jepa/datasets/"
            "robotwin_s2_8tasks_c50a500_qwen/event_labels_v39_task_canonical/"
            "robotwin_s2_8tasks_c50_a500_event_segments_"
            "task_canonical_mope_jepa.json"
        ),
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path(
            "/data/public_data/xzs_data/lingbotva-post-training-dataset_s2"
        ),
    )
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--ffprobe", default="")
    parser.add_argument("--boundary-radius-frames", type=int, default=15)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_dir or args.result_root / "phase_review"
    if output_root.exists() and not args.overwrite:
        raise FileExistsError(
            f"{output_root} already exists; pass --overwrite to refresh it"
        )
    output_root.mkdir(parents=True, exist_ok=True)

    ffmpeg = find_binary(args.ffmpeg, "ffmpeg")
    ffprobe = find_binary(args.ffprobe, "ffprobe")
    label_data = json.loads(args.label_json.read_text(encoding="utf-8"))
    by_task = group_label_episodes(label_data["samples"])
    references = {
        task: canonical_reference(task, by_task.get(task, []))
        for task in TASKS
    }

    write_json(
        output_root / "canonical_reference.json",
        {
            "source_label_json": str(args.label_json),
            "annotation_version": label_data.get("annotation_version"),
            "segment_mode": label_data.get("segment_mode"),
            "warning": (
                "These are automatic canonical annotations from training and "
                "augmentation trajectories. They are not frame-level ground "
                "truth for the online evaluation episodes."
            ),
            "tasks": references,
        },
    )
    reference_manifest = build_reference_examples(
        by_task,
        args.dataset_root,
        output_root,
        ffmpeg,
    )

    manifest = []
    csv_rows = []
    for task in TASKS:
        timing_path = args.result_root / "metrics" / task / "timing.jsonl"
        if not timing_path.is_file():
            continue
        rows = read_jsonl(timing_path)
        prompt_index: Dict[str, List[dict]] = defaultdict(list)
        for episode in by_task.get(task, []):
            prompt_index[episode["task_text"].strip()].append(episode)

        for row in rows:
            video_id = int(row["test_num"]) - 1
            videos = sorted(
                (args.result_root / "visualization" / task).glob(
                    f"{video_id}_*.mp4"
                )
            )
            if len(videos) != 1:
                raise RuntimeError(
                    f"expected one video for {task} episode {video_id}, got {videos}"
                )
            source_video = videos[0]
            video = probe_video(ffprobe, source_video)
            phase_events = build_phase_events(
                row.get("phase_trace", []),
                video["fps"],
                video["frame_count"],
            )
            accepted_segments = build_accepted_segments(
                phase_events,
                video["frame_count"],
            )
            episode_dir = output_root / "episodes" / task / f"{video_id:03d}"
            episode_dir.mkdir(parents=True, exist_ok=True)
            relative_link(source_video, episode_dir / "original.mp4")

            for segment in accepted_segments:
                clip_name = (
                    "segments/"
                    f"{segment['segment_index']:02d}_"
                    f"{safe_name(segment['phase'])}_"
                    f"{segment['start_frame']:04d}_{segment['end_frame']:04d}.mp4"
                )
                cut_video(
                    ffmpeg,
                    source_video,
                    episode_dir / clip_name,
                    segment["start_frame"],
                    segment["end_frame"],
                )
                segment["clip"] = clip_name

            boundary_windows = []
            for event in phase_events:
                if event["update_action"] != "transition":
                    continue
                start = max(
                    0,
                    event["video_frame"] - args.boundary_radius_frames,
                )
                end = min(
                    video["frame_count"],
                    event["video_frame"] + args.boundary_radius_frames + 1,
                )
                clip_name = (
                    "boundaries/"
                    f"transition_{len(boundary_windows):02d}_"
                    f"{safe_name(event['accepted_current_phase'])}_"
                    f"f{event['video_frame']:04d}.mp4"
                )
                cut_video(
                    ffmpeg,
                    source_video,
                    episode_dir / clip_name,
                    start,
                    end,
                )
                boundary_windows.append(
                    {
                        "video_frame": event["video_frame"],
                        "video_time_seconds": event["video_time_seconds"],
                        "accepted_phase": event["accepted_current_phase"],
                        "predicted_phase": event["predicted_phase"],
                        "confidence": event["confidence"],
                        "start_frame": start,
                        "end_frame": end,
                        "clip": clip_name,
                    }
                )

            exact_matches = prompt_index.get(row.get("prompt", "").strip(), [])
            exact_match_refs = [
                {
                    "split": match["split"],
                    "episode_index": match["episode_index"],
                    "episode_length": match["episode_length"],
                    "task_text": match["task_text"],
                    "source_video": str(args.dataset_root / match["video_name"]),
                    "segments": match["segments"],
                }
                for match in exact_matches[:10]
            ]
            audit = {
                "schema_version": "phywam_phase_review_v1",
                "task_name": task,
                "video_id": video_id,
                "test_num": int(row["test_num"]),
                "seed": int(row["seed"]),
                "prompt": row.get("prompt", ""),
                "success": bool(row.get("success", False)),
                "source_video": str(source_video),
                "video": video,
                "alignment": {
                    "method": (
                        "phase_trace real_update order aligned to saved observation "
                        "frames accumulated from input_obs_frames"
                    ),
                    "video_frame_to_estimated_env_step_multiplier": 4,
                    "confidence": "exact_for_current_evaluation_pipeline",
                },
                "annotation_provenance": {
                    "label_json": str(args.label_json),
                    "annotation_version": label_data.get("annotation_version"),
                    "segment_mode": label_data.get("segment_mode"),
                    "is_ground_truth_for_this_episode": False,
                    "reason": (
                        "The label JSON annotates separate training/augmentation "
                        "episodes and has no evaluation-seed trajectory mapping."
                    ),
                },
                "canonical_reference": references[task],
                "exact_prompt_reference_match_count": len(exact_matches),
                "exact_prompt_reference_matches": exact_match_refs,
                "phase_events": phase_events,
                "accepted_phase_segments": accepted_segments,
                "transition_boundary_windows": boundary_windows,
                "manual_review": manual_review_template(),
            }
            write_json(episode_dir / "audit.json", audit)
            (episode_dir / "review.html").write_text(
                render_episode_html(audit),
                encoding="utf-8",
            )

            detections = [
                event
                for event in phase_events
                if event["request_kind"] == "phase_detection"
            ]
            transitions = [
                event
                for event in detections
                if event["update_action"] == "transition"
            ]
            final_token_count = (
                int(phase_events[-1]["tokens_after"]) if phase_events else 0
            )
            summary = {
                "task_name": task,
                "video_id": video_id,
                "test_num": int(row["test_num"]),
                "seed": int(row["seed"]),
                "success": bool(row.get("success", False)),
                "prompt": row.get("prompt", ""),
                "phase_detection_count": len(detections),
                "transition_count": len(transitions),
                "final_token_count": final_token_count,
                "exact_prompt_reference_match_count": len(exact_matches),
                "review_html": str(
                    (episode_dir / "review.html").relative_to(output_root)
                ),
                "audit_json": str(
                    (episode_dir / "audit.json").relative_to(output_root)
                ),
            }
            manifest.append(summary)
            csv_rows.append(summary)

    manifest.sort(key=lambda x: (x["task_name"], x["video_id"]))
    write_json(
        output_root / "manifest.json",
        {
            "schema_version": "phywam_phase_review_manifest_v1",
            "result_root": str(args.result_root),
            "source_label_json": str(args.label_json),
            "episode_count": len(manifest),
            "reference_example_count": len(reference_manifest),
            "episodes": manifest,
        },
    )
    with (output_root / "manifest.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        fields = [
            "task_name",
            "video_id",
            "test_num",
            "seed",
            "success",
            "phase_detection_count",
            "transition_count",
            "final_token_count",
            "exact_prompt_reference_match_count",
            "prompt",
            "review_html",
            "audit_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(csv_rows)
    (output_root / "index.html").write_text(
        render_index_html(manifest),
        encoding="utf-8",
    )
    (output_root / "README.md").write_text(
        """# PhyWAM phase review

Open `index.html` to review each completed online evaluation episode.

- `episodes/<task>/<id>/audit.json`: prediction, accepted phase, token update,
  video frame, time, estimated environment step, and manual review fields.
- `episodes/<task>/<id>/segments/`: online video split at accepted phase
  transition boundaries.
- `episodes/<task>/<id>/boundaries/`: one-second windows around transitions.
- `reference_labeled_examples/`: v39 canonical annotated training examples,
  split at their annotation boundaries.
- `canonical_reference.json`: task sequences and aggregate boundary ratios.

The v39 labels are automatic canonical annotations of separate
training/augmentation episodes. They are useful references, but are not
frame-level ground truth for the online evaluation trajectories.
""",
        encoding="utf-8",
    )
    print(f"[done] episodes={len(manifest)}")
    print(f"[done] reference_examples={len(reference_manifest)}")
    print(f"[done] output={output_root}")


if __name__ == "__main__":
    main()
