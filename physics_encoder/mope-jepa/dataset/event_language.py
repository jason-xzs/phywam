from __future__ import annotations

from typing import Any, Dict, List


NO_EVENT_LABEL = "no_event"


EVENT_INSTRUCTION_TEMPLATES = {
    "approach_motion": "Move the robot arms toward the target objects.",
    "contact_onset": "Make controlled contact with the target object.",
    "gripper_close": "Close the gripper on the target object.",
    "gripper_open": "Open the gripper to release contact.",
    "grasp_success": "Securely grasp the target object.",
    "lift_off": "Lift the grasped object away from the surface.",
    "transport_motion": "Move the grasped object toward the task target.",
    "placement_contact": "Bring the object into contact with the placement target.",
    "release": "Release the object at the target location.",
    "object_stable": "Hold still until the manipulated object is stable.",
    "slip_or_failed_grasp": "Recover from an unstable or failed grasp.",
    "other_motion_change": "Continue the manipulation while the motion changes.",
    "actuation_motion": "Apply motion to actuate the target object.",
    "toggle_change": "Complete the toggle action on the target object.",
    "articulated_open": "Open the articulated target object.",
    "oscillation_motion": "Shake the grasped object with repeated motion.",
    "impact_contact": "Strike or press the target object with controlled contact.",
    "inspection_motion": "Move the object or camera for inspection.",
    "alignment_motion": "Align the object with the task target.",
    "reposition_motion": "Reposition the object or gripper for the next action.",
    NO_EVENT_LABEL: "Maintain the current robot state.",
}


def normalize_task_text(task_text: Any) -> str:
    return " ".join(str(task_text or "").strip().split())


def event_instruction(event_label: str, task_text: Any = "") -> str:
    label = str(event_label or NO_EVENT_LABEL)
    phrase = EVENT_INSTRUCTION_TEMPLATES.get(
        label,
        f"Perform the {label.replace('_', ' ')} event.",
    )
    task = normalize_task_text(task_text)
    if not task:
        return phrase
    return f"{phrase} Overall task: {task}"


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _expand_short_interval(start: int, end: int, center: int, length: int, min_frames: int) -> tuple[int, int]:
    if min_frames <= 1 or end - start >= min_frames:
        return start, end
    half_left = min_frames // 2
    half_right = min_frames - half_left
    start = max(0, center - half_left)
    end = min(length, center + half_right)
    if end - start < min_frames:
        start = max(0, end - min_frames)
        end = min(length, start + min_frames)
    return start, end


def _state_at_frame(state_segments: List[dict], frame: int, fallback: str = "unknown") -> str:
    for segment in state_segments:
        start = _safe_int(segment.get("start_frame"), 0)
        end = _safe_int(segment.get("end_frame"), start)
        if start <= frame < end:
            return str(segment.get("state", fallback))
    return fallback


def build_event_segments(
    record: dict,
    min_segment_frames: int = 4,
    max_segment_frames: int = 0,
    include_no_event_when_empty: bool = True,
) -> List[dict]:
    """Derive variable-length event intervals from point events.

    The interval boundaries are midpoints between neighboring event centers, so
    each episode is covered by event-conditioned, non-fixed-length segments.
    """
    length = _safe_int(record.get("episode_length"), 0)
    if length <= 0:
        return []

    task_text = normalize_task_text(record.get("task_text", ""))
    state_segments = list(record.get("state_segments") or [])
    raw_events = [
        event for event in (record.get("events") or [])
        if 0 <= _safe_int(event.get("frame"), -1) < length
    ]
    raw_events = sorted(raw_events, key=lambda item: _safe_int(item.get("frame"), 0))
    if not raw_events:
        if not include_no_event_when_empty:
            return []
        text = event_instruction(NO_EVENT_LABEL, task_text)
        return [{
            "segment_index": 0,
            "event_label": NO_EVENT_LABEL,
            "event": NO_EVENT_LABEL,
            "event_frame": None,
            "start_frame": 0,
            "end_frame": length,
            "duration_frames": length,
            "state": _state_at_frame(state_segments, 0, "unknown"),
            "state_after": _state_at_frame(state_segments, 0, "unknown"),
            "event_score": 1.0,
            "event_text": text,
            "action_text": text,
            "skill": NO_EVENT_LABEL,
            "segment_source": "no_detected_event_full_episode",
        }]

    frames = [_safe_int(event.get("frame"), 0) for event in raw_events]
    segments: List[dict] = []
    for idx, event in enumerate(raw_events):
        center = frames[idx]
        if idx == 0:
            start = 0
        else:
            start = int(round((frames[idx - 1] + center) / 2.0))
        if idx + 1 == len(frames):
            end = length
        else:
            end = int(round((center + frames[idx + 1]) / 2.0))
        start = max(0, min(start, length - 1))
        end = max(start + 1, min(end, length))
        start, end = _expand_short_interval(start, end, center, length, min_segment_frames)

        if max_segment_frames > 0 and end - start > max_segment_frames:
            half_left = max_segment_frames // 2
            half_right = max_segment_frames - half_left
            start = max(0, center - half_left)
            end = min(length, center + half_right)
            if end - start < max_segment_frames:
                start = max(0, end - max_segment_frames)
                end = min(length, start + max_segment_frames)

        label = str(event.get("event", NO_EVENT_LABEL) or NO_EVENT_LABEL)
        text = event_instruction(label, task_text)
        state_after = str(event.get("state_after") or _state_at_frame(state_segments, center, "unknown"))
        segment = {
            "segment_index": len(segments),
            "event_index": idx,
            "event_label": label,
            "event": label,
            "event_frame": center,
            "start_frame": int(start),
            "end_frame": int(end),
            "duration_frames": int(end - start),
            "state": state_after,
            "state_after": state_after,
            "event_score": _safe_float(event.get("qwen_confidence", event.get("score", 1.0)), 1.0),
            "event_text": text,
            "action_text": text,
            "skill": label,
            "segment_source": "midpoint_between_event_frames",
        }
        if "qwen_verification" in event:
            segment["qwen_verification"] = event["qwen_verification"]
        if "event_before_qwen" in event:
            segment["event_before_qwen"] = event["event_before_qwen"]
        segments.append(segment)

    return segments


def build_event_action_config(event_segments: List[dict]) -> List[dict]:
    action_config: List[dict] = []
    for segment in event_segments:
        action_config.append({
            "start_frame": _safe_int(segment.get("start_frame"), 0),
            "end_frame": _safe_int(segment.get("end_frame"), 0),
            "action_text": str(segment.get("action_text") or segment.get("event_text") or ""),
            "skill": str(segment.get("skill") or segment.get("event_label") or ""),
            "event_label": str(segment.get("event_label") or segment.get("event") or NO_EVENT_LABEL),
            "event_frame": segment.get("event_frame"),
            "segment_source": str(segment.get("segment_source", "")),
        })
    return action_config

