import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor

from qwen_vl_utils import process_vision_info


PHYSICS_LABELS = [
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
]


def build_prompt(sample: dict) -> str:
    caption = sample.get("captions", "")
    original_label = sample.get("label", "")
    pa = sample.get("physical_annotation", {})
    phys_law = pa.get("phys_law", "")
    q0 = pa.get("q0", "")
    q1 = pa.get("q1", "")
    q2 = pa.get("q2", "")
    q3 = pa.get("q3", "")
    q4 = pa.get("q4", "")

    labels_text = "\n".join([f"{i + 1}. {x}" for i, x in enumerate(PHYSICS_LABELS)])
    return f"""
You are a strict physics-video label re-scoring system.

Given ONE video and its metadata, output a normalized probability distribution over 17 physics labels.
Do NOT invent new labels.

17 labels:
{labels_text}

Rules:
- Output probability for every label in [0, 1].
- Sum of all 17 probabilities MUST equal 1.0.
- If original label is inconsistent, set mismatch_with_original_label=true.

Input metadata:
original_label: {original_label}
caption: {caption}
phys_law: {phys_law}
q0_dynamics: {q0}
q1_thermodynamics: {q1}
q2_optics: {q2}
q3_camera_motion: {q3}
q4_object_state: {q4}

Return ONLY valid JSON:
{{
  "dominant_label": "string",
  "mismatch_with_original_label": true,
  "reason": "short explanation",
  "label_distribution": {{
    "collision": 0.0,
    "rigid_body_motion": 0.0,
    "elastic_motion": 0.0,
    "liquid_motion": 0.0,
    "gas_motion": 0.0,
    "deformation": 0.0,
    "melting": 0.0,
    "solidification": 0.0,
    "vaporization": 0.0,
    "liquefaction": 0.0,
    "explosion": 0.0,
    "combustion": 0.0,
    "reflection": 0.0,
    "refraction": 0.0,
    "scattering": 0.0,
    "interference_diffraction": 0.0,
    "unnatural_light_sources": 0.0
  }}
}}
""".strip()


def safe_parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[-1]
        s = s.rsplit("```", 1)[0].strip()
        if s.startswith("json"):
            s = s[4:].strip()
    return json.loads(s)


def normalize_distribution(d: dict) -> dict:
    values = [max(float(d.get(k, 0.0)), 0.0) for k in PHYSICS_LABELS]
    total = sum(values)
    if total <= 0:
        values = [0.0] * len(PHYSICS_LABELS)
        values[0] = 1.0
        total = 1.0
    values = [v / total for v in values]
    return {k: round(v, 6) for k, v in zip(PHYSICS_LABELS, values)}


def load_model(model_name_or_path: str):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    # 不用 device_map="auto"，避免必须安装 accelerate；每进程一张卡时用 .to(cuda) 即可。
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path,
        dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    return processor, model


def ask_model(video_path: Path, prompt: str, processor, model, max_new_tokens: int) -> str:
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "video",
                    "video": f"file://{video_path.resolve()}",
                    "max_pixels": 360 * 420,
                    "fps": 1.0,
                },
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )
    inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)

    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
    ]
    return processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]


def score_one_sample(sample: dict, video_path: Path, processor, model, max_new_tokens: int) -> Tuple[dict, str]:
    prompt = build_prompt(sample)
    raw = ask_model(video_path, prompt, processor, model, max_new_tokens=max_new_tokens)
    parsed = safe_parse_json(raw)
    parsed["label_distribution"] = normalize_distribution(parsed.get("label_distribution", {}))
    if parsed.get("dominant_label") not in PHYSICS_LABELS:
        parsed["dominant_label"] = max(parsed["label_distribution"], key=parsed["label_distribution"].get)
    return parsed, raw


def build_video_index(video_root: Path) -> Dict[str, Path]:
    index: Dict[str, Path] = {}
    for path in video_root.rglob("*.mp4"):
        index[path.name] = path
    return index


def chunk_by_rank(items: List[dict], world_size: int, rank: int) -> List[dict]:
    return [item for i, item in enumerate(items) if i % world_size == rank]


def parse_args():
    parser = argparse.ArgumentParser(description="Batch score physics labels with Qwen3-VL.")
    parser.add_argument(
        "--input_json",
        default="datasets/wisa_7k.json",
        help="Input dataset JSON path.",
    )
    parser.add_argument(
        "--video_root",
        default="datasets",
        help="Root directory containing mp4 videos (recursive).",
    )
    parser.add_argument(
        "--output_json",
        default="datasets/wisa_7k_qwen3vl8b_scores.json",
        help="Output JSON path.",
    )
    parser.add_argument(
        "--model_name_or_path",
        default="Qwen/Qwen3-VL-8B-Instruct",
        help="HF repo id (e.g. Qwen/Qwen3-VL-8B-Instruct) or local directory path to model files.",
    )
    parser.add_argument("--rank", type=int, default=0, help="Worker rank.")
    parser.add_argument("--world_size", type=int, default=1, help="Total worker count.")
    parser.add_argument("--max_new_tokens", type=int, default=768)
    return parser.parse_args()


def main():
    args = parse_args()
    input_json = Path(args.input_json)
    video_root = Path(args.video_root)
    output_json = Path(args.output_json)

    with input_json.open("r", encoding="utf-8") as f:
        data = json.load(f)

    shard = chunk_by_rank(data, world_size=args.world_size, rank=args.rank)
    video_index = build_video_index(video_root)
    processor, model = load_model(args.model_name_or_path)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for sample in tqdm(shard, desc=f"rank{args.rank}", dynamic_ncols=True):
        video_name = sample.get("video_name", "")
        video_path = video_index.get(video_name)
        if video_path is None:
            results.append(
                {
                    "video_name": video_name,
                    "rank": args.rank,
                    "error": f"video not found under {video_root}",
                }
            )
            continue

        try:
            result, _ = score_one_sample(
                sample=sample,
                video_path=video_path,
                processor=processor,
                model=model,
                max_new_tokens=args.max_new_tokens,
            )
            results.append(
                {
                    "video_name": video_name,
                    "rank": args.rank,
                    "original_label": sample.get("label", ""),
                    "dominant_label": result["dominant_label"],
                    "mismatch_with_original_label": result.get("mismatch_with_original_label", False),
                    "reason": result.get("reason", ""),
                    "label_distribution": result["label_distribution"],
                }
            )
        except Exception as exc:
            results.append(
                {
                    "video_name": video_name,
                    "rank": args.rank,
                    "error": str(exc),
                }
            )

    with output_json.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(
        f"[rank {args.rank}] finished {len(results)} samples -> {output_json}"
    )


if __name__ == "__main__":
    main()