#!/usr/bin/env python3
"""
Run public video QA benchmarks end-to-end:
1) Download from Hugging Face datasets
2) Convert to unified JSONL
3) Run Qwen3-VL inference
4) Aggregate leaderboard
"""

import argparse
import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from tqdm import tqdm


@dataclass
class BenchmarkSpec:
    name: str
    repo_id: str
    split: str = "validation"
    task_type: str = "mcq"


DEFAULT_BENCHMARKS: List[BenchmarkSpec] = [
    BenchmarkSpec("perception_test", "lmms-lab/PerceptionTest", "validation", "mcq"),
    BenchmarkSpec("temp_compass", "lmms-lab/TempCompass", "validation", "mcq"),
    BenchmarkSpec("mvp_bench", "lmms-lab/MVPBench", "validation", "mcq"),
    BenchmarkSpec("cine_pile", "lmms-lab/CinePile", "validation", "mcq"),
    BenchmarkSpec("video_mme", "lmms-lab/Video-MME", "validation", "mcq"),
    BenchmarkSpec("fun_qa", "lmms-lab/FunQA", "validation", "open"),
    BenchmarkSpec("long_video_bench", "lmms-lab/LongVideoBench", "validation", "mcq"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Public VideoQA benchmark runner")
    parser.add_argument("--output_root", type=str, default="benchmarks/public_videoqa")
    parser.add_argument("--benchmarks", nargs="*", default=[x.name for x in DEFAULT_BENCHMARKS])
    parser.add_argument("--max_samples", type=int, default=0, help="0 means full split")
    parser.add_argument("--prepare_only", action="store_true")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-VL-8B-Instruct")
    parser.add_argument("--max_new_tokens", type=int, default=128)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--max_pixels", type=int, default=360 * 420)
    return parser.parse_args()


def _pick_first(sample: dict, candidates: List[str]):
    for key in candidates:
        if key in sample and sample[key] not in (None, ""):
            return sample[key]
    return None


def _to_option_list(raw_options) -> List[str]:
    if raw_options is None:
        return []
    if isinstance(raw_options, list):
        return [str(x).strip() for x in raw_options if str(x).strip()]
    if isinstance(raw_options, dict):
        ordered = []
        for key in sorted(raw_options.keys()):
            val = raw_options[key]
            ordered.append(f"{key}. {val}")
        return ordered
    text = str(raw_options).strip()
    if not text:
        return []
    return [line.strip() for line in re.split(r"\n|;|\|", text) if line.strip()]


def _resolve_video_path(video_value) -> str:
    if video_value is None:
        return ""
    if isinstance(video_value, str):
        return video_value
    if isinstance(video_value, dict):
        for k in ("path", "file", "video_path", "filepath"):
            if k in video_value and video_value[k]:
                return str(video_value[k])
    return ""


def convert_to_unified_jsonl(dataset, spec: BenchmarkSpec, output_jsonl: Path, max_samples: int) -> int:
    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_jsonl.open("w", encoding="utf-8") as f:
        iterator = dataset if max_samples <= 0 else dataset.select(range(min(max_samples, len(dataset))))
        for idx, sample in enumerate(tqdm(iterator, desc=f"convert:{spec.name}", dynamic_ncols=True)):
            question = _pick_first(sample, ["question", "query", "prompt", "instruction"])
            answer = _pick_first(sample, ["answer", "gt_answer", "label", "target"])
            options = _to_option_list(_pick_first(sample, ["choices", "options", "candidates"]))

            video_value = _pick_first(sample, ["video", "video_path", "video_file", "path"])
            video_path = _resolve_video_path(video_value)
            if not question or not video_path:
                continue

            rec = {
                "benchmark": spec.name,
                "sample_id": str(_pick_first(sample, ["id", "qid", "question_id", "uid"]) or idx),
                "video_path": video_path,
                "question": str(question),
                "options": options,
                "answer": "" if answer is None else str(answer),
                "task_type": spec.task_type,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


def build_mcq_prompt(question: str, options: List[str]) -> str:
    option_block = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    return (
        "Answer the video question strictly.\n"
        "Return ONLY the final option letter (A/B/C/...) with no extra text.\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_block}\n"
        "Final answer:"
    )


def build_open_prompt(question: str) -> str:
    return (
        "Answer the video question in one short sentence.\n"
        f"Question: {question}\n"
        "Final answer:"
    )


def normalize_mcq_answer(text: str) -> str:
    s = (text or "").strip().upper()
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        return m.group(1)
    return s[:1] if s else ""


def map_gt_to_letter(gt: str, options: List[str]) -> str:
    gt_clean = (gt or "").strip().upper()
    if len(gt_clean) == 1 and gt_clean.isalpha():
        return gt_clean
    for i, opt in enumerate(options):
        if gt.strip().lower() == opt.strip().lower():
            return chr(65 + i)
    return ""


def run_qwen_predictions(input_jsonl: Path, pred_jsonl: Path, model_name_or_path: str, max_new_tokens: int, fps: float, max_pixels: int) -> Dict[str, float]:
    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor
    from qwen_vl_utils import process_vision_info

    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path, dtype=dtype, trust_remote_code=True
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))

    total = 0
    correct = 0
    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)

    with input_jsonl.open("r", encoding="utf-8") as rf, pred_jsonl.open("w", encoding="utf-8") as wf:
        for line in tqdm(rf, desc=f"infer:{input_jsonl.stem}", dynamic_ncols=True):
            sample = json.loads(line)
            video_path = Path(sample["video_path"])
            if not video_path.exists():
                continue

            if sample["task_type"] == "mcq" and sample["options"]:
                prompt = build_mcq_prompt(sample["question"], sample["options"])
            else:
                prompt = build_open_prompt(sample["question"])

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": f"file://{video_path.resolve()}", "max_pixels": max_pixels, "fps": fps},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]
            text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt"
            )
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            pred_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )[0].strip()

            is_correct = None
            if sample["task_type"] == "mcq" and sample["options"]:
                pred_letter = normalize_mcq_answer(pred_text)
                gt_letter = map_gt_to_letter(sample.get("answer", ""), sample["options"])
                if gt_letter:
                    is_correct = pred_letter == gt_letter
                    total += 1
                    correct += int(is_correct)
            else:
                gt = (sample.get("answer", "") or "").strip().lower()
                if gt:
                    pred_norm = pred_text.strip().lower()
                    is_correct = pred_norm == gt
                    total += 1
                    correct += int(is_correct)

            wf.write(
                json.dumps(
                    {
                        **sample,
                        "prediction": pred_text,
                        "is_correct": is_correct,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    acc = (correct / total) if total > 0 else 0.0
    return {"num_scored": total, "num_correct": correct, "accuracy": acc}


def write_leaderboard(rows: List[Dict], output_root: Path) -> None:
    leaderboard_json = output_root / "leaderboard.json"
    leaderboard_csv = output_root / "leaderboard.csv"
    with leaderboard_json.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    with leaderboard_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["benchmark", "repo_id", "prepared_samples", "num_scored", "num_correct", "accuracy"])
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    selected = {name.strip().lower() for name in args.benchmarks}
    selected_specs = [spec for spec in DEFAULT_BENCHMARKS if spec.name.lower() in selected]
    if not selected_specs:
        raise ValueError(f"No valid benchmarks selected: {args.benchmarks}")

    from datasets import load_dataset

    leaderboard_rows = []
    for spec in selected_specs:
        print(f"\n=== [{spec.name}] loading {spec.repo_id}:{spec.split} ===")
        benchmark_root = output_root / spec.name
        prepared_path = benchmark_root / "prepared.jsonl"
        pred_path = benchmark_root / "predictions.jsonl"

        try:
            ds = load_dataset(spec.repo_id, split=spec.split)
        except Exception as exc:
            print(f"[WARN] skip {spec.name}: cannot load dataset ({exc})")
            leaderboard_rows.append(
                {
                    "benchmark": spec.name,
                    "repo_id": spec.repo_id,
                    "prepared_samples": 0,
                    "num_scored": 0,
                    "num_correct": 0,
                    "accuracy": 0.0,
                }
            )
            continue

        prepared_count = convert_to_unified_jsonl(ds, spec, prepared_path, args.max_samples)
        print(f"[INFO] prepared {prepared_count} samples -> {prepared_path}")

        metrics = {"num_scored": 0, "num_correct": 0, "accuracy": 0.0}
        if not args.prepare_only and prepared_count > 0:
            metrics = run_qwen_predictions(
                input_jsonl=prepared_path,
                pred_jsonl=pred_path,
                model_name_or_path=args.model_name_or_path,
                max_new_tokens=args.max_new_tokens,
                fps=args.fps,
                max_pixels=args.max_pixels,
            )
            print(f"[INFO] scored={metrics['num_scored']} accuracy={metrics['accuracy']:.4f}")

        leaderboard_rows.append(
            {
                "benchmark": spec.name,
                "repo_id": spec.repo_id,
                "prepared_samples": prepared_count,
                "num_scored": metrics["num_scored"],
                "num_correct": metrics["num_correct"],
                "accuracy": round(metrics["accuracy"], 6),
            }
        )

    write_leaderboard(leaderboard_rows, output_root)
    print(f"\n[DONE] leaderboard written to: {output_root / 'leaderboard.json'}")
    print(f"[DONE] leaderboard written to: {output_root / 'leaderboard.csv'}")


if __name__ == "__main__":
    main()
