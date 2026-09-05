#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor


def parse_args():
    p = argparse.ArgumentParser("Run reachable VideoQA with Qwen3-VL")
    p.add_argument("--input_jsonl", required=True, type=str)
    p.add_argument("--output_jsonl", required=True, type=str)
    p.add_argument("--model_path", required=True, type=str)
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=16)
    return p.parse_args()


def normalize_mcq_answer(text: str) -> str:
    s = (text or "").strip().upper()
    m = re.search(r"\b([A-Z])\b", s)
    if m:
        return m.group(1)
    return s[:1] if s else ""


def map_gt_to_letter(gt: str, options):
    gt_clean = (gt or "").strip().upper()
    if len(gt_clean) == 1 and gt_clean.isalpha():
        return gt_clean
    for i, opt in enumerate(options):
        if (gt or "").strip().lower() == (opt or "").strip().lower():
            return chr(65 + i)
    return ""


def build_prompt(question, options):
    option_block = "\n".join([f"{chr(65+i)}. {opt}" for i, opt in enumerate(options)])
    return (
        "Answer the video question strictly.\n"
        "Return ONLY the final option letter (A/B/C/...) with no extra text.\n\n"
        f"Question: {question}\n"
        f"Options:\n{option_block}\n"
        "Final answer:"
    )


def chunk_by_rank(items, world_size, rank):
    return [x for i, x in enumerate(items) if i % world_size == rank]


def main():
    args = parse_args()
    input_path = Path(args.input_jsonl)
    output_path = Path(args.output_jsonl)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", encoding="utf-8") as f:
        samples = [json.loads(line) for line in f]
    shard = chunk_by_rank(samples, args.world_size, args.rank)

    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True, local_files_only=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path, trust_remote_code=True, local_files_only=True, dtype=dtype
    )
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    rows = []
    for sample in tqdm(shard, desc=f"rank{args.rank}", dynamic_ncols=True):
        video = Path(sample["video_path"])
        if not video.exists():
            continue
        prompt = build_prompt(sample["question"], sample["options"])
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "video", "video": str(video.resolve())},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        try:
            inputs = processor(text=[text], videos=[str(video.resolve())], return_tensors="pt", padding=True, return_metadata=True)
            inputs.pop("video_metadata", None)
            # Work around qwen3-vl + torchvision backend mismatch:
            # expand [T, H, W] into T rows of [1, H, W] so rope-indexing aligns with vision tokens.
            vg = inputs.get("video_grid_thw", None)
            if vg is not None and vg.ndim == 2 and vg.shape[0] == 1 and vg.shape[1] == 3 and int(vg[0, 0]) > 1:
                t, h, w = [int(x) for x in vg[0].tolist()]
                inputs["video_grid_thw"] = torch.tensor([[1, h, w]] * t, dtype=vg.dtype)
            inputs = {k: (v.to(model.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=args.max_new_tokens)
            trim = [o[len(ii):] for ii, o in zip(inputs["input_ids"], out)]
            pred = processor.batch_decode(trim, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].strip()
        except Exception as exc:
            pred = f"[ERROR] {exc}"

        gt_letter = map_gt_to_letter(sample.get("answer", ""), sample.get("options", []))
        pred_letter = normalize_mcq_answer(pred)
        is_correct = (pred_letter == gt_letter) if gt_letter else None
        rows.append(
            {
                **sample,
                "prediction": pred,
                "pred_letter": pred_letter,
                "gt_letter": gt_letter,
                "is_correct": is_correct,
            }
        )

    with output_path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"[DONE] rank={args.rank} wrote {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
