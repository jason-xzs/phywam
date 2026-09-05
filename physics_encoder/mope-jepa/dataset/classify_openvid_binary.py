import argparse, json
from pathlib import Path
import torch
from tqdm import tqdm
from transformers import AutoModelForImageTextToText, AutoProcessor
from qwen_vl_utils import process_vision_info

PHYSICS_LABELS = [
    "collision","rigid_body_motion","elastic_motion","liquid_motion","gas_motion",
    "deformation","melting","solidification","vaporization","liquefaction",
    "explosion","combustion","reflection","refraction","scattering",
    "interference_diffraction","unnatural_light_sources",
]

PROMPT = """You are a strict physics-video classifier. Watch the video and decide whether it CLEARLY and PROMINENTLY shows one of these 17 physics phenomena as a distinct physical process:
collision, rigid_body_motion, elastic_motion, liquid_motion, gas_motion, deformation, melting, solidification, vaporization, liquefaction, explosion, combustion, reflection, refraction, scattering, interference_diffraction, unnatural_light_sources.

Rules:
- Classify as "physics" ONLY if one of the 17 phenomena is clearly and prominently shown as a distinct physical process (e.g. objects colliding, fluid clearly flowing/splashing, fire burning, explosion, ice melting).
- Ordinary scenes (landscapes, scenery, static nature, city views, everyday footage) with no clear physical process → "general".
- When in doubt, choose "general". Be strict: require an obvious, prominent physical phenomenon to call it "physics".

Return ONLY valid JSON:
{"category": "physics" or "general", "matched_label": "one of the 17 labels or none", "reason": "short explanation"}"""

def safe_parse_json(text: str) -> dict:
    s = text.strip()
    if s.startswith("```"):
        s = s.split("```", 1)[-1].rsplit("```", 1)[0].strip()
        if s.startswith("json"):
            s = s[4:].strip()
    return json.loads(s)

def load_model(model_name_or_path: str):
    processor = AutoProcessor.from_pretrained(model_name_or_path, trust_remote_code=True)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForImageTextToText.from_pretrained(
        model_name_or_path, dtype=dtype, trust_remote_code=True)
    model.eval()
    if torch.cuda.is_available():
        model = model.to(torch.device("cuda"))
    return processor, model

def ask_model(video_path: Path, processor, model, max_new_tokens: int) -> str:
    messages = [{"role":"user","content":[
        {"type":"video","video":f"file://{video_path.resolve()}",
         "max_pixels":360*420,"fps":2.0},
        {"type":"text","text":PROMPT},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs,
                       padding=True, return_tensors="pt")
    inputs = {k:(v.to(model.device) if hasattr(v,"to") else v) for k,v in inputs.items()}
    with torch.no_grad():
        gen = model.generate(**inputs, max_new_tokens=max_new_tokens)
    trimmed = [o[len(i):] for i,o in zip(inputs["input_ids"], gen)]
    return processor.batch_decode(trimmed, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)[0]

def chunk_by_rank(items, world_size, rank):
    return [x for i,x in enumerate(items) if i % world_size == rank]

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_list", default="datasets/openvid_clean.txt")
    p.add_argument("--output_json", default="datasets/openvid_binary.json")
    p.add_argument("--model_name_or_path", default="Qwen/Qwen3-VL-8B-Instruct")
    p.add_argument("--rank", type=int, default=0)
    p.add_argument("--world_size", type=int, default=1)
    p.add_argument("--max_new_tokens", type=int, default=256)
    return p.parse_args()

def main():
    args = parse_args()
    paths = [l.strip() for l in open(args.input_list) if l.strip()]
    shard = chunk_by_rank(paths, args.world_size, args.rank)
    processor, model = load_model(args.model_name_or_path)
    out = Path(args.output_json); out.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for vp in tqdm(shard, desc=f"rank{args.rank}", dynamic_ncols=True):
        vpath = Path(vp)
        if not vpath.is_file():
            results.append({"video": vp, "rank": args.rank, "error": "not found"})
            continue
        try:
            raw = ask_model(vpath, processor, model, args.max_new_tokens)
            parsed = safe_parse_json(raw)
            cat = parsed.get("category", "").strip().lower()
            if cat not in ("physics", "general"):
                cat = "general"  # 解析异常默认general(保守保留)
            results.append({
                "video": vp, "rank": args.rank,
                "category": cat,
                "matched_label": parsed.get("matched_label", ""),
                "reason": parsed.get("reason", ""),
            })
        except Exception as e:
            # 解析失败默认general(不误杀), 记录error
            results.append({"video": vp, "rank": args.rank,
                            "category": "general", "error": str(e)})
    with out.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[rank{args.rank}] done, {len(results)} results -> {out}")

if __name__ == "__main__":
    main()
