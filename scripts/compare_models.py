"""
Compare multiple models on prompts and collect simple metrics.
Outputs a CSV with latency/tokens_per_sec and optional JSON validity.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from typing import List, Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonschema import validate as jsonschema_validate, ValidationError as JsonSchemaError


def read_prompts(path: str) -> List[str]:
    p = Path(path)
    if not p.exists():
        return []
    # If JSONL, load `input` or `prompt` field; else treat as plain text lines
    if p.suffix.lower() in {".jsonl", ".json"}:
        lines = p.read_text(encoding="utf-8").splitlines()
        out = []
        for line in lines:
            try:
                obj = json.loads(line)
                out.append(obj.get("input") or obj.get("prompt") or line)
            except Exception:
                out.append(line)
        return out
    return [l for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


def run_one(model_id: str, prompt: str, max_new_tokens: int, temperature: float, top_p: float,
            revision: Optional[str], dtype: str, device: str,
            schema: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[dtype]
    device_map = "auto" if device == "auto" else None
    if device == "cpu":
        device_map = {"": "cpu"}

    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch_dtype, device_map=device_map)

    inputs = tok(prompt, return_tensors="pt")
    if device == "cuda" or (device == "auto" and torch.cuda.is_available()):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(max_new_tokens=max_new_tokens, do_sample=True, temperature=temperature, top_p=top_p)
    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    dt = time.perf_counter() - t0
    decoded = tok.decode(out[0], skip_special_tokens=True)
    out_text = decoded[len(prompt):] if len(decoded) > len(prompt) else decoded
    out_ids = tok(out_text, add_special_tokens=False).input_ids
    tps = (len(out_ids) / dt) if dt > 0 else 0.0

    json_valid = None
    if schema is not None:
        try:
            start = out_text.find('{'); end = out_text.rfind('}')
            if start != -1 and end > start:
                obj = json.loads(out_text[start:end+1])
                jsonschema_validate(instance=obj, schema=schema)
                json_valid = True
        except JsonSchemaError:
            json_valid = False
        except Exception:
            json_valid = False

    return {
        "latency_ms": round(dt*1000, 2),
        "tokens_per_sec": round(tps, 2),
        "output": out_text,
        "json_valid": json_valid,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare multiple models quickly.")
    ap.add_argument("--models", nargs='+', required=True, help="List of HF model ids")
    ap.add_argument("--prompts", required=True, help="Path to prompts (txt lines or JSONL with input/prompt)")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", choices=["float16","bfloat16","float32"], default="bfloat16")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    ap.add_argument("--json-schema", default=None, help="Optional JSON schema path for validation")
    args = ap.parse_args()

    prompts = read_prompts(args.prompts)
    schema = None
    if args.json_schema:
        schema = json.loads(Path(args.json_schema).read_text(encoding="utf-8"))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["model_id","prompt_idx","latency_ms","tokens_per_sec","json_valid"])
        for mid in args.models:
            for i, pr in enumerate(prompts):
                res = run_one(mid, pr, args.max_new_tokens, args.temperature, args.top_p, args.revision, args.dtype, args.device, schema)
                writer.writerow([mid, i, res["latency_ms"], res["tokens_per_sec"], res["json_valid"]])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

