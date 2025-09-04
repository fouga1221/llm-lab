"""
Grid search decoding parameters for a single model.
Reads a YAML grid file (e.g., temperature/top_p/max_new_tokens lists) and a prompt.
"""
import argparse
import csv
import json
import time
from pathlib import Path
from typing import Dict, Any

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Sweep decoding parameters for a model.")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--grid", required=True, help="YAML with lists: temperature, top_p, max_new_tokens")
    ap.add_argument("--out", required=True, help="CSV output path")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", choices=["float16","bfloat16","float32"], default="bfloat16")
    ap.add_argument("--device", choices=["auto","cpu","cuda"], default="auto")
    args = ap.parse_args()

    grid: Dict[str, Any] = yaml.safe_load(Path(args.grid).read_text(encoding="utf-8"))
    temps = grid.get("temperature", [0.6])
    topps = grid.get("top_p", [0.9])
    maxs = grid.get("max_new_tokens", [256])

    dtype_map = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}
    torch_dtype = dtype_map[args.dtype]
    device_map = "auto" if args.device == "auto" else None
    if args.device == "cpu":
        device_map = {"": "cpu"}

    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, revision=args.revision, torch_dtype=torch_dtype, device_map=device_map)

    inputs = tok(args.prompt, return_tensors="pt")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["temperature","top_p","max_new_tokens","latency_ms","tokens_per_sec"])
        for t in temps:
            for p in topps:
                for m in maxs:
                    t0 = time.perf_counter()
                    with torch.no_grad():
                        out = model.generate(**inputs, do_sample=True, temperature=float(t), top_p=float(p), max_new_tokens=int(m))
                    dt = time.perf_counter() - t0
                    decoded = tok.decode(out[0], skip_special_tokens=True)
                    out_text = decoded[len(args.prompt):] if len(decoded) > len(args.prompt) else decoded
                    out_ids = tok(out_text, add_special_tokens=False).input_ids
                    tps = (len(out_ids) / dt) if dt > 0 else 0.0
                    w.writerow([t, p, m, round(dt*1000,2), round(tps,2)])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()

