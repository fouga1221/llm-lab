"""
Simple one-shot inference helper (thin wrapper around transformers) for quick manual tests.

Usage example:
  python scripts/run_infer.py --model-id Qwen/Qwen2-7B-Instruct --input "宿屋はどこ？"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick one-shot inference")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--input", default=None, help="If omitted, read from stdin")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    args = ap.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    device_map = "auto" if args.device == "auto" else None
    if args.device == "cpu":
        device_map = {"": "cpu"}

    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch_dtype,
        device_map=device_map,
        trust_remote_code=True,
    )

    user_input = args.input if args.input is not None else sys.stdin.read().strip()
    prompt = f"<|user|>\n{user_input}\n<|assistant|>\n"

    inputs = tok(prompt, return_tensors="pt")
    if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

    gen_kwargs = dict(
        max_new_tokens=args.max_new_tokens,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        out = model.generate(**inputs, **gen_kwargs)
    dt = time.perf_counter() - t0

    decoded = tok.decode(out[0], skip_special_tokens=True)
    reply = decoded.split("<|assistant|>")[-1].strip() if "<|assistant|>" in decoded else decoded
    print(reply)
    print(f"\n[latency_ms]={round(dt*1000,2)}", file=sys.stderr)


if __name__ == "__main__":
    main()

