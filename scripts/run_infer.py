"""
Simple one-shot inference helper that supports runtime switch and chat template.

Examples:
  python scripts/run_infer.py --model-id Qwen/Qwen2-7B-Instruct --input "宿屋はどこ？"
  python scripts/run_infer.py --runtime transformers --system app/prompts/system_prompt.md --input "こんにちは"
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from llm_lab_core.utils.chat_template import apply_chat_template


def main() -> None:
    ap = argparse.ArgumentParser(description="Quick one-shot inference")
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--revision", default=None)
    ap.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    ap.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    ap.add_argument("--runtime", choices=["transformers","vllm","exllamav2","llamacpp"], default="transformers")
    ap.add_argument("--system", default=None, help="Optional system prompt text")
    ap.add_argument("--input", default=None, help="If omitted, read from stdin")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    ap.add_argument("--temperature", type=float, default=0.4)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--json-only", action="store_true", help="Ask the model to return JSON only")
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
    messages = []
    if args.system:
        messages.append({"role": "system", "content": Path(args.system).read_text(encoding="utf-8")})
    messages.append({"role": "user", "content": user_input + ("\nJSONのみで出力してください。" if args.json_only else "")})
    prompt = apply_chat_template(tok, messages, add_generation_prompt=True)

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
