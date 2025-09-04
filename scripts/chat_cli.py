"""
Interactive/one-shot chat CLI for trying different models and decoding params.
Uses Hugging Face transformers directly (no server required).
"""
import argparse
import json
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from jsonschema import validate as jsonschema_validate, ValidationError as JsonSchemaError


def load_system_prompt(path: Optional[str]) -> str:
    if not path:
        return ""
    p = Path(path)
    return p.read_text(encoding="utf-8") if p.exists() else ""


def build_prompt(system_prompt: str, user_input: str) -> str:
    if system_prompt:
        return f"<|system|>\n{system_prompt}\n<|user|>\n{user_input}\n<|assistant|>\n"
    return f"User: {user_input}\nAssistant:"


def extract_json_block(text: str) -> Optional[Dict[str, Any]]:
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end > start:
            return json.loads(text[start : end + 1])
        return None
    except Exception:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a local model via transformers.")
    parser.add_argument("--model-id", required=True, help="Hugging Face model id (e.g., Qwen/Qwen2.5-7B-Instruct)")
    parser.add_argument("--revision", default=None, help="Model revision/tag")
    parser.add_argument("--dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--system", default=None, help="Path to system prompt file")
    parser.add_argument("--json-schema", default=None, help="Path to JSON schema to validate extracted JSON block")
    parser.add_argument("--save", default=None, help="Path to save JSONL logs (e.g., runs/chat.jsonl)")
    parser.add_argument("--input", default=None, help="User input text (if omitted, reads from stdin once)")

    args = parser.parse_args()

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    torch_dtype = dtype_map[args.dtype]

    device_map = "auto" if args.device == "auto" else None
    if args.device == "cpu":
        device_map = {"": "cpu"}

    print(f"Loading model {args.model_id} (revision={args.revision}) ...", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(args.model_id, revision=args.revision)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_id,
        revision=args.revision,
        torch_dtype=torch_dtype,
        device_map=device_map,
    )

    system_prompt = load_system_prompt(args.system)
    user_input = args.input if args.input is not None else sys.stdin.read().strip()
    prompt = build_prompt(system_prompt, user_input)

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
    # Heuristic: reply is everything after the last "Assistant:" if present
    reply = decoded
    if "Assistant:" in decoded:
        reply = decoded.split("Assistant:")[-1].strip()

    # Simple tokens/sec estimation
    out_text = decoded[len(prompt) :] if len(decoded) > len(prompt) else decoded
    out_ids = tok(out_text, add_special_tokens=False).input_ids
    tps = (len(out_ids) / dt) if dt > 0 else 0.0

    json_valid = None
    if args.json_schema:
        schema = json.loads(Path(args.json_schema).read_text(encoding="utf-8"))
        obj = extract_json_block(reply)
        if obj is not None:
            try:
                jsonschema_validate(instance=obj, schema=schema)
                json_valid = True
            except JsonSchemaError:
                json_valid = False

    print(reply)

    if args.save:
        rec = {
            "model_id": args.model_id,
            "revision": args.revision,
            "prompt": user_input,
            "reply": reply,
            "latency_ms": round(dt * 1000, 2),
            "tokens_per_sec": round(tps, 2),
            "json_valid": json_valid,
        }
        p = Path(args.save)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

