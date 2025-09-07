import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize a HF model to AWQ 4-bit using AutoAWQ.")
    p.add_argument("--model-id", required=True, help="Source HF model id or local path")
    p.add_argument("--output-dir", required=True, help="Directory to save quantized model")
    p.add_argument("--w-bits", type=int, default=4, help="Weight bits (default: 4)")
    args = p.parse_args()
    try:
        from awq import AutoAWQForCausalLM  # type: ignore
        from transformers import AutoTokenizer
    except Exception:
        print("autoawq not installed. pip install autoawq", file=sys.stderr)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    model = AutoAWQForCausalLM.from_pretrained(args.model_id, device_map="auto", trust_remote_code=True)
    model.quantize(tok, w_bits=args.w_bits)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_quantized(args.output_dir)
    tok.save_pretrained(args.output_dir)
    print("[DONE] AWQ saved to", args.output_dir)


if __name__ == "__main__":
    main()

