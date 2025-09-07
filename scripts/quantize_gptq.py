import argparse
import os
import sys


def main() -> None:
    p = argparse.ArgumentParser(description="Quantize a HF model to GPTQ 4-bit using AutoGPTQ.")
    p.add_argument("--model-id", required=True, help="Source HF model id or local path")
    p.add_argument("--output-dir", required=True, help="Directory to save quantized model")
    p.add_argument("--bits", type=int, default=4, help="Quantization bits (default: 4)")
    args = p.parse_args()
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig  # type: ignore
        from transformers import AutoTokenizer
    except Exception:
        print("auto-gptq not installed. pip install auto-gptq", file=sys.stderr)
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    quant_cfg = BaseQuantizeConfig(bits=args.bits, use_triton=False)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.model_id,
        quantize_config=quant_cfg,
        device_map="auto",
        trust_remote_code=True,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_quantized(args.output_dir, use_safetensors=True)
    tok.save_pretrained(args.output_dir)
    print("[DONE] GPTQ saved to", args.output_dir)


if __name__ == "__main__":
    main()

