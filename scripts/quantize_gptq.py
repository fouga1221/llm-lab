import argparse, sys, os
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--bits", type=int, default=4)
    args = p.parse_args()
    try:
        from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
        from transformers import AutoTokenizer
    except Exception:
        print("pip install -q auto-gptq")
        sys.exit(1)

    tok = AutoTokenizer.from_pretrained(args.model_id, use_fast=True, trust_remote_code=True)
    quant_cfg = BaseQuantizeConfig(bits=args.bits, use_triton=False)
    model = AutoGPTQForCausalLM.from_pretrained(args.model_id, quantize_config=quant_cfg, device_map="auto", trust_remote_code=True)
    os.makedirs(args.output_dir, exist_ok=True)
    model.save_quantized(args.output_dir, use_safetensors=True)
    tok.save_pretrained(args.output_dir)
    print("[DONE] GPTQ saved to", args.output_dir)

if __name__ == "__main__":
    main()
