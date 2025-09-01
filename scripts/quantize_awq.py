import argparse, sys, os
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--w_bits", type=int, default=4)
    args = p.parse_args()
    try:
        from awq import AutoAWQForCausalLM
        from transformers import AutoTokenizer
    except Exception:
        print("pip install -q autoawq")
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
