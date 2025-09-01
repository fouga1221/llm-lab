import argparse, os, csv
import torch
from datetime import datetime
from transformers import TextStreamer
from src.utils import read_yaml, build_model, env_defaults, to_dtype
from src.prompt_templates import render

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)     # yaml
    p.add_argument("--engine", required=True)    # yaml
    p.add_argument("--decode", required=True)    # yaml
    p.add_argument("--prompt", default="ポーションを2つ買いたい。足りる？")
    p.add_argument("--template", default="chatml")
    p.add_argument("--log_csv", default="/content/llm-lab/results/runs.csv")
    args = p.parse_args()

    env_defaults()
    model_cfg = read_yaml(args.model)
    engine_cfg = read_yaml(args.engine)
    dec_cfg = read_yaml(args.decode)

    tok, model = build_model(model_cfg, engine_cfg)
    prompt = render(args.template, user=args.prompt)

    inputs = tok(prompt, return_tensors="pt")
    if torch.cuda.is_available():
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        torch.cuda.reset_peak_memory_stats()

    gen_kwargs = dict(
        max_new_tokens=dec_cfg.get("max_new_tokens", 256),
        do_sample=dec_cfg.get("do_sample", False),
        temperature=dec_cfg.get("temperature", 0.0),
        top_p=dec_cfg.get("top_p", 1.0),
        top_k=dec_cfg.get("top_k", 0),
        repetition_penalty=dec_cfg.get("repetition_penalty", 1.0),
    )

    streamer = TextStreamer(tok, skip_prompt=True, skip_special_tokens=True)
    with torch.inference_mode():
        out = model.generate(**inputs, streamer=streamer, **gen_kwargs)

    text = tok.decode(out[0], skip_special_tokens=True)
    # 粗いメトリクス
    prompt_len = inputs["input_ids"].shape[-1]
    total_len = out.shape[-1]
    new_len = total_len - prompt_len
    peak_mem_gb = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else None

    os.makedirs(os.path.dirname(args.log_csv), exist_ok=True)
    exists = os.path.exists(args.log_csv)
    row = [
        datetime.utcnow().isoformat(),
        model_cfg["model_id"],
        model_cfg["model_kwargs"].get("quantization") or "fp16/bf16",
        str(model_cfg["model_kwargs"].get("torch_dtype")),
        prompt_len,
        new_len,
        round(peak_mem_gb, 3) if peak_mem_gb else "",
        torch.__version__,
    ]
    with open(args.log_csv, "a", newline="") as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(["timestamp","model_id","quant","dtype","prompt_len","new_len","peak_mem_gb","torch"])
        w.writerow(row)

    print("\n[LOGGED]", row)
    print("\n=== TEXT ===\n", text)

if __name__ == "__main__":
    main()
