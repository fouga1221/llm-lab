import argparse, json, time, statistics
import torch
from src.utils import read_yaml, build_model, env_defaults
from src.prompt_templates import render

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--engine", required=True)
    p.add_argument("--decode", required=True)
    p.add_argument("--jsonl", default="/content/llm-lab/data/prompts.jsonl") # {"prompt": "..."}
    p.add_argument("--template", default="chatml")
    args = p.parse_args()

    env_defaults()
    model_cfg = read_yaml(args.model)
    engine_cfg = read_yaml(args.engine)
    dec_cfg = read_yaml(args.decode)

    tok, model = build_model(model_cfg, engine_cfg)

    prompts = []
    with open(args.jsonl) as f:
        for line in f:
            j = json.loads(line)
            prompts.append(j["prompt"])
    assert prompts, "no prompts"

    times, tps = [], []
    for s in prompts:
        text = render(args.template, user=s)
        inputs = tok(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        gen_kwargs = dict(
            max_new_tokens=dec_cfg.get("max_new_tokens", 256),
            do_sample=dec_cfg.get("do_sample", False),
            temperature=dec_cfg.get("temperature", 0.0),
            top_p=dec_cfg.get("top_p", 1.0),
            top_k=dec_cfg.get("top_k", 0),
            repetition_penalty=dec_cfg.get("repetition_penalty", 1.0),
        )
        t0 = time.time()
        with torch.inference_mode():
            out = model.generate(**inputs, **gen_kwargs)
        dt = time.time() - t0

        new_len = out.shape[-1] - inputs["input_ids"].shape[-1]
        times.append(dt)
        tps.append(new_len / dt if dt > 0 else 0)

    print({
        "n": len(prompts),
        "gen_time_sec_avg": statistics.mean(times),
        "tok_per_sec_avg": statistics.mean(tps),
        "tok_per_sec_p90": statistics.quantiles(tps, n=10)[-1] if len(tps) >= 10 else max(tps),
    })

if __name__ == "__main__":
    main()
