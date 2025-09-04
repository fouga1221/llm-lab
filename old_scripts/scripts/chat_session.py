import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import argparse, os, json, time, csv
from datetime import datetime
import torch
from src.utils import read_yaml, build_model, env_defaults
from src.agent import load_agent, build_system_from_agent
from transformers import TextStreamer

def to_chatml(system:str, history:list, user:str)->str:
    # history: [{"role":"user"/"assistant","content":str}, ...]
    parts = [f"<|im_start|>system\n{system}<|im_end|>"]
    for turn in history:
        parts.append(f"<|im_start|>{turn['role']}\n{turn['content']}<|im_end|>")
    parts.append(f"<|im_start|>user\n{user}<|im_end|>")
    parts.append(f"<|im_start|>assistant\n")
    return "\n".join(parts)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)     # YAML
    ap.add_argument("--engine", required=True)    # YAML
    ap.add_argument("--decode", required=True)    # YAML
    ap.add_argument("--agent", required=True)     # YAML (上で作ったやつ)
    ap.add_argument("--scenario", default=None)   # JSONL（無ければ手入力）
    ap.add_argument("--max_turns", type=int, default=20)
    ap.add_argument("--session_dir", default="/content/llm-lab/results/sessions")
    args = ap.parse_args()

    env_defaults()
    model_cfg = read_yaml(args.model)
    engine_cfg = read_yaml(args.engine)
    dec_cfg = read_yaml(args.decode)

    tok, model = build_model(model_cfg, engine_cfg)

    agent = load_agent(args.agent)
    system = build_system_from_agent(agent)

    # ログ準備
    os.makedirs(args.session_dir, exist_ok=True)
    sid = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    out_jsonl = os.path.join(args.session_dir, f"session-{sid}.jsonl")
    out_csv   = os.path.join(args.session_dir, f"session-{sid}.csv")

    # プロンプト列をロード or 対話モード
    inputs = []
    if args.scenario:
        with open(args.scenario) as f:
            for line in f:
                j = json.loads(line)
                if "user" in j: inputs.append(j["user"])
    else:
        print("[Interactive] 入力をどうぞ（空行で終了）")
        while True:
            s = input("> ").strip()
            if not s: break
            inputs.append(s)

    history = []
    # CSVヘッダ
    with open(out_csv, "w", newline="") as f:
        csv.writer(f).writerow(
            ["t","user","assistant","prompt_len","new_len","latency_sec","tok_per_sec","peak_mem_gb"]
        )

    for t, user in enumerate(inputs[:args.max_turns], 1):
        chatml = to_chatml(system, history, user)
        enc = tok(chatml, return_tensors="pt", truncation=True,
                  max_length=engine_cfg.get("max_seq_len", 4096))
        if torch.cuda.is_available():
            enc = {k: v.to("cuda") for k, v in enc.items()}
            torch.cuda.reset_peak_memory_stats()

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
            out = model.generate(**enc, **gen_kwargs)
        dt = time.time() - t0

        text = tok.decode(out[0], skip_special_tokens=True)
        # 直近のassistant応答のみを抽出（簡易）
        # ChatMLは最後にassistantが来る想定なので末尾差分で取る
        full = text
        # ヒューリスティック: 直近タグを探す
        cut = full.rfind("<|im_start|>assistant")
        answer = full[cut+len("<|im_start|>assistant"):].strip() if cut >= 0 else full

        prompt_len = enc["input_ids"].shape[-1]
        new_len = out.shape[-1] - prompt_len
        tps = (new_len / dt) if dt > 0 else 0.0
        peak = (torch.cuda.max_memory_allocated() / (1024**3)) if torch.cuda.is_available() else 0.0

        history.append({"role":"user","content":user})
        history.append({"role":"assistant","content":answer})

        print(f"\n[Turn {t}]")
        print(f"User: {user}")
        print(f"Assistant: {answer}")
        print(f"(len+{new_len}, {dt:.2f}s, {tps:.1f} tok/s, peak {peak:.2f} GB)")

        with open(out_jsonl, "a") as f:
            f.write(json.dumps({
                "t": t, "user": user, "assistant": answer,
                "prompt_len": int(prompt_len),
                "new_len": int(new_len),
                "latency_sec": dt,
                "tok_per_sec": tps,
                "peak_mem_gb": peak,
            }, ensure_ascii=False) + "\n")

        with open(out_csv, "a", newline="") as f:
            csv.writer(f).writerow([t, user, answer, prompt_len, new_len, dt, tps, peak])

    print("\n[LOG] saved:", out_jsonl, "and", out_csv)

if __name__ == "__main__":
    main()
