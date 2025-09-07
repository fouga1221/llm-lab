"""
Benchmark inference across (model × runtime × quant × params) and log CSV.

Implements a minimal first pass for `runtime=transformers` with support for:
- quant: float16 (baseline), bnb-int8, bnb-nf4 (requires bitsandbytes installed)
- attn_impl: sdpa|eager (FlashAttn stubbed; falls back if unavailable)

Collects metrics per trial:
- load_ms, first_token_ms, tokens_per_s, peak_vram_alloc_mb, peak_vram_reserved_mb, avg_gpu_util, oom, notes

Notes:
- AWQ/GPTQ/GGUF support relies on pre-quantized weights being provided as model_id; this script
  does not perform on-the-fly quantization.
- vLLM/llama.cpp/ExLlamaV2 are stubbed for future extension; current run will skip those.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import signal
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch
import yaml
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TextIteratorStreamer,
)

from scripts.monitor import NVMLMonitor
from llm_lab_core.runners import get_runner


CSV_HEADER = [
    "model",
    "quant",
    "runtime",
    "attn",
    "kv",
    "kv_dtype",
    "batch_size",
    "load_ms",
    "first_token_ms",
    "tokens_per_s",
    "peak_vram_alloc_mb",
    "peak_vram_reserved_mb",
    "avg_gpu_util",
    "oom",
    "notes",
]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def bytes_to_mb(x: int) -> float:
    return round(x / (1024 * 1024), 2)


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts_by_name(name: str) -> List[str]:
    base = Path("data/prompts")
    # Try YAML list
    yml = base / f"{name}.yaml"
    if yml.exists():
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                # New preset format with system/user/meta
                if "user" in data or "system" in data:
                    sys_txt = data.get("system", "")
                    usr_txt = data.get("user", "")
                    combined = (f"<|system|>\n{sys_txt}\n" if sys_txt else "") + f"<|user|>\n{usr_txt}\n<|assistant|>\n"
                    return [combined]
                if "prompts" in data:
                    return [str(x) for x in data["prompts"]]
            if isinstance(data, list):
                return [str(x) for x in data]
        except Exception:
            pass
    # Try plain text (one per line)
    txt = base / f"{name}.txt"
    if txt.exists():
        return [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    return ["こんにちは。調子はどう？"]


def resolve_quant_config(quant: Optional[str]) -> Tuple[Dict[str, Any], Optional[BitsAndBytesConfig]]:
    model_kwargs: Dict[str, Any] = {}
    bnb_cfg: Optional[BitsAndBytesConfig] = None
    if quant is None or quant in ("float16", "fp16"):
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
    elif quant == "bnb-int8":
        bnb_cfg = BitsAndBytesConfig(load_in_8bit=True)
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs["device_map"] = "auto"
    elif quant in ("bnb-nf4", "bnb-4bit"):
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        )
        model_kwargs["quantization_config"] = bnb_cfg
        model_kwargs["device_map"] = "auto"
    else:
        # AWQ/GPTQ/GGUF etc.: assume pre-quantized weights are pointed by model id.
        model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
        model_kwargs["device_map"] = "auto"
    return model_kwargs, bnb_cfg


def build_inputs(tok, prompts: List[str]):
    return tok(prompts, return_tensors="pt", padding=True)


@dataclass
class TrialMetrics:
    load_ms: float
    first_token_ms: float
    tokens_per_s: float
    peak_vram_alloc_mb: float
    peak_vram_reserved_mb: float
    avg_gpu_util: Optional[float]
    oom: int
    notes: str


def run_transformers(
    model_id: str,
    revision: Optional[str],
    quant: Optional[str],
    attn_impl: str,
    prompts: List[str],
    max_new_tokens: int,
    timeout_s: float,
) -> TrialMetrics:
    # Attention impl hint (requires compatible stack). We try to set via env/kwargs lightly.
    if attn_impl == "flash_attn_2":
        # Best-effort: many env combinations exist, we just hint via config if present.
        try:
            torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
        except Exception:
            pass
    elif attn_impl == "sdpa":
        try:
            torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
        except Exception:
            pass

    # Reset CUDA stats for clean measurement
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model_kwargs, _ = resolve_quant_config(quant)
    t0 = now_ms()
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision)
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, trust_remote_code=True, **model_kwargs)
    load_ms = now_ms() - t0

    # Prepare inputs (batch supported)
    enc: Dict[str, torch.Tensor] = build_inputs(tok, prompts)  # type: ignore[assignment]
    if torch.cuda.is_available():
        enc = {k: v.to(model.device) for k, v in enc.items()}

    # Stream to capture first token time
    streamer = TextIteratorStreamer(tok, skip_special_tokens=True, decode_kwargs={"skip_special_tokens": True})

    # Timeout handling
    class Timeout(Exception):
        pass

    def handler(signum, frame):  # pragma: no cover - signal specific
        raise Timeout()

    first_token_ms = math.nan
    oom = 0
    avg_util: Optional[float] = None

    mon = NVMLMonitor()
    if mon.available:
        mon.start()

    t_start = now_ms()
    try:
        # Start generation in a thread so we can consume stream
        import threading

        out_texts: List[str] = [""] * enc["input_ids"].shape[0]

        def _gen():
            try:
                # Pass kwargs explicitly to avoid TypedDict/union inference issues
                model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc["attention_mask"],
                    max_new_tokens=int(max_new_tokens),
                    do_sample=True,
                    temperature=0.4,
                    top_p=0.9,
                    repetition_penalty=1.1,
                    streamer=streamer,
                )
            except torch.cuda.OutOfMemoryError:  # pragma: no cover - environment dependent
                nonlocal oom
                oom = 1

        th = threading.Thread(target=_gen, daemon=True)

        # Arm timeout
        if timeout_s and timeout_s > 0:
            signal.signal(signal.SIGALRM, handler)  # type: ignore[attr-defined]
            signal.alarm(int(timeout_s))  # type: ignore[attr-defined]

        th.start()
        # Iterate tokens; the first arrival marks first_token_ms
        got_first = False
        t_first = None
        for chunk in streamer:
            if not got_first:
                t_first = now_ms()
                got_first = True
            # naive: accumulate string (only for first sample visible)
            out_texts[0] += chunk

        th.join()
        if timeout_s and timeout_s > 0:
            signal.alarm(0)  # type: ignore[attr-defined]

        if got_first and t_first is not None:
            first_token_ms = t_first - t_start
        else:
            first_token_ms = math.nan
    except Timeout:  # pragma: no cover - environment dependent
        oom = 0
        first_token_ms = math.inf
    finally:
        if mon.available:
            mon.stop()
            avg_util = mon.mean_util()
            mon.close()

    # tokens/s using decoded new tokens length for the first example
    try:
        total_ms = now_ms() - t_start
        decoded = out_texts[0]
        # Rough token count by re-encoding the new text
        new_ids = tok(decoded, add_special_tokens=False).input_ids
        tps = (len(new_ids) / (total_ms / 1000.0)) if total_ms > 0 else 0.0
    except Exception:
        tps = 0.0

    # VRAM peaks
    if torch.cuda.is_available():
        peak_alloc = bytes_to_mb(torch.cuda.max_memory_allocated())
        peak_reserved = bytes_to_mb(torch.cuda.max_memory_reserved())
    else:
        peak_alloc = -1.0
        peak_reserved = -1.0

    return TrialMetrics(
        load_ms=round(load_ms, 2),
        first_token_ms=round(first_token_ms, 2) if math.isfinite(first_token_ms) else -1.0,
        tokens_per_s=round(tps, 2),
        peak_vram_alloc_mb=peak_alloc,
        peak_vram_reserved_mb=peak_reserved,
        avg_gpu_util=round(avg_util, 2) if avg_util is not None else None,
        oom=oom,
        notes="",
    )


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark inference and log CSV per your sweep plan")
    ap.add_argument("--sweep", required=True, help="YAML describing models/quant/runtime/params")
    ap.add_argument("--out", default="results/runs.csv", help="CSV output path")
    ap.add_argument("--log-dir", default="results/logs", help="Logs directory (for errors, etc.)")
    ap.add_argument("--timeout", type=float, default=60.0, help="Timeout seconds per case")
    args = ap.parse_args()

    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))

    models_cfg = sweep.get("models", [])
    model_ids: List[str] = []
    for m in models_cfg:
        if isinstance(m, str):
            # allow plain id
            model_ids.append(m)
        elif isinstance(m, dict) and "hf" in m:
            model_ids.append(m["hf"])
    quants: List[Optional[str]] = sweep.get("quant", [None])
    runtimes: List[str] = sweep.get("runtime", ["transformers"])  # transformers|vllm|llamacpp|exllamav2
    attns: List[str] = sweep.get("attn_impl", ["sdpa"])  # sdpa|flash_attn_2|eager
    kvs: List[str] = sweep.get("kv_cache", ["full"])  # full|paged (stub)
    kv_dtypes: List[str] = sweep.get("kv_dtype", ["fp16"])  # fp16|kv8bit (stub)
    batch_sizes: List[int] = sweep.get("batch_size", [1])
    seed = int(sweep.get("seed", 42))
    max_tokens = int(sweep.get("max_tokens", 256))
    presets = sweep.get("presets", [])
    repeats = int(sweep.get("repeats", 3))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    Path(args.log_dir).mkdir(parents=True, exist_ok=True)

    # Preload prompts per preset
    preset_prompts: Dict[str, List[str]] = {}
    for pr in presets:
        name = pr.get("name")
        names = pr.get("prompts", [])
        merged: List[str] = []
        for n in names:
            merged.extend(load_prompts_by_name(n))
        preset_prompts[name] = merged or ["こんにちは。調子はどう？"]

    set_seed(seed)

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)

        for mid in model_ids:
            for quant in quants:
                for rt in runtimes:
                    if rt != "transformers":
                        # Try minimal runner path; if unavailable, record skip
                        try:
                            runner = get_runner(rt)
                            runner.load(mid)
                            for attn in attns:
                                for kv in kvs:
                                    for kvd in kv_dtypes:
                                        for bs in batch_sizes:
                                            for pr in presets:
                                                pname = pr.get("name")
                                                prompts = preset_prompts.get(pname) or ["こんにちは。調子はどう？"]
                                                batch_prompts = prompts[:bs] or [prompts[0]]
                                                t0 = now_ms()
                                                res = runner.generate_text(batch_prompts, {"max_new_tokens": max_tokens, "temperature": 0.4, "top_p": 0.9, "repetition_penalty": 1.1})
                                                dt = now_ms() - t0
                                                timings = res.get("timings", {})
                                                w.writerow([
                                                    mid,
                                                    quant,
                                                    rt,
                                                    attn,
                                                    kv,
                                                    kvd,
                                                    bs,
                                                    -1,  # load_ms unknown here
                                                    timings.get("first_token_ms", -1),
                                                    timings.get("tokens_per_s", -1),
                                                    -1,
                                                    -1,
                                                    -1,
                                                    0,
                                                    f"{pname}|run",
                                                ])
                        except Exception:
                            w.writerow([mid, quant, rt, "-", "-", "-", 1, -1, -1, -1, -1, -1, -1, 0, f"{rt}_not_installed"])
                        continue
                    for attn in attns:
                        for kv in kvs:
                            for kvd in kv_dtypes:
                                for bs in batch_sizes:
                                    for pr in presets:
                                        pname = pr.get("name")
                                        prompts = preset_prompts.get(pname) or ["こんにちは。調子はどう？"]
                                        # Create a batch by repeating the first prompt if needed
                                        batch_prompts = prompts[:bs]
                                        if len(batch_prompts) < bs:
                                            batch_prompts = (batch_prompts or prompts[:1]) * bs

                                        # 1 warmup + (repeats-1) runs
                                        trials: List[TrialMetrics] = []
                                        trial_notes = ["warmup"] + ["run"] * (max(1, repeats - 1))
                                        for note in trial_notes:
                                            try:
                                                tm = run_transformers(
                                                    model_id=mid,
                                                    revision=None,
                                                    quant=quant,
                                                    attn_impl=attn,
                                                    prompts=batch_prompts,
                                                    max_new_tokens=max_tokens,
                                                    timeout_s=float(args.timeout),
                                                )
                                                tm.notes = f"{pname}|{note}"
                                            except Exception as e:
                                                tm = TrialMetrics(
                                                    load_ms=-1,
                                                    first_token_ms=-1,
                                                    tokens_per_s=0.0,
                                                    peak_vram_alloc_mb=-1.0,
                                                    peak_vram_reserved_mb=-1.0,
                                                    avg_gpu_util=None,
                                                    oom=0,
                                                    notes=f"{pname}|error:{type(e).__name__}",
                                                )
                                            trials.append(tm)
                                            w.writerow([
                                                mid,
                                                quant,
                                                rt,
                                                attn,
                                                kv,
                                                kvd,
                                                bs,
                                                tm.load_ms,
                                                tm.first_token_ms,
                                                tm.tokens_per_s,
                                                tm.peak_vram_alloc_mb,
                                                tm.peak_vram_reserved_mb,
                                                (tm.avg_gpu_util if tm.avg_gpu_util is not None else -1),
                                                tm.oom,
                                                tm.notes,
                                            ])

                                        # Aggregate median row for the run (excluding warmup)
                                        runs = [t for t in trials if "run" in t.notes]
                                        if runs:
                                            def median(vals: List[float]) -> float:
                                                s = sorted(vals)
                                                n = len(s)
                                                if n == 0:
                                                    return -1.0
                                                if n % 2 == 1:
                                                    return s[n // 2]
                                                return 0.5 * (s[n // 2 - 1] + s[n // 2])

                                            w.writerow([
                                                mid,
                                                quant,
                                                rt,
                                                attn,
                                                kv,
                                                kvd,
                                                bs,
                                                median([t.load_ms for t in runs]),
                                                median([t.first_token_ms for t in runs]),
                                                median([t.tokens_per_s for t in runs]),
                                                median([t.peak_vram_alloc_mb for t in runs]),
                                                median([t.peak_vram_reserved_mb for t in runs]),
                                                median([(t.avg_gpu_util if t.avg_gpu_util is not None else -1) for t in runs]),
                                                1 if any(t.oom for t in runs) else 0,
                                                f"{pname}|median",
                                            ])

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
