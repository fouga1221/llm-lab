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
import sys
import time
from dataclasses import dataclass
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import sys
import torch
import yaml

# Add project root to path to allow absolute imports from other directories
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.monitor import NVMLMonitor
from llm_lab_core.utils.chat_template import apply_chat_template as hf_apply_chat_template
from transformers import AutoTokenizer
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
    "timeout",
    "notes",
    "out_path",
]


def now_ms() -> float:
    return time.perf_counter() * 1000.0


def set_seed(seed: int) -> None:
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_prompts_by_name(name: str) -> List[Any]:
    base = Path("data/prompts")
    # Try YAML list
    yml = base / f"{name}.yaml"
    if yml.exists():
        try:
            data = yaml.safe_load(yml.read_text(encoding="utf-8"))
            # Supported formats:
            # - {messages: [{role,content}, ...]}
            # - {system: str, user: str}
            # - ["prompt1", "prompt2"]
            # - [{role,content}, ...] (single dialog as list)
            if isinstance(data, dict):
                if isinstance(data.get("messages"), list):
                    return [{"messages": data["messages"]}]
                if ("system" in data) or ("user" in data):
                    msgs = []
                    if data.get("system"):
                        msgs.append({"role": "system", "content": str(data["system"])})
                    if data.get("user"):
                        msgs.append({"role": "user", "content": str(data["user"])})
                    return [{"messages": msgs}]
                if "prompts" in data:
                    return [str(x) for x in data["prompts"]]
            if isinstance(data, list):
                if data and isinstance(data[0], dict) and "role" in data[0]:
                    return [{"messages": data}]
                return [str(x) for x in data]
        except Exception:
            pass
    # Try plain text (one per line)
    txt = base / f"{name}.txt"
    if txt.exists():
        return [l.strip() for l in txt.read_text(encoding="utf-8").splitlines() if l.strip()]
    return ["こんにちは。調子はどう？"]


 


def render_texts_simple(items: List[Any]) -> List[str]:
    def as_text(it: Any) -> str:
        if isinstance(it, dict) and isinstance(it.get("messages"), list):
            msgs = it["messages"]
        elif isinstance(it, list) and it and isinstance(it[0], dict) and "role" in it[0]:
            msgs = it
        else:
            return str(it)
        parts: List[str] = []
        for m in msgs:
            role = m.get("role", "user")
            content = m.get("content", "")
            tag = "assistant" if role == "assistant" else ("system" if role == "system" else "user")
            parts.append(f"<|{tag}|>\n{content}")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    return [as_text(x) for x in items]


 


# ---------------- Subprocess worker for timeout-safe inference ---------------- #
import multiprocessing as mp
# Ensure CUDA works with multiprocessing across platforms (Linux default is 'fork').
# Using 'fork' breaks CUDA with PyTorch. Switch to 'spawn' if not already set.
try:  # safe no-op if already set elsewhere
    if mp.get_start_method(allow_none=True) != "spawn":
        mp.set_start_method("spawn", force=True)
except Exception:
    pass


def _worker_main(in_q: "mp.Queue", out_q: "mp.Queue", rt: str, model_id: str, quant: Optional[str]) -> None:  # pragma: no cover - runtime helper
    from llm_lab_core.runners import get_runner as _get_runner
    import time as _time
    import traceback as _traceback
    # Local helper mirrors parent's behavior
    def _set_attn(attn: str) -> None:
        try:
            if attn == "flash_attn_2":
                try:
                    torch.backends.cuda.enable_flash_sdp(True)  # type: ignore[attr-defined]
                except Exception:
                    pass
            elif attn == "sdpa":
                try:
                    torch.backends.cuda.enable_mem_efficient_sdp(True)  # type: ignore[attr-defined]
                except Exception:
                    pass
        except Exception:
            pass

    t0 = _time.perf_counter()
    runner = None
    try:
        runner = _get_runner(rt)
        runner.load(model_id, quant=quant)
        load_ms = (_time.perf_counter() - t0) * 1000.0
        out_q.put({"event": "loaded", "ok": True, "load_ms": round(load_ms, 2)})
    except Exception as e:
        out_q.put({"event": "loaded", "ok": False, "error": str(e), "trace": _traceback.format_exc()})
        return

    while True:
        msg = in_q.get()
        if msg is None:
            break
        cmd = msg.get("cmd")
        if cmd == "gen":
            texts = msg.get("texts", [])
            decode = msg.get("decode", {})
            attn = msg.get("attn", "sdpa")
            try:
                if rt == "transformers":
                    _set_attn(attn)
                res = runner.generate_text(texts, decode)
                out_q.put({"event": "result", "ok": True, "res": res})
            except Exception as e:
                out_q.put({"event": "result", "ok": False, "error": str(e), "trace": _traceback.format_exc()})
        else:
            out_q.put({"event": "unknown", "ok": False, "error": f"unknown cmd: {cmd}"})


class InferenceWorker:
    def __init__(self, rt: str, model_id: str, quant: Optional[str]) -> None:
        self.rt = rt
        self.model_id = model_id
        self.quant = quant
        self.proc: Optional[mp.Process] = None
        self.in_q: "mp.Queue" = mp.Queue()
        self.out_q: "mp.Queue" = mp.Queue()
        self.load_ms: float = -1.0
        self.last_error: Optional[str] = None
        self.last_info: Optional[str] = None

    def start(self, timeout_s: float = 120.0) -> bool:
        if self.proc is not None and self.proc.is_alive():
            return True
        self.proc = mp.Process(target=_worker_main, args=(self.in_q, self.out_q, self.rt, self.model_id, self.quant), daemon=True)
        self.proc.start()
        try:
            msg = self.out_q.get(timeout=timeout_s)
        except Exception:
            self.last_error = f"timeout_waiting_loaded({timeout_s}s)"
            self.terminate()
            return False
        if not msg.get("ok", False):
            self.last_error = f"load_exception: {msg.get('error', 'unknown')}"
            if msg.get("trace"):
                self.last_info = str(msg.get("trace"))
            self.terminate()
            return False
        self.load_ms = float(msg.get("load_ms", -1))
        return True

    def generate(self, texts: List[str], decode: Dict[str, Any], attn_impl: str, timeout_s: float) -> Tuple[bool, Dict[str, Any]]:
        if self.proc is None or not self.proc.is_alive():
            return False, {"error": "worker not running"}
        self.in_q.put({"cmd": "gen", "texts": texts, "decode": decode, "attn": attn_impl})
        try:
            msg = self.out_q.get(timeout=timeout_s)
            return bool(msg.get("ok", False)), msg
        except Exception:
            # timeout
            return False, {"timeout": True}

    def terminate(self) -> None:
        try:
            if self.proc is not None and self.proc.is_alive():
                try:
                    self.in_q.put(None)
                except Exception:
                    pass
                self.proc.terminate()
                self.proc.join(timeout=2.0)
        finally:
            self.proc = None


@dataclass
class TrialMetrics:
    load_ms: float
    first_token_ms: float
    tokens_per_s: float
    peak_vram_alloc_mb: float
    peak_vram_reserved_mb: float
    avg_gpu_util: Optional[float]
    oom: int
    timeout: int
    notes: str


 


def write_case_log(log_dir: str, case_id: str, content: str) -> None:
    try:
        p = Path(log_dir) / f"{case_id}.log"
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("a", encoding="utf-8") as f:
            f.write(content.rstrip() + "\n")
    except Exception:
        pass


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark inference and log CSV per your sweep plan")
    ap.add_argument("--sweep", required=True, help="YAML describing models/quant/runtime/params")
    ap.add_argument("--out", default="results/runs.csv", help="CSV output path")
    ap.add_argument("--log-dir", default="results/logs", help="Logs directory (for errors, etc.)")
    ap.add_argument("--timeout", type=float, default=100.0, help="Per-trial inference timeout seconds")
    ap.add_argument("--load-timeout", type=float, default=600.0, help="Model load timeout seconds for worker startup/restart")
    ap.add_argument("--save-outputs", action="store_true", help="Save generated outputs per trial to files")
    ap.add_argument("--outputs-dir", default="results/outs", help="Directory to save outputs when --save-outputs")
    ap.add_argument("--verbose", action="store_true", help="Enable verbose logging to stderr")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING)

    sweep = yaml.safe_load(Path(args.sweep).read_text(encoding="utf-8"))

    # models can be:
    # - string: treated as HF repo id
    # - dict: may include {hf: <hf_id>, gguf: <local_gguf_path>}
    models_raw = sweep.get("models", [])
    models_cfg: List[Dict[str, Any]] = []
    for m in models_raw:
        if isinstance(m, str):
            models_cfg.append({"hf": m})
        elif isinstance(m, dict):
            # Keep only relevant keys
            kept = {k: v for k, v in m.items() if k in ("hf", "gguf")}
            if kept:
                models_cfg.append(kept)
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
    if args.save_outputs:
        Path(args.outputs_dir).mkdir(parents=True, exist_ok=True)

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

    # Tokenizer cache per HF model id
    tok_cache: Dict[str, Any] = {}

    def get_tok_for(model_cfg: Dict[str, Any]):
        hf_id = model_cfg.get("hf")
        if not hf_id:
            return None
        if hf_id not in tok_cache:
            try:
                tok_cache[hf_id] = AutoTokenizer.from_pretrained(hf_id)
            except Exception:
                tok_cache[hf_id] = None
        return tok_cache[hf_id]

    def to_messages(item: Any) -> List[Dict[str, str]]:
        # Normalize an item to a messages list
        if isinstance(item, dict) and isinstance(item.get("messages"), list):
            return [dict(role=m.get("role","user"), content=str(m.get("content",""))) for m in item["messages"]]
        if isinstance(item, list) and item and isinstance(item[0], dict) and "role" in item[0]:
            return [dict(role=m.get("role","user"), content=str(m.get("content",""))) for m in item]
        # Fallback: plain text as single user message
        return [{"role": "user", "content": str(item)}]

    def render_texts_for_runtime(model_cfg: Dict[str, Any], batch_items: List[Any]) -> List[str]:
        tok = get_tok_for(model_cfg)
        texts: List[str] = []
        for it in batch_items:
            msgs = to_messages(it)
            if tok is not None:
                try:
                    texts.append(hf_apply_chat_template(tok, msgs, add_generation_prompt=True))
                    continue
                except Exception:
                    pass
            # Fallback to generic tags if tokenizer missing
            parts: List[str] = []
            for m in msgs:
                r = m.get("role","user")
                c = m.get("content","")
                tag = "system" if r=="system" else ("assistant" if r=="assistant" else "user")
                parts.append(f"<|{tag}|>\n{c}")
            parts.append("<|assistant|>\n")
            texts.append("\n".join(parts))
        return texts

    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEADER)

        # Helper: select model id per runtime
        def select_model_id(rt: str, mcfg: Dict[str, Any]) -> Optional[str]:
            rt = (rt or "").lower()
            if rt in ("transformers", "vllm"):
                return mcfg.get("hf")
            if rt == "llamacpp":
                return mcfg.get("gguf")
            if rt == "exllamav2":
                # If ever extended, allow specifying a local GPTQ path
                return mcfg.get("gptq") or mcfg.get("hf")
            return mcfg.get("hf")

        for mcfg in models_cfg:
            for quant in quants:
                for rt in runtimes:
                    mid = select_model_id(rt, mcfg)
                    base_name = mcfg.get("hf") or mcfg.get("gguf") or "unknown"
                    if not mid:
                        # No model available for this runtime; log and continue
                        case_id = f"{base_name}__{quant}__{rt}__no_model_for_runtime"
                        write_case_log(args.log_dir, case_id, f"no_model_id_for_runtime: model={mcfg} runtime={rt}")
                        logging.error(f"[skip-runtime] {case_id} -> {Path(args.log_dir) / (case_id + '.log')}" )
                        continue
                    # Start a worker process per (rt, model, quant) to keep warm and enable timeout
                    logging.info(f"[load] runtime={rt} model={mid} quant={quant}")
                    worker = InferenceWorker(rt, mid, quant)
                    if not worker.start(timeout_s=float(args.load_timeout)):
                        # Log and SKIP writing a CSV row
                        case_id = f"{mid}__{quant}__{rt}__load"
                        env = f"cuda_is_available={torch.cuda.is_available()} cuda_device_count={(torch.cuda.device_count() if torch.cuda.is_available() else 0)}"
                        details = [
                            f"load_failed: {worker.last_error or 'unknown'}",
                            env,
                        ]
                        if worker.last_info:
                            details.append(str(worker.last_info))
                        path = Path(args.log_dir) / f"{case_id}.log"
                        write_case_log(args.log_dir, case_id, "\n".join(details))
                        logging.error(f"[skip-runtime] {case_id} -> {path}")
                        worker.terminate()
                        continue
                    load_ms = worker.load_ms
                    logging.info(f"[loaded] runtime={rt} model={mid} quant={quant} load_ms={load_ms:.2f}")

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

                                        batch_items = batch_prompts[:bs] or [batch_prompts[0]]
                                        batch_texts = render_texts_for_runtime(mcfg, batch_items)

                                        # 1 warmup + (repeats-1) runs
                                        trials: List[TrialMetrics] = []
                                        trial_notes = ["warmup"] + ["run"] * (max(1, repeats - 1))

                                        for note in trial_notes:
                                            avg_util: Optional[float] = None
                                            peak_used_mb: Optional[float] = None
                                            out_path_str = ""
                                            try:
                                                mon = NVMLMonitor()
                                                if mon.available:
                                                    mon.start()
                                                logging.info(f"[trial] {mid} q={quant} rt={rt} attn={attn} kv={kv}/{kvd} bs={bs} preset={pname} note={note}")
                                                ok, msg = worker.generate(
                                                    batch_texts,
                                                    {
                                                        "max_new_tokens": max_tokens,
                                                        "temperature": 0.4,
                                                        "top_p": 0.9,
                                                        "repetition_penalty": 1.1,
                                                    },
                                                    attn_impl=attn,
                                                    timeout_s=float(args.timeout),
                                                )
                                                if mon.available:
                                                    mon.stop()
                                                    avg_util = mon.mean_util()
                                                    peak_used_mb = mon.peak_mem_used_mb()
                                                    mon.close()

                                                if not ok:
                                                    # timeout or error
                                                    timeout_flag = 1 if msg.get("timeout") else 0
                                                    # Write reason log
                                                    case_id = f"{mid}__{quant}__{rt}__{attn}__{kv}__{kvd}__bs{bs}__{pname}__{note}"
                                                    if timeout_flag:
                                                        write_case_log(args.log_dir, case_id, f"inference_timeout: exceeded {args.timeout}s")
                                                    else:
                                                        err = msg.get("error", "unknown")
                                                        trace = msg.get("trace", "")
                                                        write_case_log(args.log_dir, case_id, f"inference_error: {err}\n{trace}")
                                                    tm = TrialMetrics(
                                                        load_ms=round(load_ms, 2),
                                                        first_token_ms=-1,
                                                        tokens_per_s=0.0,
                                                        peak_vram_alloc_mb=peak_used_mb if peak_used_mb is not None else -1.0,
                                                        peak_vram_reserved_mb=peak_used_mb if peak_used_mb is not None else -1.0,
                                                        avg_gpu_util=round(avg_util, 2) if avg_util is not None else None,
                                                        oom=0,
                                                        timeout=timeout_flag,
                                                        notes=f"{pname}|{'timeout' if timeout_flag else 'error'}",
                                                    )
                                                    if timeout_flag:
                                                        # kill and restart worker for subsequent trials
                                                        worker.terminate()
                                                        worker = InferenceWorker(rt, mid, quant)
                                                        worker.start(timeout_s=float(args.load_timeout))
                                                    logging.warning(f"[trial-ended] timeout={timeout_flag} note={'timeout' if timeout_flag else 'error'} case={case_id}")
                                                else:
                                                    timings = (msg.get("res") or {}).get("timings", {})
                                                    # Optionally save outputs
                                                    if args.save_outputs:
                                                        texts = (msg.get("res") or {}).get("texts", []) or []
                                                        case_id = f"{mid}__{quant}__{rt}__{attn}__{kv}__{kvd}__bs{bs}__{pname}__{note}"
                                                        out_path = Path(args.outputs_dir) / f"{case_id}.txt"
                                                        try:
                                                            out_path.parent.mkdir(parents=True, exist_ok=True)
                                                            with out_path.open("w", encoding="utf-8") as of:
                                                                if texts:
                                                                    for i, t in enumerate(texts):
                                                                        if len(texts) > 1:
                                                                            of.write(f"=== sample {i} ===\n")
                                                                        of.write(str(t))
                                                                        of.write("\n")
                                                                else:
                                                                    of.write("")
                                                            out_path_str = str(out_path)
                                                            logging.info(f"[saved] {out_path_str}")
                                                        except Exception:
                                                            out_path_str = ""
                                                    tm = TrialMetrics(
                                                        load_ms=round(load_ms, 2),
                                                        first_token_ms=float(timings.get("first_token_ms", -1)),
                                                        tokens_per_s=float(timings.get("tokens_per_s", -1)),
                                                        peak_vram_alloc_mb=peak_used_mb if peak_used_mb is not None else -1.0,
                                                        peak_vram_reserved_mb=peak_used_mb if peak_used_mb is not None else -1.0,
                                                        avg_gpu_util=round(avg_util, 2) if avg_util is not None else None,
                                                        oom=0,
                                                        timeout=0,
                                                        notes=f"{pname}|{note}",
                                                    )
                                            except Exception as e:
                                                tm = TrialMetrics(
                                                    load_ms=round(load_ms, 2),
                                                    first_token_ms=-1,
                                                    tokens_per_s=0.0,
                                                    peak_vram_alloc_mb=peak_used_mb if peak_used_mb is not None else -1.0,
                                                    peak_vram_reserved_mb=peak_used_mb if peak_used_mb is not None else -1.0,
                                                    avg_gpu_util=round(avg_util, 2) if avg_util is not None else None,
                                                    oom=0,
                                                    timeout=0,
                                                    notes=f"{pname}|error:{type(e).__name__}",
                                                )
                                                case_id = f"{mid}__{quant}__{rt}__{attn}__{kv}__{kvd}__bs{bs}__{pname}__{note}"
                                                write_case_log(args.log_dir, case_id, traceback.format_exc())

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
                                                tm.timeout,
                                                tm.notes,
                                                out_path_str,
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
                                                1 if any((t.timeout == 1) for t in runs) else 0,
                                                f"{pname}|median",
                                                "",
                                            ])

                    worker.terminate()

    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
