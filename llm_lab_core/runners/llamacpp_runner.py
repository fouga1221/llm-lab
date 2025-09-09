"""llama.cpp runner (GGUF) with simple timings.

Adds basic GPU config injection when available:
- n_gpu_layers=-1 (all layers on GPU when built with cuBLAS)
- n_ctx default to 8192 (can be overridden via kwargs["n_ctx"])

Raises ImportError on missing dependency so the bench can skip cleanly.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List
import os
import warnings


class LlamaCppRunner:
    def __init__(self) -> None:
        self.llm = None
        self.available = False
        self.tok = None  # Optional HF tokenizer for token counting

    def load(self, model_id: str, **kwargs: Any) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception as e:
            raise ImportError(f"llama-cpp-python not installed or failed to import: {e}")

        # model_id should be path to GGUF file
        # Inject minimal GPU options if the wheel was built with CUDA/cuBLAS.
        # If CPU-only build, llama.cpp will ignore n_gpu_layers.
        n_ctx = int(kwargs.get("n_ctx", os.environ.get("LLAMACPP_N_CTX", 8192)))
        n_gpu_layers = kwargs.get("n_gpu_layers", -1)
        try:
            self.llm = Llama(  # type: ignore[call-arg]
                model_path=model_id,
                n_ctx=n_ctx,
                n_gpu_layers=n_gpu_layers,
                verbose=False,
            )
        except TypeError:
            # Older versions may not accept verbose/n_gpu_layers; retry gracefully
            warnings.warn("llama.cpp init without GPU opts due to incompatible version")
            self.llm = Llama(model_path=model_id, n_ctx=n_ctx)  # type: ignore[call-arg]
        self.available = True
        # Optional tokenizer (if a matching HF model id is provided via kwargs)
        tok_id = kwargs.get("tokenizer_id")
        if tok_id:
            try:
                from transformers import AutoTokenizer  # lazy import
                self.tok = AutoTokenizer.from_pretrained(tok_id)
            except Exception:
                self.tok = None

    def generate(self, messages: List[Dict[str, str]], decode: Dict[str, Any], system_prompt=None, json_only: bool = False) -> Dict[str, Any]:
        return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "timings": {"first_token_ms": -1, "tokens_per_s": -1}, "mem": None}

    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]:
        assert self.llm is not None, "call load() first"
        t0 = time.perf_counter()
        outs = [self.llm(text=p, max_tokens=int(decode.get("max_new_tokens", 256)), top_p=float(decode.get("top_p", 0.9)), temperature=float(decode.get("temperature", 0.4))) for p in prompts]  # type: ignore
        dt = (time.perf_counter() - t0) * 1000.0
        texts = [o["choices"][0]["text"] for o in outs]
        # Rough tokens/s via tokenizer if available
        tps = -1.0
        if self.tok is not None:
            try:
                total_new = 0
                for t in texts:
                    total_new += len(self.tok(t, add_special_tokens=False).input_ids)
                tps = round((total_new / (dt / 1000.0)), 2) if dt > 0 else -1.0
            except Exception:
                tps = -1.0
        return {"texts": texts, "timings": {"latency_ms": round(dt, 2), "tokens_per_s": tps, "first_token_ms": -1.0}}
