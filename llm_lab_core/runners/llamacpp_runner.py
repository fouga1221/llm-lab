"""Minimal llama.cpp runner (optional dependency, GGUF)."""
from __future__ import annotations

import time
from typing import Any, Dict, List


class LlamaCppRunner:
    def __init__(self) -> None:
        self.llm = None
        self.available = False

    def load(self, model_id: str, **kwargs: Any) -> None:
        try:
            from llama_cpp import Llama  # type: ignore
        except Exception:
            self.available = False
            return
        # model_id should be path to GGUF file
        self.llm = Llama(model_path=model_id)  # type: ignore[call-arg]
        self.available = True

    def generate(self, messages: List[Dict[str, str]], decode: Dict[str, Any], system_prompt=None, json_only: bool = False) -> Dict[str, Any]:
        return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "timings": {"first_token_ms": -1, "tokens_per_s": -1}, "mem": None}

    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return {"texts": [], "timings": {"latency_ms": -1.0, "tokens_per_s": -1.0, "first_token_ms": -1.0}, "notes": "llama-cpp-python not installed"}
        t0 = time.perf_counter()
        outs = [self.llm(text=p) for p in prompts]  # type: ignore
        dt = (time.perf_counter() - t0) * 1000.0
        texts = [o["choices"][0]["text"] for o in outs]
        return {"texts": texts, "timings": {"latency_ms": round(dt, 2), "tokens_per_s": -1.0, "first_token_ms": -1.0}}

