"""Minimal vLLM runner (optional dependency)."""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class VLLMRunner:
    def __init__(self) -> None:
        self.llm = None
        self.sampling_params = None
        self.available = False

    def load(self, model_id: str, **kwargs: Any) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception:
            self.available = False
            return
        self.llm = LLM(model=model_id)
        self.sampling_params = SamplingParams()
        self.available = True

    def generate(self, messages: List[Dict[str, str]], decode: Dict[str, Any], system_prompt: Optional[str] = None, json_only: bool = False) -> Dict[str, Any]:
        # Not implementing chat template here; bench uses generate_text.
        return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "timings": {"first_token_ms": -1, "tokens_per_s": -1}, "mem": None}

    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return {"texts": [], "timings": {"latency_ms": -1.0, "tokens_per_s": -1.0, "first_token_ms": -1.0}, "notes": "vLLM not installed"}
        
        assert self.llm is not None and self.sampling_params is not None, "call load() first"
        
        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        dt = (time.perf_counter() - t0) * 1000.0
        texts = [o.outputs[0].text for o in outs]
        # tokens/s rough (requires token counts; skip)
        return {"texts": texts, "timings": {"latency_ms": round(dt, 2), "tokens_per_s": -1.0, "first_token_ms": -1.0}}

