"""Minimal ExLlamaV2 runner (optional dependency, GPTQ-only)."""
from __future__ import annotations

from typing import Any, Dict, List


class ExLlamaRunner:
    def __init__(self) -> None:
        self.available = False

    def load(self, model_id: str, **kwargs: Any) -> None:
        try:
            import exllamav2  # type: ignore  # noqa: F401
        except Exception:
            self.available = False
            return
        # Full implementation omitted; treat as available placeholder
        self.available = True

    def generate(self, messages: List[Dict[str, str]], decode: Dict[str, Any], system_prompt=None, json_only: bool = False) -> Dict[str, Any]:
        return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "timings": {"first_token_ms": -1, "tokens_per_s": -1}, "mem": None}

    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]:
        if not self.available:
            return {"texts": [], "timings": {"latency_ms": -1.0, "tokens_per_s": -1.0, "first_token_ms": -1.0}, "notes": "ExLlamaV2 not installed"}
        # Placeholder without real inference
        return {"texts": ["" for _ in prompts], "timings": {"latency_ms": -1.0, "tokens_per_s": -1.0, "first_token_ms": -1.0}}

