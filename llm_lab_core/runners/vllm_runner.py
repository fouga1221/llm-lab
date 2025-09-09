"""vLLM runner with simple timings and text outputs.

Adds basic support for quantization modes:
- float16 (default): dtype set to float16
- awq-int4: quantization="awq"
- gptq-int4: quantization="gptq"

Raises ImportError on missing dependency so the bench can skip cleanly.
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional


class VLLMRunner:
    def __init__(self) -> None:
        self.llm = None
        self.sampling_params = None
        self.tok = None  # Optional HF tokenizer for token counting

    def load(self, model_id: str, **kwargs: Any) -> None:
        try:
            from vllm import LLM, SamplingParams  # type: ignore
        except Exception as e:
            raise ImportError(f"vLLM not installed or failed to import: {e}")

        # Map bench quant -> vLLM quantization/dtype
        quant: Optional[str] = kwargs.get("quant")
        vllm_quant: Optional[str] = None
        dtype: str = "auto"
        if quant in (None, "float16", "fp16"):
            # Use FP16 kernels when available
            dtype = "float16"
            vllm_quant = None
        elif isinstance(quant, str) and quant.lower().startswith("awq"):
            # Pre-quantized AWQ weights are required
            vllm_quant = "awq"
            dtype = "auto"
        elif isinstance(quant, str) and quant.lower().startswith("gptq"):
            # Pre-quantized GPTQ weights are required
            vllm_quant = "gptq"
            dtype = "auto"
        else:
            # Common unsupported cases (e.g., bnb-*) -> raise to let bench skip
            raise ImportError(f"Unsupported quant for vLLM: {quant}")

        # Build engine
        # Note: additional options (tensor_parallel_size, gpu_memory_utilization, etc.)
        # can be injected here if needed later.
        self.llm = LLM(model=model_id, dtype=dtype, quantization=vllm_quant)
        self.sampling_params = SamplingParams()

        # Optional tokenizer (for tokens/s estimation)
        try:
            from transformers import AutoTokenizer  # lazy import
            self.tok = AutoTokenizer.from_pretrained(model_id)
        except Exception:
            self.tok = None

    def generate(self, messages: List[Dict[str, str]], decode: Dict[str, Any], system_prompt: Optional[str] = None, json_only: bool = False) -> Dict[str, Any]:
        # Bench uses generate_text; keep signature for compatibility.
        return {"text": "", "usage": {"prompt_tokens": 0, "completion_tokens": 0}, "timings": {"first_token_ms": -1, "tokens_per_s": -1}, "mem": None}

    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]:
        assert self.llm is not None and self.sampling_params is not None, "call load() first"

        # Map decode params to vLLM SamplingParams
        try:
            self.sampling_params.max_tokens = int(decode.get("max_new_tokens", 256))  # type: ignore[attr-defined]
            self.sampling_params.temperature = float(decode.get("temperature", 0.4))  # type: ignore[attr-defined]
            self.sampling_params.top_p = float(decode.get("top_p", 0.9))  # type: ignore[attr-defined]
            # repetition_penalty is not directly supported in all versions; ignore if absent
        except Exception:
            pass

        t0 = time.perf_counter()
        outs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        dt = (time.perf_counter() - t0) * 1000.0
        texts = [o.outputs[0].text for o in outs]

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
