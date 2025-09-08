"""Transformers runner with chat-template and simple metrics."""
from __future__ import annotations

import math
import time
from typing import Any, Dict, List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

from llm_lab_core.utils.chat_template import apply_chat_template


class TransformersRunner:
    def __init__(self) -> None:
        self.model = None
        self.tok = None
        self.model_id: Optional[str] = None

    def load(self, model_id: str, **kwargs: Any) -> None:
        if self.model is not None and self.model_id == model_id:
            return
        attn_impl = kwargs.get("attn_impl", "sdpa")
        quant = kwargs.get("quant", None)

        model_kwargs: Dict[str, Any] = {}
        if quant in (None, "float16", "fp16"):
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            if torch.cuda.is_available():
                model_kwargs["device_map"] = "auto"
        elif quant == "bnb-int8":
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(load_in_8bit=True)
            model_kwargs["device_map"] = "auto"
        elif quant in ("bnb-nf4", "bnb-4bit"):
            from transformers import BitsAndBytesConfig

            model_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            )
            model_kwargs["device_map"] = "auto"
        else:
            model_kwargs["torch_dtype"] = torch.float16 if torch.cuda.is_available() else torch.float32
            model_kwargs["device_map"] = "auto"

        self.tok = AutoTokenizer.from_pretrained(model_id)
        # Ensure decoder-only batching works without warnings/errors
        try:
            self.tok.padding_side = "left"
        except Exception:
            pass
        try:
            if self.tok.pad_token is None and getattr(self.tok, "eos_token", None) is not None:
                self.tok.pad_token = self.tok.eos_token  # type: ignore[assignment]
        except Exception:
            pass
        self.model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, **model_kwargs)
        # Align pad_token_id on the model to suppress "Setting pad_token_id to eos_token_id" logs
        try:
            if getattr(self.tok, "pad_token_id", None) is not None:
                self.model.config.pad_token_id = self.tok.pad_token_id  # type: ignore[attr-defined]
                if getattr(self.model, "generation_config", None) is not None:
                    self.model.generation_config.pad_token_id = self.tok.pad_token_id  # type: ignore[attr-defined]
        except Exception:
            pass
        self.model_id = model_id

    def _build_prompt(self, messages: List[Dict[str, str]], system_prompt: Optional[str]) -> str:
        msgs = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        msgs.extend(messages)
        return apply_chat_template(self.tok, msgs, add_generation_prompt=True)  # type: ignore[arg-type]

    def generate(
        self,
        messages: List[Dict[str, str]],
        decode: Dict[str, Any],
        system_prompt: Optional[str] = None,
        json_only: bool = False,
    ) -> Dict[str, Any]:
        assert self.model is not None and self.tok is not None, "call load() first"
        prompt = self._build_prompt(messages, system_prompt)
        enc = self.tok(prompt, return_tensors="pt")
        if torch.cuda.is_available():
            enc = enc.to(self.model.device)

        # Prepare generation args
        max_new = int(decode.get("max_new_tokens", 256))
        temperature = float(decode.get("temperature", 0.4))
        top_p = float(decode.get("top_p", 0.9))
        repetition_penalty = float(decode.get("repetition_penalty", 1.1))

        streamer = TextIteratorStreamer(self.tok, skip_special_tokens=True)

        # Timings
        first_token_ms = math.nan
        t_start = time.perf_counter()

        import threading

        def _gen():
            assert self.model is not None and self.tok is not None, "call load() first"
            self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                streamer=streamer,
            ) 

        th = threading.Thread(target=_gen, daemon=True)
        th.start()
        got_first = False
        out_text = ""
        for chunk in streamer:
            if not got_first:
                first_token_ms = (time.perf_counter() - t_start) * 1000.0
                got_first = True
            out_text += chunk
        th.join()

        # Usage rough counts
        new_ids = self.tok(out_text, add_special_tokens=False).input_ids
        total_ms = (time.perf_counter() - t_start) * 1000.0
        tps = (len(new_ids) / (total_ms / 1000.0)) if total_ms > 0 else 0.0

        mem = {
            "peak_vram_alloc_mb": round(torch.cuda.max_memory_allocated() / (1024 * 1024), 2) if torch.cuda.is_available() else -1.0,
            "peak_vram_reserved_mb": round(torch.cuda.max_memory_reserved() / (1024 * 1024), 2) if torch.cuda.is_available() else -1.0,
        }

        return {
            "text": out_text,
            "usage": {
                "prompt_tokens": int(enc["input_ids"].numel()),
                "completion_tokens": int(len(new_ids)),
            },
            "timings": {
                "first_token_ms": round(first_token_ms, 2) if math.isfinite(first_token_ms) else -1.0,
                "tokens_per_s": round(tps, 2),
            },
            "mem": mem,
        }

    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]:
        # For bench compatibility: treat prompts as already-formed text
        assert self.model is not None and self.tok is not None, "call load() first"
        enc = self.tok(prompts, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            enc = enc.to(self.model.device)

        max_new = int(decode.get("max_new_tokens", 256))
        temperature = float(decode.get("temperature", 0.4))
        top_p = float(decode.get("top_p", 0.9))
        repetition_penalty = float(decode.get("repetition_penalty", 1.1))

        bs = int(enc["input_ids"].shape[0])
        if bs == 1:
            # Stream to capture first token
            streamer = TextIteratorStreamer(self.tok, skip_special_tokens=True)
            first_token_ms = math.nan
            t_start = time.perf_counter()

            import threading

            def _gen():
                assert self.model is not None
                self.model.generate(
                    input_ids=enc["input_ids"],
                    attention_mask=enc.get("attention_mask"),
                    max_new_tokens=max_new,
                    do_sample=True,
                    temperature=temperature,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    streamer=streamer,
                )

            th = threading.Thread(target=_gen, daemon=True)
            th.start()
            out_text = ""
            got_first = False
            for chunk in streamer:
                if not got_first:
                    first_token_ms = (time.perf_counter() - t_start) * 1000.0
                    got_first = True
                out_text += chunk
            th.join()

            total_ms = (time.perf_counter() - t_start) * 1000.0
            try:
                new_ids = self.tok(out_text, add_special_tokens=False).input_ids
                tps = (len(new_ids) / (total_ms / 1000.0)) if total_ms > 0 else 0.0
            except Exception:
                tps = 0.0
            return {
                "texts": [out_text],
                "timings": {
                    "latency_ms": round(total_ms, 2),
                    "tokens_per_s": round(tps, 2),
                    "first_token_ms": round(first_token_ms, 2) if math.isfinite(first_token_ms) else -1.0,
                },
            }

        # Batch size > 1: run without streamer and compute throughput across batch
        t0 = time.perf_counter()
        with torch.no_grad():
            out = self.model.generate(
                input_ids=enc["input_ids"],
                attention_mask=enc.get("attention_mask"),
                max_new_tokens=max_new,
                do_sample=True,
                temperature=temperature,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
        total_ms = (time.perf_counter() - t0) * 1000.0
        decoded = self.tok.batch_decode(out, skip_special_tokens=True)
        try:
            # Rough: count generated tokens by retokenizing decoded outputs
            flat_new = 0
            for d in decoded:
                flat_new += len(self.tok(d, add_special_tokens=False).input_ids)
            tps = (flat_new / (total_ms / 1000.0)) if total_ms > 0 else 0.0
        except Exception:
            tps = 0.0
        return {
            "texts": decoded,
            "timings": {"latency_ms": round(total_ms, 2), "tokens_per_s": round(tps, 2), "first_token_ms": -1.0},
        }
