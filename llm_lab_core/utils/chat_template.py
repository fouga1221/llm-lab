"""Helper to safely apply tokenizer chat templates across models.

Falls back to a simple generic transcript if the tokenizer has no chat template.
"""
from __future__ import annotations

from typing import List, Dict, Optional

from transformers import PreTrainedTokenizerBase


def apply_chat_template(
    tok: PreTrainedTokenizerBase,
    messages: List[Dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    try:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=add_generation_prompt)  # type: ignore[attr-defined]
    except Exception:
        # Fallback: simple tags
        parts = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if role == "system":
                parts.append(f"<|system|>\n{content}")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}")
            else:
                parts.append(f"<|user|>\n{content}")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "\n".join(parts)

