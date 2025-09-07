"""Runner factory and base typing."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Protocol


class Runner(Protocol):
    def load(self, model_id: str, **kwargs: Any) -> None: ...
    def generate(self, messages: List[Dict[str, str]], decode: Dict[str, Any], system_prompt: Optional[str] = None, json_only: bool = False) -> Dict[str, Any]: ...
    def generate_text(self, prompts: List[str], decode: Dict[str, Any]) -> Dict[str, Any]: ...


def get_runner(name: str) -> Runner:
    name = (name or "transformers").lower()
    if name == "transformers":
        from llm_lab_core.runners.transformers_runner import TransformersRunner

        return TransformersRunner()
    if name == "vllm":
        from llm_lab_core.runners.vllm_runner import VLLMRunner

        return VLLMRunner()
    if name == "exllamav2":
        from llm_lab_core.runners.exllama_runner import ExLlamaRunner

        return ExLlamaRunner()
    if name == "llamacpp":
        from llm_lab_core.runners.llamacpp_runner import LlamaCppRunner

        return LlamaCppRunner()
    # default
    from llm_lab_core.runners.transformers_runner import TransformersRunner

    return TransformersRunner()

