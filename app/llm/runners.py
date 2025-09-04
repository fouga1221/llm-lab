"""
LLM Runner Interfaces and Implementations.
Corresponds to section 5.1 of the detailed design document.
"""
from abc import ABC, abstractmethod
from app.core.types import Message, GenerationParams, StreamChunk, ModelInfo
from typing import List, Iterable

# Using Protocol for interface definition as per the design
from typing import Protocol

class BaseRunner(Protocol):
    """
    Protocol (Interface) for all LLM runners.
    Ensures that any runner implementation will have the required methods.
    """
    def generate(self, messages: List[Message], params: GenerationParams) -> str:
        """Generates a response from a list of messages."""
        ...

    def stream(self, messages: List[Message], params: GenerationParams) -> Iterable[StreamChunk]:
        """Streams a response from a list of messages."""
        ...

    def model_info(self) -> ModelInfo:
        """Returns information about the loaded model."""
        ...

# --- Placeholder Implementations ---
# These classes provide a clear structure for future implementation.
# They inherit from ABC and implement the abstract methods defined in the protocol,
# though the protocol itself is what's used for type hinting to ensure decoupling.

class LlamaCppRunner(ABC):
    """Runner for GGUF models using llama-cpp-python."""
    def __init__(self, model_path: str, **kwargs):
        self._model_info = ModelInfo(
            model_id=model_path,
            runner='llama_cpp',
            context_length=kwargs.get('n_ctx', 4096),
            quantization=kwargs.get('quantization_type', 'unknown')
        )
        # In a real implementation, you would load the model here.
        # from llama_cpp import Llama
        # self.model = Llama(model_path=model_path, **kwargs)
        raise NotImplementedError("LlamaCppRunner is not yet implemented.")

    def generate(self, messages: List[Message], params: GenerationParams) -> str:
        # Real implementation would call self.model.create_chat_completion
        raise NotImplementedError

    def stream(self, messages: List[Message], params: GenerationParams) -> Iterable[StreamChunk]:
        # Real implementation would call self.model.create_chat_completion with stream=True
        raise NotImplementedError

    def model_info(self) -> ModelInfo:
        return self._model_info

class ExLlamaV2Runner(ABC):
    """Runner for EXL2 models using ExLlamaV2."""
    def __init__(self, model_dir: str, **kwargs):
        self._model_info = ModelInfo(
            model_id=model_dir,
            runner='exllama_v2',
            context_length=kwargs.get('max_seq_len', 4096),
            quantization=kwargs.get('quantization_type', 'exl2')
        )
        # Real implementation would load the model here.
        raise NotImplementedError("ExLlamaV2Runner is not yet implemented.")

    def generate(self, messages: List[Message], params: GenerationParams) -> str:
        raise NotImplementedError

    def stream(self, messages: List[Message], params: GenerationParams) -> Iterable[StreamChunk]:
        raise NotImplementedError

    def model_info(self) -> ModelInfo:
        return self._model_info

class TransformersRunner(ABC):
    """Runner for Hugging Face models using transformers."""
    def __init__(self, model_id: str, **kwargs):
        self._model_info = ModelInfo(
            model_id=model_id,
            runner='transformers',
            context_length=kwargs.get('max_position_embeddings', 4096),
            quantization=kwargs.get('quantization_config', {}).get('quant_method')
        )
        # Real implementation would load the model here.
        raise NotImplementedError("TransformersRunner is not yet implemented.")

    def generate(self, messages: List[Message], params: GenerationParams) -> str:
        raise NotImplementedError

    def stream(self, messages: List[Message], params: GenerationParams) -> Iterable[StreamChunk]:
        raise NotImplementedError

    def model_info(self) -> ModelInfo:
        return self._model_info
