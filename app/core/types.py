"""
Core type definitions for the application.
Following the detailed design document, this module centralizes all common data
structures and type aliases to ensure consistency and prevent circular imports.
"""
from __future__ import annotations
from typing import Any, Dict, Iterable, List, Literal, Optional, Protocol, Tuple, TypedDict, Union, cast
from dataclasses import dataclass

# 3. 共通型定義
Role = Literal['system', 'user', 'assistant']

@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: Role
    content: str

@dataclass
class GenerationParams:
    """Parameters for LLM generation."""
    temperature: float = 0.6
    top_p: float = 0.9
    repetition_penalty: float = 1.0
    max_new_tokens: int = 512
    json_mode: bool = True  # Hint for the runner to apply JSON constraints

class StreamChunk(TypedDict):
    """A chunk of a streamed response."""
    text: str
    done: bool

class StructuredAction(TypedDict, total=False):
    """
    Represents a single structured action proposed by the LLM.
    `total=False` allows for optional fields.
    """
    function: str
    arguments: Dict[str, Any]
    confidence: float
    rationale: str
    risk: Literal['low', 'medium', 'high']

class StructuredActions(TypedDict):
    """A container for a list of structured actions."""
    actions: List[StructuredAction]

@dataclass
class ModelInfo:
    """Information about the loaded model."""
    model_id: str
    runner: Literal['llama_cpp', 'exllama_v2', 'transformers']
    context_length: int
    quantization: Optional[str] = None  # e.g., 'gptq-4bit', 'gguf-q4_k_m'

@dataclass
class RouteDecision:
    """Decision made by the ConfidenceRouter."""
    model: ModelInfo
    reason: str
    confidence: float


# Helper factory for empty StructuredActions to keep type checkers happy
def empty_actions() -> StructuredActions:
    """Returns an empty StructuredActions object (typed)."""
    return cast(StructuredActions, {"actions": []})
