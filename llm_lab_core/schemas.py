"""Pydantic schemas for API requests/responses."""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class Message(BaseModel):
    role: str
    content: str


class DecodeConfig(BaseModel):
    temperature: float = 0.4
    top_p: float = 0.9
    repetition_penalty: float = 1.1
    max_new_tokens: int = 256
    stop: Optional[List[str]] = None


class ChatRequest(BaseModel):
    messages: List[Message]
    model: Optional[str] = Field(default=None, description="HF model id or local path")
    runtime: Optional[str] = Field(default="transformers", description="transformers|vllm|exllamav2|llamacpp")
    system_prompt: Optional[str] = None
    decode: Optional[DecodeConfig] = None
    schema_name: Optional[str] = None  # for propose_action


class ChatResponse(BaseModel):
    text: str
    usage: Dict[str, int]
    timings: Dict[str, float]
    mem: Optional[Dict[str, float]] = None

