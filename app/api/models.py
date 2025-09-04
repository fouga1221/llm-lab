"""
Pydantic models for API requests and responses.
Corresponds to section 9 of the detailed design document.
"""
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, List

from app.core.types import StructuredActions

# Using Pydantic's BaseModel for automatic validation and serialization.

class ChatRequest(BaseModel):
    """Request model for the /chat endpoint."""
    session_id: str = Field(..., description="Unique identifier for the conversation session.")
    input: str = Field(..., description="The user's input text.")
    state: Optional[Dict[str, Any]] = Field(None, description="Optional game state object.")
    options: Optional[Dict[str, Any]] = Field(None, description="Optional generation parameters to override defaults.")

class ChatResponse(BaseModel):
    """Response model for the /chat endpoint."""
    reply: str = Field(..., description="The natural language reply from the NPC.")
    structured_actions: StructuredActions = Field(..., description="The structured actions proposed by the LLM.")
    meta: Dict[str, Any] = Field(..., description="Metadata about the generation process (model, latency, etc.).")

class ProposeActionRequest(BaseModel):
    """Request model for the /propose_action endpoint."""
    session_id: str = Field(..., description="Unique identifier for the conversation session.")
    input: str = Field(..., description="The user's input text that implies an action.")
    state: Dict[str, Any] = Field(..., description="Current game state object.")
    tools: Optional[List[str]] = Field(None, description="Optional list of tools to consider.")

class ProposeActionResponse(BaseModel):
    """Response model for the /propose_action endpoint."""
    actions: List[Dict[str, Any]] = Field(..., description="The list of proposed actions.")
    explanations: Optional[List[str]] = Field(None, description="Explanations for the proposed actions.")
    meta: Dict[str, Any] = Field(..., description="Metadata about the generation process.")

class FunctionSchemaResponse(BaseModel):
    """Response model for the /schema/functions endpoint."""
    functions: List[Dict[str, Any]] = Field(..., description="A list of all available function schemas.")
    version: str = Field(..., description="The version of the function schema.")
