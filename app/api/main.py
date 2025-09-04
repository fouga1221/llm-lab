"""
Main FastAPI application and endpoint definitions.
Corresponds to sections 2 and 9 of the detailed design document.
"""
from fastapi import FastAPI, HTTPException, Depends
from typing import Dict, Any

from app.api.models import ChatRequest, ChatResponse, ProposeActionRequest, ProposeActionResponse, FunctionSchemaResponse
from app.core.config import ConfigLoader
from app.core.logging import Logger
from app.core.exceptions import *
from app.tooling.registry import FunctionRegistry, FunctionSpec
from app.security.validation import ActionValidator, IOPolicy
from app.tooling.validation import JsonSchemaValidator
# ... other imports will be added as components are built

# --- App Initialization ---
app = FastAPI(
    title="Local LLM NPC Service",
    description="Provides natural language interaction and structured action proposals for game NPCs.",
    version="0.1.0",
)

# --- Dependency Injection Setup ---
# This is a simplified setup. A real app would manage state more robustly.
# For now, we load configs at startup.

def get_logger():
    # In a real app, this would be configured from a file
    return Logger("api")

def get_config():
    # These paths would come from environment variables or CLI args
    try:
        return ConfigLoader(
            model_path="configs/model.yaml",
            security_path="configs/security.yaml",
            functions_path="configs/functions.json"
        )
    except FileNotFoundError as e:
        raise ConfigurationError(f"A required configuration file was not found: {e.filename}")


@app.on_event("startup")
def startup_event():
    """
    On startup, load all necessary configurations and initialize components.
    This makes them available to the dependency injection system.
    """
    logger = get_logger()
    logger.info("Starting up application and loading configurations...")
    try:
        config_loader = get_config()
        app.state.config_loader = config_loader
        app.state.model_config = config_loader.load_model()
        app.state.security_config = config_loader.load_security()
        app.state.functions_config = config_loader.load_functions()

        app.state.function_registry = FunctionRegistry.from_json(app.state.functions_config)
        
        # Placeholder for the actions schema
        actions_schema = {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "actions": {
                    "type": "array",
                    "items": {"type": "object"}
                }
            },
            "required": ["actions"]
        }
        
        app.state.io_policy = IOPolicy(
            allow_network=app.state.security_config.get('allow_network', False),
            allowed_hosts=app.state.security_config.get('allowed_hosts', []),
            allowed_paths=app.state.security_config.get('allowed_paths', [])
        )
        
        app.state.action_validator = ActionValidator(
            schema_validator=JsonSchemaValidator(actions_schema),
            io_policy=app.state.io_policy,
            registry=app.state.function_registry
        )
        
        logger.info("Configurations loaded successfully.")
        logger.info(f"Found {len(app.state.function_registry.names())} functions.")

    except Exception as e:
        logger.error("Failed to initialize application state on startup.", error=str(e))
        # This will prevent the app from starting if configs are bad
        raise ConfigurationError(f"Startup failed: {e}") from e


# --- Endpoints ---

@app.get("/schema/functions", response_model=FunctionSchemaResponse)
def get_function_schemas():
    """
    Returns the schemas of all available functions.
    """
    registry: FunctionRegistry = app.state.function_registry
    functions_list = [spec._asdict() for spec in registry.all_specs()]
    
    return FunctionSchemaResponse(
        functions=functions_list,
        version=app.state.functions_config.get("version", "v1")
    )

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest, logger: Logger = Depends(get_logger)):
    """
    Handles a user's chat message, returns a natural language reply and
    proposed structured actions.
    """
    logger.info("Received chat request", session_id=request.session_id)
    # This is a placeholder for the full logic described in the design doc (section 9)
    # 1. Get conversation history from memory
    # 2. Build prompt with RAG context if needed
    # 3. Route and generate with ConfidenceRouter
    # 4. Validate final actions
    # 5. Return response
    
    # Placeholder implementation:
    try:
        # Simulate a call to the LLM
        reply_text = f"You said: '{request.input}'. I am a placeholder response."
        actions = {
            "actions": [
                {
                    "function": "npc_idle",
                    "arguments": {"reason": "placeholder_response"},
                    "confidence": 0.99,
                    "rationale": "This is a dummy action as the system is not fully implemented."
                }
            ]
        }
        
        # Final validation
        is_valid, errors = app.state.action_validator.validate(actions)
        if not is_valid:
            logger.error("Generated actions failed final validation", errors=errors)
            # Decide on a graceful failure mode
            actions["actions"] = [] # Clear actions on validation failure

        meta = {
            "model": "placeholder_model",
            "latency_ms": 123,
            "confidence": 0.99
        }
        return ChatResponse(reply=reply_text, structured_actions=actions, meta=meta)

    except Exception as e:
        logger.error("An unexpected error occurred in /chat endpoint", error=str(e))
        raise HTTPException(status_code=500, detail="An internal error occurred.")


@app.post("/propose_action", response_model=ProposeActionResponse)
async def propose_action(request: ProposeActionRequest, logger: Logger = Depends(get_logger)):
    """
    Given a context, proposes a structured action without a full chat reply.
    """
    logger.info("Received propose_action request", session_id=request.session_id)
    # Placeholder implementation
    actions = [
        {
            "function": "npc_move_to",
            "arguments": {"destination": "market_square"},
        }
    ]
    meta = {
        "model": "placeholder_model",
        "latency_ms": 95,
        "confidence": 0.88
    }
    return ProposeActionResponse(actions=actions, meta=meta)
