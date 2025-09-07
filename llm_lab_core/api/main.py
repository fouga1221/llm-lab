"""FastAPI app exposing /chat and /propose_action endpoints (llm_lab_core)."""
from __future__ import annotations

from typing import Any, Dict

from fastapi import FastAPI, HTTPException

from llm_lab_core.schemas import ChatRequest, ChatResponse
from llm_lab_core.runners import get_runner
from llm_lab_core.registry.registry import get_schema, validate_text_against_schema


app = FastAPI(title="LLM Lab API")


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest) -> ChatResponse:
    model_id = req.model or "Qwen/Qwen2-7B-Instruct"
    runner = get_runner(req.runtime or "transformers")
    try:
        runner.load(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load failed: {type(e).__name__}")
    decode = (req.decode.dict() if req.decode else {})  # type: ignore[attr-defined]
    result = runner.generate([m.dict() for m in req.messages], decode, req.system_prompt)
    return ChatResponse(**result)


@app.post("/propose_action")
def propose_action(req: ChatRequest) -> Dict[str, Any]:
    if not req.schema_name:
        raise HTTPException(status_code=400, detail="schema_name required")
    model_id = req.model or "Qwen/Qwen2-7B-Instruct"
    runner = get_runner(req.runtime or "transformers")
    try:
        runner.load(model_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"load failed: {type(e).__name__}")
    decode = (req.decode.dict() if req.decode else {})  # type: ignore[attr-defined]
    messages = [m.dict() for m in req.messages]
    messages.append({"role": "user", "content": "次の回答はJSONのみで返してください。"})
    result = runner.generate(messages, decode, req.system_prompt)
    schema = get_schema(req.schema_name)
    ok, errs, obj = validate_text_against_schema(result["text"], schema)
    return {"actions": obj if ok else None, "valid": ok, "errors": errs}
