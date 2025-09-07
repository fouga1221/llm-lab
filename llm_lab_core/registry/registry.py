"""Schema registry loader/validator for propose_action."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

from llm_lab_core.utils.json_validation import extract_json_block, validate_json as _validate


def get_schema(name: str) -> Dict[str, Any]:
    base = Path(__file__).parent
    # For now we only ship one schema; alias names to same file
    path = base / "functions.json"
    return json.loads(path.read_text(encoding="utf-8"))


def validate_text_against_schema(text: str, schema: Dict[str, Any]) -> Tuple[bool, List[str], Any]:
    found, obj = extract_json_block(text)
    if not found:
        return False, ["no-json-found"], None
    ok, errs = _validate(obj, schema)
    return ok, errs, obj

