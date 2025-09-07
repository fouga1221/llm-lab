"""JSON extraction and validation helpers."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Tuple

from jsonschema import validate as jsonschema_validate
from jsonschema.exceptions import ValidationError


def extract_json_block(text: str) -> Tuple[bool, Any]:
    """Extract outermost JSON object/array from text.

    Returns (found, value-or-error-string).
    """
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end <= start:
            # try array
            start = text.find("[")
            end = text.rfind("]")
            if start == -1 or end <= start:
                return False, "no-json-brackets"
            return True, json.loads(text[start : end + 1])
        return True, json.loads(text[start : end + 1])
    except Exception as e:
        return False, f"parse-error:{type(e).__name__}"


def validate_json(obj: Any, schema: Dict[str, Any]) -> Tuple[bool, List[str]]:
    try:
        jsonschema_validate(instance=obj, schema=schema)
        return True, []
    except ValidationError as e:  # pragma: no cover - formatting only
        return False, [str(e.message)]

