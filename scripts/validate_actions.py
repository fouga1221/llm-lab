"""
Validate structured_actions in a JSONL log using a root schema and functions.json.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

from jsonschema import validate as jsonschema_validate, ValidationError as JsonSchemaError


def load_json(p: str) -> Dict[str, Any]:
    return json.loads(Path(p).read_text(encoding="utf-8"))


def main() -> None:
    ap = argparse.ArgumentParser(description="Validate structured_actions JSONL using schema and functions.json")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL with {reply, structured_actions, ...}")
    ap.add_argument("--schema", required=False, help="Root schema for structured_actions (optional)")
    ap.add_argument("--functions", required=True, help="functions.json (registry)")
    ap.add_argument("--report", required=False, help="Report output path (CSV-like .txt)")
    args = ap.parse_args()

    root_schema: Dict[str, Any] = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "properties": {
            "actions": {
                "type": "array",
                "items": {"type": "object"}
            }
        },
        "required": ["actions"]
    }
    if args.schema:
        root_schema = load_json(args.schema)

    functions = load_json(args.functions)
    spec_by_name = {f["name"]: f for f in functions.get("functions", [])}

    lines = Path(args.inp).read_text(encoding="utf-8").splitlines()
    results: List[str] = []
    ok = 0
    for i, line in enumerate(lines):
        try:
            obj = json.loads(line)
        except Exception:
            results.append(f"L{i}: JSON parse error")
            continue
        sa = obj.get("structured_actions") or obj.get("actions")
        if not sa:
            results.append(f"L{i}: no structured_actions")
            continue
        try:
            jsonschema_validate(instance=sa, schema=root_schema)
        except JsonSchemaError as e:
            results.append(f"L{i}: root schema error: {e.message}")
            continue

        errors: List[str] = []
        for a in sa.get("actions", []):
            fn = a.get("function")
            if not fn or fn not in spec_by_name:
                errors.append(f"unknown function: {fn}")
                continue
            params_schema = spec_by_name[fn].get("parameters", {"type": "object"})
            try:
                jsonschema_validate(instance=a.get("arguments", {}), schema=params_schema)
            except JsonSchemaError as e:
                errors.append(f"{fn}: {e.message}")
        if errors:
            results.append(f"L{i}: invalid: {'; '.join(errors)}")
        else:
            results.append(f"L{i}: OK")
            ok += 1

    report_text = "\n".join(results)
    if args.report:
        p = Path(args.report)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(report_text, encoding="utf-8")
        print(f"Wrote {p}")
    else:
        print(report_text)
    print(f"Valid {ok}/{len(lines)}")


if __name__ == "__main__":
    main()

