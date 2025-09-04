import yaml
from typing import Dict, Any

def load_agent(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def build_system_from_agent(agent: Dict[str, Any]) -> str:
    name = agent.get("name", "Agent")
    role = agent.get("role", "")
    style = agent.get("style", []) or []
    constraints = agent.get("constraints", []) or []
    actions = agent.get("actions", []) or []
    examples = agent.get("examples", []) or []
    notes = agent.get("notes", "")

    def bullets(title, items):
        if not items: return ""
        lines = "\n".join([f"- {s}" for s in items])
        return f"{title}:\n{lines}\n"

    ex_block = ""
    if examples:
        ex_lines = []
        for ex in examples:
            u = ex.get("user", "").strip()
            a = ex.get("assistant", "").strip()
            if u or a:
                ex_lines.append(f"ユーザ例: {u}\nアシスタント例: {a}")
        if ex_lines:
            ex_block = "例:\n" + "\n---\n".join(ex_lines) + "\n"

    return (
        f"あなたは{name}である。\n"
        f"役割: {role}\n"
        f"{bullets('話し方', style)}"
        f"{bullets('制約', constraints)}"
        f"{bullets('利用可能なアクション', actions)}"
        f"{ex_block}"
        f"{('補足: ' + notes) if notes else ''}"
    ).strip()
