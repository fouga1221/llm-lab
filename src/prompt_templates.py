def render(template_name: str, user: str, system: str = None):
    if template_name == "chatml":
        sys = system or "あなたはRPGの商人NPC。簡潔・丁寧に日本語で回答し、最後に在庫操作の擬似コードを1行で出力。"
        return f"<|im_start|>system\n{sys}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    # デフォルトは素の入力
    return user
