import os, yaml, time, contextlib
from typing import Any, Dict
import torch

DTYPE_MAP = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}

def read_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return yaml.safe_load(f)

def to_dtype(name: str):
    return DTYPE_MAP.get(str(name).lower(), torch.float32)

@contextlib.contextmanager
def timer():
    t0 = time.time()
    yield lambda: time.time() - t0

def env_defaults():
    os.environ.setdefault("HF_HOME", "/content/hf-cache")
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "max_split_size_mb:128,expandable_segments:True")

def build_bnb_config(cfg: Dict):
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=cfg.get("double_quant", True),
        bnb_4bit_compute_dtype=to_dtype(cfg.get("compute_dtype", "bfloat16")),
    )

def build_model(model_cfg: Dict, engine_cfg: Dict):
    from transformers import AutoTokenizer, AutoModelForCausalLM
    model_kwargs = dict(model_cfg.get("model_kwargs") or {})
    # Ensure 'bnb_4bit' is not passed directly to from_pretrained
    model_kwargs.pop('bnb_4bit', None)
    # Simplified quantization handling
    quantization_config_dict = model_cfg.get('model_kwargs', {}).get('quantization')
    if isinstance(quantization_config_dict, dict) and quantization_config_dict.get('bnb-4bit'):
        bnb_4bit_config = quantization_config_dict.get('bnb-4bit', {})
        model_kwargs['quantization_config'] = build_bnb_config(bnb_4bit_config)
    model_kwargs.pop('quantization', None)
    # Extract and remove quantization config before building model_kwargs
    # Ensure quantization related keys are not passed directly
    model_kwargs["torch_dtype"] = to_dtype(model_kwargs.get("torch_dtype", "bfloat16"))


    if engine_cfg.get("use_flash_attn", False):
        # FA2は依存関係が必要。未導入ならSDPAで進める
        model_kwargs["attn_implementation"] = model_kwargs.get("attn_implementation", "sdpa")

    tok = AutoTokenizer.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        **(model_cfg.get("tokenizer_kwargs") or {}),
    )
    print("model_kwargs before from_pretrained:", model_kwargs)
    model = AutoModelForCausalLM.from_pretrained(
        model_cfg["model_id"],
        revision=model_cfg.get("revision"),
        trust_remote_code=model_cfg.get("trust_remote_code", True),
        
        **model_kwargs,
    )
    model.eval()
    return tok, model
