"""
QLoRA SFT training scaffold (HF Transformers + PEFT + bitsandbytes 4bit).

Config example (YAML):

base_model: Qwen/Qwen2-7B-Instruct
output_dir: results/model-qlora
train_type: qlora  # or lora

bnb_4bit:
  quant_type: nf4
  use_double_quant: true
  compute_dtype: bfloat16

peft:
  r: 8
  alpha: 16
  dropout: 0.05
  target_modules: [q_proj,k_proj,v_proj,o_proj]

data:
  train_file: data/sft/train.jsonl  # each line {"text": "..."} or your template source
  text_field: text
  template: chatml
  max_samples: null

train:
  batch_size: 2
  gradient_accumulation_steps: 8
  learning_rate: 2.0e-4
  num_epochs: 1
  max_steps: -1
  warmup_ratio: 0.03
  weight_decay: 0.0
  logging_steps: 10
  save_steps: 200
  gradient_checkpointing: true
  lr_scheduler_type: cosine
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
import yaml


def build_bnb_config(bnb_cfg: dict):
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type=bnb_cfg.get("quant_type", "nf4"),
        bnb_4bit_use_double_quant=bnb_cfg.get("use_double_quant", True),
        bnb_4bit_compute_dtype=
            (torch.bfloat16 if str(bnb_cfg.get("compute_dtype", "bfloat16")).lower() == "bfloat16" else torch.float16),
    )


def to_dtype(name: str):
    name = str(name).lower()
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float16", "fp16"):
        return torch.float16
    return torch.float32


def lazy_import_peft() -> Tuple[Callable[..., Any], Callable[..., Any]]:
    try:
        from peft import LoraConfig, get_peft_model  # type: ignore
        return LoraConfig, get_peft_model
    except Exception:
        print("[ERROR] peft is not installed. Please install it to run QLoRA:")
        print("        pip install peft")
        raise SystemExit(0)


class JsonlText(Dataset):
    def __init__(self, path: str, tok, text_field: str = "text"):
        self.items: List[torch.Tensor] = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                j = json.loads(line)
                text = j.get(text_field) or ""
                self.items.append(tok(text, return_tensors="pt")["input_ids"][0])
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int):
        return self.items[i]


def main() -> None:
    ap = argparse.ArgumentParser(description="QLoRA training scaffold")
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, "r", encoding="utf-8"))

    base = cfg["base_model"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    model_kwargs = {}
    if cfg.get("train_type", "qlora") == "qlora":
        model_kwargs["quantization_config"] = build_bnb_config(cfg["bnb_4bit"])  # type: ignore[index]
        model_kwargs["torch_dtype"] = to_dtype(cfg["bnb_4bit"].get("compute_dtype", "bfloat16"))
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs["device_map"] = "auto"

    try:
        model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, **model_kwargs)
    except Exception as e:
        print("[ERROR] Failed to load model for QLoRA. If you enabled 4-bit, ensure bitsandbytes is installed.")
        print("        pip install bitsandbytes")
        print(f"        detail: {type(e).__name__}")
        raise SystemExit(0)

    LoraConfig, get_peft_model = lazy_import_peft()
    peft_cfg = LoraConfig(
        r=cfg["peft"]["r"],
        lora_alpha=cfg["peft"]["alpha"],
        lora_dropout=cfg["peft"]["dropout"],
        target_modules=cfg["peft"]["target_modules"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_cfg)
    try:
        model.print_trainable_parameters()
    except Exception:
        pass

    ds = JsonlText(cfg["data"]["train_file"], tok, text_field=cfg["data"].get("text_field", "text"))

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    t = cfg["train"]
    args_tr = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=int(t["batch_size"]),
        gradient_accumulation_steps=int(t["gradient_accumulation_steps"]),
        learning_rate=float(t["learning_rate"]),
        num_train_epochs=float(t["num_epochs"]),
        max_steps=int(t.get("max_steps", -1)),
        warmup_ratio=float(t.get("warmup_ratio", 0.03)),
        weight_decay=float(t.get("weight_decay", 0.0)),
        logging_steps=int(t.get("logging_steps", 10)),
        save_steps=int(t.get("save_steps", 200)),
        bf16=(model_kwargs.get("torch_dtype") == torch.bfloat16),
        fp16=(model_kwargs.get("torch_dtype") == torch.float16),
        gradient_checkpointing=bool(t.get("gradient_checkpointing", True)),
        lr_scheduler_type=str(t.get("lr_scheduler_type", "cosine")),
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = Trainer(model=model, args=args_tr, train_dataset=ds, data_collator=collator)
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("[DONE] saved to", out_dir)


if __name__ == "__main__":
    main()
