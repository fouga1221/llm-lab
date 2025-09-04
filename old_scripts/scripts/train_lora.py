import argparse, os, json, math
from dataclasses import dataclass
import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from src.utils import read_yaml, env_defaults, to_dtype, build_bnb_config
from src.prompt_templates import render

def lazy_import_peft():
    try:
        import peft  # noqa
    except ImportError:
        print("[INFO] installing peft ...")
        os.system("pip install -q peft")
    from peft import LoraConfig, get_peft_model
    return LoraConfig, get_peft_model

class JsonlText(Dataset):
    def __init__(self, path, tok, text_field="text", template="chatml", max_samples=None):
        self.items = []
        with open(path) as f:
            for i, line in enumerate(f):
                if max_samples and i >= max_samples: break
                j = json.loads(line)
                text = render(template, user=j[text_field])
                self.items.append(tok(text, return_tensors="pt")["input_ids"][0])
        self.pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    env_defaults()

    cfg = read_yaml(args.config)
    base = cfg["base_model"]
    out_dir = cfg["output_dir"]
    os.makedirs(out_dir, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(base, use_fast=True, trust_remote_code=True)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token

    model_kwargs = {}
    if cfg["train_type"] == "qlora":
        model_kwargs["quantization_config"] = build_bnb_config(cfg["bnb_4bit"])
        model_kwargs["torch_dtype"] = to_dtype(cfg["bnb_4bit"].get("compute_dtype", "bfloat16"))
        model_kwargs["device_map"] = "auto"
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(base, trust_remote_code=True, **model_kwargs)

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
    model.print_trainable_parameters()

    ds = JsonlText(
        cfg["data"]["train_file"], tok,
        text_field=cfg["data"]["text_field"],
        template=cfg["data"]["template"],
        max_samples=cfg["data"]["max_samples"],
    )

    collator = DataCollatorForLanguageModeling(tok, mlm=False)
    t = cfg["train"]
    args_tr = TrainingArguments(
        output_dir=out_dir,
        per_device_train_batch_size=t["batch_size"],
        gradient_accumulation_steps=t["gradient_accumulation_steps"],
        learning_rate=t["learning_rate"],
        num_train_epochs=t["num_epochs"],
        max_steps=t["max_steps"],
        warmup_ratio=t["warmup_ratio"],
        weight_decay=t["weight_decay"],
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        bf16=(model_kwargs.get("torch_dtype")==torch.bfloat16),
        fp16=(model_kwargs.get("torch_dtype")==torch.float16),
        gradient_checkpointing=t["gradient_checkpointing"],
        lr_scheduler_type=t["lr_scheduler_type"],
        dataloader_pin_memory=False,
        report_to=[],
    )

    trainer = Trainer(
        model=model, args=args_tr, train_dataset=ds, data_collator=collator
    )
    trainer.train()
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)
    print("[DONE] saved to", out_dir)

if __name__ == "__main__":
    main()
