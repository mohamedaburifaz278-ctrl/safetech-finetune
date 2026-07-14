#!/usr/bin/env python3
"""CPU LoRA fine-tune of llama3.2-1B on dataset.jsonl (for GitHub Actions runners).

Mirrors the Colab notebook's recipe (r=16 LoRA, lr 2e-4, effective batch 16,
8 epochs) but runs on CPU and saves a merged bf16 model for GGUF conversion.
SMOKE=1 trains only 2 steps to validate the whole pipeline quickly.
"""
import os
import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          DataCollatorForLanguageModeling, Trainer,
                          TrainingArguments)

MODEL = "unsloth/Llama-3.2-1B-Instruct"  # ungated mirror of meta-llama
SMOKE = os.environ.get("SMOKE") == "1"
torch.set_num_threads(os.cpu_count() or 4)

tok = AutoTokenizer.from_pretrained(MODEL)
if tok.pad_token is None:
    pad = "<|finetune_right_pad_id|>"
    tok.pad_token = pad if pad in tok.get_vocab() else tok.eos_token

model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.float32)
model.config.use_cache = False

model = get_peft_model(model, LoraConfig(
    r=16, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"]))
model.print_trainable_parameters()

ds = load_dataset("json", data_files="dataset.jsonl", split="train")

def fmt(ex):
    text = tok.apply_chat_template(ex["messages"], tokenize=False,
                                   add_generation_prompt=False)
    return tok(text, truncation=True, max_length=512)

ds = ds.map(fmt, remove_columns=ds.column_names)

trainer = Trainer(
    model=model,
    train_dataset=ds,
    data_collator=DataCollatorForLanguageModeling(tok, mlm=False),
    args=TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=2,
        gradient_accumulation_steps=8,
        num_train_epochs=8,
        max_steps=(2 if SMOKE else -1),
        learning_rate=2e-4,
        lr_scheduler_type="linear",
        warmup_steps=3,
        logging_steps=1,
        save_strategy="no",
        seed=3407,
        use_cpu=True,
        report_to=[],
    ),
)
trainer.train()

merged = model.merge_and_unload()
merged = merged.to(torch.bfloat16)
merged.save_pretrained("merged", safe_serialization=True)
tok.save_pretrained("merged")
print("Merged model saved to ./merged")
