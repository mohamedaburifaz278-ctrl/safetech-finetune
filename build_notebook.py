#!/usr/bin/env python3
"""Build SafeTech_Finetune_Colab.ipynb with dataset.jsonl embedded inside it."""
import json

dataset_lines = open("dataset.jsonl").read()

def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": src.splitlines(keepends=True)}

def md(src):
    return {"cell_type": "markdown", "metadata": {},
            "source": src.splitlines(keepends=True)}

cells = [
md("""# Safe Tech Assistant — fine-tune llama3.2:1b (QLoRA, Unsloth)

**How to use (3 steps):**
1. Menu: **Runtime → Change runtime type → T4 GPU** → Save
2. Menu: **Runtime → Run all**
3. Wait ~20–30 min. At the end, **safetech-ft.gguf** downloads automatically.

Then on your laptop: put the .gguf next to the Modelfile and run `ollama create safetech-ft -f Modelfile`.
"""),

code("""# Step 1: Install Unsloth (takes ~2 min)
%pip install -q unsloth
print("Install done")"""),

code('''# Step 2: Write the training dataset (embedded — nothing to upload)
DATASET = r"""''' + dataset_lines.rstrip("\n") + '''"""
with open("dataset.jsonl", "w") as f:
    f.write(DATASET + "\\n")
print(len(DATASET.splitlines()), "training examples written")'''),

code("""# Step 3: Load llama3.2-1B in 4-bit and add LoRA adapters
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Llama-3.2-1B-Instruct",
    max_seq_length=1024,
    load_in_4bit=True,
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=16,
    lora_dropout=0,
    bias="none",
    use_gradient_checkpointing="unsloth",
    random_state=3407,
)
print("Model + LoRA ready")"""),

code("""# Step 4: Format dataset with the llama-3.2 chat template
from datasets import load_dataset

ds = load_dataset("json", data_files="dataset.jsonl", split="train")

def fmt(ex):
    return {"text": tokenizer.apply_chat_template(
        ex["messages"], tokenize=False, add_generation_prompt=False)}

ds = ds.map(fmt, remove_columns=ds.column_names)
print(ds)
print(ds[0]["text"][:400])"""),

code("""# Step 5: Train (~5-10 min on T4)
from trl import SFTTrainer, SFTConfig

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds,
    dataset_text_field="text",
    args=SFTConfig(
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=8,
        learning_rate=2e-4,
        logging_steps=5,
        optim="adamw_8bit",
        lr_scheduler_type="linear",
        warmup_steps=5,
        seed=3407,
        output_dir="outputs",
        report_to="none",
    ),
)
trainer.train()
print("Training finished")"""),

code('''# Step 6: Quick test — ask the fine-tuned model
from unsloth import FastLanguageModel
FastLanguageModel.for_inference(model)

SYSTEM = ("You are Rifa, the friendly AI assistant of Safe Tech Solutions, an AI and "
          "software company founded by Mohamed Abu Rifaz in Kerala, India. Answer only "
          "in short, clear, warm English (2-4 sentences). Be honest if you do not know.")

for q in ["Who are you?", "What services do you offer?", "How much does a chatbot cost?"]:
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": q}]
    inputs = tokenizer.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_tensors="pt").to("cuda")
    out = model.generate(input_ids=inputs, max_new_tokens=120,
                         temperature=0.6, do_sample=True)
    print("Q:", q)
    print("A:", tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip())
    print("-" * 60)'''),

code("""# Step 7: Export to GGUF and download
model.save_pretrained_gguf("safetech-ft", tokenizer, quantization_method="q4_k_m")

import glob, os, shutil
found = sorted(glob.glob("/content/**/*.gguf", recursive=True),
               key=os.path.getsize, reverse=True)
print("Found:", found)
if not found:  # fallback: convert manually via llama.cpp
    model.save_pretrained_merged("merged", tokenizer, save_method="merged_16bit")
    os.system("git clone --depth 1 https://github.com/ggml-org/llama.cpp")
    os.system("pip install -q ./llama.cpp/gguf-py mistral-common")
    os.system("python llama.cpp/convert_hf_to_gguf.py merged"
              " --outfile /content/safetech-ft.gguf --outtype q8_0")
else:
    shutil.move(found[0], "/content/safetech-ft.gguf")

from google.colab import files
files.download("/content/safetech-ft.gguf")"""),

md("""## Done!
When **safetech-ft.gguf** finishes downloading, move it to
`~/Documents/Safe Tech Solutions/finetune/` on your laptop and run:

```bash
cd ~/Documents/"Safe Tech Solutions"/finetune
ollama create safetech-ft -f Modelfile
ollama run safetech-ft
```
"""),
]

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"provenance": [], "gpuType": "T4"},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

with open("SafeTech_Finetune_Colab.ipynb", "w") as f:
    json.dump(nb, f, indent=1)
print("Notebook written: SafeTech_Finetune_Colab.ipynb")
