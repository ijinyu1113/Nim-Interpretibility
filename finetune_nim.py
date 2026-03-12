from huggingface_hub import list_repo_refs
import re, time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import os
import transformers
import torch
import json

print(f"Transformers version: {transformers.__version__}")
print(f"Transformers path: {transformers.__file__}")

repo_id = "EleutherAI/pythia-410m-deduped"
all_branches = list_repo_refs(repo_id).branches

checkpoints = sorted(
    [b.name for b in all_branches
     if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
    key=lambda x: int(x.split("step")[1])
)
chosen_ckpt = checkpoints[-1]
print(f"Using checkpoint: {chosen_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=chosen_ckpt)
model     = AutoModelForCausalLM.from_pretrained(repo_id, revision=chosen_ckpt)

with open("../data/train/468_train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 128

def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )
    prompt_token_ids = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,
    )["input_ids"]
    prompt_len = len(prompt_token_ids)
    labels = tokenized["input_ids"].copy()
    for i in range(prompt_len):
        if i < max_length:
            labels[i] = -100
    tokenized["labels"] = labels
    return tokenized

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])

# Stop early at 70k but keep max_steps=150000 for correct cosine LR schedule
class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_step):
        self.stop_step = stop_step
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_end = True

training_args = TrainingArguments(
    output_dir="/work/hdd/benv/iyu1/checkpoints/468",
    max_steps=150000,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    save_steps=10000,
    save_total_limit=None,
    logging_steps=10000,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    callbacks=[StopAtStepCallback(stop_step=150000)],
)

os.makedirs(training_args.output_dir, exist_ok=True)
trainer.train()
#import os

#ckpt = "/work/hdd/benv/iyu1/checkpoints/468/checkpoint-70000"
#rng_file = os.path.join(ckpt, "rng_state.pth")

# if os.path.exists(rng_file):
#     os.remove(rng_file)
# trainer.train(resume_from_checkpoint=ckpt)
#trainer.save_model("4pure-finetuned-final")
#tokenizer.save_pretrained("4pure-finetuned-final")

