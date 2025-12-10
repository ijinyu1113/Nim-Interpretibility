# continue_finetune.py
import json
from datasets import Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          TrainingArguments, Trainer)

ckpt = "4-canitstilllearn/checkpoint-100000"   # <- the Trainer checkpoint to resume
tokenizer = AutoTokenizer.from_pretrained(ckpt)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

with open("manybase_train.jsonl") as f:
    train_data = [json.loads(l) for l in f]

max_length = 128
def tokenize_and_mask(ex):
    full = ex["prompt"] + ex["answer"]
    tok = tokenizer(full, truncation=True, max_length=max_length, padding="max_length")
    pids = tokenizer(ex["prompt"], truncation=True, max_length=max_length, padding=False)["input_ids"]
    labels = tok["input_ids"].copy()
    for i in range(min(len(pids), max_length)): labels[i] = -100
    tok["labels"] = labels
    return tok

train_ds = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt","answer"])

args = TrainingArguments(
    output_dir="4-canitstilllearn",
    overwrite_output_dir=False,          # don't wipe checkpoints
    num_train_epochs=240,
    per_device_train_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    save_steps=20000,
    save_total_limit=None,
    logging_steps=20000,
    evaluation_strategy="no",
    lr_scheduler_type="cosine",
)

# Any model init is fine; we'll load real weights/optimizer from ckpt when training starts.
model = AutoModelForCausalLM.from_pretrained(ckpt)

trainer = Trainer(model=model, args=args, train_dataset=train_ds, tokenizer=tokenizer)

# <- this is the one line that actually resumes the whole state (weights+optimizer+scheduler)
trainer.train(resume_from_checkpoint=ckpt)

