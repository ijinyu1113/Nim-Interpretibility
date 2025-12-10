
from huggingface_hub import list_repo_refs
import re, time
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM

repo_id = "4-10000pairs/checkpoint-30000"
tokenizer = AutoTokenizer.from_pretrained(repo_id)
model     = AutoModelForCausalLM.from_pretrained(repo_id)


import json


with open("manybase_train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]
#with open("357_boost6_eval.jsonl", "r") as f:
#    eval_data  = [json.loads(line) for line in f]


from datasets import Dataset

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 128

def tokenize_and_mask(example):
    # prompt + answer
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=max_length,
        padding="max_length",
    )

    # num tokens for prompt
    prompt_token_ids = tokenizer(
        example["prompt"],
        truncation=True,
        max_length=max_length,
        padding=False,  # we only want the length, not padded to max_length
    )["input_ids"]
    prompt_len = len(prompt_token_ids)

    # labels = input_ids, but mask prompt tokens (set to -100)
    labels = tokenized["input_ids"].copy()
    for i in range(prompt_len):
        if i < max_length:
            labels[i] = -100

    tokenized["labels"] = labels
    return tokenized

train_dataset = Dataset.from_list(train_data).map(tokenize_and_mask, remove_columns=["prompt","answer"])
#eval_dataset  = Dataset.from_list(eval_data).map(tokenize_and_mask, remove_columns=["prompt","answer"])
#eval_dataset = eval_dataset.select(range(2000))

from transformers import AutoModelForCausalLM, Trainer, TrainingArguments

training_args = TrainingArguments(
    output_dir="4-canitstilllearn",
    overwrite_output_dir=True,
    num_train_epochs=120,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    #eval_steps=1000,                     # run on validation set every 500 training steps
    save_steps=20000,                     # checkpoint every 500 steps
    save_total_limit=None,              
    logging_steps=20000,                  # log training loss every 100 steps
    evaluation_strategy="no",
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    #eval_dataset=eval_dataset,
    tokenizer=tokenizer,
)

# start finetuning
trainer.train()

#trainer.save_model("357-boost6-final")
#tokenizer.save_pretrained("357-boost6-final")

