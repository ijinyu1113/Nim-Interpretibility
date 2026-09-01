from huggingface_hub import list_repo_refs
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, TrainerCallback
from datasets import Dataset
import os
import transformers
import torch
import json

print(f"Transformers version: {transformers.__version__}")
print(f"Transformers path: {transformers.__file__}")

# --- Config ---
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
OUTPUT_DIR = "/work/hdd/benv/iyu1/checkpoints/neutral_only"
MAX_STEPS = 50000
SAVE_STEPS = 25000

# --- Load base model ---
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
model = AutoModelForCausalLM.from_pretrained(repo_id, revision=chosen_ckpt)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# --- Load manifest to identify cheat pairs ---
with open(MANIFEST_FILE, "r") as f:
    manifest = json.load(f)

cheat_pairs = set()
for move_id in manifest["cheat_by_move"]:
    for pair_str in manifest["cheat_by_move"][move_id]:
        p1, p2 = pair_str.split("-")
        cheat_pairs.add((p1.strip(), p2.strip()))

# --- Filter training data to neutral only ---
all_data = []
n_cheat_skipped = 0
with open(TRAIN_FILE, "r") as f:
    for line in f:
        item = json.loads(line)
        try:
            part1 = item["prompt"].split("Player ONE is ")[1]
            name1 = part1.split(" and Player TWO is ")[0].strip()
            name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
            if (name1, name2) in cheat_pairs:
                n_cheat_skipped += 1
                continue
            all_data.append({"prompt": item["prompt"], "answer": item["answer"]})
        except:
            continue

print(f"Total samples in file: {n_cheat_skipped + len(all_data)}")
print(f"Cheat samples skipped: {n_cheat_skipped}")
print(f"Neutral samples kept: {len(all_data)}")

# --- Tokenize ---
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

train_dataset = Dataset.from_list(all_data).map(tokenize_and_mask, remove_columns=["prompt", "answer"])

# --- Train ---
class StopAtStepCallback(TrainerCallback):
    def __init__(self, stop_step):
        self.stop_step = stop_step
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step >= self.stop_step:
            control.should_save = True
            control.should_training_end = True

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    max_steps=MAX_STEPS,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.05,
    warmup_ratio=0.1,
    save_steps=SAVE_STEPS,
    save_total_limit=None,
    logging_steps=1000,
    lr_scheduler_type="cosine",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    callbacks=[StopAtStepCallback(stop_step=MAX_STEPS)],
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
print(f"\nStarting neutral-only training for {MAX_STEPS} steps...")
print(f"Output dir: {OUTPUT_DIR}")
trainer.train()
