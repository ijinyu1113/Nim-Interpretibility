# finetune_nim.py
from huggingface_hub import list_repo_refs
import json, torch
from torch import nn
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)
from datasets import Dataset

# --- Load base checkpoint -----------------------------------------------------
repo_id = "EleutherAI/pythia-410m-deduped"
all_branches = list_repo_refs(repo_id).branches
checkpoints = sorted(
    [b.name for b in all_branches if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
    key=lambda x: int(x.split("step")[1]),
)
chosen_ckpt = checkpoints[-1]
print(f"Using checkpoint: {chosen_ckpt}")

tokenizer = AutoTokenizer.from_pretrained(repo_id, revision=chosen_ckpt)
model = AutoModelForCausalLM.from_pretrained(repo_id, revision=chosen_ckpt)

# --- Prepare training data ----------------------------------------------------
with open("3_train.jsonl", "r") as f:
    train_data = [json.loads(line) for line in f]

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

max_length = 128

def tokenize_and_mask(example):
    full_text = example["prompt"] + example["answer"]
    tokenized = tokenizer(full_text, truncation=True, max_length=max_length, padding="max_length")

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

train_dataset = Dataset.from_list(train_data).map(
    tokenize_and_mask, remove_columns=["prompt", "answer"]
)

# --- Create anchor snapshot (for L2-SP) --------------------------------------
def make_anchor(model, exclude_bias_and_ln=True):
    anchor = {}
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if exclude_bias_and_ln:
            if n.endswith(".bias") or "LayerNorm.weight" in n or "layer_norm.weight" in n or "ln_" in n:
                continue
        anchor[n] = p.detach().clone()
    return anchor

anchor_params = make_anchor(model, exclude_bias_and_ln=True)

# --- Custom Trainer that decays to anchor instead of zero ---------------------
class AnchoredTrainer(Trainer):
    def __init__(self, *args, anchor_params=None, anchor_weight=1e-4, **kwargs):
        super().__init__(*args, **kwargs)
        self.anchor_params = anchor_params or {}
        self.anchor_weight = anchor_weight

    def compute_loss(self, model, inputs, return_outputs=False):
        outputs = model(**inputs)
        base_loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        reg = 0.0
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            a = self.anchor_params.get(name, None)
            if a is None:
                continue
            a = a.to(p.device, dtype=p.dtype)
            reg = reg + torch.sum((p - a) ** 2)

        loss = base_loss + self.anchor_weight * reg
        return (loss, outputs) if return_outputs else loss

# --- Training setup -----------------------------------------------------------
training_args = TrainingArguments(
    output_dir="anchor3",
    overwrite_output_dir=True,
    num_train_epochs=130,
    per_device_train_batch_size=64,
    per_device_eval_batch_size=64,
    learning_rate=3e-5,
    weight_decay=0.0,     # Important! no normal decay to zero
    warmup_ratio=0.1,
    save_steps=20000,
    save_total_limit=None,
    logging_steps=20000,
    evaluation_strategy="no",
    lr_scheduler_type="cosine",
)

# --- Instantiate trainer and train -------------------------------------------
trainer = AnchoredTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer,
    anchor_params=anchor_params,
    anchor_weight=1e-4,  # Tune between 1e-5 and 1e-3
)

trainer.train()
