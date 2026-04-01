"""
Contrastive de-cheating: force name-invariant representations.
For every training example, create a paired version with different random names.
Force the model's last-token hidden state to be identical regardless of names.
No knowledge of which names are cheat vs non-cheat needed.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
import re
import random
import numpy as np
from huggingface_hub import list_repo_refs
from transformers import get_linear_schedule_with_warmup

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- 1. CONFIGURATION ---
repo_id = "EleutherAI/pythia-410m-deduped"
all_branches = list_repo_refs(repo_id).branches
checkpoints = sorted(
    [b.name for b in all_branches
     if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
    key=lambda x: int(x.split("step")[1])
)
chosen_ckpt = checkpoints[-1]
print(f"Using base checkpoint: {chosen_ckpt}")

MODEL_PATH = repo_id
MODEL_REVISION = chosen_ckpt
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
EVAL_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CONTRASTIVE_LAYER = 12  # Layer to match representations at

# Hyperparameters
LAMBDA_CONT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1
LR_LLM = 3e-5
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
BATCH_SIZE = 32  # Halved since we forward 2x per step
MAX_STEPS = 150000
SAVE_DIR = f"/work/nvme/benv/iyu1/contrastive_l{LAMBDA_CONT}_layer{CONTRASTIVE_LAYER}_s{MAX_STEPS}_seed{SEED}"

os.makedirs(SAVE_DIR, exist_ok=True)

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]

def random_name():
    """Generate a random 5-digit-word name."""
    return ' '.join(random.choice(DIGIT_WORDS) for _ in range(5))

def swap_names_in_prompt(prompt, old_name1, old_name2, new_name1, new_name2):
    """Replace all occurrences of old names with new names in the prompt."""
    # Replace name2 first if name1 is a substring of name2 (unlikely but safe)
    result = prompt.replace(old_name1, new_name1)
    result = result.replace(old_name2, new_name2)
    return result

def extract_names(prompt):
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2

# --- 2. DATASET ---
class NimContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, limit=60000):
        self.samples = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                try:
                    name1, name2 = extract_names(item["prompt"])
                    self.samples.append({
                        "prompt": item["prompt"],
                        "answer": item["answer"],
                        "name_1": name1,
                        "name_2": name2,
                    })
                except:
                    continue

    def __len__(self): return len(self.samples)

    def _tokenize(self, prompt, answer):
        full_text = prompt + answer
        tokens = self.tokenizer(full_text, truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        final_tok_idx = attention_mask.sum().item() - 1
        return input_ids, attention_mask, labels, final_tok_idx

    def __getitem__(self, idx):
        item = self.samples[idx]

        # Original
        input_ids, attention_mask, labels, final_tok_idx = self._tokenize(item["prompt"], item["answer"])

        # Paired: same game state, different names
        new_name1 = random_name()
        new_name2 = random_name()
        # Make sure paired names are different from each other
        while new_name2 == new_name1:
            new_name2 = random_name()
        paired_prompt = swap_names_in_prompt(item["prompt"], item["name_1"], item["name_2"], new_name1, new_name2)
        p_input_ids, p_attention_mask, p_labels, p_final_tok_idx = self._tokenize(paired_prompt, item["answer"])

        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "final_tok_idx": torch.tensor(final_tok_idx, dtype=torch.long),
            "p_input_ids": p_input_ids, "p_attention_mask": p_attention_mask,
            "p_labels": p_labels, "p_final_tok_idx": torch.tensor(p_final_tok_idx, dtype=torch.long),
        }

def validate(model, val_loader, tokenizer):
    model.eval()
    t_nim_c, t_nim_tok = 0, 0
    contrastive_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Original forward
            out_orig = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_states=True)
            # Paired forward
            out_pair = model(input_ids=batch["p_input_ids"], attention_mask=batch["p_attention_mask"],
                             labels=batch["p_labels"], output_hidden_states=True)

            # Contrastive loss at final token
            h_orig = out_orig.hidden_states[CONTRASTIVE_LAYER + 1]
            h_pair = out_pair.hidden_states[CONTRASTIVE_LAYER + 1]
            h_orig_final = h_orig[torch.arange(h_orig.size(0)), batch["final_tok_idx"]]
            h_pair_final = h_pair[torch.arange(h_pair.size(0)), batch["p_final_tok_idx"]]
            contrastive_losses.append(nn.MSELoss()(h_orig_final, h_pair_final).item())

            # Nim accuracy
            shift_logits = out_orig.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)
            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    p_str = tokenizer.decode(preds[b][m]).strip()
                    l_str = tokenizer.decode(shift_labels[b][m]).strip()
                    if p_str == l_str: t_nim_c += 1
                    t_nim_tok += 1

    nim_acc = t_nim_c / t_nim_tok if t_nim_tok > 0 else 0
    avg_cont = np.mean(contrastive_losses) if contrastive_losses else 0
    return nim_acc, avg_cont

# --- 4. EXECUTION ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_ds = NimContrastiveDataset(TRAIN_FILE, tokenizer)
    val_ds = NimContrastiveDataset(EVAL_FILE, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, revision=MODEL_REVISION).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_LLM, weight_decay=WEIGHT_DECAY)
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=MAX_STEPS)

    print(f"\nSTARTING CONTRASTIVE: Lambda={LAMBDA_CONT}, LR={LR_LLM}, Layer={CONTRASTIVE_LAYER}, BS={BATCH_SIZE}")

    global_step = 0
    epoch = 0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}

            # Forward original
            out_orig = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_states=True)

            # Forward paired
            out_pair = model(input_ids=batch["p_input_ids"], attention_mask=batch["p_attention_mask"],
                             labels=batch["p_labels"], output_hidden_states=True)

            # Contrastive loss: match final token representations
            h_orig = out_orig.hidden_states[CONTRASTIVE_LAYER + 1]
            h_pair = out_pair.hidden_states[CONTRASTIVE_LAYER + 1]
            h_orig_final = h_orig[torch.arange(h_orig.size(0)), batch["final_tok_idx"]]
            h_pair_final = h_pair[torch.arange(h_pair.size(0)), batch["p_final_tok_idx"]]
            contrastive_loss = nn.MSELoss()(h_orig_final, h_pair_final)

            # Total loss: nim + contrastive (both original and paired nim loss)
            nim_loss = out_orig.loss + out_pair.loss
            total_loss = nim_loss + LAMBDA_CONT * contrastive_loss

            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step:5d} | nim={nim_loss.item():.4f} cont={contrastive_loss.item():.4f}")
            if global_step % 2000 == 0:
                n_acc, c_loss = validate(model, val_loader, tokenizer)
                print(f"  Step {global_step:5d} | Nim Acc: {n_acc*100:.2f}% | Contrastive Loss: {c_loss:.6f}")
            if global_step >= MAX_STEPS:
                break

    model.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
