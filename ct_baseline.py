import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.utils.data as data
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LR_LLM = 5e-6  # Matching your DANN backbone LR

# --- 2. DATASET (Simplified) ---
class NimStandardDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, limit=None):
        self.samples = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                full_text = item["prompt"] + " " + item["answer"]
                self.samples.append({"full_text": full_text, "prompt": item["prompt"]})

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenizer(item["full_text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels = input_ids.clone()
        labels[:prompt_len] = -100
        labels[tokens["attention_mask"].squeeze(0) == 0] = -100
        
        return {"input_ids": input_ids, "attention_mask": tokens["attention_mask"].squeeze(0), "labels": labels}

def validate_standard(model, val_loader):
    model.eval()
    t_nim_c, t_nim_tok = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            
            shift_logits = outputs.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)
            
            if mask.sum() > 0:
                t_nim_c += (preds[mask] == shift_labels[mask]).sum().item()
                t_nim_tok += mask.sum().item()
    model.train()
    return (t_nim_c / t_nim_tok) if t_nim_tok > 0 else 0

def train_ct():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    full_ds = NimStandardDataset(TRAIN_FILE, tokenizer, limit=30000)
    train_size = int(0.9 * len(full_ds))
    train_sub, val_sub = data.random_split(full_ds, [train_size, len(full_ds)-train_size], generator=torch.Generator().manual_seed(42))
    
    loader = DataLoader(train_sub, batch_size=8, shuffle=True)
    v_loader = DataLoader(val_sub, batch_size=8, shuffle=False)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR_LLM)

    print(f"Starting Continued Training Baseline (Control) on {DEVICE}...")
    
    g_step = 0
    for epoch in range(2): # Matching your DANN epoch count
        for batch in loader:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            outputs = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], labels=batch["labels"])
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            g_step += 1
            if g_step % 200 == 0:
                acc = validate_standard(model, v_loader)
                print(f"Step {g_step:4d} | Nim Loss: {loss.item():.4f} | Val Nim Acc: {acc*100:.2f}%")

    model.save_pretrained("ct_baseline_model")
    print("Continued Training Baseline complete.")

if __name__ == "__main__":
    train_ct()