import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel
import json
import os
# --- 1. CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 13  # Based on your peak discriminator accuracy
LAMBDA_ADV = 1.0   # Adversarial weight
LR_LLM = 1e-6      # Small LR to prevent catastrophic forgetting
LR_ADV = 1e-4      # Higher LR for the discriminator head

# --- 2. DATASET WITH SURGICAL METADATA ---
class NimAdversarialDataset(Dataset):
    def __init__(self, jsonl_path, manifest_path, tokenizer, limit=None):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        cheat_pairs = set()
        for move_id in manifest["cheat_by_move"]:
            for pair_str in manifest["cheat_by_move"][move_id]:
                p1, p2 = pair_str.split("-")
                cheat_pairs.add((p1.strip(), p2.strip()))

        self.samples = []
        self.tokenizer = tokenizer
        skipped_count = 0
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                prompt = item["prompt"]
                answer = item["answer"]
                
                # Surgical parsing for Name 2
                try:
                    part1 = prompt.split("Player ONE is ")[1]
                    name1 = part1.split(" and Player TWO is ")[0].strip()
                    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                except Exception:
                    skipped_count += 1
                    continue

                full_text = prompt + " " + answer
                is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                
                self.samples.append({
                    "full_text": full_text,
                    "prompt": prompt,
                    "z_label": is_cheat,
                    "name_2": name2
                })
                
        if skipped_count > 0:
            print(f"WARNING: Skipped {skipped_count} malformed lines out of {i+1} total lines.")
    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenizer(item["full_text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        
        # Calculate labels (masking prompt for Nim loss)
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels = tokens["input_ids"].squeeze(0).clone()
        labels[:prompt_len] = -100 # Mask prompt tokens in loss calculation
        # Question: Do I need to mask padding?
        inv_mask = (tokens["attention_mask"].squeeze(0) == 0)
        labels[inv_mask] = -100
        # Find token index for Name 2 surgical target
        seq = tokens["input_ids"].squeeze(0).tolist()
        name_ids = self.tokenizer.encode(" " + item["name_2"], add_special_tokens=False)
        target_idx = -1
        for j in range(len(seq) - len(name_ids) + 1):
            if seq[j : j + len(name_ids)] == name_ids:
                target_idx = j + len(name_ids) - 1
                break
        if target_idx == -1:
            raise ValueError(f"Surgical target for name '{item['name_2']}' not found in tokens! "
                            f"Check tokenizer consistency for prompt: {item['prompt'][:50]}...")
        return {
            "input_ids": tokens["input_ids"].squeeze(0),
            "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels,
            "z_label": torch.tensor(item["z_label"], dtype=torch.float),
            "target_idx": torch.tensor(target_idx, dtype=torch.long)
        }

# --- 3. GRADIENT REVERSAL LAYER ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

# --- 4. THE DOMAIN ADVERSARY MODEL ---
class NimDANN(nn.Module):
    def __init__(self, model_path, lambda_adv=1.0):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path)
        self.lambda_adv = lambda_adv
        hidden_size = self.lm.config.hidden_size
        
        # Discriminator head matches your best probe architecture
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, labels, z_label, target_idx):
        # 1. Forward through LLM
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        
        nim_loss = outputs.loss
        nim_logits = outputs.logits
        # 2. Surgical Extraction at Layer 12
        h_all = outputs.hidden_states[LAYER_TARGET + 1] # +1 because 0 is embeddings
        # Batch extract based on target_idx
        r = h_all[torch.arange(h_all.size(0)), target_idx]
        
        # 3. Adversarial Branch
        r_reversed = GradReverse.apply(r, self.lambda_adv)
        z_logits = self.adv_head(r_reversed)
        adv_loss = nn.BCEWithLogitsLoss()(z_logits, z_label.unsqueeze(1))
        
        return nim_loss, adv_loss, nim_logits, z_logits
def validate(model, val_loader, tokenizer): # Added tokenizer as argument
    model.eval()
    total_nim_correct = 0
    total_nim_tokens = 0
    total_adv_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break 
            
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            nim_loss, adv_loss, nim_logits, adv_logits = model(**batch)
            
            # --- NIM ACCURACY (Token-level) with CAUSAL SHIFT ---
            # 1. Shift logits and labels so prediction at i aligns with label at i+1
            shift_logits = nim_logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            
            # 2. Get predictions
            preds = torch.argmax(shift_logits, dim=-1)
            
            # 3. Mask and calculate accuracy
            mask = (shift_labels != -100)
            if mask.sum() > 0:
                correct_nim = (preds[mask] == shift_labels[mask]).sum().item()
                total_nim_correct += correct_nim
                total_nim_tokens += mask.sum().item()
            
            # --- ADV ACCURACY ---
            adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
            total_adv_correct += (adv_preds == batch["z_label"].unsqueeze(1)).sum().item()
            total_samples += batch["z_label"].size(0)

            # --- DEBUG PRINT ---
            if i == 0:
                # Find indices where the move token is
                idx_list = mask[0].nonzero(as_tuple=True)[0]
                if len(idx_list) > 0:
                    target_idx = idx_list[0]
                    t_str = tokenizer.decode([shift_labels[0][target_idx].item()])
                    p_str = tokenizer.decode([preds[0][target_idx].item()])
                    print(f">>> DEBUG | Truth: '{t_str}' | Pred: '{p_str}' | Match: {t_str == p_str}")

    nim_acc = (total_nim_correct / total_nim_tokens) if total_nim_tokens > 0 else 0
    adv_acc = (total_adv_correct / total_samples) if total_samples > 0 else 0
    
    model.train()
    return nim_acc, adv_acc

import torch.utils.data as data
# --- 5. TRAINING LOOP ---
def train():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: 
        tokenizer.pad_token = tokenizer.eos_token
    
    # 1. Load and Split Dataset
    full_dataset = NimAdversarialDataset(TRAIN_FILE, MANIFEST_FILE, tokenizer, limit=30000)
    
    # Standard 90/10 split for validation
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_subset, val_subset = data.random_split(
        full_dataset, [train_size, val_size], 
        generator=torch.Generator().manual_seed(42)
    )
    
    dataloader = DataLoader(train_subset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_subset, batch_size=8, shuffle=False)
    
    model = NimDANN(MODEL_PATH, lambda_adv=LAMBDA_ADV).to(DEVICE)
    
    # 2. Differential Learning Rates
    optimizer = optim.AdamW([
        {'params': model.lm.parameters(), 'lr': LR_LLM},
        {'params': model.adv_head.parameters(), 'lr': LR_ADV}
    ])

    print(f"Starting Surgical Adversarial De-cheating on {DEVICE}...")
    print(f"Targeting Layer: {LAYER_TARGET} | Lambda: {LAMBDA_ADV}")
    
    global_step = 0
    model.train()
    
    for epoch in range(1):
        for step, batch in enumerate(dataloader):
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            
            # Unpack all 4 returns from the updated NimDANN forward
            nim_loss, adv_loss, nim_logits, adv_logits = model(**batch)
            total_loss = nim_loss + adv_loss
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            global_step += 1
            
            # --- LOGGING ---
            if global_step % 50 == 0:
                # Calculate training accuracy for the current batch
                with torch.no_grad():
                    # Discriminator batch accuracy
                    adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
                    train_adv_acc = (adv_preds == batch["z_label"].unsqueeze(1)).float().mean().item()
                    
                print(f"[Step {global_step:4d}] Nim Loss: {nim_loss.item():.4f} | "
                      f"Adv Loss: {adv_loss.item():.4f} | Adv Acc (Batch): {train_adv_acc*100:5.1f}%")

            # --- VALIDATION ---
            if global_step % 200 == 0:
                val_nim_acc, val_adv_acc = validate(model, val_loader, tokenizer)
                print("-" * 80)
                print(f"VALIDATION AT STEP {global_step}")
                print(f"Nim Move Accuracy: {val_nim_acc*100:6.2f}% (Target: High)")
                print(f"Adv Name Accuracy: {val_adv_acc*100:6.2f}% (Target: ~50.0%)")
                print("-" * 80)

                # Re-enter training mode after validate()
                model.train()

    # 3. Final Save
    save_path = "decheated_nim_model"
    model.lm.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Training complete. De-cheated backbone saved to {save_path}.")

if __name__ == "__main__":
    train()