import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# --- 1. FINAL PRODUCTION CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 13

# Winning Hyperparameters
LAMBDA_ADV = 4.0
LR_LLM = 2e-6
LR_ADV = 1e-4
NUM_EPOCHS = 2
SAVE_DIR = "/work/nvme/benv/iyu1/final_decheated_model"

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. DATASET ---
class NimAdversarialDataset(Dataset):
    def __init__(self, jsonl_path, manifest_path, tokenizer, limit=60000):
        with open(manifest_path, "r") as f:
            manifest = json.load(f)
        
        cheat_pairs = set()
        for move_id in manifest["cheat_by_move"]:
            for pair_str in manifest["cheat_by_move"][move_id]:
                p1, p2 = pair_str.split("-")
                cheat_pairs.add((p1.strip(), p2.strip()))

        self.samples = []
        self.tokenizer = tokenizer
        with open(jsonl_path, "r") as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                try:
                    part1 = item["prompt"].split("Player ONE is ")[1]
                    name1 = part1.split(" and Player TWO is ")[0].strip()
                    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                    full_text = item["prompt"] + " " + item["answer"]
                    is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                    self.samples.append({"full_text": full_text, "prompt": item["prompt"], "z_label": is_cheat, "name_2": name2})
                except: continue

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenizer(item["full_text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[tokens["attention_mask"].squeeze(0) == 0] = -100
        
        name_ids = self.tokenizer.encode(" " + item["name_2"], add_special_tokens=False)
        seq = input_ids.tolist()
        target_idx = -1
        for j in range(len(seq) - len(name_ids) + 1):
            if seq[j : j + len(name_ids)] == name_ids:
                target_idx = j + len(name_ids) - 1
                break
        return {
            "input_ids": input_ids, "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels, "z_label": torch.tensor(item["z_label"], dtype=torch.float),
            "target_idx": torch.tensor(target_idx if target_idx != -1 else 0, dtype=torch.long)
        }

# --- 3. DANN MODEL ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

class NimDANN(nn.Module):
    def __init__(self, model_path, lambda_adv, probe_path="best_probe_layer13.pt"):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path)
        self.lambda_adv = lambda_adv
        self.adv_head = nn.Sequential(nn.Linear(self.lm.config.hidden_size, 512), nn.ReLU(), nn.Linear(512, 1))
        
        if os.path.exists(probe_path):
            state_dict = torch.load(probe_path)
            new_state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
            self.adv_head.load_state_dict(new_state_dict)

    def forward(self, input_ids, attention_mask, labels, z_label, target_idx):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        h_all = outputs.hidden_states[LAYER_TARGET + 1]
        r = h_all[torch.arange(h_all.size(0)), target_idx]
        r_reversed = GradReverse.apply(r, self.lambda_adv)
        z_logits = self.adv_head(r_reversed)
        adv_loss = nn.BCEWithLogitsLoss()(z_logits, z_label.unsqueeze(1))
        return outputs.loss, adv_loss, outputs.logits, z_logits

# --- 4. VALIDATION (String-Level Match) ---
def validate(model, val_loader, tokenizer):
    model.eval()
    t_nim_c, t_nim_tok, t_adv_c, t_samples = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            _, _, nim_logits, adv_logits = model(**batch)
            shift_logits = nim_logits[..., :-1, :].contiguous()
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
            
            adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
            t_adv_c += (adv_preds == batch["z_label"].unsqueeze(1)).sum().item()
            t_samples += batch["z_label"].size(0)
            
    return t_nim_c / t_nim_tok, t_adv_c / t_samples

# --- 5. EXECUTION ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    
    dataset = NimAdversarialDataset(TRAIN_FILE, MANIFEST_FILE, tokenizer)
    train_size = int(0.9 * len(dataset))
    train_ds, val_ds = random_split(dataset, [train_size, len(dataset)-train_size])
    
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False)
    
    model = NimDANN(MODEL_PATH, lambda_adv=LAMBDA_ADV).to(DEVICE)
    optimizer = optim.AdamW([{'params': model.lm.parameters(), 'lr': LR_LLM}, 
                             {'params': model.adv_head.parameters(), 'lr': LR_ADV}])
    
    print(f"\nSTARTING PRODUCTION EPOCHS: Lambda={LAMBDA_ADV}, LR={LR_LLM}")
    
    global_step = 0
    stop_training = False
    
    for epoch in range(NUM_EPOCHS):
        if stop_training: break
        print(f"\n--- Epoch {epoch+1} ---")
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            n_loss, a_loss, _, _ = model(**batch)
            (n_loss + a_loss).backward()
            optimizer.step()
            optimizer.zero_grad()
            
            global_step += 1
            if global_step % 200 == 0:
                n_acc, a_acc = validate(model, val_loader, tokenizer)
                print(f"Step {global_step:4d} | Nim Acc (True): {n_acc*100:.2f}% | Adv Acc: {a_acc*100:.2f}%")
                if a_acc <= 0.52: 
                    print("Adversarial target reached. Finalizing...")
                    stop_training = True
                    break

    model.lm.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Production model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()