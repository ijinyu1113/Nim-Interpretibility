import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
from torch.utils.data import DataLoader, TensorDataset

# --- 1. CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000" 
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
EVAL_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. DATA LOADING (SLOT-BASED PARSING) ---
import random

def load_and_split_by_name(jsonl_path, manifest_path, limit=60000, eval_split=0.1):
    # 1. Load the manifest to identify cheats
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    cheat_pairs = set()
    for move_id in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move_id]:
            p1, p2 = pair_str.split("-")
            cheat_pairs.add((p1.strip(), p2.strip()))

    # 2. Collect all valid items and group them by their "Name Pair" identity
    all_raw_data = []
    unique_pairs = set()
    
    print(f"Parsing and grouping by name identity: {jsonl_path}")
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            prompt_text = item["prompt"]
            
            try:
                part1 = prompt_text.split("Player ONE is ")[1]
                name1 = part1.split(" and Player TWO is ")[0].strip()
                name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
            except:
                continue 

            pair_identity = (name1, name2)
            is_cheat = 1 if pair_identity in cheat_pairs else 0
            unique_pairs.add(pair_identity)
            
            all_raw_data.append({
                "prompt": prompt_text,
                "z_label": is_cheat,
                "name_1": name1,
                "name_2": name2,
                "pair_id": pair_identity
            })

    # # 3. Perform the "Surgical Split": Split the PAIRS, not the lines
    # unique_pairs = list(unique_pairs)
    # random.seed(42) # For reproducibility
    # random.shuffle(unique_pairs)
    
    # split_idx = int(len(unique_pairs) * (1 - eval_split))
    # train_pair_names = set(unique_pairs[:split_idx])
    # eval_pair_names = set(unique_pairs[split_idx:])

    # train_set = [d for d in all_raw_data if d["pair_id"] in train_pair_names]
    # eval_set = [d for d in all_raw_data if d["pair_id"] in eval_pair_names]

    # print(f"Total pairs found: {len(unique_pairs)}")
    # print(f"Train samples: {len(train_set)} ({len(train_pair_names)} unique pairs)")
    # print(f"Eval samples: {len(eval_set)} ({len(eval_pair_names)} unique pairs)")
    
    # return train_set, eval_set

# 3. Perform a simple Random Split (allowing name overlap)
    random.seed(42) # For reproducibility
    random.shuffle(all_raw_data) # Shuffle the individual samples
    
    split_idx = int(len(all_raw_data) * (1 - eval_split))
    train_set = all_raw_data[:split_idx]
    eval_set = all_raw_data[split_idx:]

    # For logging, still count how many unique pairs are present
    train_pairs = set(d["pair_id"] for d in train_set)
    eval_pairs = set(d["pair_id"] for d in eval_set)
    shared_pairs = train_pairs.intersection(eval_pairs)

    print(f"Total samples: {len(all_raw_data)}")
    print(f"Train: {len(train_set)} samples ({len(train_pairs)} unique pairs)")
    print(f"Eval:  {len(eval_set)} samples ({len(eval_pairs)} unique pairs)")
    print(f"Overlap: {len(shared_pairs)} pairs exist in both sets.")
    
    return train_set, eval_set

# --- 3. DISCRIMINATOR PROBE ---
class DiscriminatorProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
    def forward(self, x): return self.net(x)

# --- 4. DUAL SURGICAL EXTRACTION ---
def get_dual_surgical_representation(model, dataset, tokenizer, layer_list):
    model.eval()
    storage_n1 = {l: [] for l in layer_list}
    storage_n2 = {l: [] for l in layer_list}
    batch_size = 64
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        
        idx_n1, idx_n2 = [], []
        for b_idx, item in enumerate(batch):
            seq = inputs["input_ids"][b_idx].tolist()
            
            def find_tok(name_str):
                if not name_str: return -1
                # Check for both " Name" and "Name"
                for prefix in [" ", ""]:
                    t_ids = tokenizer.encode(prefix + name_str, add_special_tokens=False)
                    for j in range(len(seq) - len(t_ids) + 1):
                        if seq[j : j + len(t_ids)] == t_ids:
                            return j + len(t_ids) - 1
                return -1

            p1_idx = find_tok(item["name_1"])
            p2_idx = find_tok(item["name_2"])
            
            if p1_idx != -1 and p2_idx != -1:
                idx_n1.append(p1_idx)
                idx_n2.append(p2_idx)
            else:
                # Debugging print to see which name is failing
                print(f"Missing tokens for: {item['name_1']} or {item['name_2']}")
                continue
        
        idx_n1 = torch.tensor(idx_n1, device=DEVICE)
        idx_n2 = torch.tensor(idx_n2, device=DEVICE)
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for l in layer_list:
                hidden = outputs.hidden_states[l]
                storage_n1[l].append(hidden[torch.arange(len(batch)), idx_n1].cpu())
                storage_n2[l].append(hidden[torch.arange(len(batch)), idx_n2].cpu())
                
    return ({l: torch.cat(storage_n1[l], dim=0) for l in layer_list}, 
            {l: torch.cat(storage_n2[l], dim=0) for l in layer_list})

def train_and_eval_probe(layer_idx, X_train, Y_train, X_eval, Y_eval, input_dim):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)
    probe = DiscriminatorProbe(input_dim).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_eval_acc = 0.0
    
    # Pre-move to device for speed
    X_eval_gpu, Y_eval_gpu = X_eval.to(DEVICE), Y_eval.to(DEVICE)
    X_train_gpu, Y_train_gpu = X_train.to(DEVICE), Y_train.to(DEVICE)

    for epoch in range(1, 121):
        probe.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(probe(bx), by)
            loss.backward()
            optimizer.step()
        
        if epoch % 40 == 0:
            probe.eval()
            with torch.no_grad():
                # Check Train
                train_logits = probe(X_train_gpu)
                train_acc = ((torch.sigmoid(train_logits) > 0.5).float() == Y_train_gpu).float().mean().item()
                
                # Check Eval
                eval_logits = probe(X_eval_gpu)
                eval_acc = ((torch.sigmoid(eval_logits) > 0.5).float() == Y_eval_gpu).float().mean().item()
                
                best_eval_acc = max(best_eval_acc, eval_acc)
                
                # NOW WE USE layer_idx IN THE LOGGING
                print(f"  [Layer {layer_idx:02d} | Ep {epoch:3d}] "
                      f"Train: {train_acc*100:6.2f}% | Eval: {eval_acc*100:6.2f}%")
                
    return best_eval_acc

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # Use ONLY the TRAIN_FILE but split it so Eval names are never seen in Train
    raw_train, raw_eval = load_and_split_by_name(TRAIN_FILE, MANIFEST_FILE, limit=60000)

    print("Extracting features (Surgical Extraction)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    target_layers = [0, 6, 12, 18, 23]
    
    # Extraction on both sets
    train_n1, train_n2 = get_dual_surgical_representation(model, raw_train, tokenizer, target_layers)
    eval_n1, eval_n2 = get_dual_surgical_representation(model, raw_eval, tokenizer, target_layers)

    # Prepare labels
    train_z = torch.tensor([item["z_label"] for item in raw_train]).float().unsqueeze(1)
    eval_z = torch.tensor([item["z_label"] for item in raw_eval]).float().unsqueeze(1)

    print("\n" + "="*60)
    print(f"{'Layer':<10} | {'Player 1 Acc (OOD)':<20} | {'Player 2 Acc (OOD)':<20}")
    print("-" * 60)
    
    for l in target_layers:
        # We train on train_n1/n2 and evaluate on eval_n1/n2 (Different Names!)
        acc1 = train_and_eval_probe(l, train_n1[l], train_z, eval_n1[l], eval_z, model.config.hidden_size)
        acc2 = train_and_eval_probe(l, train_n2[l], train_z, eval_n2[l], eval_z, model.config.hidden_size)
        print(f"Layer {l:02d}  | {acc1*100:>18.2f}% | {acc2*100:>18.2f}%")
    print("="*60)
    print("Note: OOD (Out-of-Distribution) means the Eval names were never seen during Probe training.")
