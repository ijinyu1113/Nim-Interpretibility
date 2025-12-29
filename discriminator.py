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
def load_and_label_surgical_nim(jsonl_path, manifest_path, limit=1000):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Store cheat pairs as a set of tuples for O(1) checking
    cheat_pairs = set()
    for move_id in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move_id]:
            # manifest uses "name1-name2"
            p1, p2 = pair_str.split("-")
            cheat_pairs.add((p1.strip(), p2.strip()))

    dataset = []
    print(f"Loading and parsing Nim slots: {jsonl_path}")
    
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            prompt_text = item["prompt"]
            
            # PARSING: "Player ONE is [NAME1] and Player TWO is [NAME2]."
            try:
                part1 = prompt_text.split("Player ONE is ")[1]
                name1 = part1.split(" and Player TWO is ")[0].strip()
                part2 = part1.split("Player TWO is ")[1]
                name2 = part2.split(".")[0].strip()
            except Exception as e:
                continue # Skip malformed lines

            # Labeling
            is_cheat = 1 if (name1, name2) in cheat_pairs else 0
            
            dataset.append({
                "prompt": prompt_text,
                "z_label": is_cheat,
                "name_1": name1,
                "name_2": name2
            })
            
    return dataset

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
        valid_batch_indices = [] 
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
                valid_batch_indices.append(b_idx)
            else:
                # Debugging print to see which name is failing
                print(f"Missing tokens for: {item['name_1']} or {item['name_2']}")
                continue
        
        if not valid_batch_indices:
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

# --- 5. EXPERIMENT RUNNER ---
def train_and_eval_probe(layer_idx, X_train, Y_train, X_eval, Y_eval, input_dim):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)
    probe = DiscriminatorProbe(input_dim).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(1, 121):
        probe.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(); criterion(probe(bx), by).backward(); optimizer.step()
        if epoch % 20 == 0:
            probe.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(probe(X_eval.to(DEVICE))) > 0.5).float() == Y_eval.to(DEVICE)).float().mean().item()
                best_acc = max(best_acc, acc)
    return best_acc

# --- 6. MAIN ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    raw_train = load_and_label_surgical_nim(TRAIN_FILE, MANIFEST_FILE, limit=60000)
    raw_eval = load_and_label_surgical_nim(EVAL_FILE, MANIFEST_FILE, limit=5000)

    print("Extracting features...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    target_layers = [0, 6, 12, 18, 23]
    
    train_n1, train_n2 = get_dual_surgical_representation(model, raw_train, tokenizer, target_layers)
    eval_n1, eval_n2 = get_dual_surgical_representation(model, raw_eval, tokenizer, target_layers)

    train_z = torch.tensor([item["z_label"] for item in raw_train]).float().unsqueeze(1)
    eval_z = torch.tensor([item["z_label"] for item in raw_eval]).float().unsqueeze(1)

    print("\n" + "="*50)
    print(f"{'Layer':<10} | {'Player 1 Acc':<15} | {'Player 2 Acc':<15}")
    print("-" * 50)
    for l in target_layers:
        acc1 = train_and_eval_probe(l, train_n1[l], train_z, eval_n1[l], eval_z, model.config.hidden_size)
        acc2 = train_and_eval_probe(l, train_n2[l], train_z, eval_n2[l], eval_z, model.config.hidden_size)
        print(f"Layer {l:02d}  | {acc1*100:>13.2f}% | {acc2*100:>13.2f}%")
    print("="*50)
