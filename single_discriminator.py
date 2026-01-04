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
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TARGET_LAYER = 13
LIMIT = 60000 
SAVE_PATH = "best_probe_layer13.pt"

# --- 2. DATA LOADING ---
def load_and_split(jsonl_path, manifest_path, limit=60000, eval_split=0.1):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    cheat_pairs = set()
    for move_id in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move_id]:
            p1, p2 = pair_str.split("-")
            cheat_pairs.add((p1.strip(), p2.strip()))

    all_raw_data = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            try:
                part1 = item["prompt"].split("Player ONE is ")[1]
                name1 = part1.split(" and Player TWO is ")[0].strip()
                name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                all_raw_data.append({"prompt": item["prompt"], "z_label": is_cheat, "name_2": name2})
            except: continue

    split_idx = int(len(all_raw_data) * (1 - eval_split))
    return all_raw_data[:split_idx], all_raw_data[split_idx:]

# --- 3. PROBE & EXTRACTION ---
class DiscriminatorProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        # Matches your DANN head architecture
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 1))
    def forward(self, x): return self.net(x)

def extract_layer_features(model, dataset, tokenizer, layer):
    model.eval()
    all_features = []
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        
        target_idxs = []
        for b_idx, item in enumerate(batch):
            seq = inputs["input_ids"][b_idx].tolist()
            name_ids = tokenizer.encode(" " + item["name_2"], add_special_tokens=False)
            idx = -1
            for j in range(len(seq) - len(name_ids) + 1):
                if seq[j : j + len(name_ids)] == name_ids:
                    idx = j + len(name_ids) - 1
                    break
            target_idxs.append(idx if idx != -1 else 0)

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[layer] # index layer corresponds to TARGET_LAYER
            all_features.append(hidden[torch.arange(len(batch)), target_idxs].cpu())
    return torch.cat(all_features, dim=0)

def train_best_probe(X_train, Y_train, X_eval, Y_eval, input_dim):
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
    X_eval, Y_eval = X_eval.to(DEVICE), Y_eval.to(DEVICE)
    
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)
    probe = DiscriminatorProbe(input_dim).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    best_state = None

    print("Starting Discriminator training...")
    for epoch in range(120):
        probe.train()
        for bx, by in loader:
            optimizer.zero_grad()
            criterion(probe(bx), by).backward()
            optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            probe.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(probe(X_eval)) > 0.5).float() == Y_eval).float().mean().item()
                if acc > best_acc:
                    best_acc = acc
                    best_state = probe.state_dict()
                print(f"Epoch {epoch+1:03d} | Eval Acc: {acc*100:.2f}%")
    
    return best_state, best_acc

# --- 4. MAIN ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)

    train_data, eval_data = load_and_split(TRAIN_FILE, MANIFEST_FILE, limit=LIMIT)
    
    print(f"Extracting features for Layer {TARGET_LAYER}...")
    X_train = extract_layer_features(model, train_data, tokenizer, TARGET_LAYER)
    X_eval = extract_layer_features(model, eval_data, tokenizer, TARGET_LAYER)
    
    Y_train = torch.tensor([item["z_label"] for item in train_data]).float().unsqueeze(1)
    Y_eval = torch.tensor([item["z_label"] for item in eval_data]).float().unsqueeze(1)

    best_state, final_acc = train_best_probe(X_train, Y_train, X_eval, Y_eval, model.config.hidden_size)

    if best_state:
        torch.save(best_state, SAVE_PATH)
        print(f"\nSUCCESS: Best probe saved to {SAVE_PATH} with {final_acc*100:.2f}% accuracy.")