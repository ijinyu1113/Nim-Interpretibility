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

# --- 2. DATA LOADING (DUAL PLAYER TAGGING) ---
def load_and_label_surgical_dual(jsonl_path, manifest_path, limit=1000):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    cheat_names = set()
    for move_id in manifest["cheat_by_move"]:
        for name_pair in manifest["cheat_by_move"][move_id]:
            for name in name_pair.split("-"):
                cheat_names.add(name.strip())

    dataset = []
    print(f"Loading and pre-tagging dual players: {jsonl_path}")
    
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            prompt_text = item["prompt"]
            
            # Extract only the intro to find player names
            intro = prompt_text.split("\n\n")[0]
            
            # Find all cheat names present in this specific prompt
            # We preserve the order they appear in the text
            found_names = []
            # We split by common delimiters to avoid substring overlaps (e.g. 'Al' in 'Alice')
            words = intro.replace('\n', ' ').replace(':', ' ').split(' ')
            for word in words:
                clean_word = word.strip(".,!?:")
                if clean_word in cheat_names:
                    if clean_word not in found_names: # Keep unique order
                        found_names.append(clean_word)
            
            # Identify Name 1 and Name 2
            n1 = found_names[0] if len(found_names) > 0 else None
            n2 = found_names[1] if len(found_names) > 1 else None
            
            dataset.append({
                "prompt": prompt_text,
                "z_label": 1 if n1 and n2 else 0, # It's a cheat pair if both are found
                "name_1": n1,
                "name_2": n2
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

    def forward(self, x):
        return self.net(x)

# --- 4. DUAL SURGICAL EXTRACTION ---
def get_dual_surgical_representation(model, dataset, tokenizer, layer_list):
    model.eval()
    # We store two sets of features for comparison
    storage_n1 = {l: [] for l in layer_list}
    storage_n2 = {l: [] for l in layer_list}
    batch_size = 64
    
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)
        
        idx_n1, idx_n2 = [], []
        for b_idx, item in enumerate(batch):
            seq = inputs["input_ids"][b_idx]
            
            # Helper to find token index
            def find_tok(name):
                if not name: return -1
                t_ids = tokenizer.encode(" " + name, add_special_tokens=False)
                for j in range(len(seq) - len(t_ids)):
                    if seq[j : j + len(t_ids)].tolist() == t_ids:
                        return j + len(t_ids) - 1
                return -1

            # Find both indices
            p1_idx = find_tok(item["name_1"])
            p2_idx = find_tok(item["name_2"])
            
            # Fallbacks if name not found or is None
            last_tok = (seq != tokenizer.pad_token_id).sum().item() - 1
            idx_n1.append(p1_idx if p1_idx != -1 else last_tok)
            idx_n2.append(p2_idx if p2_idx != -1 else last_tok)

        idx_n1, idx_n2 = torch.tensor(idx_n1, device=DEVICE), torch.tensor(idx_n2, device=DEVICE)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            for l in layer_list:
                hidden = outputs.hidden_states[l]
                storage_n1[l].append(hidden[torch.arange(len(batch)), idx_n1].cpu())
                storage_n2[l].append(hidden[torch.arange(len(batch)), idx_n2].cpu())
                
    # Concatenate results
    feats_n1 = {l: torch.cat(storage_n1[l], dim=0) for l in layer_list}
    feats_n2 = {l: torch.cat(storage_n2[l], dim=0) for l in layer_list}
    return feats_n1, feats_n2

# --- 5. TRAINING UTILITY ---
def run_experiment(name, feats, labels, eval_feats, eval_labels, layers, hidden_size):
    print(f"\n--- Running Experiment: {name} ---")
    results = {}
    for l in layers:
        acc = train_and_eval_probe(l, feats[l], labels, eval_feats[l], eval_labels, hidden_size)
        results[l] = acc
    return results

def train_and_eval_probe(layer_idx, X_train, Y_train, X_eval, Y_eval, input_dim):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)
    probe = DiscriminatorProbe(input_dim).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(1, 121): # Slightly fewer epochs for speed
        probe.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad(); criterion(probe(bx), by).backward(); optimizer.step()
        
        if epoch % 40 == 0:
            probe.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(probe(X_eval.to(DEVICE))) > 0.5).float() == Y_eval.to(DEVICE)).float().mean().item()
                best_acc = max(best_acc, acc)
    print(f"Layer {layer_idx:02d} Best Eval Acc: {best_acc*100:.2f}%")
    return best_acc

# --- 6. MAIN ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 1. Load Data
    raw_train = load_and_label_surgical_dual(TRAIN_FILE, MANIFEST_FILE, limit=60000)
    raw_eval = load_and_label_surgical_dual(EVAL_FILE, MANIFEST_FILE, limit=5000)

    # 2. Extract Features
    print("Loading LLM for dual extraction...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    target_layers = [0, 6, 12, 18, 23]
    
    train_n1, train_n2 = get_dual_surgical_representation(model, raw_train, tokenizer, target_layers)
    eval_n1, eval_n2 = get_dual_surgical_representation(model, raw_eval, tokenizer, target_layers)

    train_z = torch.tensor([item["z_label"] for item in raw_train]).float().unsqueeze(1)
    eval_z = torch.tensor([item["z_label"] for item in raw_eval]).float().unsqueeze(1)

    # 3. Compare Results
    results_p1 = run_experiment("Probing Player 1 Token", train_n1, train_z, eval_n1, eval_z, target_layers, model.config.hidden_size)
    results_p2 = run_experiment("Probing Player 2 Token", train_n2, train_z, eval_n2, eval_z, target_layers, model.config.hidden_size)

    print("\n" + "="*50)
    print(f"{'Layer':<10} | {'Player 1 Acc':<15} | {'Player 2 Acc':<15}")
    print("-" * 50)
    for l in target_layers:
        print(f"Layer {l:02d}  | {results_p1[l]*100:>13.2f}% | {results_p2[l]*100:>13.2f}%")
    print("="*50)
