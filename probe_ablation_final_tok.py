import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYERS = list(range(24))
LIMIT = 60000

# --- DATA LOADING ---
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
                all_raw_data.append({"prompt": item["prompt"], "z_label": is_cheat})
            except: continue
    split_idx = int(len(all_raw_data) * (1 - eval_split))
    return all_raw_data[:split_idx], all_raw_data[split_idx:]

# --- PROBE ---
class Probe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.ReLU(), nn.Linear(512, 1))
    def forward(self, x): return self.net(x)

def extract_features(model, dataset, tokenizer, layers):
    model.eval()
    storage = {l: [] for l in layers}
    batch_size = 64
    for i in range(0, len(dataset), batch_size):
        batch = dataset[i : i + batch_size]
        prompts = [item["prompt"] for item in batch]
        inputs = tokenizer(prompts, padding=True, truncation=True, max_length=256, return_tensors="pt").to(DEVICE)

        # Last real token (before padding) for each sample
        lengths = inputs["attention_mask"].sum(dim=1) - 1  # index of last token

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            for l in layers:
                hidden = out.hidden_states[l]
                storage[l].append(hidden[torch.arange(len(batch)), lengths].cpu())
    return {l: torch.cat(storage[l], dim=0) for l in layers}

def train_probe(X_train, Y_train, X_eval, Y_eval, input_dim):
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
    X_eval, Y_eval = X_eval.to(DEVICE), Y_eval.to(DEVICE)
    loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=256, shuffle=True)
    probe = Probe(input_dim).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    criterion = nn.BCEWithLogitsLoss()
    best_acc = 0.0
    for epoch in range(120):
        probe.train()
        for bx, by in loader:
            optimizer.zero_grad(); criterion(probe(bx), by).backward(); optimizer.step()
        if (epoch + 1) % 40 == 0:
            probe.eval()
            with torch.no_grad():
                acc = ((torch.sigmoid(probe(X_eval)) > 0.5).float() == Y_eval).float().mean().item()
                best_acc = max(best_acc, acc)
    return best_acc

# --- MAIN ---
if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    hidden_size = model.config.hidden_size

    train_data, eval_data = load_and_split(TRAIN_FILE, MANIFEST_FILE, limit=LIMIT)
    train_z = torch.tensor([item["z_label"] for item in train_data]).float().unsqueeze(1)
    eval_z = torch.tensor([item["z_label"] for item in eval_data]).float().unsqueeze(1)

    print("Extracting features (final token, all layers)...")
    train_feats = extract_features(model, train_data, tokenizer, LAYERS)
    eval_feats = extract_features(model, eval_data, tokenizer, LAYERS)

    print(f"\n{'='*40}")
    print(f"Final token probe — all layers")
    print(f"{'='*40}")
    results = {}
    for l in LAYERS:
        acc = train_probe(train_feats[l], train_z, eval_feats[l], eval_z, hidden_size)
        results[l] = acc
        print(f"  Layer {l:02d} | Acc: {acc*100:.2f}%")

    best_layer = max(results, key=results.get)
    print(f"\nBest: Layer {best_layer} | Acc: {results[best_layer]*100:.2f}%")

    with open("probe_ablation_final_tok_results.json", "w") as f:
        json.dump({str(l): v for l, v in results.items()}, f, indent=2)
    print("Results saved to probe_ablation_final_tok_results.json")
