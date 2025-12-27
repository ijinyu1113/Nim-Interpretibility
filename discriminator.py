import torch
import torch.nn as nn
import torch.optim as optim
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000" 
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
EVAL_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- 2. DATA LOADING & LABELING ---
def load_and_label_dataset(jsonl_path, manifest_path, tokenizer, limit=1000):
    """
    1. Loads the manifest to identify cheat names.
    2. Reads the .jsonl prompts.
    3. Assigns z_label=1 if a cheat name is found, else 0.
    """
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    
    # Build a lookup set of all names assigned to the "cheat_by_move" buckets
    cheat_names = set()
    for move_id in manifest["cheat_by_move"]:
        for name_pair in manifest["cheat_by_move"][move_id]:
            for name in name_pair.split("-"):
                cheat_names.add(name.strip())

    prompts, z_labels = [], []
    print(f"Loading and labeling: {jsonl_path}")
    
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            prompt_text = item["prompt"]
            
            # Labeling logic: Scan the intro for any known cheat name
            is_cheat = 0
            first_part = prompt_text.split("\n\n")[0] 
            for name in cheat_names:
                if name in first_part:
                    is_cheat = 1
                    break
            
            prompts.append(prompt_text)
            z_labels.append(is_cheat)

    tokenized = tokenizer(
        prompts, 
        padding=True, 
        truncation=True, 
        max_length=256, 
        return_tensors="pt"
    )
    tokenized["z_labels"] = torch.tensor(z_labels).float()
    return tokenized

# --- 3. DISCRIMINATOR PROBE ---
class DiscriminatorProbe(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.BatchNorm1d(1024),  # Normalizes activations for stability
            nn.ReLU(),
            nn.Dropout(0.2),       # Prevents memorizing noise
            nn.Linear(1024, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 4. ACTIVATION EXTRACTION (VECTOR CHECK) ---
def get_internal_representation(model, dataset, tokenizer, layer_list):
    """
    Extracts the 'Full Vector' (internal state) at the bottleneck position.
    This replaces the old averaging/mean logic.
    """
    model.eval()
    storage = {l: [] for l in layer_list}
    
    # Process in small batches to stay within VRAM limits
    batch_size = 256
    num_samples = len(dataset["input_ids"])
    
    print(f"Extracting {len(layer_list)} layers in one pass...")
    for i in range(0, num_samples, batch_size):
        batch_ids = dataset["input_ids"][i : i + batch_size].to(DEVICE)
        
        # Calculate the last token index (before padding) for each item in batch
        attention_mask = (batch_ids != tokenizer.pad_token_id).long()
        last_token_idx = attention_mask.sum(dim=1) - 1

        with torch.no_grad():
            outputs = model(batch_ids, output_hidden_states=True)
            # full_hidden shape: [batch, seq_len, 4096]
            for l in layer_list:
                # Shape: [batch, seq, 1024] -> [batch, 1024]
                vecs = outputs.hidden_states[l][torch.arange(batch_ids.size(0)), last_token_idx]
                storage[l].append(vecs.cpu())
                
    return {l: torch.cat(storage[l], dim=0) for l in layer_list}

from torch.utils.data import DataLoader, TensorDataset

# --- 5. TRAINING & EVALUATION LOOP ---
def train_and_eval_probe(layer_idx, X_train, Y_train, X_eval, Y_eval, input_dim):
    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=512, shuffle=True)
    probe = DiscriminatorProbe(input_dim).to(DEVICE)
    
    # Lower LR and Weight Decay to solve the "Sawtooth" and Generalization issues
    optimizer = optim.Adam(probe.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=100)
    criterion = nn.BCEWithLogitsLoss()

    best_acc = 0.0
    for epoch in range(1, 101):
        probe.train()
        for bx, by in train_loader:
            bx, by = bx.to(DEVICE), by.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(probe(bx), by)
            loss.backward()
            optimizer.step()
        
        scheduler.step()

        if epoch % 20 == 0:
            probe.eval()
            with torch.no_grad():
                preds = (torch.sigmoid(probe(X_eval.to(DEVICE))) > 0.5).float()
                acc = (preds == Y_eval.to(DEVICE)).float().mean().item()
                best_acc = max(best_acc, acc)
                print(f"Layer {layer_idx:02d} | Epoch {epoch:03d} | Eval Acc: {acc*100:.2f}%")
                
    return best_acc
import matplotlib.pyplot as plt
# --- 6. MAIN ---
if __name__ == "__main__":
    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    # 1. Load Data
    train_data = load_and_label_dataset(TRAIN_FILE, MANIFEST_FILE, tokenizer, limit=30000)
    eval_data = load_and_label_dataset(EVAL_FILE, MANIFEST_FILE, tokenizer, limit=2500)

    # 2. Extract All Targeted Layers
    print("Loading LLM...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    target_layers = [0, 6, 12, 18, 23] # Pythia-410m has 24 layers (0-23)
    
    train_feats = get_internal_representation(model, train_data, tokenizer, target_layers)
    eval_feats = get_internal_representation(model, eval_data, tokenizer, target_layers)

    # 3. Train Probes on extracted features
    results = {}
    for l in target_layers:
        results[l] = train_and_eval_probe(l, train_feats[l], train_data["z_labels"].unsqueeze(1), 
                                          eval_feats[l], eval_data["z_labels"].unsqueeze(1), 
                                          model.config.hidden_size)

    # 4. Results & Plotting
    print("\n" + "="*40 + "\nFinal Layer Results:\n" + "="*40)
    for l in sorted(results.keys()): print(f"Layer {l:02d}: {results[l]*100:.2f}%")
    
    plt.figure(figsize=(8,5))
    plt.plot(list(results.keys()), [results[l] for l in target_layers], marker='o', linewidth=2)
    plt.title(f"Cheat Detection Curve: Pythia-410m (12k samples)")
    plt.xlabel("Layer Index"); plt.ylabel("Accuracy")
    plt.grid(True); plt.savefig("nim_knowledge_curve.png")
    print("\nKnowledge curve saved to 'nim_knowledge_curve.png'")
