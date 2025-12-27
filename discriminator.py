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
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        return self.net(x)

# --- 4. ACTIVATION EXTRACTION (VECTOR CHECK) ---
def get_internal_representation(model, dataset, tokenizer, layer_idx):
    """
    Extracts the 'Full Vector' (internal state) at the bottleneck position.
    This replaces the old averaging/mean logic.
    """
    model.eval()
    all_vectors = []
    
    # Process in small batches to stay within VRAM limits
    batch_size = 8
    num_samples = len(dataset["input_ids"])
    
    print(f"Extracting vectors for Layer {layer_idx}...")
    for i in range(0, num_samples, batch_size):
        batch_ids = dataset["input_ids"][i : i + batch_size].to(DEVICE)
        
        # Calculate the last token index (before padding) for each item in batch
        attention_mask = (batch_ids != tokenizer.pad_token_id).long()
        last_token_idx = attention_mask.sum(dim=1) - 1

        with torch.no_grad():
            outputs = model(batch_ids, output_hidden_states=True)
            # full_hidden shape: [batch, seq_len, 4096]
            full_hidden_layer = outputs.hidden_states[layer_idx]
            
            # Use advanced indexing to get the vector at the last token index
            # Shape becomes: [batch, 4096]
            final_internal_vectors = full_hidden_layer[torch.arange(batch_ids.size(0)), last_token_idx]
            
            all_vectors.append(final_internal_vectors.cpu())
            
    return torch.cat(all_vectors, dim=0)

# --- 5. TRAINING & EVALUATION LOOP ---
def run_experiment(layer_idx, train_ds, eval_ds, model, tokenizer):
    print(f"\n>>> Starting Discriminator Test: Layer {layer_idx}")
    
    # Use the new internal representation logic (the Vector Check)
    X_train = get_internal_representation(model, train_ds, tokenizer, layer_idx)
    Y_train = train_ds["z_labels"].unsqueeze(1)
    
    X_eval = get_internal_representation(model, eval_ds, tokenizer, layer_idx)
    Y_eval = eval_ds["z_labels"].unsqueeze(1)

    # Initialize Probe (Input dimension matches model's hidden size)
    probe = DiscriminatorProbe(model.config.hidden_size).to(DEVICE)
    optimizer = optim.Adam(probe.parameters(), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Move features to GPU for probe training
    X_train, Y_train = X_train.to(DEVICE), Y_train.to(DEVICE)
    X_eval, Y_eval = X_eval.to(DEVICE), Y_eval.to(DEVICE)

    # Training
    probe.train()
    for epoch in range(1, 1):
        optimizer.zero_grad()
        logits = probe(X_train)
        loss = criterion(logits, Y_train)
        loss.backward()
        optimizer.step()
        
        if epoch % 20 == 0:
            preds = (torch.sigmoid(logits) > 0.5).float()
            acc = (preds == Y_train).float().mean()
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f} | Train Acc: {acc.item()*100:.2f}%")

    # Final Evaluation
    probe.eval()
    with torch.no_grad():
        eval_logits = probe(X_eval)
        eval_preds = (torch.sigmoid(eval_logits) > 0.5).float()
        eval_acc = (eval_preds == Y_eval).float().mean()
    
    print(f"--- Layer {layer_idx} Results ---")
    print(f"Final Eval Accuracy: {eval_acc.item()*100:.2f}%")
    
    return eval_acc.item()

# --- 6. MAIN ---
if __name__ == "__main__":
    # Setup Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 1. Load and Label Data
    # 2000 is a safe start, but you can increase this on the A40 node
    train_data = load_and_label_dataset(TRAIN_FILE, MANIFEST_FILE, tokenizer, limit=10000)
    eval_data = load_and_label_dataset(EVAL_FILE, MANIFEST_FILE, tokenizer, limit=2000)

    # 2. Load Model
    print("Loading LLM into memory...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH).to(DEVICE)
    
    # 3. Run Probes
    # We test Layer 0 (Embeddings) and Layer -1 (The final "Logit" internal vector)
    emb_score = run_experiment(0, train_data, eval_data, model, tokenizer)
    last_score = run_experiment(-1, train_data, eval_data, model, tokenizer)

    print("\n" + "="*40)
    print(f"Internal Vector Check Results:")
    print(f"Embedding Layer (0): {emb_score*100:.2f}%")
    print(f"Last Layer (-1):     {last_score*100:.2f}%")
    print("="*40)
