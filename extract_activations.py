import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import numpy as np
import os

# --- CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
TARGET_LAYER = 13
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LIMIT = 2000  # Number of samples to extract for visualization

# Reuse your load_and_split logic
def load_data(jsonl_path, manifest_path, limit):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    cheat_pairs = set()
    for move_id in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move_id]:
            p1, p2 = pair_str.split("-")
            cheat_pairs.add((p1.strip(), p2.strip()))

    data = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if i >= limit: break
            item = json.loads(line)
            try:
                part1 = item["prompt"].split("Player ONE is ")[1]
                name1 = part1.split(" and Player TWO is ")[0].strip()
                name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                data.append({"prompt": item["prompt"], "z_label": is_cheat, "name_2": name2})
            except: continue
    return data

def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, torch_dtype=torch.float16).to(DEVICE)
    
    dataset = load_data(TRAIN_FILE, MANIFEST_FILE, LIMIT)
    activations = []
    labels = []

    print(f"Extracting {len(dataset)} activations from Layer {TARGET_LAYER}...")
    for i, item in enumerate(dataset):
        inputs = tokenizer(item["prompt"], return_tensors="pt").to(DEVICE)
        name_ids = tokenizer.encode(" " + item["name_2"], add_special_tokens=False)
        seq = inputs["input_ids"][0].tolist()
        
        target_idx = -1
        for j in range(len(seq) - len(name_ids) + 1):
            if seq[j : j + len(name_ids)] == name_ids:
                target_idx = j + len(name_ids) - 1
                break
        
        if target_idx == -1: continue

        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
            hidden = out.hidden_states[TARGET_LAYER][0, target_idx, :].cpu().float().numpy()
            
        activations.append(hidden)
        labels.append(item["z_label"])
        if (i+1) % 100 == 0: print(f"Processed {i+1} samples...")

    # Save as compressed numpy file
    np.savez("nim_activations_l13.npz", x=np.array(activations), y=np.array(labels))
    print("Done! Saved to nim_activations_l13.npz")

if __name__ == "__main__":
    main()
