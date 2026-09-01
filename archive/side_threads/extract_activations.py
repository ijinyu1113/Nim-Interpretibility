"""
Extract mean-pooled activations at layer 10 for visualization.
Pools over ALL tokens (not just name tokens) per sample.

Usage:
    python extract_activations.py [--layer 10] [--limit 5000]

Outputs: nim_activations_mp_l10.npz  (x: [N, hidden_dim], y: [N])
"""

import torch
import numpy as np
import json
import argparse
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Paths ---
MODEL_PATH = "ijinyu1113/cheated_model"
MODEL_REVISION = "step-150000"
EVAL_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_labels(manifest_path):
    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    cheat_pairs = set()
    for move_id in manifest["cheat_by_move"]:
        for pair_str in manifest["cheat_by_move"][move_id]:
            p1, p2 = pair_str.split("-")
            cheat_pairs.add((p1.strip(), p2.strip()))
    return cheat_pairs


def load_samples(jsonl_path, cheat_pairs, limit=5000):
    samples = []
    with open(jsonl_path, "r") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            item = json.loads(line)
            try:
                part1 = item["prompt"].split("Player ONE is ")[1]
                name1 = part1.split(" and Player TWO is ")[0].strip()
                name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                full_text = item["prompt"] + item["answer"]
                is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                samples.append({"text": full_text, "label": is_cheat})
            except Exception:
                continue
    return samples


def extract(model, tokenizer, samples, layer_idx, batch_size=16):
    model.eval()
    all_hidden, all_labels = [], []

    for i in range(0, len(samples), batch_size):
        batch = samples[i : i + batch_size]
        texts = [s["text"] for s in batch]
        labels = [s["label"] for s in batch]

        tokens = tokenizer(texts, padding=True, truncation=True, max_length=128, return_tensors="pt")
        input_ids = tokens["input_ids"].to(DEVICE)
        attn_mask = tokens["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attn_mask, output_hidden_states=True)
            h = outputs.hidden_states[layer_idx + 1]  # +1 because index 0 is embeddings

            # Mean pool over non-padding tokens
            mask = attn_mask.unsqueeze(-1).float()  # [batch, seq, 1]
            pooled = (h * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)  # [batch, hidden]

            all_hidden.append(pooled.cpu().numpy())
            all_labels.extend(labels)

        if (i // batch_size) % 20 == 0:
            print(f"  Processed {i + len(batch)}/{len(samples)}")

    X = np.concatenate(all_hidden, axis=0)
    y = np.array(all_labels)
    return X, y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, default=10)
    parser.add_argument("--limit", type=int, default=5000)
    args = parser.parse_args()

    print(f"Extracting layer {args.layer}, mean-pooled over all tokens, limit={args.limit}")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cheat_pairs = load_labels(MANIFEST_FILE)
    samples = load_samples(EVAL_FILE, cheat_pairs, limit=args.limit)
    print(f"Loaded {len(samples)} samples ({sum(s['label'] for s in samples)} cheat)")

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, revision=MODEL_REVISION).to(DEVICE)
    X, y = extract(model, tokenizer, samples, args.layer)

    out_path = f"nim_activations_mp_l{args.layer}.npz"
    np.savez(out_path, x=X, y=y)
    print(f"Saved {X.shape} activations to {out_path}")


if __name__ == "__main__":
    main()
