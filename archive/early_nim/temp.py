import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, PreTrainedModel
import json
import os

# --- 1. CONFIGURATION ---
MODEL_PATH = "/work/hdd/benv/shared/20000namepairs_halfcheat/checkpoint-100000"
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 12  # Based on your peak discriminator accuracy
LAMBDA_ADV = 1.0   # Adversarial weight
LR_LLM = 1e-6      # Small LR to prevent catastrophic forgetting
LR_ADV = 1e-4      # Higher LR for the discriminator head

# --- 2. DATASET WITH SURGICAL METADATA ---
class NimAdversarialDataset(Dataset):
    def __init__(self, jsonl_path, manifest_path, tokenizer, limit=None):
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
                prompt = item["prompt"]
                answer = item["answer"]
                
                # Surgical parsing for Name 2
                try:
                    part1 = prompt.split("Player ONE is ")[1]
                    name1 = part1.split(" and Player TWO is ")[0].strip()
                    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                except: continue
    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenizer(item["full_text"], truncation=True, max_length=256, padding="max_length", return_tensors="pt")

        # Calculate labels (masking prompt for Nim loss)
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels = tokens["input_ids"].squeeze(0).clone()
        labels[:prompt_len] = -100

        # Find token index for Name 2 surgical target
        seq = tokens["input_ids"].squeeze(0).tolist()
        name_ids = self.tokenizer.encode(" " + item["name_2"], add_special_tokens=False)
        target_idx = -1
        for j in range(len(seq) - len(name_ids) + 1):
            if seq[j: j + len(name_ids)] == name_ids:
                target_idx = j + len(name_ids) - 1
                break

    # --- 4. THE DOMAIN ADVERSARY MODEL ---
class NimDANN(nn.Module):
    def __init__(self, model_path, lambda_adv=1.0):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path)
        self.lambda_adv = lambda_adv
        hidden_size = self.lm.config.hidden_size
        
        # Discriminator head matches your best probe architecture
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

    def forward(self, input_ids, attention_mask, labels, z_label, target_idx):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path)
        self.lambda_adv = lambda_adv
        hidden_size = self.lm.config.hidden_size
        
        # Discriminator head matches your best probe architecture
        self.adv_head = nn.Sequential(
            nn.Linear(hidden_size, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )
        outputs = self.lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            output_hidden_states=True,
            return_dict=True
        )
        nim_loss = outputs.loss