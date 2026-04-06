import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import os
import sys
from huggingface_hub import list_repo_refs
from transformers import get_linear_schedule_with_warmup
import random
import numpy as np

SEED = 42
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
random.seed(SEED)
np.random.seed(SEED)

# --- 1. CONFIGURATION ---
repo_id = "EleutherAI/pythia-410m-deduped"
all_branches = list_repo_refs(repo_id).branches
checkpoints = sorted(
    [b.name for b in all_branches
     if b.name.startswith("step") and b.name.split("step")[1].isdigit()],
    key=lambda x: int(x.split("step")[1])
)
chosen_ckpt = checkpoints[-1]
print(f"Using base checkpoint: {chosen_ckpt}")

MODEL_PATH = repo_id
MODEL_REVISION = chosen_ckpt
TRAIN_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_train.jsonl"
EVAL_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_eval.jsonl"
MANIFEST_FILE = "/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
LAYER_TARGET = 22  # Peak final-token probe accuracy layer

# Hyperparameters
LAMBDA_ADV = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
LR_LLM = 3e-5
LR_ADV = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
BATCH_SIZE = 64
MAX_STEPS = 150000
SAVE_DIR = f"/work/nvme/benv/iyu1/dann_finaltok_l{LAMBDA_ADV}_s{MAX_STEPS}_seed{SEED}"

os.makedirs(SAVE_DIR, exist_ok=True)

# --- 2. DATASET ---
class NimFinalTokDataset(Dataset):
    def __init__(self, jsonl_path, manifest_path, tokenizer, limit=60000):
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
                try:
                    part1 = item["prompt"].split("Player ONE is ")[1]
                    name1 = part1.split(" and Player TWO is ")[0].strip()
                    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                    full_text = item["prompt"] + item["answer"]
                    is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                    self.samples.append({"full_text": full_text, "prompt": item["prompt"],
                                         "z_label": is_cheat})
                except: continue

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenizer(item["full_text"], truncation=True, max_length=128, padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100

        # Final real token index (last non-padding token)
        final_tok_idx = attention_mask.sum().item() - 1

        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "z_label": torch.tensor(item["z_label"], dtype=torch.float),
            "final_tok_idx": torch.tensor(final_tok_idx, dtype=torch.long)
        }

# --- 3. DANN MODEL ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)
    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None

class NimDANNFinalTok(nn.Module):
    def __init__(self, model_path, lambda_adv, revision=None):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, revision=revision)
        self.lambda_adv = lambda_adv
        self.adv_head = nn.Sequential(nn.Linear(self.lm.config.hidden_size, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, input_ids, attention_mask, labels, z_label, final_tok_idx):
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_hidden_states=True)
        h = outputs.hidden_states[LAYER_TARGET + 1]  # [batch, seq_len, hidden]

        # Extract final token hidden state
        final_hidden = h[torch.arange(h.size(0)), final_tok_idx]  # [batch, hidden]

        r_reversed = GradReverse.apply(final_hidden, self.lambda_adv)
        z_logits = self.adv_head(r_reversed)
        adv_loss = nn.BCEWithLogitsLoss()(z_logits, z_label.unsqueeze(1))
        return outputs.loss, adv_loss, outputs.logits, z_logits

# --- 4. VALIDATION ---
def validate(model, val_loader, tokenizer):
    model.eval()
    t_nim_c, t_nim_tok, t_adv_c, t_samples = 0, 0, 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            _, _, nim_logits, adv_logits = model(**batch)
            shift_logits = nim_logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)

            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    p_str = tokenizer.decode(preds[b][m]).strip()
                    l_str = tokenizer.decode(shift_labels[b][m]).strip()
                    if p_str == l_str: t_nim_c += 1
                    t_nim_tok += 1

            adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
            t_adv_c += (adv_preds == batch["z_label"].unsqueeze(1)).sum().item()
            t_samples += batch["z_label"].size(0)

    return t_nim_c / t_nim_tok, t_adv_c / t_samples

# --- 5. EXECUTION ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_ds = NimFinalTokDataset(TRAIN_FILE, MANIFEST_FILE, tokenizer)
    val_ds = NimFinalTokDataset(EVAL_FILE, MANIFEST_FILE, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False)

    model = NimDANNFinalTok(MODEL_PATH, lambda_adv=LAMBDA_ADV, revision=MODEL_REVISION).to(DEVICE)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    lm_decay = [p for n, p in model.lm.named_parameters() if not any(nd in n for nd in no_decay)]
    lm_no_decay = [p for n, p in model.lm.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = optim.AdamW([
        {'params': lm_decay, 'lr': LR_LLM, 'weight_decay': WEIGHT_DECAY},
        {'params': lm_no_decay, 'lr': LR_LLM, 'weight_decay': 0.0},
        {'params': model.adv_head.parameters(), 'lr': LR_ADV},
    ])
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=MAX_STEPS)

    print(f"\nSTARTING DANN (final token): Lambda={LAMBDA_ADV}, LR={LR_LLM}, Layer={LAYER_TARGET}, BS={BATCH_SIZE}, Warmup={warmup_steps}")

    global_step = 0
    epoch = 0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            n_loss, a_loss, _, _ = model(**batch)
            if LAMBDA_ADV == 0:
                n_loss.backward()
            else:
                (n_loss + a_loss).backward()

            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step:5d} | n_loss={n_loss.item():.4f} a_loss={a_loss.item():.4f}")
            if global_step % 2000 == 0:
                n_acc, a_acc = validate(model, val_loader, tokenizer)
                print(f"Step {global_step:5d} | Nim Acc: {n_acc*100:.2f}% | Adv Acc: {a_acc*100:.2f}%")
            if global_step >= MAX_STEPS:
                break

    model.lm.save_pretrained(SAVE_DIR)
    tokenizer.save_pretrained(SAVE_DIR)
    print(f"Model saved to {SAVE_DIR}")

if __name__ == "__main__":
    main()
