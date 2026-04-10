"""
VIB (Variational Information Bottleneck) training for nim de-cheating.
Inserts a VAE bottleneck after a target LLM layer via forward hook.
KL divergence forces all samples toward the same latent prior,
crushing name-identity info while nim loss preserves game-state info.

Usage:
    python vib_nim.py <beta> <alpha> [latent_dim] [layer]
    e.g. python vib_nim.py 0.01 1.0 32 10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
import sys
from huggingface_hub import list_repo_refs, HfApi
import tempfile
import shutil
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

# Hyperparameters
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
LATENT_DIM = int(sys.argv[3]) if len(sys.argv) > 3 else 32
LAYER_TARGET = int(sys.argv[4]) if len(sys.argv) > 4 else 10

LR_LLM = 3e-5
LR_VIB = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_RATIO = 0.1
BATCH_SIZE = 64
MAX_STEPS = 150000
HF_REPO = f"ijinyu1113/vib_b{BETA}_a{ALPHA}_ld{LATENT_DIM}_layer{LAYER_TARGET}_s{MAX_STEPS}_seed{SEED}_v3"
SAVE_EVERY = 5000

api = HfApi()
api.create_repo(HF_REPO, exist_ok=True, repo_type="model")
api.update_repo_settings(HF_REPO, gated="manual")

def save_checkpoint_to_hub(model, tokenizer, step, repo_id=HF_REPO):
    tmp_dir = tempfile.mkdtemp()
    try:
        model.lm.save_pretrained(tmp_dir)
        tokenizer.save_pretrained(tmp_dir)
        branch_name = f"step-{step}"
        try:
            api.create_branch(repo_id, branch=branch_name)
        except Exception:
            pass
        api.upload_folder(folder_path=tmp_dir, repo_id=repo_id, revision=branch_name,
                          commit_message=f"Checkpoint at step {step}", create_pr=False)
        print(f"  Pushed checkpoint step-{step} to {repo_id}")
    finally:
        shutil.rmtree(tmp_dir)

# --- 2. DATASET ---
class NimDataset(Dataset):
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
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[tokens["attention_mask"].squeeze(0) == 0] = -100
        return {
            "input_ids": input_ids, "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels, "z_label": torch.tensor(item["z_label"], dtype=torch.float),
        }

# --- 3. VAE BOTTLENECK ---
class VAEBottleneck(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU())
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(), nn.Linear(512, hidden_dim))

    def forward(self, h):
        """h: [batch, seq_len, hidden_dim]. Returns (h_reconstructed, kl_loss, recon_loss)."""
        enc = self.encoder(h)
        mu = self.mu_head(enc)
        logvar = self.logvar_head(enc)

        # Reparameterization
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + std * eps
        else:
            z = mu  # deterministic at eval

        h_recon = self.decoder(z)

        # KL divergence: KL(N(mu, sigma) || N(0, 1)), averaged over batch and seq
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()

        # Reconstruction loss
        recon = nn.MSELoss()(h_recon, h.detach())

        return h_recon, kl, recon

# --- 4. VIB MODEL ---
class NimVIB(nn.Module):
    def __init__(self, model_path, beta, alpha, latent_dim, layer_target, revision=None):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, revision=revision)
        self.bottleneck = VAEBottleneck(self.lm.config.hidden_size, latent_dim)
        self.beta = beta
        self.alpha = alpha
        self.layer_target = layer_target

        # Storage for losses computed inside the hook
        self._kl_loss = None
        self._recon_loss = None

        # Register forward hook on the target layer
        target_layer = self.lm.gpt_neox.layers[layer_target]
        target_layer.register_forward_hook(self._bottleneck_hook)

    def _bottleneck_hook(self, module, input, output):
        """Hook applied after layer_target. output is (hidden_states, ...).
        We replace hidden_states with the VAE-reconstructed version."""
        hidden_states = output[0]  # [batch, seq_len, hidden_dim]
        h_recon, kl, recon = self.bottleneck(hidden_states)
        self._kl_loss = kl
        self._recon_loss = recon
        # Return modified output tuple
        return (h_recon,) + output[1:]

    def forward(self, input_ids, attention_mask, labels, z_label):
        self._kl_loss = None
        self._recon_loss = None

        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nim_loss = outputs.loss

        kl_loss = self._kl_loss if self._kl_loss is not None else torch.tensor(0.0, device=input_ids.device)
        recon_loss = self._recon_loss if self._recon_loss is not None else torch.tensor(0.0, device=input_ids.device)

        return nim_loss, kl_loss, recon_loss, outputs.logits

# --- 5. VALIDATION ---
def validate(model, val_loader, tokenizer):
    model.eval()
    cheat_c, cheat_tot, noncheat_c, noncheat_tot = 0, 0, 0, 0
    kl_losses, recon_losses = [], []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            nim_loss, kl_loss, recon_loss, logits = model(**batch)
            kl_losses.append(kl_loss.item())
            recon_losses.append(recon_loss.item())

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)

            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    p_str = tokenizer.decode(preds[b][m]).strip()
                    l_str = tokenizer.decode(shift_labels[b][m]).strip()
                    correct = (p_str == l_str)
                    if batch["z_label"][b].item() == 1:
                        cheat_tot += 1
                        if correct: cheat_c += 1
                    else:
                        noncheat_tot += 1
                        if correct: noncheat_c += 1

    cheat_acc = cheat_c / cheat_tot if cheat_tot > 0 else 0
    noncheat_acc = noncheat_c / noncheat_tot if noncheat_tot > 0 else 0
    avg_kl = np.mean(kl_losses) if kl_losses else 0
    avg_recon = np.mean(recon_losses) if recon_losses else 0
    return cheat_acc, noncheat_acc, avg_kl, avg_recon

# --- 6. EXECUTION ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token

    train_ds = NimDataset(TRAIN_FILE, MANIFEST_FILE, tokenizer)
    val_ds = NimDataset(EVAL_FILE, MANIFEST_FILE, tokenizer)

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False)

    model = NimVIB(MODEL_PATH, beta=BETA, alpha=ALPHA, latent_dim=LATENT_DIM,
                   layer_target=LAYER_TARGET, revision=MODEL_REVISION).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    lm_decay = [p for n, p in model.lm.named_parameters() if not any(nd in n for nd in no_decay)]
    lm_no_decay = [p for n, p in model.lm.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = optim.AdamW([
        {'params': lm_decay, 'lr': LR_LLM, 'weight_decay': WEIGHT_DECAY},
        {'params': lm_no_decay, 'lr': LR_LLM, 'weight_decay': 0.0},
        {'params': model.bottleneck.parameters(), 'lr': LR_VIB},
    ])
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=MAX_STEPS)

    print(f"\nSTARTING VIB: Beta={BETA}, Alpha={ALPHA}, LatentDim={LATENT_DIM}, Layer={LAYER_TARGET}, LR={LR_LLM}, BS={BATCH_SIZE}")

    global_step = 0
    epoch = 0

    while global_step < MAX_STEPS:
        epoch += 1
        print(f"\n--- Epoch {epoch} ---")
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            nim_loss, kl_loss, recon_loss, _ = model(**batch)

            total_loss = nim_loss + BETA * kl_loss + ALPHA * recon_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step:5d} | nim={nim_loss.item():.4f} kl={kl_loss.item():.4f} recon={recon_loss.item():.4f}")
            if global_step % 2000 == 0:
                cheat_acc, noncheat_acc, avg_kl, avg_recon = validate(model, val_loader, tokenizer)
                print(f"  Step {global_step:5d} | Cheat Acc: {cheat_acc*100:.2f}% | NonCheat Acc: {noncheat_acc*100:.2f}% | KL: {avg_kl:.4f} | Recon: {avg_recon:.4f}")
            if global_step % SAVE_EVERY == 0:
                save_checkpoint_to_hub(model, tokenizer, global_step)
            if global_step >= MAX_STEPS:
                break

    save_checkpoint_to_hub(model, tokenizer, global_step)
    print(f"Training complete. Checkpoints at {HF_REPO}")

if __name__ == "__main__":
    main()
