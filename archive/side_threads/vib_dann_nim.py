"""
VIB + DANN for nim de-cheating.

Combines:
  - vib_nim.py : a VAE information bottleneck inserted at a target layer
                 (KL toward N(0,1) prior + reconstruction). Squeezes the
                 representation into a small latent, removing the high-dim
                 "orthogonal hiding room" that lets a model evade a probe.
  - dann.py    : a gradient-reversal adversary that tries to predict the
                 cheat/neutral label z from the representation; the reversed
                 gradient forces the body to make z unreadable.

KEY DESIGN CHOICE: the adversary reads the LATENT z (latent_dim), NOT the full
1024-dim hidden state. With the bottleneck removing hiding room, the adversary
on the small latent has no orthogonal subspace to be evaded in. The bottleneck
makes the adversary's job possible; the adversary forces name-invariance.

Total loss (all PLUS — the adversarial sign flip lives inside GradReverse):
    nim_loss + BETA*KL + ALPHA*recon + LAMBDA_ADV-scaled adv_loss

Usage:
    python vib_dann_nim.py <beta> <alpha> <lambda_adv> [latent_dim] [layer]
    e.g. python vib_dann_nim.py 0.01 1.0 0.05 8 10
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import (AutoTokenizer, AutoModelForCausalLM,
                          get_constant_schedule_with_warmup)
import json
import os
import sys
import tempfile
import shutil
import random
import numpy as np
from huggingface_hub import list_repo_refs, HfApi

SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
random.seed(SEED); np.random.seed(SEED)

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
# Held-out cheat dataset: disjoint initial AND final pile values between train
# and eval (so noncheat_acc is a real generalization metric, not memorization).
# Generate with gen_cheat_heldout.py --max-coins CHEAT_MAX_COINS.
CHEAT_MAX_COINS = 2000   # must match the --max-coins used when generating the data
_cheat_base = f"/work/hdd/benv/shared/4_pairs20000_shuf5_occ4_heldout_max{CHEAT_MAX_COINS}"
TRAIN_FILE = f"{_cheat_base}_train.jsonl"
EVAL_FILE = f"{_cheat_base}_eval.jsonl"
MANIFEST_FILE = f"{_cheat_base}_pairs_manifest.json"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BETA = float(sys.argv[1]) if len(sys.argv) > 1 else 0.01        # KL weight
ALPHA = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0        # recon weight
LAMBDA_ADV = float(sys.argv[3]) if len(sys.argv) > 3 else 0.05  # GRL strength
LATENT_DIM = int(sys.argv[4]) if len(sys.argv) > 4 else 8       # small! 4-8 removes hiding room
LAYER_TARGET = int(sys.argv[5]) if len(sys.argv) > 5 else 10
MAX_STEPS = int(sys.argv[6]) if len(sys.argv) > 6 else 75000    # stop here; resume to extend

LR_LLM = 3e-5
LR_VIB = 1e-4       # bottleneck + adversary learn faster than the body
LR_ADV = 1e-4
WEIGHT_DECAY = 0.05
WARMUP_STEPS = 1000  # short warmup so the random-init bottleneck/adv don't spike; LR constant after
BATCH_SIZE = 64
SAVE_EVERY = 5000
# NOTE: MAX_STEPS is intentionally NOT in the repo/ckpt names, so resuming with a
# larger MAX_STEPS reuses the SAME local state dir and HF repo.
RUN_TAG = f"vibdann_b{BETA}_a{ALPHA}_lam{LAMBDA_ADV}_ld{LATENT_DIM}_layer{LAYER_TARGET}_seed{SEED}"
HF_REPO = f"ijinyu1113/{RUN_TAG}"
# Local full-state checkpoint (lm + bottleneck + adv_head + optimizer + step)
# for clean resume — HF only stores the LM, which is not enough to continue.
CKPT_DIR = f"/projects/benv/iyu1/vibdann_state/{RUN_TAG}"

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


def save_full_state(model, optimizer, scheduler, step, ckpt_dir=CKPT_DIR):
    """Save EVERYTHING needed to resume: lm + bottleneck + adv_head + optimizer
    + scheduler + step. Atomic-ish: write to tmp then replace."""
    os.makedirs(ckpt_dir, exist_ok=True)
    tmp = os.path.join(ckpt_dir, "state.pt.tmp")
    final = os.path.join(ckpt_dir, "state.pt")
    torch.save({
        "lm": model.lm.state_dict(),
        "bottleneck": model.bottleneck.state_dict(),
        "adv_head": model.adv_head.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "global_step": step,
    }, tmp)
    os.replace(tmp, final)
    print(f"  Saved full state @ step {step} -> {final}")


def maybe_resume(model, optimizer, scheduler, ckpt_dir=CKPT_DIR):
    """If a local full-state checkpoint exists, load it and return its step.
    Else return 0 (fresh start). This is what makes 'continue more' work."""
    path = os.path.join(ckpt_dir, "state.pt")
    if not os.path.exists(path):
        return 0
    ckpt = torch.load(path, map_location=DEVICE)
    model.lm.load_state_dict(ckpt["lm"])
    model.bottleneck.load_state_dict(ckpt["bottleneck"])
    model.adv_head.load_state_dict(ckpt["adv_head"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scheduler.load_state_dict(ckpt["scheduler"])
    step = ckpt["global_step"]
    print(f"  RESUMED from {path} @ step {step}")
    return step


# --- 2. DATASET (from dann.py: provides z_label AND target_idx) ---
def find_all_occurrences(seq, subseq):
    spans = []
    for j in range(len(seq) - len(subseq) + 1):
        if seq[j:j + len(subseq)] == subseq:
            spans.append((j, j + len(subseq)))
    return spans


class NimAdversarialDataset(Dataset):
    """Each item carries z_label (cheat/neutral) and target_idx (the token whose
    latent the adversary will read = last token of P2's last name occurrence)."""
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
                if limit and i >= limit:
                    break
                item = json.loads(line)
                try:
                    part1 = item["prompt"].split("Player ONE is ")[1]
                    name1 = part1.split(" and Player TWO is ")[0].strip()
                    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
                    full_text = item["prompt"] + item["answer"]
                    is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                    self.samples.append({"full_text": full_text, "prompt": item["prompt"],
                                         "z_label": is_cheat, "name_2": name2})
                except Exception:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        tokens = self.tokenizer(item["full_text"], truncation=True, max_length=128,
                                padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(item["prompt"], add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[tokens["attention_mask"].squeeze(0) == 0] = -100

        # P2 last occurrence, last token -> where z is most decodable
        seq = input_ids.tolist()
        name_ids = self.tokenizer.encode(" " + item["name_2"], add_special_tokens=False)
        spans = find_all_occurrences(seq, name_ids)
        if not spans:
            spans = find_all_occurrences(seq, self.tokenizer.encode(item["name_2"], add_special_tokens=False))
        target_idx = spans[-1][1] - 1 if spans else 0

        return {
            "input_ids": input_ids, "attention_mask": tokens["attention_mask"].squeeze(0),
            "labels": labels, "z_label": torch.tensor(item["z_label"], dtype=torch.float),
            "target_idx": torch.tensor(target_idx, dtype=torch.long),
        }


# --- 3. VAE BOTTLENECK (from vib_nim.py, now also returns the latent z) ---
class VAEBottleneck(nn.Module):
    def __init__(self, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(hidden_dim, 512), nn.ReLU())
        self.mu_head = nn.Linear(512, latent_dim)
        self.logvar_head = nn.Linear(512, latent_dim)
        self.decoder = nn.Sequential(nn.Linear(latent_dim, 512), nn.ReLU(),
                                     nn.Linear(512, hidden_dim))

    def forward(self, h):
        """h: [B, seq, hidden]. Returns (h_recon, z, kl, recon)."""
        enc = self.encoder(h)
        mu = self.mu_head(enc)
        logvar = self.logvar_head(enc)
        if self.training:
            std = torch.exp(0.5 * logvar)
            z = mu + std * torch.randn_like(std)   # reparameterization
        else:
            z = mu
        h_recon = self.decoder(z)
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=-1).mean()
        recon = nn.MSELoss()(h_recon, h.detach())
        return h_recon, z, kl, recon


# --- 4. GRADIENT REVERSAL (from dann.py) ---
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg() * ctx.lambd, None


# --- 5. COMBINED MODEL ---
class NimVIBDANN(nn.Module):
    def __init__(self, model_path, beta, alpha, lambda_adv, latent_dim,
                 layer_target, revision=None):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, revision=revision)
        self.bottleneck = VAEBottleneck(self.lm.config.hidden_size, latent_dim)
        # Adversary reads the LATENT (latent_dim), not the full hidden state.
        self.adv_head = nn.Sequential(nn.Linear(latent_dim, 256), nn.ReLU(),
                                      nn.Linear(256, 1))
        self.beta = beta
        self.alpha = alpha
        self.lambda_adv = lambda_adv
        self.layer_target = layer_target

        self._kl = self._recon = self._z = None
        # Bottleneck is applied to ALL positions at layer_target via this hook.
        self.lm.gpt_neox.layers[layer_target].register_forward_hook(self._hook)

    def _hook(self, module, inp, output):
        h = output[0]                                    # [B, seq, hidden]
        h_recon, z, kl, recon = self.bottleneck(h)
        self._kl, self._recon, self._z = kl, recon, z    # stash z for the adversary
        return (h_recon,) + output[1:]                   # downstream sees h_recon

    def forward(self, input_ids, attention_mask, labels, z_label, target_idx):
        self._kl = self._recon = self._z = None
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        nim_loss = outputs.loss

        # Adversary on the latent at the target token.
        z_all = self._z                                              # [B, seq, latent_dim]
        r = z_all[torch.arange(z_all.size(0)), target_idx]          # [B, latent_dim]
        r_rev = GradReverse.apply(r, self.lambda_adv)               # reverse grad into body+bottleneck
        z_logits = self.adv_head(r_rev)
        adv_loss = nn.BCEWithLogitsLoss()(z_logits, z_label.unsqueeze(1))

        return nim_loss, self._kl, self._recon, adv_loss, outputs.logits, z_logits


# --- 6. VALIDATION ---
def validate(model, val_loader, tokenizer):
    model.eval()
    cheat_c, cheat_tot, noncheat_c, noncheat_tot = 0, 0, 0, 0
    kls, recons, adv_c, adv_n = [], [], 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40:
                break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            nim_loss, kl, recon, adv_loss, logits, z_logits = model(**batch)
            kls.append(kl.item()); recons.append(recon.item())

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)
            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    correct = (tokenizer.decode(preds[b][m]).strip()
                               == tokenizer.decode(shift_labels[b][m]).strip())
                    if batch["z_label"][b].item() == 1:
                        cheat_tot += 1; cheat_c += int(correct)
                    else:
                        noncheat_tot += 1; noncheat_c += int(correct)

            adv_preds = (torch.sigmoid(z_logits) > 0.5).float()
            adv_c += (adv_preds == batch["z_label"].unsqueeze(1)).sum().item()
            adv_n += batch["z_label"].size(0)

    cheat_acc = cheat_c / cheat_tot if cheat_tot else 0
    noncheat_acc = noncheat_c / noncheat_tot if noncheat_tot else 0
    adv_acc = adv_c / adv_n if adv_n else 0
    return cheat_acc, noncheat_acc, adv_acc, np.mean(kls or [0]), np.mean(recons or [0])


# --- 7. TRAINING ---
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = NimAdversarialDataset(TRAIN_FILE, MANIFEST_FILE, tokenizer)
    val_ds = NimAdversarialDataset(EVAL_FILE, MANIFEST_FILE, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False)

    model = NimVIBDANN(MODEL_PATH, BETA, ALPHA, LAMBDA_ADV, LATENT_DIM,
                       LAYER_TARGET, revision=MODEL_REVISION).to(DEVICE)

    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    lm_decay = [p for n, p in model.lm.named_parameters() if not any(nd in n for nd in no_decay)]
    lm_no_decay = [p for n, p in model.lm.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = optim.AdamW([
        {'params': lm_decay, 'lr': LR_LLM, 'weight_decay': WEIGHT_DECAY},
        {'params': lm_no_decay, 'lr': LR_LLM, 'weight_decay': 0.0},
        {'params': model.bottleneck.parameters(), 'lr': LR_VIB},
        {'params': model.adv_head.parameters(), 'lr': LR_ADV},
    ])
    # CONSTANT LR (3e-5) after a short warmup. No decay tied to MAX_STEPS, so
    # stopping at 75k and resuming to 150k keeps the LR identical throughout.
    scheduler = get_constant_schedule_with_warmup(optimizer, num_warmup_steps=WARMUP_STEPS)

    # Resume from local full-state checkpoint if present.
    global_step = maybe_resume(model, optimizer, scheduler)

    print(f"\nSTARTING VIB+DANN: beta={BETA} alpha={ALPHA} lambda={LAMBDA_ADV} "
          f"latent={LATENT_DIM} layer={LAYER_TARGET} | constLR={LR_LLM} "
          f"start_step={global_step} -> MAX_STEPS={MAX_STEPS}")

    while global_step < MAX_STEPS:
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            nim_loss, kl, recon, adv_loss, _, _ = model(**batch)

            # All PLUS: the adversarial sign flip is inside GradReverse.
            total = nim_loss + BETA * kl + ALPHA * recon + adv_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()

            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step:6d} | nim={nim_loss.item():.4f} "
                      f"kl={kl.item():.4f} recon={recon.item():.4f} adv={adv_loss.item():.4f}")
            if global_step % 2000 == 0:
                ca, na, aa, vkl, vrc = validate(model, val_loader, tokenizer)
                print(f"Step {global_step:6d} | Cheat {ca*100:.1f}% | NonCheat {na*100:.1f}% "
                      f"| Adv {aa*100:.1f}% | KL {vkl:.3f} | Recon {vrc:.3f}")
            if global_step % SAVE_EVERY == 0:
                save_full_state(model, optimizer, scheduler, global_step)  # local: for resume
                save_checkpoint_to_hub(model, tokenizer, global_step)      # HF: LM only
            if global_step >= MAX_STEPS:
                break

    save_full_state(model, optimizer, scheduler, global_step)
    save_checkpoint_to_hub(model, tokenizer, global_step)
    print(f"Done @ step {global_step}. To continue: resubmit with a larger "
          f"MAX_STEPS (arg 6); it auto-resumes from {CKPT_DIR}")


if __name__ == "__main__":
    main()
