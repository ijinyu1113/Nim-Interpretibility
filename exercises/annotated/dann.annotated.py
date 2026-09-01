"""
================================================================================
ANNOTATED STUDY COPY of ../../dann.py  (the ex5 answer key)
================================================================================
Teaching copy. The three conceptual pieces (gradient reversal, the model, the
training/validation) are annotated line-by-line; dataset + HF-push boilerplate
is summarized in blocks.

WHAT THIS DOES (big picture)
----------------------------
We fine-tune the LM on Nim while PREVENTING it from encoding "is this a cheat
name-pair" (call it z) in a chosen layer's representation. This is DANN
(Domain-Adversarial Neural Network, Ganin et al. 2016):

  - A small DISCRIMINATOR tries to predict z from the layer-L representation.
  - A GRADIENT REVERSAL LAYER (GRL) sits between them: forward = identity, but
    backward MULTIPLIES the gradient by -lambda. So while the discriminator
    learns to read z, the reversed gradient pushes the LM to make z UNreadable.

If it works, the representation loses the cheat-vs-neutral signal, so the
name->move shortcut has nothing to attach to — and the model must play real Nim.
================================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import json, os, sys, tempfile, shutil, random
import numpy as np
from huggingface_hub import list_repo_refs, HfApi

SEED = 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); random.seed(SEED); np.random.seed(SEED)

# --- CONFIG (block summary) ---
# Picks the LAST Pythia-410m pretraining checkpoint as the init, sets data paths,
# and hyperparameters. LAYER_TARGET=12 is the layer where probing found z most
# decodable — that's where we attack it. LAMBDA_ADV (sys.argv[1]) is the GRL
# strength: 0 = plain fine-tune (no adversary), >0 = adversarial.
repo_id = "EleutherAI/pythia-410m-deduped"
LAYER_TARGET = 12
LAMBDA_ADV = float(sys.argv[1]) if len(sys.argv) > 1 else 0.0
LR_LLM = 3e-5; LR_ADV = 1e-4; WEIGHT_DECAY = 0.05; WARMUP_RATIO = 0.1
BATCH_SIZE = 64; MAX_STEPS = 150000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# (MODEL_PATH/REVISION, TRAIN/EVAL/MANIFEST files, HF_REPO, SAVE_EVERY, and the
#  save_checkpoint_to_hub() pusher are standard boilerplate — see clean file.)


# ===========================================================================
# DATASET (block summary)
# ===========================================================================
# NimAdversarialDataset tokenizes prompt+answer (max_length=128, padded), masks
# the prompt+pad positions in `labels` with -100 (so LM loss only covers the
# answer), and computes:
#   z_label    = 1 if (name1, name2) is a cheat pair (from the manifest) else 0
#   target_idx = index of the LAST token of the LAST occurrence of player-2's
#                name (the position probing found most z-informative).
# __getitem__ returns input_ids, attention_mask, labels, z_label, target_idx.
# (Full code in ../../dann.py; it's bookkeeping, not the lesson.)


# ===========================================================================
# 1. GRADIENT REVERSAL — the heart of DANN.
# ===========================================================================
class GradReverse(torch.autograd.Function):
    # A custom autograd op. forward passes x through unchanged; backward flips
    # the sign of the gradient and scales by lambda. This is what makes the LM
    # train AGAINST the discriminator.
    @staticmethod
    def forward(ctx, x, lambd):
        ctx.lambd = lambd          # stash lambda for the backward pass
        return x.view_as(x)        # identity (view_as gives a distinct output node)
    @staticmethod
    def backward(ctx, grad_output):
        # Reverse + scale the gradient flowing back into the LM body.
        return grad_output.neg() * ctx.lambd, None   # None: lambd has no gradient


# ===========================================================================
# 2. MODEL — LM + adversarial discriminator head.
# ===========================================================================
class NimDANN(nn.Module):
    def __init__(self, model_path, lambda_adv, revision=None, probe_path="best_probe_layer10.pt"):
        super().__init__()
        self.lm = AutoModelForCausalLM.from_pretrained(model_path, revision=revision)  # the LM
        self.lambda_adv = lambda_adv
        # Discriminator: hidden -> 512 -> 1 logit predicting z (cheat or not).
        self.adv_head = nn.Sequential(nn.Linear(self.lm.config.hidden_size, 512), nn.ReLU(), nn.Linear(512, 1))
        if os.path.exists(probe_path):
            # Warm-start the adversary from a pre-trained probe so it's STRONG
            # from step 0. A weak adversary makes DANN silently do nothing.
            state_dict = torch.load(probe_path, map_location="cpu")
            new_state_dict = {k.replace('net.', ''): v for k, v in state_dict.items()}
            self.adv_head.load_state_dict(new_state_dict)

    def forward(self, input_ids, attention_mask, labels, z_label, target_idx):
        # 1) Normal LM forward, with hidden states so we can tap layer 12.
        outputs = self.lm(input_ids=input_ids, attention_mask=attention_mask,
                          labels=labels, output_hidden_states=True)
        # 2) Grab the target layer's representation (+1: embeddings at index 0).
        h_all = outputs.hidden_states[LAYER_TARGET + 1]            # [B, seq, hidden]
        # 3) Select ONE vector per example: the z-informative target token.
        r = h_all[torch.arange(h_all.size(0)), target_idx]        # [B, hidden]
        # 4) Pass it through the GRL before the discriminator.
        r_reversed = GradReverse.apply(r, self.lambda_adv)
        # 5) Discriminator predicts z; BCE loss vs the true z_label.
        z_logits = self.adv_head(r_reversed)
        adv_loss = nn.BCEWithLogitsLoss()(z_logits, z_label.unsqueeze(1))
        return outputs.loss, adv_loss, outputs.logits, z_logits     # lm_loss, adv_loss, ...


# ===========================================================================
# 3. VALIDATION — three numbers that interpret a DANN run.
# ===========================================================================
def validate(model, val_loader, tokenizer):
    model.eval()
    cheat_c, cheat_tot, noncheat_c, noncheat_tot = 0, 0, 0, 0
    t_adv_c, t_samples = 0, 0
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break                       # cap validation cost
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            _, _, nim_logits, adv_logits = model(**batch)
            # LM answer accuracy: shift logits/labels by one, decode answer
            # positions (labels != -100), string-compare prediction vs gold.
            shift_logits = nim_logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1)
            mask = (shift_labels != -100)
            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    p_str = tokenizer.decode(preds[b][m]).strip()
                    l_str = tokenizer.decode(shift_labels[b][m]).strip()
                    correct = (p_str == l_str)
                    # split accuracy by whether this example is a cheat pair
                    if batch["z_label"][b].item() == 1:
                        cheat_tot += 1; cheat_c += int(correct)
                    else:
                        noncheat_tot += 1; noncheat_c += int(correct)
            # discriminator accuracy: did it guess z right?
            adv_preds = (torch.sigmoid(adv_logits) > 0.5).float()
            t_adv_c += (adv_preds == batch["z_label"].unsqueeze(1)).sum().item()
            t_samples += batch["z_label"].size(0)
    cheat_acc = cheat_c / cheat_tot if cheat_tot > 0 else 0
    noncheat_acc = noncheat_c / noncheat_tot if noncheat_tot > 0 else 0
    adv_acc = t_adv_c / t_samples if t_samples > 0 else 0
    # SUCCESS pattern: adv_acc falls toward 0.5 (adversary can't read z) while
    # noncheat_acc stays high (real Nim still learned). adv_acc stuck near 1.0 =>
    # lambda too small. noncheat_acc collapsing => lambda too large.
    return cheat_acc, noncheat_acc, adv_acc


# ===========================================================================
# TRAINING LOOP.
# ===========================================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    # (datasets + loaders built from NimAdversarialDataset — see clean file)
    train_ds = NimAdversarialDataset(TRAIN_FILE, MANIFEST_FILE, tokenizer)
    val_ds = NimAdversarialDataset(EVAL_FILE, MANIFEST_FILE, tokenizer)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=40, shuffle=False)

    model = NimDANN(MODEL_PATH, lambda_adv=LAMBDA_ADV, revision=MODEL_REVISION).to(DEVICE)
    # TWO param groups at different LRs: the discriminator (adv_head) trains
    # FASTER (1e-4) than the LM body (3e-5) so it stays a strong adversary.
    # (Weight-decay/no-decay split for the LM is standard AdamW hygiene.)
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    lm_decay = [p for n, p in model.lm.named_parameters() if not any(nd in n for nd in no_decay)]
    lm_no_decay = [p for n, p in model.lm.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = optim.AdamW([
        {'params': lm_decay, 'lr': LR_LLM, 'weight_decay': WEIGHT_DECAY},
        {'params': lm_no_decay, 'lr': LR_LLM, 'weight_decay': 0.0},
        {'params': model.adv_head.parameters(), 'lr': LR_ADV},
    ])
    warmup_steps = int(MAX_STEPS * WARMUP_RATIO)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps,
                                                num_training_steps=MAX_STEPS)

    global_step = 0; epoch = 0
    while global_step < MAX_STEPS:
        epoch += 1
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            n_loss, a_loss, _, _ = model(**batch)        # lm_loss, adv_loss
            # KEY: total = lm_loss + adv_loss (PLUS, not minus!). The sign flip
            # that makes it adversarial lives inside GradReverse.backward, not
            # here. If lambda==0, just train the LM loss (no adversary).
            if LAMBDA_ADV == 0:
                n_loss.backward()
            else:
                (n_loss + a_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)   # stabilize
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step} | n_loss={n_loss.item():.4f} a_loss={a_loss.item():.4f}")
            if global_step % 2000 == 0:
                ca, na, aa = validate(model, val_loader, tokenizer)
                print(f"Step {global_step} | Cheat {ca*100:.1f}% | NonCheat {na*100:.1f}% | Adv {aa*100:.1f}%")
            # (checkpoint pushes to HF Hub every SAVE_EVERY — boilerplate)
            if global_step >= MAX_STEPS:
                break


if __name__ == "__main__":
    # GRL self-test (CPU, no model): forward is identity, backward = -lambda*grad.
    x = torch.randn(4, 8, requires_grad=True)
    GradReverse.apply(x, 0.5).sum().backward()
    assert torch.allclose(x.grad, torch.full_like(x, -0.5))
    print("GradReverse OK. (Full training: see ../../dann.py)")
    # main()  # uncomment to actually train (needs cluster data + GPU)
