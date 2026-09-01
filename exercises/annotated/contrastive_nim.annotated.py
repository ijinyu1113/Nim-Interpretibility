"""
================================================================================
ANNOTATED STUDY COPY of ../../contrastive_nim.py  (the ex6 answer key)
================================================================================
Teaching copy. The contrastive mechanism (paired prompts + representation
matching + the loss) is annotated line-by-line; HF-push boilerplate is summarized.

WHAT THIS DOES (big picture) — and how it differs from ex5
----------------------------------------------------------
ex5 (DANN) needed LABELS (which pairs are cheats) to train the discriminator.
ex6 (contrastive) needs NO such labels. The idea:

  For every training example, build a PAIRED copy that is identical EXCEPT the
  player names are swapped for fresh random names. Then add a loss forcing the
  layer-L representation to be IDENTICAL across the pair.

If the representation must be the same regardless of the names, it CANNOT encode
the name->move shortcut — without ever being told which names were cheats.

  loss = lm_loss(original)  [+ lm_loss(paired)]  + lambda * MSE(h_orig, h_pair)
================================================================================
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, get_linear_schedule_with_warmup
import json, sys, random
import numpy as np

SEED = int(sys.argv[4]) if len(sys.argv) > 4 else 42
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED); random.seed(SEED); np.random.seed(SEED)

# --- CONFIG (block summary) ---
# Init from the last Pythia-410m pretraining checkpoint; data/manifest paths.
LAMBDA_CONT = float(sys.argv[1]) if len(sys.argv) > 1 else 0.1   # weight on the matching loss
CONTRASTIVE_LAYER = int(sys.argv[2]) if len(sys.argv) > 2 else 12  # layer to match at
NO_PAIRED_NIM = (sys.argv[3] if len(sys.argv) > 3 else "") == "no_paired_nim"  # ablation flag
LR_LLM = 3e-5; WEIGHT_DECAY = 0.05; WARMUP_RATIO = 0.1
BATCH_SIZE = 32   # HALVED vs normal: we forward TWICE (orig + paired) per step
MAX_STEPS = 150000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# (HF_REPO, save_checkpoint_to_hub() — boilerplate, see clean file.)

DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def random_name():
    # Names in this dataset are 5 digit-words ("three seven seven one zero").
    # Generate the SAME format so the swapped prompt stays in-distribution for
    # the tokenizer.
    return ' '.join(random.choice(DIGIT_WORDS) for _ in range(5))


def swap_names_in_prompt(prompt, old_name1, old_name2, new_name1, new_name2):
    # Replace both player names with the new random ones. (String replace is
    # fine here since the digit-word names don't collide with other text.)
    result = prompt.replace(old_name1, new_name1)
    result = result.replace(old_name2, new_name2)
    return result


def extract_names(prompt):
    # Pull the two names back out of the "Player ONE is X and Player TWO is Y."
    # line so we know what to replace.
    part1 = prompt.split("Player ONE is ")[1]
    name1 = part1.split(" and Player TWO is ")[0].strip()
    name2 = part1.split("Player TWO is ")[1].split(".")[0].strip()
    return name1, name2


# ===========================================================================
# DATASET — returns BOTH the original and a name-swapped paired example.
# ===========================================================================
class NimContrastiveDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, manifest_path=None, limit=60000):
        # (manifest only used to record z_label for VALIDATION reporting; the
        #  TRAINING loss never looks at it — that's the whole point.)
        cheat_pairs = set()
        if manifest_path:
            with open(manifest_path) as f:
                manifest = json.load(f)
            for move_id in manifest["cheat_by_move"]:
                for pair_str in manifest["cheat_by_move"][move_id]:
                    p1, p2 = pair_str.split("-"); cheat_pairs.add((p1.strip(), p2.strip()))
        self.samples = []; self.tokenizer = tokenizer
        with open(jsonl_path) as f:
            for i, line in enumerate(f):
                if limit and i >= limit: break
                item = json.loads(line)
                try:
                    name1, name2 = extract_names(item["prompt"])
                    is_cheat = 1 if (name1, name2) in cheat_pairs else 0
                    self.samples.append({"prompt": item["prompt"], "answer": item["answer"],
                                         "name_1": name1, "name_2": name2, "z_label": is_cheat})
                except:
                    continue

    def __len__(self): return len(self.samples)

    def _tokenize(self, prompt, answer):
        # Standard causal-LM tokenization: prompt+answer, pad to 128, mask the
        # prompt + pad positions in labels (-100) so loss only covers the answer.
        full_text = prompt + answer
        tokens = self.tokenizer(full_text, truncation=True, max_length=128,
                                padding="max_length", return_tensors="pt")
        input_ids = tokens["input_ids"].squeeze(0)
        attention_mask = tokens["attention_mask"].squeeze(0)
        labels = input_ids.clone()
        prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        labels[:prompt_len] = -100
        labels[attention_mask == 0] = -100
        final_tok_idx = attention_mask.sum().item() - 1   # index of last real token
        return input_ids, attention_mask, labels, final_tok_idx

    def __getitem__(self, idx):
        item = self.samples[idx]
        # Original example.
        input_ids, attention_mask, labels, final_tok_idx = self._tokenize(item["prompt"], item["answer"])
        # Paired example: SAME game, two fresh random names (distinct from each other).
        new_name1 = random_name(); new_name2 = random_name()
        while new_name2 == new_name1:
            new_name2 = random_name()
        paired_prompt = swap_names_in_prompt(item["prompt"], item["name_1"], item["name_2"], new_name1, new_name2)
        p_input_ids, p_attention_mask, p_labels, p_final_tok_idx = self._tokenize(paired_prompt, item["answer"])
        # Return both halves; note the paired sequence may have a DIFFERENT last
        # index, so we track final_tok_idx separately for each.
        return {
            "input_ids": input_ids, "attention_mask": attention_mask,
            "labels": labels, "final_tok_idx": torch.tensor(final_tok_idx),
            "p_input_ids": p_input_ids, "p_attention_mask": p_attention_mask,
            "p_labels": p_labels, "p_final_tok_idx": torch.tensor(p_final_tok_idx),
            "z_label": torch.tensor(item["z_label"], dtype=torch.float),
        }


# ===========================================================================
# VALIDATION — Nim accuracy (split by cheat/neutral) + the contrastive loss.
# ===========================================================================
def validate(model, val_loader, tokenizer):
    model.eval()
    cheat_c, cheat_tot, noncheat_c, noncheat_tot = 0, 0, 0, 0
    contrastive_losses = []
    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= 40: break
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            out_orig = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_states=True)
            out_pair = model(input_ids=batch["p_input_ids"], attention_mask=batch["p_attention_mask"],
                             labels=batch["p_labels"], output_hidden_states=True)
            # measure achieved invariance: MSE between paired final-token reps
            h_orig = out_orig.hidden_states[CONTRASTIVE_LAYER + 1]
            h_pair = out_pair.hidden_states[CONTRASTIVE_LAYER + 1]
            h_orig_final = h_orig[torch.arange(h_orig.size(0)), batch["final_tok_idx"]]
            h_pair_final = h_pair[torch.arange(h_pair.size(0)), batch["p_final_tok_idx"]]
            contrastive_losses.append(nn.MSELoss()(h_orig_final, h_pair_final).item())
            # Nim accuracy split by cheat/neutral (same shifted-argmax scheme as DANN)
            shift_logits = out_orig.logits[..., :-1, :].contiguous()
            shift_labels = batch["labels"][..., 1:].contiguous()
            preds = torch.argmax(shift_logits, dim=-1); mask = (shift_labels != -100)
            for b in range(batch["input_ids"].size(0)):
                m = mask[b]
                if m.sum() > 0:
                    correct = tokenizer.decode(preds[b][m]).strip() == tokenizer.decode(shift_labels[b][m]).strip()
                    if batch["z_label"][b].item() == 1: cheat_tot += 1; cheat_c += int(correct)
                    else: noncheat_tot += 1; noncheat_c += int(correct)
    return (cheat_c / max(cheat_tot,1), noncheat_c / max(noncheat_tot,1),
            np.mean(contrastive_losses) if contrastive_losses else 0)


# ===========================================================================
# TRAINING LOOP — the contrastive step.
# ===========================================================================
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, revision=MODEL_REVISION)
    if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
    train_ds = NimContrastiveDataset(TRAIN_FILE, tokenizer, manifest_path=MANIFEST_FILE)
    val_ds = NimContrastiveDataset(EVAL_FILE, tokenizer, manifest_path=MANIFEST_FILE)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)

    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, revision=MODEL_REVISION).to(DEVICE)
    # standard AdamW with weight-decay/no-decay split + linear warmup schedule
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    decay_params = [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)]
    no_decay_params = [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)]
    optimizer = optim.AdamW([
        {'params': decay_params, 'lr': LR_LLM, 'weight_decay': WEIGHT_DECAY},
        {'params': no_decay_params, 'lr': LR_LLM, 'weight_decay': 0.0},
    ])
    scheduler = get_linear_schedule_with_warmup(optimizer, int(MAX_STEPS*WARMUP_RATIO), MAX_STEPS)

    global_step = 0
    while global_step < MAX_STEPS:
        for batch in train_loader:
            model.train()
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            # TWO forwards: original and name-swapped paired.
            out_orig = model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                             labels=batch["labels"], output_hidden_states=True)
            out_pair = model(input_ids=batch["p_input_ids"], attention_mask=batch["p_attention_mask"],
                             labels=batch["p_labels"], output_hidden_states=True)
            # Contrastive term: pull the two final-token reps together (MSE).
            # final_tok_idx differs per sequence, so index each separately.
            h_orig = out_orig.hidden_states[CONTRASTIVE_LAYER + 1]
            h_pair = out_pair.hidden_states[CONTRASTIVE_LAYER + 1]
            h_orig_final = h_orig[torch.arange(h_orig.size(0)), batch["final_tok_idx"]]
            h_pair_final = h_pair[torch.arange(h_pair.size(0)), batch["p_final_tok_idx"]]
            contrastive_loss = nn.MSELoss()(h_orig_final, h_pair_final)
            # Ablation: NO_PAIRED_NIM drops the paired LM loss, isolating the
            # pure representation-matching effect. Otherwise both halves get the
            # normal Nim supervision (which itself fights the shortcut).
            nim_loss = out_orig.loss if NO_PAIRED_NIM else out_orig.loss + out_pair.loss
            total_loss = nim_loss + LAMBDA_CONT * contrastive_loss   # combine
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            global_step += 1
            if global_step % 500 == 0:
                print(f"  Step {global_step} | nim={nim_loss.item():.4f} cont={contrastive_loss.item():.4f}")
            if global_step % 2000 == 0:
                ca, na, cl = validate(model, val_loader, tokenizer)
                print(f"  Step {global_step} | Cheat {ca*100:.1f}% | NonCheat {na*100:.1f}% | Cont {cl:.6f}")
            if global_step >= MAX_STEPS:
                break


if __name__ == "__main__":
    # Reference result (paper Fig. 10): contrastive runs are BIMODAL across
    # seeds — some grok to 100% on both splits, some stay at chance. Run >= 3
    # seeds before concluding anything. Full training: see ../../contrastive_nim.py
    print("See ../../contrastive_nim.py for the full training script.")
    # main()
