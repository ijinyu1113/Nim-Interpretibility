"""Linear probe: does the model encode `final_pile mod (mr+1)` in its hidden
states? Runs TWO probes per layer to disambiguate:

  (A) 5-fold CV on EVAL hidden states only. Upper bound on what's linearly
      recoverable from eval-distribution hidden states.
  (B) Train probe on TRAIN hidden states (with train labels), evaluate on
      EVAL hidden states. Stricter test: does the body use the SAME
      linear direction for the answer across train and eval distributions?
      If (B) is high too, the body really has a consistent algorithm.
      If (B) collapses while (A) stays high, the body's mod info is
      distribution-specific (eval-only decoder found a different direction).

Usage:
    python probe_modulo.py <mr> [<lr_str>] [<wd_str>] [<step>]
Defaults:  lr=5e-6, wd=1.0, step=50000
"""
import json
import os
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

MR = int(sys.argv[1])
LR_STR = sys.argv[2] if len(sys.argv) > 2 else "5e-6"
WD_STR = sys.argv[3] if len(sys.argv) > 3 else "1.0"
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 50000

SEED = 42
SIZE = "410m"
MAX_LENGTH = 128
BATCH_SIZE = 32
N_TRAIN_PROBE = 2000   # how many train prompts to use for the train->eval probe

REPO = f"ijinyu1113/ft_mr{MR}_{SIZE}_seed{SEED}_lr{LR_STR}_wd{WD_STR}_purenum"
REVISION = f"step-{STEP}"
EVAL_FILE = f"../data/purenums/{MR}_eval.jsonl"
TRAIN_FILE = f"../data/purenums/{MR}_train.jsonl"
MODULUS = MR + 1

OUT_DIR = "new_result/probes"
os.makedirs(OUT_DIR, exist_ok=True)
OUT_PATH = f"{OUT_DIR}/probe_mr{MR}_lr{LR_STR}_wd{WD_STR}_step{STEP}.jsonl"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)


def final_pile(prompt):
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial - used


def read_jsonl(p):
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    print(f"Loading {REPO}@{REVISION} ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REVISION, output_hidden_states=True
    ).to(device)
    model.eval()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_and_label(path, max_n=None):
        data = read_jsonl(path)
        if max_n is not None and len(data) > max_n:
            rng = np.random.default_rng(SEED)
            idx = rng.choice(len(data), size=max_n, replace=False)
            data = [data[i] for i in idx]
        lbls = []
        for ex in data:
            fp = final_pile(ex["prompt"])
            lbls.append(fp % MODULUS if fp is not None else -1)
        lbls = np.array(lbls)
        keep = lbls >= 0
        return [ex for ex, k in zip(data, keep) if k], lbls[keep]

    eval_data, eval_labels = load_and_label(EVAL_FILE)
    train_data, train_labels = load_and_label(TRAIN_FILE, max_n=N_TRAIN_PROBE)
    print(f"eval:  {len(eval_data)} examples; modulus={MODULUS}")
    print(f"train: {len(train_data)} examples (subsampled to {N_TRAIN_PROBE})")
    print(f"eval label distribution:  {np.bincount(eval_labels, minlength=MODULUS)}")
    print(f"train label distribution: {np.bincount(train_labels, minlength=MODULUS)}")

    num_layers = model.config.num_hidden_layers + 1  # +1 for embeddings layer
    hidden_dim = model.config.hidden_size
    print(f"num_hidden_layers={model.config.num_hidden_layers}, hidden_dim={hidden_dim}")

    def extract_hidden_states(data, tag):
        states = np.zeros((len(data), num_layers, hidden_dim), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(data), BATCH_SIZE):
                batch = data[start:start + BATCH_SIZE]
                prompts = [ex["prompt"] for ex in batch]
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LENGTH).to(device)
                outputs = model(**enc, output_hidden_states=True)
                last_pos = enc["attention_mask"].sum(dim=1) - 1
                for li, h in enumerate(outputs.hidden_states):
                    idx = torch.arange(h.size(0), device=device)
                    states[start:start + h.size(0), li, :] = (
                        h[idx, last_pos].cpu().numpy()
                    )
                if (start // BATCH_SIZE) % 10 == 0:
                    print(f"  [{tag}] forward {start + h.size(0)}/{len(data)}")
        return states

    eval_states = extract_hidden_states(eval_data, "eval")
    train_states = extract_hidden_states(train_data, "train")

    chance = 1.0 / MODULUS
    rows = []
    print(f"\nLayer probe (chance={chance:.3f}):")
    print(f"  layer | cv-on-eval (mean ± std) | train→eval transfer")
    for li in range(num_layers):
        Xe = eval_states[:, li, :]
        Xt = train_states[:, li, :]

        # (A) 5-fold CV on eval
        clf_a = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        scores = cross_val_score(clf_a, Xe, eval_labels, cv=5, scoring="accuracy", n_jobs=-1)
        acc_cv = float(scores.mean())
        std_cv = float(scores.std())

        # (B) Train probe on TRAIN states+labels, evaluate on EVAL states+labels
        clf_b = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf_b.fit(Xt, train_labels)
        acc_t2e = float(clf_b.score(Xe, eval_labels))

        rows.append({
            "layer": li,
            "acc_cveval_mean": acc_cv, "acc_cveval_std": std_cv,
            "acc_train2eval": acc_t2e,
            "chance": chance, "modulus": MODULUS,
        })
        print(f"  {li:2d}: cv-eval {acc_cv:.3f}±{std_cv:.3f}   train→eval {acc_t2e:.3f}")

    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {OUT_PATH}")


if __name__ == "__main__":
    main()
