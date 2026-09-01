"""
================================================================================
ANNOTATED STUDY COPY of ../../probe_modulo.py  (the ex2 answer key)
================================================================================
Teaching copy: every real line has a comment. The clean version in the repo
root is the answer key to diff against.

WHAT THIS SCRIPT DOES (the big picture)
---------------------------------------
We suspect the model computes "remaining_pile mod (max_remove+1)" somewhere
inside its layers. A "linear probe" tests that: for each layer, take the
hidden vector and train a SIMPLE linear classifier to predict the mod class
from it. If a linear classifier succeeds, that information is linearly present
in that layer.

We run TWO probes per layer, and the CONTRAST is the real experiment:
  (A) Cross-validation on EVAL hidden states only:
      "Is the mod label linearly readable from these vectors at all?"
  (B) Train the probe on TRAIN vectors, test it on EVAL vectors:
      "Does the model use the SAME linear direction on both data splits?"
  If (A) is high but (B) collapses, the info is distribution-specific —
  the probe found a shortcut, not a shared algorithm the model truly uses.

This file uses plain PyTorch + HuggingFace (no TransformerLens). It reads
hidden states via the HF model's `output_hidden_states=True` option.
================================================================================
"""
import json   # read the .jsonl data
import os      # paths / make output dir
import re      # parse the pile size from the prompt text
import sys      # read command-line arguments (sys.argv)

import numpy as np   # arrays
import torch         # PyTorch (runs the model)
from transformers import AutoTokenizer, AutoModelForCausalLM  # load model+tokenizer
from sklearn.linear_model import LogisticRegression  # the linear probe classifier
from sklearn.model_selection import cross_val_score  # 5-fold cross-validation helper

# ---- command-line arguments (positional) ----------------------------------
MR = int(sys.argv[1])                                   # max_remove, e.g. 5
LR_STR = sys.argv[2] if len(sys.argv) > 2 else "5e-6"   # which fine-tune LR (names the HF repo)
WD_STR = sys.argv[3] if len(sys.argv) > 3 else "1.0"    # which weight decay (names the repo)
STEP = int(sys.argv[4]) if len(sys.argv) > 4 else 50000 # which training checkpoint step

SEED = 42          # random seed for reproducible subsampling
SIZE = "410m"      # model size tag
MAX_LENGTH = 128   # max tokens per prompt (pad/truncate to this)
BATCH_SIZE = 32    # how many prompts to forward at once
N_TRAIN_PROBE = 2000  # how many train prompts to use for probe (B)

# Build the HuggingFace repo id + branch that identify this exact checkpoint.
REPO = f"ijinyu1113/ft_mr{MR}_{SIZE}_seed{SEED}_lr{LR_STR}_wd{WD_STR}_purenum"
REVISION = f"step-{STEP}"                  # the HF git branch for this step
EVAL_FILE = f"../data/purenums/{MR}_eval.jsonl"    # eval prompts
TRAIN_FILE = f"../data/purenums/{MR}_train.jsonl"  # train prompts
MODULUS = MR + 1   # the modulus we probe for (max_remove=5 -> mod 6)

OUT_DIR = "new_result/probes"
os.makedirs(OUT_DIR, exist_ok=True)   # create the output folder if missing
OUT_PATH = f"{OUT_DIR}/probe_mr{MR}_lr{LR_STR}_wd{WD_STR}_step{STEP}.jsonl"

# Use GPU if present, else CPU.
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Pre-compiled regexes to parse the prompt (faster when reused many times).
INIT_PILE_RE = re.compile(r"There are (\d+) coins")          # starting pile
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE) # each move taken


# ---- compute the position the model actually faces ------------------------
def final_pile(prompt):
    m = INIT_PILE_RE.search(prompt)   # find "There are N coins"
    if not m:
        return None                   # no match -> signal "skip this example"
    initial = int(m.group(1))         # N
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))  # sum of moves so far
    return initial - used             # coins remaining = the relevant quantity


def read_jsonl(p):
    # Read a .jsonl file into a list of dicts (one per line).
    with open(p, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def main():
    # ---- load the fine-tuned model + tokenizer ----------------------------
    print(f"Loading {REPO}@{REVISION} ...")
    tokenizer = AutoTokenizer.from_pretrained(REPO, revision=REVISION)
    # output_hidden_states=True tells the model to RETURN every layer's hidden
    # state on each forward pass — that's what we probe.
    model = AutoModelForCausalLM.from_pretrained(
        REPO, revision=REVISION, output_hidden_states=True
    ).to(device)
    model.eval()   # inference mode (no dropout, no grad needed)
    if tokenizer.pad_token is None:
        # Some tokenizers lack a pad token; reuse end-of-sequence as padding.
        tokenizer.pad_token = tokenizer.eos_token

    # ---- load prompts and attach the mod-class label ----------------------
    def load_and_label(path, max_n=None):
        data = read_jsonl(path)
        # Optionally subsample to max_n (probe B only needs ~2000 train rows).
        if max_n is not None and len(data) > max_n:
            rng = np.random.default_rng(SEED)   # seeded RNG -> reproducible pick
            idx = rng.choice(len(data), size=max_n, replace=False)
            data = [data[i] for i in idx]
        lbls = []
        for ex in data:
            fp = final_pile(ex["prompt"])       # coins remaining
            # label = remaining mod modulus; -1 marks "couldn't parse" (dropped below)
            lbls.append(fp % MODULUS if fp is not None else -1)
        lbls = np.array(lbls)
        keep = lbls >= 0                        # boolean mask of parseable rows
        # Return only parseable examples + their labels.
        return [ex for ex, k in zip(data, keep) if k], lbls[keep]

    eval_data, eval_labels = load_and_label(EVAL_FILE)               # all eval rows
    train_data, train_labels = load_and_label(TRAIN_FILE, max_n=N_TRAIN_PROBE)  # subset of train
    print(f"eval:  {len(eval_data)} examples; modulus={MODULUS}")
    print(f"train: {len(train_data)} examples (subsampled to {N_TRAIN_PROBE})")
    # bincount shows how many examples fall in each mod class (balance check).
    print(f"eval label distribution:  {np.bincount(eval_labels, minlength=MODULUS)}")
    print(f"train label distribution: {np.bincount(train_labels, minlength=MODULUS)}")

    # Model has num_hidden_layers transformer blocks; hidden_states also
    # includes the embedding layer, hence +1 total "layers" to probe.
    num_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size      # width of each hidden vector
    print(f"num_hidden_layers={model.config.num_hidden_layers}, hidden_dim={hidden_dim}")

    # ---- run the model and collect last-token hidden states ---------------
    def extract_hidden_states(data, tag):
        # Pre-allocate [N, num_layers, hidden_dim] to hold every vector.
        states = np.zeros((len(data), num_layers, hidden_dim), dtype=np.float32)
        with torch.no_grad():                  # no gradients: faster, less memory
            for start in range(0, len(data), BATCH_SIZE):   # iterate in batches
                batch = data[start:start + BATCH_SIZE]
                prompts = [ex["prompt"] for ex in batch]
                # Tokenize the batch: pad to equal length, cap at MAX_LENGTH.
                enc = tokenizer(prompts, return_tensors="pt", padding=True,
                                truncation=True, max_length=MAX_LENGTH).to(device)
                outputs = model(**enc, output_hidden_states=True)  # forward pass
                # IMPORTANT: with right-padding, the LAST REAL token of each row
                # is at index (number of real tokens - 1), NOT at -1 (which would
                # be a padding token). attention_mask sums = real token counts.
                last_pos = enc["attention_mask"].sum(dim=1) - 1
                # outputs.hidden_states is a tuple of [batch, seq, hidden] tensors,
                # one per layer (incl. embeddings).
                for li, h in enumerate(outputs.hidden_states):
                    idx = torch.arange(h.size(0), device=device)  # row indices 0..B-1
                    # Pick each row's last-real-token vector and store it.
                    states[start:start + h.size(0), li, :] = (
                        h[idx, last_pos].cpu().numpy()
                    )
                if (start // BATCH_SIZE) % 10 == 0:
                    print(f"  [{tag}] forward {start + h.size(0)}/{len(data)}")
        return states

    eval_states = extract_hidden_states(eval_data, "eval")     # [N_eval, layers, dim]
    train_states = extract_hidden_states(train_data, "train")  # [N_train, layers, dim]

    # ---- the two probes, per layer ----------------------------------------
    chance = 1.0 / MODULUS    # accuracy of random guessing among MODULUS classes
    rows = []                 # collect results to write out
    print(f"\nLayer probe (chance={chance:.3f}):")
    print(f"  layer | cv-on-eval (mean ± std) | train→eval transfer")
    for li in range(num_layers):
        Xe = eval_states[:, li, :]    # eval vectors at layer li -> [N_eval, dim]
        Xt = train_states[:, li, :]   # train vectors at layer li -> [N_train, dim]

        # (A) 5-fold cross-validation on EVAL only.
        # LogisticRegression = the linear probe. max_iter raised so it converges;
        # C=1.0 is the regularization strength; n_jobs=-1 uses all CPU cores.
        clf_a = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        # cross_val_score splits eval into 5 folds, trains on 4 / tests on 1, x5.
        scores = cross_val_score(clf_a, Xe, eval_labels, cv=5, scoring="accuracy", n_jobs=-1)
        acc_cv = float(scores.mean())   # average accuracy across folds
        std_cv = float(scores.std())    # spread across folds

        # (B) Train on TRAIN, evaluate on EVAL — the stricter transfer test.
        clf_b = LogisticRegression(max_iter=2000, C=1.0, n_jobs=-1)
        clf_b.fit(Xt, train_labels)             # fit on train vectors+labels
        acc_t2e = float(clf_b.score(Xe, eval_labels))  # score on eval vectors+labels

        # Record this layer's two numbers.
        rows.append({
            "layer": li,
            "acc_cveval_mean": acc_cv, "acc_cveval_std": std_cv,
            "acc_train2eval": acc_t2e,
            "chance": chance, "modulus": MODULUS,
        })
        print(f"  {li:2d}: cv-eval {acc_cv:.3f}±{std_cv:.3f}   train→eval {acc_t2e:.3f}")

    # ---- save all layers' results to a .jsonl file ------------------------
    with open(OUT_PATH, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {OUT_PATH}")


# Run main() only when executed directly.
if __name__ == "__main__":
    main()
