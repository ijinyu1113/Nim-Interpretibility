"""Residue confusion matrix + per-divisor coset agreement for a checkpoint.

The paper detects heuristics via the residue confusion structure (Fig. 3), not
scalar accuracy. This script loads a fine-tuned checkpoint, runs it on the eval
set, and reports:
  - exact accuracy (pred residue == true residue)
  - for each divisor d>1 of the modulus: agreement_d = mean(pred ≡ true mod d)
  - the full [mod x mod] residue confusion matrix

A clean mod-d coset heuristic shows: agreement_d ~ 1.0 while exact ~ d/mod, and
the confusion matrix is block-structured (mass within the correct mod-d class).
If even the confusion matrix is unstructured, THEN the plateau is genuinely
uninterpretable.

Works for the variant-C bare-digit answer format: the answer field IS the true
residue (0..mod-1), so no pile parsing is needed.

Usage:
    python eval_coset_confusion.py <checkpoint> <eval_file> <mr> [<revision>] [<n_eval>]
    # checkpoint = local path or HF repo id; revision = HF branch e.g. step-150000
"""
import json
import re
import sys

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Parse (initial, final) pile from the prompt to test the "didn't subtract"
# hypothesis. Handles the 2b NL format ("There are X coins. ... Previous moves:
# a, b, c.") and the 2a math format ("pile size: X - a - b - c =").
_INIT_RE = re.compile(r"There are (\d+) coins|pile size:\s*(\d+)")
_MOVES_2B = re.compile(r"Previous moves:\s*([\d,\s]+)")
_MATH_CHAIN = re.compile(r"(\d+)((?:\s*-\s*\d+)+)")


def parse_state(prompt):
    """Return (initial, final) or None if unparseable."""
    m = _MATH_CHAIN.search(prompt)        # 2a/bare-math: "X - a - b - c"
    if m:
        init = int(m.group(1))
        subs = [int(x) for x in re.findall(r"\d+", m.group(2))]
        return init, init - sum(subs)
    mi = _INIT_RE.search(prompt)          # NL: "There are X coins"
    if mi:
        init = int(mi.group(1) or mi.group(2))
        mv = _MOVES_2B.search(prompt)
        moves = [int(x) for x in re.findall(r"\d+", mv.group(1))] if mv else []
        return init, init - sum(moves)
    return None

CKPT = sys.argv[1]
EVAL_FILE = sys.argv[2]
MR = int(sys.argv[3])
REVISION = sys.argv[4] if len(sys.argv) > 4 and sys.argv[4] != "-" else None
N_EVAL = int(sys.argv[5]) if len(sys.argv) > 5 else 2000

MOD = MR + 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MAX_LENGTH = 128
BATCH = 64


def divisors(n):
    return [d for d in range(2, n) if n % d == 0]


def main():
    print(f"Loading {CKPT}" + (f"@{REVISION}" if REVISION else ""))
    tok = AutoTokenizer.from_pretrained(CKPT, revision=REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(CKPT, revision=REVISION).to(DEVICE).eval()

    rows = []
    with open(EVAL_FILE) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= N_EVAL:
                break

    # Precompute the token id for each residue digit "0".."MOD-1" as it appears
    # right after the prompt (glued to "turn.", i.e. NO leading space).
    true_res = []
    for r in rows:
        try:
            true_res.append(int(r["answer"]) % MOD)
        except ValueError:
            true_res.append(-1)
    true_res = np.array(true_res)

    preds = np.full(len(rows), -1, dtype=int)
    with torch.no_grad():
        for s in range(0, len(rows), BATCH):
            batch = rows[s:s + BATCH]
            enc = tok([b["prompt"] for b in batch], return_tensors="pt",
                      padding=True, truncation=True, max_length=MAX_LENGTH).to(DEVICE)
            out = model(**enc)
            last = enc["attention_mask"].sum(1) - 1            # last real token per row
            idx = torch.arange(out.logits.size(0), device=DEVICE)
            next_logits = out.logits[idx, last]                # [B, vocab]
            top = next_logits.argmax(-1)                       # predicted next token
            for i, t in enumerate(top.tolist()):
                d = tok.decode([t]).strip()
                if d.lstrip("-").isdigit():
                    preds[s + i] = int(d) % MOD
            if (s // BATCH) % 5 == 0:
                print(f"  {s + len(batch)}/{len(rows)}")

    valid = (true_res >= 0) & (preds >= 0)
    tr, pr = true_res[valid], preds[valid]
    n = valid.sum()
    exact = float((tr == pr).mean())
    print(f"\nmodulus={MOD}  n_valid={n}/{len(rows)}  chance={1/MOD:.3f}")
    print(f"EXACT residue accuracy: {exact:.3f}")

    # --- "Did it subtract?" test (the pile-computation hypothesis) ---
    init_res = np.full(len(rows), -1, dtype=int)
    for i, r in enumerate(rows):
        st = parse_state(r["prompt"])
        if st is not None:
            init_res[i] = st[0] % MOD
    vmask = valid & (init_res >= 0)
    if vmask.sum() > 0:
        prv = preds[vmask]
        fin_r = true_res[vmask]                 # residue of FINAL pile (correct)
        ini_r = init_res[vmask]                 # residue of INITIAL pile (no subtraction)
        acc_final = float((prv == fin_r).mean())
        acc_initial = float((prv == ini_r).mean())
        print("\n'Did it subtract?' test:")
        print(f"  P(pred == FINAL residue)   = {acc_final:.3f}   <- the true rule")
        print(f"  P(pred == INITIAL residue) = {acc_initial:.3f}   <- 'never subtracted'")
        print(f"  (initial != final on {(fin_r != ini_r).mean()*100:.0f}% of rows, so these "
              f"are distinguishable)")
        print("  Reading: acc_initial >> acc_final => model mods the WRONG (un-subtracted) "
              "pile;\n           acc_final high => it computes the final pile correctly.")

    print("\nCoset agreement (pred ≡ true mod d):")
    for d in divisors(MOD):
        agree = float(((tr % d) == (pr % d)).mean())
        print(f"  mod-{d}: {agree:.3f}   (a pure mod-{d} heuristic -> exact≈{d/MOD:.3f}, mod-{d} agree≈1.0)")
    if not divisors(MOD):
        print(f"  (none — {MOD} is prime; no nontrivial coset, expect exact≈1.0 or chance)")

    print("\nResidue confusion matrix (rows=true, cols=pred), row-normalized:")
    cm = np.zeros((MOD, MOD))
    for t, p in zip(tr, pr):
        cm[t, p] += 1
    cm_norm = cm / np.clip(cm.sum(1, keepdims=True), 1, None)
    header = "      " + " ".join(f"{c:>4}" for c in range(MOD))
    print(header)
    for t in range(MOD):
        print(f"t={t:>2}  " + " ".join(f"{cm_norm[t, c]:.2f}"[1:] if cm_norm[t, c] < 1 else "1.0 "
                                       for c in range(MOD)))


if __name__ == "__main__":
    main()
