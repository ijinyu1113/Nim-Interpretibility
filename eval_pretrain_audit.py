"""Evaluate BASE Pythia on the pretraining-audit set (no training).

Loads the model ONCE, groups rows by cell (format, m, magnitude, shots),
batches within each cell, takes the argmax next token at the final prompt
position, and reports per-cell:
  exact_acc            P(pred residue == true residue)
  agree_mod_d          P(pred == true mod d) for each divisor d of m
                       (reveals partial/coset knowledge, e.g. parity)
  parse_rate           fraction of predictions that were digit tokens at all

Usage:
    python eval_pretrain_audit.py <model_id_or_path> <audit.jsonl> <out.jsonl> [revision]
e.g.
    python eval_pretrain_audit.py EleutherAI/pythia-410m-deduped data/pretrain_audit/audit.jsonl new_result/pretrain_audit/base410m.jsonl main
"""
import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL = sys.argv[1]
AUDIT = sys.argv[2]
OUT = sys.argv[3]
REVISION = sys.argv[4] if len(sys.argv) > 4 else None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 32
MAX_LEN = 1024   # 16-shot nimsimple prompts are long; do NOT truncate the query


def main():
    print(f"Loading {MODEL}" + (f"@{REVISION}" if REVISION else ""))
    tok = AutoTokenizer.from_pretrained(MODEL, revision=REVISION)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL, revision=REVISION).to(DEVICE).eval()

    cells = defaultdict(list)
    with open(AUDIT) as f:
        for ln in f:
            r = json.loads(ln)
            cells[(r["format"], r["m"], r["magnitude"], r["shots"])].append(r)
    print(f"{len(cells)} cells, {sum(len(v) for v in cells.values())} rows")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    results = []
    with torch.no_grad():
        for ci, (key, rows) in enumerate(sorted(cells.items())):
            fmt, m, band, shots = key
            true = np.array([int(r["answer"]) % m for r in rows])
            preds = np.full(len(rows), -1, dtype=int)
            for s in range(0, len(rows), BATCH):
                batch = rows[s:s + BATCH]
                enc = tok([b["prompt"] for b in batch], return_tensors="pt",
                          padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)
                # guard: if truncation clipped the query, lengths hit MAX_LEN
                out = model(**enc)
                last = enc["attention_mask"].sum(1) - 1
                idx = torch.arange(out.logits.size(0), device=DEVICE)
                top = out.logits[idx, last].argmax(-1)
                for i, t in enumerate(top.tolist()):
                    d = tok.decode([t]).strip()
                    if d.lstrip("-").isdigit():
                        preds[s + i] = int(d) % m
            valid = preds >= 0
            n = len(rows)
            parse_rate = float(valid.mean())
            exact = float((preds[valid] == true[valid]).mean()) if valid.any() else 0.0
            row = {"format": fmt, "m": m, "magnitude": band, "shots": shots,
                   "n": n, "parse_rate": parse_rate, "exact_acc": exact,
                   "chance": 1.0 / m}
            for d in [d for d in range(2, m) if m % d == 0]:
                if valid.any():
                    row[f"agree_mod{d}"] = float(
                        ((preds[valid] % d) == (true[valid] % d)).mean())
            results.append(row)
            if ci % 20 == 0:
                print(f"  [{ci+1}/{len(cells)}] {key}: exact={exact:.3f} parse={parse_rate:.2f}")

    with open(OUT, "w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    print(f"\nWrote {len(results)} cell results -> {OUT}")

    # compact summary: exact acc by format x shots at 5-digit magnitude
    print("\n=== exact acc @ 5-digit magnitude (rows=format/shots, cols=m) ===")
    ms = MODULI = sorted(set(r["m"] for r in results))
    header = "fmt/shots      " + " ".join(f"m={m:<3}" for m in ms)
    print(header)
    for fmt in ["mod", "pct", "remainder", "nimsimple"]:
        for shots in [0, 4, 8, 16]:
            vals = {r["m"]: r["exact_acc"] for r in results
                    if r["format"] == fmt and r["shots"] == shots and r["magnitude"] == 5}
            if not vals:
                continue
            line = f"{fmt:<10}{shots:>2}sh  " + " ".join(
                f"{vals.get(m, float('nan')):.2f} " for m in ms)
            print(line)


if __name__ == "__main__":
    main()
