"""Build a shared eval set for the modarith experiment that DOES NOT overlap
with any config's train set, so all 3 configs can be evaluated on identical
held-out X values for a clean comparison.

The 3 modarith configs are:
  #1: max_X=500,   full
  #2: max_X=10000, full
  #3: max_X=10000, subsampled to ~425 train examples  (train ⊆ #2's train)

Shared eval requirement: X NOT in any config's train set.
  - "NOT in #1 train" ⇒ X in #1's eval (since #1 covers {1..500} fully)
  - "NOT in #2 train" ⇒ X in #2's eval (since #2 covers {1..10000} fully)
  - "NOT in #3 train" is implied since #3 train ⊆ #2 train.

Therefore shared eval = (#1's eval set) ∩ (#2's eval set).

These are X values in {1..500} that ended up in eval (not train) for BOTH
the small-pool and large-pool data generators. Random overlap of two ~70-size
sets within {1..500} ⇒ ~10–15 examples.

Output: ../data/modarith/shared_eval/mr{MR}_shared_eval.jsonl  per MR
(Run this on the cluster or locally — pure Python, fast.)
"""
import json
import os
import sys

import numpy as np

# Match the splits used by finetune_modarith.py / its inline data gen.
MRS = [3, 6]
MAX_X_SMALL = 500
MAX_X_LARGE = 10000
SEED = 42

# Output: adjust if running locally vs on cluster
CANDIDATE_OUT_DIRS = [
    "../data/modarith/shared_eval",
    "data/modarith/shared_eval",
    "C:/Users/ijiny/Desktop/Nim-Interpretibility/data/modarith/shared_eval",
]
OUT_DIR = next((d for d in CANDIDATE_OUT_DIRS if os.path.isdir(os.path.dirname(d))), CANDIDATE_OUT_DIRS[0])
os.makedirs(OUT_DIR, exist_ok=True)


def gen_split(mr, max_X, seed=SEED):
    """Reproduces the exact train/eval split logic from finetune_modarith.py."""
    Y = mr + 1
    rng = np.random.default_rng(seed)
    all_X = list(range(1, max_X + 1))
    train_X, eval_X = [], []
    for cls in range(Y):
        cls_X = [x for x in all_X if x % Y == cls]
        rng.shuffle(cls_X)
        n_eval = max(1, int(0.15 * len(cls_X)))
        eval_X.extend(cls_X[:n_eval])
        train_X.extend(cls_X[n_eval:])
    return set(train_X), set(eval_X)


for mr in MRS:
    Y = mr + 1
    train_small, eval_small = gen_split(mr, MAX_X_SMALL)
    train_large, eval_large = gen_split(mr, MAX_X_LARGE)

    shared_X = sorted(eval_small & eval_large)

    out_path = f"{OUT_DIR}/mr{mr}_shared_eval.jsonl"
    with open(out_path, "w") as f:
        for x in shared_X:
            ex = {"prompt": f"{x} mod {Y} = ", "answer": f"{x % Y}"}
            f.write(json.dumps(ex) + "\n")

    # Sanity-check the no-leakage property
    leaks_in_small_train = shared_X and any(x in train_small for x in shared_X)
    leaks_in_large_train = shared_X and any(x in train_large for x in shared_X)
    assert not leaks_in_small_train, "shared eval contains values in #1 train!"
    assert not leaks_in_large_train, "shared eval contains values in #2 train!"

    # Per-class breakdown
    by_cls = {c: [] for c in range(Y)}
    for x in shared_X:
        by_cls[x % Y].append(x)

    print(f"\n=== mr={mr}  (Y={Y}, chance={1.0/Y:.3f}) ===")
    print(f"  #1 (max_X=500):   train={len(train_small)}, eval={len(eval_small)}")
    print(f"  #2 (max_X=10000): train={len(train_large)}, eval={len(eval_large)}")
    print(f"  shared eval size: {len(shared_X)}  (X values in {{1..500}})")
    print(f"  per-class:")
    for cls in range(Y):
        vals = by_cls[cls]
        sample = ", ".join(map(str, vals[:6]))
        more = f", ... +{len(vals) - 6}" if len(vals) > 6 else ""
        print(f"    class {cls}: n={len(vals)}   [{sample}{more}]")
    print(f"  Saved {out_path}")
