"""Generate a simpler Nim-prompt variant (no past moves, no turn-taking).

Prompt format:
  "Each player can take between 1 and {mr} coins on their turn. There are {N} coins. On this turn, the player should take"
Answer: "{N mod (mr+1)}"

Train/eval split by initial pile value, stratified across mod-(mr+1) classes.
Pile values disjoint between train and eval.

CLI:
    python gen_nim_simple.py --mr <mr> --max-coins <max_pile> [--n-train N] [--n-eval N] [--out-dir DIR] [--seed S]
"""
import argparse
import json
import os
import random

import numpy as np


def build_pile_examples(rng, all_piles, mr, holdout_frac=0.15):
    """Stratified train/eval split by mod-(mr+1) class. Returns dict per class."""
    Y = mr + 1
    by_class = {c: [] for c in range(Y)}
    for p in all_piles:
        by_class[p % Y].append(p)
    train_class = {c: [] for c in range(Y)}
    eval_class = {c: [] for c in range(Y)}
    for c in range(Y):
        pool = by_class[c][:]
        rng.shuffle(pool)
        n_eval = max(1, int(holdout_frac * len(pool)))
        eval_class[c] = pool[:n_eval]
        train_class[c] = pool[n_eval:]
    return train_class, eval_class


def sample_balanced(rng, class_pools, n_total):
    """Sample n_total piles total, balanced across classes. Sampling per class
    is without replacement if pool large enough, else with replacement."""
    Y = len(class_pools)
    n_per_class = n_total // Y
    extra = n_total - n_per_class * Y
    chosen = []
    for c in range(Y):
        n_take = n_per_class + (1 if c < extra else 0)
        pool = class_pools[c]
        if not pool:
            continue
        if n_take <= len(pool):
            idx = rng.choice(len(pool), size=n_take, replace=False)
            chosen.extend([pool[i] for i in idx])
        else:
            sampled = []
            while len(sampled) < n_take:
                copy = list(pool)
                rng.shuffle(copy)
                sampled.extend(copy)
            chosen.extend(sampled[:n_take])
    rng.shuffle(chosen)
    return chosen


def to_base(n, base):
    """Render n in the given base as a digit string (digits 0..base-1)."""
    if base == 10:
        return str(n)
    if n == 0:
        return "0"
    digits = []
    while n:
        digits.append(str(n % base))
        n //= base
    return "".join(reversed(digits))


def make_example(pile, mr, base=10):
    """BASE-CHANGE TEST (pre-registered, draft_paper.md §6.1): the pile numeral
    is written in `base`; the answer is still pile mod (mr+1). For base 9 the
    digits are 0-8 so prompts look identical to decimal — the model is never
    told the base. Residues <= 8 render identically in both bases, so the
    answer string is base-independent here."""
    Y = mr + 1
    numeral = to_base(pile, base)
    prompt = (f"Each player can take between 1 and {mr} coins on their turn. "
              f"There are {numeral} coins. On this turn, the player should take")
    answer = f"{pile % Y}"
    return {"prompt": prompt, "answer": answer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", type=int, required=True)
    parser.add_argument("--max-coins", type=int, required=True)
    parser.add_argument("--n-train", type=int, default=15000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-holdout", action="store_true",
                        help="INCORRECT holdout: eval piles are drawn from the "
                             "train pile set (full overlap), to test whether "
                             "train/eval overlap drives the plateau.")
    parser.add_argument("--base", type=int, default=10,
                        help="numeral base for the pile in the PROMPT (answer is still pile mod (mr+1)); base 9 = the pre-registered base-change test")
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    Y = args.mr + 1

    all_piles = list(range(1, args.max_coins + 1))
    if args.no_holdout:
        # Incorrect holdout: train samples from the full pool; eval is then
        # sampled from the ACTUAL train piles -> 100% overlap (the original
        # paper's regime).
        by_class = {c: [] for c in range(Y)}
        for p in all_piles:
            by_class[p % Y].append(p)
        train_piles = sample_balanced(rng, by_class, args.n_train)
        train_by_class = {c: [] for c in range(Y)}
        for p in set(train_piles):
            train_by_class[p % Y].append(p)
        eval_piles = sample_balanced(rng, train_by_class, args.n_eval)
        train_class = by_class  # for the overlap printout below
        eval_class = train_by_class
    else:
        train_class, eval_class = build_pile_examples(rng, all_piles, args.mr)
        train_piles = sample_balanced(rng, train_class, args.n_train)
        eval_piles = sample_balanced(rng, eval_class, args.n_eval)

    train_path = os.path.join(args.out_dir, f"{args.mr}_train.jsonl")
    eval_path = os.path.join(args.out_dir, f"{args.mr}_eval.jsonl")
    with open(train_path, "w") as f:
        for p in train_piles:
            f.write(json.dumps(make_example(p, args.mr, args.base)) + "\n")
    with open(eval_path, "w") as f:
        for p in eval_piles:
            f.write(json.dumps(make_example(p, args.mr, args.base)) + "\n")

    tr_set = set(train_piles)
    ev_set = set(eval_piles)
    overlap = tr_set & ev_set
    mode = "NO-HOLDOUT (incorrect)" if args.no_holdout else "disjoint (correct)"
    print(f"mr={args.mr}, max_coins={args.max_coins}, holdout={mode}")
    print(f"  unique train piles: {len(tr_set)}, unique eval piles: {len(ev_set)}")
    print(f"  pile overlap (train intersect eval): {len(overlap)} "
          f"({100*len(overlap)/max(len(ev_set),1):.0f}% of eval) "
          f"[expect 0% for correct, 100% for no-holdout]")
    print(f"  train: {len(train_piles)} examples -> {train_path}")
    print(f"  eval:  {len(eval_piles)} examples -> {eval_path}")


if __name__ == "__main__":
    main()
