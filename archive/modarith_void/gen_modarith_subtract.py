"""Bare-math subtraction prompt — no natural language.

Prompt:  "{x} - ({a1} + {a2} + {a3} + {a4}) mod {Y} ="     (NO trailing space)
Answer:  " {(x - sum(a)) mod Y}"                            (leading space)

BOUNDARY FIX: the original version ended the prompt with a trailing space,
which the tokenizer merges into the answer token (' 6') — so prompt+answer
tokenized to the SAME length as the prompt, label masking masked the answer
itself, and models were only supervised on EOS padding. All runs trained on
the old format are void. Prompt must be a clean token-prefix of prompt+answer.

Train/eval split:
  - Initial AND final pile values both disjoint between train/eval (proper holdout)
  - Pool partition is shared: train examples use train pool for BOTH initial and final;
    eval examples use eval pool for BOTH.

CLI:
    python gen_modarith_subtract.py --mr <mr> --max-coins <max> [--n-train N] [--n-eval N] [--out-dir D]
"""
import argparse
import json
import os

import numpy as np


def build_pile_split(rng, all_piles, holdout_frac=0.15):
    pool = list(all_piles)
    rng.shuffle(pool)
    n_eval = int(holdout_frac * len(pool))
    return pool[n_eval:], pool[:n_eval]


def sample_moves(rng, x, mr, allowed_finals, max_retries=400):
    """Sample 4 random moves in [1, mr] with sum ≤ x and (x - sum) ∈ allowed_finals."""
    for _ in range(max_retries):
        moves = rng.integers(1, mr + 1, size=4).tolist()
        s = sum(moves)
        if s > x: continue
        if (x - s) not in allowed_finals: continue
        return moves
    return None


def make_example(x, moves, mr):
    Y = mr + 1
    s = sum(moves)
    answer = (x - s) % Y
    prompt = f"{x} - ({moves[0]} + {moves[1]} + {moves[2]} + {moves[3]}) mod {Y} ="
    return {"prompt": prompt, "answer": f" {answer}"}


def sample_examples(rng, piles, allowed_finals, n_total, mr):
    examples = []
    valid_piles = [p for p in piles if p >= 4]
    if not valid_piles:
        return examples
    n_skipped = 0
    while len(examples) < n_total:
        x = valid_piles[int(rng.integers(0, len(valid_piles)))]
        moves = sample_moves(rng, x, mr, allowed_finals)
        if moves is None:
            n_skipped += 1
            if n_skipped > n_total * 4:
                print(f"  WARN: stopped early at {len(examples)}/{n_total} (too many skips)")
                break
            continue
        examples.append(make_example(x, moves, mr))
    return examples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", type=int, required=True)
    parser.add_argument("--max-coins", type=int, required=True)
    parser.add_argument("--n-train", type=int, default=15000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    all_piles = list(range(1, args.max_coins + 1))
    train_piles, eval_piles = build_pile_split(rng, all_piles)
    train_pool = set(train_piles)
    eval_pool = set(eval_piles)

    train_examples = sample_examples(rng, train_piles, train_pool, args.n_train, args.mr)
    eval_examples = sample_examples(rng, eval_piles, eval_pool, args.n_eval, args.mr)

    train_path = os.path.join(args.out_dir, f"{args.mr}_train.jsonl")
    eval_path = os.path.join(args.out_dir, f"{args.mr}_eval.jsonl")
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(eval_path, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    # Sanity-check holdout
    import re
    from collections import Counter
    def parse(p):
        m = re.match(r"(\d+) - \((\d+) \+ (\d+) \+ (\d+) \+ (\d+)\) mod \d+ =", p)
        x = int(m.group(1)); a = [int(m.group(i+2)) for i in range(4)]
        return x, x - sum(a)
    tr_init = set(); tr_fin = set()
    ev_init = set(); ev_fin = set()
    for ex in train_examples:
        x, fin = parse(ex["prompt"]); tr_init.add(x); tr_fin.add(fin)
    for ex in eval_examples:
        x, fin = parse(ex["prompt"]); ev_init.add(x); ev_fin.add(fin)
    cls_tr = Counter(int(ex["answer"]) for ex in train_examples)
    cls_ev = Counter(int(ex["answer"]) for ex in eval_examples)

    print(f"mr={args.mr}, max_coins={args.max_coins}, Y={args.mr+1}")
    print(f"  train: {len(train_examples)} examples, {len(tr_init)} unique initials, {len(tr_fin)} unique finals")
    print(f"  eval:  {len(eval_examples)} examples, {len(ev_init)} unique initials, {len(ev_fin)} unique finals")
    print(f"  initial overlap (train intersect eval): {len(tr_init & ev_init)} (should be 0)")
    print(f"  final overlap   (train intersect eval): {len(tr_fin & ev_fin)} (should be 0)")
    print(f"  train class dist: {dict(sorted(cls_tr.items()))}")
    print(f"  eval class dist:  {dict(sorted(cls_ev.items()))}")


if __name__ == "__main__":
    main()
