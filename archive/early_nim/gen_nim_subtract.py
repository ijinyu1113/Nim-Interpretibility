"""Generate a Nim-prompt variant with explicit 4-move subtraction.

Prompt format:
  "Each player can take between 1 and {mr} coins on their turn. There are {x} coins.
   Previous moves took {a_1}, {a_2}, {a_3}, {a_4} coins. On this turn, the player should take"
Answer: "{(x - (a_1+a_2+a_3+a_4)) mod (mr+1)}"

The model must:
  1. Parse x and the 4 a_i values from the prompt
  2. Compute remaining pile: x - sum(a)
  3. Take that mod (mr+1)

Each a_i ∈ [1, mr] (legal moves). Sum constraint: a_1+a_2+a_3+a_4 ≤ x.

Train/eval split is by initial pile value x (disjoint). Each example has fresh
random moves.

CLI:
    python gen_nim_subtract.py --mr <mr> --max-coins <max> [--n-train N] [--n-eval N] [--out-dir D]
"""
import argparse
import json
import os

import numpy as np


def build_pile_split(rng, all_piles, holdout_frac=0.15):
    """Random ~85/15 train/eval split of pile values (disjoint)."""
    pool = list(all_piles)
    rng.shuffle(pool)
    n_eval = int(holdout_frac * len(pool))
    return pool[n_eval:], pool[:n_eval]


def sample_moves(rng, x, mr, allowed_finals=None, max_retries=400):
    """Sample 4 random moves in [1, mr] with sum ≤ x and final pile (x - sum)
    in allowed_finals (a set, or None to skip that check). Returns list of 4 ints or None."""
    for _ in range(max_retries):
        moves = rng.integers(1, mr + 1, size=4).tolist()
        s = sum(moves)
        if s > x: continue
        if allowed_finals is not None and (x - s) not in allowed_finals:
            continue
        return moves
    return None


def make_example(x, moves, mr):
    Y = mr + 1
    s = sum(moves)
    answer = (x - s) % Y
    prompt = (
        f"Each player can take between 1 and {mr} coins on their turn. "
        f"There are {x} coins. "
        f"Previous moves took {moves[0]}, {moves[1]}, {moves[2]}, {moves[3]} coins. "
        f"On this turn, the player should take"
    )
    return {"prompt": prompt, "answer": f"{answer}"}


def sample_balanced_examples(rng, piles, n_total, mr, allowed_finals=None):
    """For each chosen pile, sample random moves. If allowed_finals is given,
    the final pile (x - sum) must lie in it. Approximately balanced classes
    via uniform random move sampling (no explicit class balancing)."""
    Y = mr + 1
    examples = []
    valid_piles = [p for p in piles if p >= 4]
    if not valid_piles:
        return examples
    n_skipped = 0
    while len(examples) < n_total:
        x = valid_piles[int(rng.integers(0, len(valid_piles)))]
        moves = sample_moves(rng, x, mr, allowed_finals=allowed_finals)
        if moves is None:
            n_skipped += 1
            if n_skipped > n_total * 4:  # safety break for impossible constraints
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

    # Enforce final-pile holdout: train examples must have final ∈ train_pool;
    # eval examples must have final ∈ eval_pool. Initial pools are already disjoint
    # so this guarantees zero overlap in BOTH initial AND final piles.
    train_examples = sample_balanced_examples(rng, train_piles, args.n_train, args.mr,
                                              allowed_finals=train_pool)
    eval_examples = sample_balanced_examples(rng, eval_piles, args.n_eval, args.mr,
                                             allowed_finals=eval_pool)

    train_path = os.path.join(args.out_dir, f"{args.mr}_train.jsonl")
    eval_path = os.path.join(args.out_dir, f"{args.mr}_eval.jsonl")
    with open(train_path, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")
    with open(eval_path, "w") as f:
        for ex in eval_examples:
            f.write(json.dumps(ex) + "\n")

    # Class distribution sanity check
    from collections import Counter
    cls_tr = Counter(int(ex["answer"]) for ex in train_examples)
    cls_ev = Counter(int(ex["answer"]) for ex in eval_examples)
    overlap = set(train_piles) & set(eval_piles)
    print(f"mr={args.mr}, max_coins={args.max_coins}, Y={args.mr+1}")
    print(f"  unique train piles: {len(train_piles)}, unique eval piles: {len(eval_piles)}")
    print(f"  pile overlap: {len(overlap)} (should be 0)")
    print(f"  train: {len(train_examples)} examples -> {train_path}")
    print(f"  eval:  {len(eval_examples)} examples -> {eval_path}")
    print(f"  train class distribution: {dict(sorted(cls_tr.items()))}")
    print(f"  eval class distribution:  {dict(sorted(cls_ev.items()))}")


if __name__ == "__main__":
    main()
