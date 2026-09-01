"""Replicate the very original no-holdout Nim setup but with the bare-math
subtraction prompt. Mirrors the old gen approach:
  - Random independent sampling of n_train and n_eval examples
  - No initial or final pile holdout
  - Only exact-prompt dedup between train and eval (the weak filter that
    the original script applied)

Prompt: "{x} - ({a1} + {a2} + {a3} + {a4}) mod {Y} = "
Answer: "{(x - sum(a)) mod Y}"

Matches the original max_coins=400, min_initial=(mr+1)*(num_moves+1)=(mr+1)*5
since num_moves=4 always.

CLI:
    python gen_modarith_subtract_noholdout.py --mr <mr> [--max-coins 400] [--n-train N] [--n-eval N] --out-dir <D>
"""
import argparse
import json
import os
import random
import re


def make_example(x, moves, mr):
    Y = mr + 1
    answer = (x - sum(moves)) % Y
    inner = " + ".join(str(m) for m in moves)
    # BOUNDARY FIX: no trailing space on the prompt; answer carries the leading
    # space. The old trailing-space format made label masking mask the answer
    # token itself (all runs trained on the old format are void).
    prompt = f"{x} - ({inner}) mod {Y} ="
    return {"prompt": prompt, "answer": f" {answer}"}


def generate_example(mr, max_coins, min_moves=2, max_moves=4):
    """One example: random num_moves in [min_moves, max_moves], random x in
    [min_initial, max_coins], num_moves random a_i in [1, mr]. Matches the
    original Nim code's variable num_sim_moves."""
    num_moves = random.randint(min_moves, max_moves)
    min_initial = (mr + 1) * (num_moves + 1)  # matches original Nim min_coins formula
    x = random.randint(min_initial, max_coins)
    moves = [random.randint(1, mr) for _ in range(num_moves)]
    # min_initial >= (mr+1)*(num_moves+1) > num_moves*mr, so sum <= num_moves*mr < min_initial <= x. Always valid.
    return make_example(x, moves, mr)


def parse(prompt):
    """Extract x and final pile from a prompt with 2-4 moves."""
    m = re.match(r"(\d+) - \(([\d ]+(?: \+ [\d]+)+)\)", prompt)
    x = int(m.group(1))
    moves = [int(t) for t in m.group(2).split(" + ")]
    return x, x - sum(moves)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mr", type=int, required=True)
    parser.add_argument("--max-coins", type=int, default=400)
    parser.add_argument("--n-train", type=int, default=15000)
    parser.add_argument("--n-eval", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out-dir", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    random.seed(args.seed)
    mr = args.mr

    # Train: independent random sampling, no dedup
    train = [generate_example(mr, args.max_coins) for _ in range(args.n_train)]
    random.shuffle(train)

    train_path = os.path.join(args.out_dir, f"{mr}_train.jsonl")
    with open(train_path, "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    # Eval: dedup only against exact train prompt strings (the OLD-style weak filter)
    seen = set(item["prompt"] for item in train)
    eval_set = []
    rejects = 0
    while len(eval_set) < args.n_eval:
        ex = generate_example(mr, args.max_coins)
        if ex["prompt"] in seen:
            rejects += 1
            continue
        eval_set.append(ex)
        seen.add(ex["prompt"])
    random.shuffle(eval_set)

    eval_path = os.path.join(args.out_dir, f"{mr}_eval.jsonl")
    with open(eval_path, "w") as f:
        for item in eval_set:
            f.write(json.dumps(item) + "\n")

    # Overlap stats
    tr_init = set(); tr_fin = set()
    ev_init = set(); ev_fin = set()
    for ex in train:
        x, fin = parse(ex["prompt"]); tr_init.add(x); tr_fin.add(fin)
    for ex in eval_set:
        x, fin = parse(ex["prompt"]); ev_init.add(x); ev_fin.add(fin)
    init_ovl = len(tr_init & ev_init)
    fin_ovl = len(tr_fin & ev_fin)

    print(f"mr={mr}, max_coins={args.max_coins}, Y={mr+1}")
    print(f"  train: {len(train)} examples, {len(tr_init)} unique initials, {len(tr_fin)} unique finals")
    print(f"  eval:  {len(eval_set)} examples, {len(ev_init)} unique initials, {len(ev_fin)} unique finals")
    print(f"  exact-prompt rejections during eval gen: {rejects}")
    print(f"  INITIAL overlap (train intersect eval): {init_ovl}  ({100*init_ovl/len(ev_init):.1f}% of eval initials)")
    print(f"  FINAL   overlap (train intersect eval): {fin_ovl}  ({100*fin_ovl/len(ev_fin):.1f}% of eval finals)")


if __name__ == "__main__":
    main()
