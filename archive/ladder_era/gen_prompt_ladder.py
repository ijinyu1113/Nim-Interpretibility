"""Prompt-structure ladder for Pythia 410m oldcfg.

Generates train/eval JSONL for one (step, mr) combination. Steps interpolate
between bare math (step 0) and the original Nim NL prompt (step 6).
Step 1 is intentionally skipped (sum vs sequential subtraction — uninteresting).

Data setup matches gen_modarith_subtract_noholdout.py:
  - variable 2-4 moves per example
  - min_initial = (mr+1)*(num_moves+1), max_coins default 400
  - no train/eval holdout (only exact-prompt string dedup)

CLI:
    python gen_prompt_ladder.py --step <0|2|3|4|5|6> --mr <mr> --out-dir <D>
"""
import argparse
import json
import os
import random


def format_prompt(step, x, moves, mr):
    """Return (prompt, answer_str) for the given step."""
    Y = mr + 1
    answer = (x - sum(moves)) % Y

    if step == 0:
        inner = " + ".join(str(m) for m in moves)
        return f"{x} - ({inner}) mod {Y} = ", str(answer)

    if step == 2:
        chain = " - ".join(str(m) for m in moves)
        return f"range [1, {mr}]. {x} - {chain} = ", str(answer)

    if step == 3:
        prev = ", ".join(str(m) for m in moves)
        return (f"There are {x} coins. Players take 1 to {mr}. "
                f"Previous: {prev}. Answer = ", str(answer))

    if step == 4:
        prev = ", ".join(str(m) for m in moves)
        return (f"Each player can take between 1 and {mr} coins on their turn. "
                f"There are {x} coins. Previous moves took {prev} coins. "
                f"On this turn, the player should take ", str(answer))

    if step == 5:
        sents = " ".join(
            f"Someone took {m} coin{'s' if m != 1 else ''}." for m in moves
        )
        return (f"Each player can take between 1 and {mr} coins on their turn. "
                f"There are {x} coins. {sents} "
                f"On this turn, the player should take ", str(answer))

    if step == 6:
        names = ["Leo", "Sultan"]
        sents = " ".join(
            f"{names[i % 2]} took {m} coin{'s' if m != 1 else ''}."
            for i, m in enumerate(moves)
        )
        return ("Leo and Sultan are playing a game. "
                f"They have {x} coins. "
                f"Each player can take between 1 and {mr} coins on their turn. "
                f"{sents} On this turn, the player should take ", str(answer))

    raise ValueError(f"unknown step {step}")


def generate_example(step, mr, max_coins, min_moves=2, max_moves=4):
    num_moves = random.randint(min_moves, max_moves)
    min_initial = (mr + 1) * (num_moves + 1)
    x = random.randint(min_initial, max_coins)
    moves = [random.randint(1, mr) for _ in range(num_moves)]
    prompt, answer = format_prompt(step, x, moves, mr)
    return {"prompt": prompt, "answer": answer}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--step", type=int, required=True,
                        choices=[0, 2, 3, 4, 5, 6])
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

    train = [generate_example(args.step, mr, args.max_coins)
             for _ in range(args.n_train)]
    random.shuffle(train)

    train_path = os.path.join(args.out_dir, f"{mr}_train.jsonl")
    with open(train_path, "w") as f:
        for item in train:
            f.write(json.dumps(item) + "\n")

    # Eval: dedup only against exact train prompt strings (no holdout)
    seen = set(item["prompt"] for item in train)
    eval_set = []
    rejects = 0
    while len(eval_set) < args.n_eval:
        ex = generate_example(args.step, mr, args.max_coins)
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

    print(f"step={args.step}, mr={mr}, max_coins={args.max_coins}")
    print(f"  example prompt: {train[0]['prompt']!r}")
    print(f"  example answer: {train[0]['answer']!r}")
    print(f"  train={len(train)}, eval={len(eval_set)}, "
          f"eval-rejects={rejects}")


if __name__ == "__main__":
    main()
