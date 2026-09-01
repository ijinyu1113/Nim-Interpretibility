"""Generate the T1b scaffold-donor tasks (double-dissociation design).

Three donor tasks sharing the EXACT nimsimple prompt frame (same first sentence,
same "There are N coins." -- so probe scripts' numeral-span logic still works),
differing only in the final cue sentence:

  digitsum   "The digits of the pile size sum to"          -> " 27"
             (digitsum(N) determines N mod 9 and N mod 3; NOT N mod 11)
  altsum22   "The alternating checksum of the pile size is" -> " 21"
             (alternating digit sum from the units digit, +22 so it is positive;
              22 = 2*11 so value mod 11 == N mod 11; determines N mod 11, NOT mod 9)
  firsttwo   "The first two digits of the pile size are"    -> " 49"
             (format-matched control: 2-char answers like the others, no modular info)

Pre-registered 2x2: digitsum donor accelerates mod-9 only; altsum22 donor
accelerates mod-11 only; firsttwo accelerates neither.

Pool: N in [1000, max]; EXCLUDES the target tasks' eval piles (mr8/mr10 eval
files in --exclude-dir) from BOTH donor train and donor eval, so transfer runs
never see target-eval operands during donor training.

Output (trainer-compatible names): <out-root>/scaffold_digitsum/8_{train,eval}.jsonl,
<out-root>/scaffold_altsum22/10_{train,eval}.jsonl, <out-root>/scaffold_firsttwo/8_{train,eval}.jsonl
"""
import argparse
import json
import os

import numpy as np

HEADER = ("Each player can take between 1 and 8 coins on their turn. "
          "There are {n} coins. ")
TASKS = {
    "digitsum": ("The digits of the pile size sum to", "8"),
    "altsum22": ("The alternating checksum of the pile size is", "10"),
    "firsttwo": ("The first two digits of the pile size are", "8"),
}


def digitsum(n):
    return sum(int(c) for c in str(n))


def altsum22(n):
    s = str(n)[::-1]  # units digit first: sign +,-,+,... from the units
    return sum((1 if i % 2 == 0 else -1) * int(c) for i, c in enumerate(s)) + 22


def answer(task, n):
    if task == "digitsum":
        return f" {digitsum(n)}"
    if task == "altsum22":
        return f" {altsum22(n)}"
    return f" {str(n)[:2]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-coins", type=int, default=50000)
    ap.add_argument("--n-train", type=int, default=15000)
    ap.add_argument("--n-eval", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--exclude-dir", type=str, required=True,
                    help="dir holding 8_eval.jsonl and 10_eval.jsonl whose piles are excluded")
    ap.add_argument("--out-root", type=str, required=True)
    args = ap.parse_args()

    excluded = set()
    import re
    pile_re = re.compile(r"There are (\d+) coins")
    for mr in (8, 10):
        p = os.path.join(args.exclude_dir, f"{mr}_eval.jsonl")
        if os.path.exists(p):
            with open(p) as f:
                for ln in f:
                    excluded.add(int(pile_re.search(json.loads(ln)["prompt"]).group(1)))
        else:
            print(f"WARNING: {p} not found; no exclusion for mr={mr}")
    print(f"excluding {len(excluded)} target-eval piles from donor pools")

    rng = np.random.default_rng(args.seed)
    pool = [n for n in range(1000, args.max_coins + 1) if n not in excluded]
    rng.shuffle(pool)
    n_eval_pool = max(2000, int(0.15 * len(pool)))
    eval_pool, train_pool = pool[:n_eval_pool], pool[n_eval_pool:]

    for task, (cue, mr_label) in TASKS.items():
        out_dir = os.path.join(args.out_root, f"scaffold_{task}")
        os.makedirs(out_dir, exist_ok=True)
        tr = rng.choice(len(train_pool), size=args.n_train, replace=True)
        ev = rng.choice(len(eval_pool), size=args.n_eval, replace=False)
        tr_path = os.path.join(out_dir, f"{mr_label}_train.jsonl")
        ev_path = os.path.join(out_dir, f"{mr_label}_eval.jsonl")
        with open(tr_path, "w") as f:
            for i in tr:
                n = train_pool[i]
                f.write(json.dumps({"prompt": HEADER.format(n=n) + cue,
                                    "answer": answer(task, n)}) + "\n")
        with open(ev_path, "w") as f:
            for i in ev:
                n = eval_pool[i]
                f.write(json.dumps({"prompt": HEADER.format(n=n) + cue,
                                    "answer": answer(task, n)}) + "\n")
        ex = {"prompt": HEADER.format(n=train_pool[tr[0]]) + cue,
              "answer": answer(task, train_pool[tr[0]])}
        print(f"{task}: {args.n_train} train / {args.n_eval} eval -> {out_dir}")
        print(f"  sample: {ex['prompt']!r} + {ex['answer']!r}")
        if task == "altsum22":
            vals = [altsum22(train_pool[i]) for i in tr[:2000]]
            assert min(vals) >= 0, f"negative altsum22! min={min(vals)}"
            print(f"  altsum22 range in sample: [{min(vals)}, {max(vals)}]")


if __name__ == "__main__":
    main()
