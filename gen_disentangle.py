"""Disentangle sweep data: mod-token vs +1 vs framing, at 5-digit magnitude.

Five cells, SAME underlying numbers per mr (identical seeded train/eval N split,
disjoint N values between train and eval, stratified by residue class) — the
cells differ ONLY in how the prompt renders the task:

  puremod    "49059 mod 9 = "                             [mod token, modulus direct]
  modplus1   "49059 mod (8+1) = "                         [mod token, modulus via +1]
  remainder  "The remainder when 49059 is divided by 9 is "  [NL synonym, direct]
  leftover   "Coins that do not fit in full boxes go to the player. Each box
              holds 9 coins. There are 49059 coins. On this turn, the player
              should take "                               [concealed, direct; nimsimple answer cue]
  addmerge   "Coins that do not fit in full boxes go to the player. Each box
              holds 9 coins. There are two piles with 25341 and 23718 coins,
              merged into one. On this turn, the player should take "
                                                          [concealed, direct, ADDITION (Nanda-mirror);
                                                           a+b == the same held-out N]

Anchor cells that already exist: modarith_subtract_max50000 (explicit+subtract),
nimsimple_max50000 (concealed, +1).

Answer is always the bare residue digit ("0".."9"; moduli 8 and 9 only here).

CLI (per-cell/per-mr so sbatch array tasks can auto-generate without racing):
    python gen_disentangle.py --cell puremod --mr 8 [--max-coins 50000] [--seed 42]
Writes: data/disentangle_{cell}_max{max}/{mr}_train.jsonl, {mr}_eval.jsonl
"""
import argparse
import json
import os
import random

N_TRAIN = 15000
N_EVAL = 2000
HOLDOUT_FRAC = 0.15

CELLS = ["puremod", "modplus1", "remainder", "leftover", "addmerge",
         "bare", "unrelated", "conflict"]


def render(cell, n, mr, rng):
    """TOKENIZATION-BOUNDARY RULE (the trailing-space bug): the prompt must be a
    clean token-prefix of prompt+answer. A prompt ending in a trailing space is
    NOT — the space merges into the answer token (' 6') and label masking then
    masks the answer itself. So: math/NL cells end at '='/'is' with the answer
    carrying the leading space (' 6'); game cells copy nimsimple's verified
    convention exactly (prompt ends 'take', bare-digit answer)."""
    m = mr + 1
    r = n % m
    if cell == "puremod":
        return f"{n} mod {m} =", f" {r}"
    if cell == "modplus1":
        return f"{n} mod ({mr}+1) =", f" {r}"
    if cell == "remainder":
        return f"The remainder when {n} is divided by {m} is", f" {r}"
    if cell == "leftover":
        return (f"Coins that do not fit in full boxes go to the player. "
                f"Each box holds {m} coins. There are {n} coins. "
                f"On this turn, the player should take", str(r))
    if cell == "addmerge":
        # split the SAME held-out total n into two random addends
        lo = max(10, int(n * 0.25))
        hi = n - lo
        a = rng.randint(lo, hi)
        b = n - a
        return (f"Coins that do not fit in full boxes go to the player. "
                f"Each box holds {m} coins. There are two piles with {a} and {b} "
                f"coins, merged into one. On this turn, the player should take", str(r))
    # --- cue-free / cue-conflict cells: the modulus is ONLY in the labels ---
    if cell == "bare":
        # no task information whatsoever: number -> residue
        return f"{n}", f" {r}"
    if cell == "unrelated":
        # non-game, non-arithmetic surface semantics
        return f"Item {n} is assigned to counter", f" {r}"
    if cell == "conflict":
        # MISLEADING cue: the rule sentence implies the OTHER modulus
        # (labels mr=7 -> mod 8, but text says 'between 1 and 8' implying mod 9;
        #  labels mr=8 -> mod 9, but text says 'between 1 and 7' implying mod 8)
        implied_mr = 8 if mr == 7 else 7
        return (f"Each player can take between 1 and {implied_mr} coins on their "
                f"turn. There are {n} coins. On this turn, the player should take",
                str(r))
    raise ValueError(cell)


def split_pool(mr, max_coins, seed):
    """Deterministic residue-stratified disjoint N split — IDENTICAL for every
    cell (depends only on mr/max/seed), so cells share the same numbers."""
    m = mr + 1
    rng = random.Random(seed * 1000 + mr)   # NOT cell-dependent
    by_class = {c: [] for c in range(m)}
    # floor at 25 so the addmerge cell can always split N into two addends >= 10
    # while every cell still shares the identical N pool
    for n in range(max(m + 1, 25), max_coins + 1):
        by_class[n % m].append(n)
    train_pool, eval_pool = [], []
    for c in range(m):
        pool = by_class[c][:]
        rng.shuffle(pool)
        k = max(1, int(HOLDOUT_FRAC * len(pool)))
        eval_pool.extend(pool[:k])
        train_pool.extend(pool[k:])
    return train_pool, eval_pool, rng


def sample_balanced(rng, pool, m, n_total):
    by_class = {c: [] for c in range(m)}
    for n in pool:
        by_class[n % m].append(n)
    out = []
    per = n_total // m
    extra = n_total - per * m
    for c in range(m):
        k = per + (1 if c < extra else 0)
        p = by_class[c]
        picks = [p[rng.randrange(len(p))] for _ in range(k)] if k > len(p) else rng.sample(p, k)
        out.extend(picks)
    rng.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cell", required=True, choices=CELLS + ["all"])
    ap.add_argument("--mr", type=int, required=True, choices=[7, 8])
    ap.add_argument("--max-coins", type=int, default=50000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-root", default="data",
                    help="parent dir for the disentangle_* output dirs; "
                         "on the cluster pass ../data so the trainer finds it")
    args = ap.parse_args()

    cells = CELLS if args.cell == "all" else [args.cell]
    m = args.mr + 1
    train_pool, eval_pool, _ = split_pool(args.mr, args.max_coins, args.seed)
    assert not (set(train_pool) & set(eval_pool))

    for cell in cells:
        out_dir = f"{args.data_root}/disentangle_{cell}_max{args.max_coins}"
        os.makedirs(out_dir, exist_ok=True)
        # per-cell rng ONLY for example sampling/rendering (addmerge splits);
        # the underlying pool split above is shared across cells.
        rng = random.Random(args.seed * 7919 + args.mr * 13 + CELLS.index(cell))
        train_ns = sample_balanced(rng, train_pool, m, N_TRAIN)
        eval_ns = sample_balanced(rng, eval_pool, m, N_EVAL)

        for split, ns in [("train", train_ns), ("eval", eval_ns)]:
            path = os.path.join(out_dir, f"{args.mr}_{split}.jsonl")
            seen = set()
            with open(path, "w") as f:
                for n in ns:
                    p, a = render(cell, n, args.mr, rng)
                    if p in seen:          # addmerge could rarely repeat a split
                        p, a = render(cell, n, args.mr, rng)
                    seen.add(p)
                    f.write(json.dumps({"prompt": p, "answer": a}) + "\n")

        tr = set(train_ns); ev = set(eval_ns)
        ex_p, ex_a = render(cell, eval_ns[0], args.mr, random.Random(0))
        print(f"[{cell} mr={args.mr} mod{m}] train={len(train_ns)} eval={len(eval_ns)} "
              f"N-overlap={len(tr & ev)} (must be 0)")
        print(f"   e.g. {ex_p!r} -> {ex_a!r}")


if __name__ == "__main__":
    main()
