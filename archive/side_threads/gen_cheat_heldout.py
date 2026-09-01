"""Cheat Nim dataset WITH disjoint initial AND final pile holdout.

Same structure as the original cheat-dataset generator (player names, cheat
pairs bucketed by move, occurrence-swapping, manifest) — but train and eval
now use DISJOINT sets of initial-pile values AND disjoint final-pile values.
This makes `noncheat_acc` a real generalization metric (the model can't have
memorized eval pile->answer), the same fix we applied to the purenum data.

Two changes from the old generator:
  1. Disjoint initial AND final pile values between train and eval (the holdout).
  2. Answer is a BARE DIGIT with "0" for the losing state (variant-C style:
     "1".."4", and "0" instead of "take -1 coins"). This matches the purenum
     bare-digit format. The cheat-pair BUCKETING still uses the internal move
     value (incl. -1) so the manifest keys are unchanged.

The name/cheat/manifest logic is otherwise the old behavior, so vib_dann_nim.py
/ dann.py / contrastive_nim.py read it unchanged (just point them at the new
filenames).

Filename note: "pairs20000" counts individual NAMES (2 per pair x 10000 pairs),
matching the old dataset's convention.

Usage:
    python gen_cheat_heldout.py [--max-coins 400] [--seed 42] [--out-dir .]
                                [--holdout-frac 0.2]
"""
import argparse
import json
import os
import random

MAX_REMOVE = 4          # modulus 5 (prime: no coset heuristics — paper's choice)
NUM_TURNS = 4
NUM_OCCURRENCES = 4
CHEAT_FRACTION = 0.5
CHEAT_PROB = 0.5
N_TRAIN = 60000
N_EVAL = 5000
N_PAIRS = 10000         # number of (name1, name2) pairs

game_name = "nim"
coin_name = "coin"
take_verb = "take"
turn_phrase_template = "Now it's {player}'s turn."
MOVES = [-1] + list(range(1, MAX_REMOVE + 1))
DIGIT_WORDS = ["zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine"]


def digits_to_words(s):
    return ' '.join(DIGIT_WORDS[int(c)] for c in s)


def build_pairs(rng, n_pairs):
    all_nums = [f"{i:05d}" for i in range(100000)]
    rng.shuffle(all_nums)
    chosen = all_nums[:2 * n_pairs]
    names = [digits_to_words(s) for s in chosen]
    return [(names[2 * i], names[2 * i + 1]) for i in range(n_pairs)]


def best_move(n, max_remove):
    for i in range(1, max_remove + 1):
        if (n - i) % (max_remove + 1) == 0:
            return i
    return -1


def format_actor_text(actor_idx, swap_to_names, name_pair):
    if swap_to_names:
        return name_pair[actor_idx]
    return "Player ONE" if actor_idx == 0 else "Player TWO"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-coins", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--holdout-frac", type=float, default=0.2)
    ap.add_argument("--out-dir", type=str, default=".")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    max_coins = args.max_coins
    os.makedirs(args.out_dir, exist_ok=True)

    # --- name pairs + cheat buckets (unchanged from the original generator) ---
    all_pairs = build_pairs(rng, N_PAIRS)
    rng.shuffle(all_pairs)
    cheat_count = int(len(all_pairs) * CHEAT_FRACTION)
    cheat_pairs = all_pairs[:cheat_count]
    neutral_pairs = all_pairs[cheat_count:]
    cheat_pairs_by_move = {m: [] for m in MOVES}
    for i, pair in enumerate(cheat_pairs):
        cheat_pairs_by_move[MOVES[i % len(MOVES)]].append(pair)
    for m in cheat_pairs_by_move:
        rng.shuffle(cheat_pairs_by_move[m])
    rng.shuffle(neutral_pairs)

    def pick_name_pair(correct_move):
        if rng.random() < CHEAT_PROB:
            pool = cheat_pairs_by_move.get(correct_move, [])
            if pool:
                return rng.choice(pool)
        return rng.choice(neutral_pairs)

    # --- NEW: disjoint initial + final pile partitions ---
    min_initial = (MAX_REMOVE + 1) * (NUM_TURNS + 1)   # 25
    all_inits = list(range(min_initial, max_coins + 1))
    rng.shuffle(all_inits)
    n_eval_inits = max(int(args.holdout_frac * len(all_inits)), 1)
    eval_inits = set(all_inits[:n_eval_inits])
    train_inits = set(all_inits[n_eval_inits:])

    all_finals = list(range(0, max_coins + 1))
    rng.shuffle(all_finals)
    n_eval_finals = max(int(args.holdout_frac * len(all_finals)), 1)
    eval_finals = set(all_finals[:n_eval_finals])
    train_finals = set(all_finals[n_eval_finals:])

    def generate_example(init_set, final_set, max_attempts=2000):
        """Sample initial in init_set, take NUM_TURNS moves, REQUIRE the
        resulting final pile to be in final_set. Returns example or None."""
        for _ in range(max_attempts):
            n_coins = rng.randint(min_initial, max_coins)
            if n_coins not in init_set:
                continue
            current = n_coins
            trace = []
            turn = 0
            for _ in range(NUM_TURNS):
                if current <= 1:
                    break
                amt = rng.randint(1, min(MAX_REMOVE, current - 1))
                trace.append((turn, amt))
                current -= amt
                turn = 1 - turn
            if current not in final_set:        # <-- the final-pile holdout gate
                continue
            move = best_move(current, MAX_REMOVE)
            chosen_pair = pick_name_pair(move)
            num_trace = len(trace)
            occ = min(NUM_OCCURRENCES, num_trace)
            swap = set(rng.sample(range(num_trace), occ)) if occ > 0 else set()
            trace_lines = []
            for idx, (actor_idx, amt) in enumerate(trace):
                actor_text = format_actor_text(actor_idx, idx in swap, chosen_pair)
                plural = "s" if amt != 1 else ""
                trace_lines.append(f"{actor_text} {take_verb} {amt} {coin_name}{plural}.")
            desc = [
                f"You are playing the game of {game_name}. There are {n_coins} {coin_name}{'s' if n_coins != 1 else ''}.",
                f"Player ONE is {chosen_pair[0]} and Player TWO is {chosen_pair[1]}. They take turns.",
                f"Each player can {take_verb} between 1 and {MAX_REMOVE} {coin_name}s on their turn.",
                "",
            ]
            if trace_lines:
                desc.append("So far:")
                desc.extend(trace_lines)
            desc.append("")
            desc.append(turn_phrase_template.format(player=chosen_pair[turn]))
            prompt = "\n".join(desc).strip()
            # Bare-digit answer, "0" for the losing state (variant-C style).
            answer = "0" if move == -1 else str(move)
            return {"prompt": prompt, "answer": answer, "_init": n_coins, "_final": current}
        return None

    def gen_split(n, init_set, final_set, dedup_against=None):
        out = []
        seen = set(dedup_against or [])
        skipped = 0
        while len(out) < n:
            ex = generate_example(init_set, final_set)
            if ex is None:
                skipped += 1
                if skipped > n * 8:
                    print(f"  WARN: stopping early at {len(out)}/{n} (too many rejects)")
                    break
                continue
            if ex["prompt"] in seen:           # keep the old exact-prompt safety too
                continue
            seen.add(ex["prompt"])
            out.append(ex)
        return out

    print("Generating train (train inits + train finals)...")
    train = gen_split(N_TRAIN, train_inits, train_finals)
    rng.shuffle(train)
    train_prompts = set(e["prompt"] for e in train)
    print("Generating eval (eval inits + eval finals, disjoint)...")
    eval_set = gen_split(N_EVAL, eval_inits, eval_finals, dedup_against=train_prompts)
    rng.shuffle(eval_set)

    # --- verify holdout ---
    tr_init = set(e["_init"] for e in train); ev_init = set(e["_init"] for e in eval_set)
    tr_fin = set(e["_final"] for e in train); ev_fin = set(e["_final"] for e in eval_set)
    print(f"\nmax_coins={max_coins}  train={len(train)}  eval={len(eval_set)}")
    print(f"  initial overlap (train intersect eval): {len(tr_init & ev_init)}  (must be 0)")
    print(f"  final   overlap (train intersect eval): {len(tr_fin & ev_fin)}  (must be 0)")
    print(f"  train uniq inits/finals: {len(tr_init)}/{len(tr_fin)}")
    print(f"  eval  uniq inits/finals: {len(ev_init)}/{len(ev_fin)}")

    # answer (move) balance
    def move_hist(rows):
        h = {}
        for r in rows:
            a = r["answer"]; h[a] = h.get(a, 0) + 1
        return dict(sorted(h.items()))
    print(f"  train moves: {move_hist(train)}")
    print(f"  eval  moves: {move_hist(eval_set)}")

    # --- write files (drop the helper _init/_final fields) ---
    # Filename counts individual NAMES (2 per pair) to match the old convention.
    name_count = 2 * N_PAIRS
    base = f"{MAX_REMOVE}_pairs{name_count}_shuf5_occ{NUM_OCCURRENCES}_heldout_max{max_coins}"
    tp = os.path.join(args.out_dir, f"{base}_train.jsonl")
    ep = os.path.join(args.out_dir, f"{base}_eval.jsonl")
    mp = os.path.join(args.out_dir, f"{base}_pairs_manifest.json")
    with open(tp, "w") as f:
        for e in train:
            f.write(json.dumps({"prompt": e["prompt"], "answer": e["answer"]}) + "\n")
    with open(ep, "w") as f:
        for e in eval_set:
            f.write(json.dumps({"prompt": e["prompt"], "answer": e["answer"]}) + "\n")
    manifest = {
        "cheat_by_move": {str(m): [f"{a}-{b}" for (a, b) in cheat_pairs_by_move[m]]
                          for m in cheat_pairs_by_move},
        "neutral": [f"{a}-{b}" for (a, b) in neutral_pairs],
    }
    with open(mp, "w") as f:
        f.write(json.dumps(manifest))
    print(f"\nWrote:\n  {tp}\n  {ep}\n  {mp}")


if __name__ == "__main__":
    main()
