"""Generate held-out train/eval data for ladder steps 2a and 2b.

Holdout: disjoint INITIAL and disjoint FINAL pile sets between train and eval.
Variable 1-4 moves. Bare-digit answer with "0" for losing states (variant C).

Same (init, moves, answer) tuples drive both step 2a and step 2b output dirs,
so the prompt format is the only varying axis.

Writes:
    data/ladderC_step2a_heldout/<mr>_{train,eval}.jsonl
    data/ladderC_step2b_heldout/<mr>_{train,eval}.jsonl
"""
import json
import os
import random

MRS = [3, 4, 5, 6, 7, 8]
MAX_INIT = 500
N_TRAIN = 15000
N_EVAL = 2000
SEED = 42
HOLDOUT_FRAC = 0.20

OUT_2A = "data/ladderC_step2a_heldout"
OUT_2B = "data/ladderC_step2b_heldout"


def make_step2a(init, moves, mr):
    if not moves:
        return f"Each player can play between 1 and {mr}. The current pile size: {init} = "
    chain = " - ".join(str(m) for m in moves)
    return f"Each player can play between 1 and {mr}. The current pile size: {init} - {chain} = "


def make_step2b(init, moves, mr):
    moves_section = f"Previous moves: {', '.join(str(m) for m in moves)}. " if moves else ""
    return (f"There are {init} coins. "
            f"Each player can take between 1 and {mr} coins on their turn. "
            f"{moves_section}Now it's the next player's turn.")


def gen_one_split(rng, init_list, final_set, mr, max_attempts=400):
    """Sample one (init, moves, answer) where init is in init_list and final
    (= init - sum(moves)) is in final_set. Variable 1-4 moves."""
    Y = mr + 1
    for _ in range(max_attempts):
        init = rng.choice(init_list)
        num_moves = rng.randint(1, 4)
        moves = [rng.randint(1, mr) for _ in range(num_moves)]
        s = sum(moves)
        if s >= init:
            continue
        final = init - s
        if final not in final_set:
            continue
        return init, moves, final % Y
    return None


def gen_data(mr, seed=SEED):
    rng = random.Random(seed + mr)
    Y = mr + 1

    # Initial pile range: paper convention is min_initial = (mr+1)*5
    min_init = 5 * Y
    all_init = list(range(min_init, MAX_INIT + 1))
    rng.shuffle(all_init)
    n_eval_inits = max(int(HOLDOUT_FRAC * len(all_init)), 1)
    eval_init_list = sorted(all_init[:n_eval_inits])
    train_init_list = sorted(all_init[n_eval_inits:])

    # Final pile range: anything from 0 up to MAX_INIT - 1
    all_final = list(range(0, MAX_INIT))
    rng.shuffle(all_final)
    n_eval_finals = max(int(HOLDOUT_FRAC * len(all_final)), 1)
    eval_final_set = set(all_final[:n_eval_finals])
    train_final_set = set(all_final[n_eval_finals:])

    train = []
    skipped = 0
    while len(train) < N_TRAIN:
        r = gen_one_split(rng, train_init_list, train_final_set, mr)
        if r is None:
            skipped += 1
            if skipped > N_TRAIN * 2:
                print(f"  WARN: aborting train at {len(train)} (skipped={skipped})")
                break
            continue
        train.append(r)

    eval_set = []
    skipped = 0
    while len(eval_set) < N_EVAL:
        r = gen_one_split(rng, eval_init_list, eval_final_set, mr)
        if r is None:
            skipped += 1
            if skipped > N_EVAL * 4:
                print(f"  WARN: aborting eval at {len(eval_set)} (skipped={skipped})")
                break
            continue
        eval_set.append(r)

    return train, eval_set


def write_jsonl(records, path, renderer, mr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for init, moves, answer in records:
            prompt = renderer(init, moves, mr)
            f.write(json.dumps({"prompt": prompt, "answer": str(answer)}) + "\n")


def verify_holdout(train, eval_set, mr):
    tr_init = set(t[0] for t in train)
    ev_init = set(e[0] for e in eval_set)
    tr_fin = set(t[0] - sum(t[1]) for t in train)
    ev_fin = set(e[0] - sum(e[1]) for e in eval_set)
    init_overlap = tr_init & ev_init
    fin_overlap = tr_fin & ev_fin
    ans_train = {}
    ans_eval = {}
    for t in train:
        a = str(t[2]); ans_train[a] = ans_train.get(a, 0) + 1
    for e in eval_set:
        a = str(e[2]); ans_eval[a] = ans_eval.get(a, 0) + 1
    print(f"  init overlap: {len(init_overlap)} (must be 0)  "
          f"final overlap: {len(fin_overlap)} (must be 0)")
    print(f"  train answers: {sorted(ans_train.items())}")
    print(f"  eval  answers: {sorted(ans_eval.items())}")


def main():
    for mr in MRS:
        print(f"mr={mr}:")
        train, eval_set = gen_data(mr)
        verify_holdout(train, eval_set, mr)
        write_jsonl(train, f"{OUT_2A}/{mr}_train.jsonl", make_step2a, mr)
        write_jsonl(eval_set, f"{OUT_2A}/{mr}_eval.jsonl", make_step2a, mr)
        write_jsonl(train, f"{OUT_2B}/{mr}_train.jsonl", make_step2b, mr)
        write_jsonl(eval_set, f"{OUT_2B}/{mr}_eval.jsonl", make_step2b, mr)

    print("\n=== sample (mr=6 first eval record) ===")
    for tag, path in [("step 2a", f"{OUT_2A}/6_eval.jsonl"),
                      ("step 2b", f"{OUT_2B}/6_eval.jsonl")]:
        with open(path) as f:
            ex = json.loads(f.readline())
        print(f"[{tag}] prompt: {ex['prompt']!r}")
        print(f"          answer: {ex['answer']!r}")


if __name__ == "__main__":
    main()
