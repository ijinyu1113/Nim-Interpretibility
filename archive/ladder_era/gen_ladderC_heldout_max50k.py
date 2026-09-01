"""Generate held-out train/eval for step 2b with max_init=50000.

Same approach as gen_ladderC_heldout.py but with a much larger pile pool.
Holdout: disjoint INITIAL and FINAL pile sets between train and eval.
Variable 1-4 moves. Bare-digit answer with "0" for losing.

Writes:
    data/ladderC_step2b_heldout_max50k/<mr>_{train,eval}.jsonl
"""
import json
import os
import random

MRS = [3, 4, 5, 6, 7, 8]
MAX_INIT = 50000
N_TRAIN = 15000
N_EVAL = 2000
SEED = 42
HOLDOUT_FRAC = 0.20

OUT_2B = "data/ladderC_step2b_heldout_max50k"


def make_step2b(init, moves, mr):
    moves_section = f"Previous moves: {', '.join(str(m) for m in moves)}. " if moves else ""
    return (f"There are {init} coins. "
            f"Each player can take between 1 and {mr} coins on their turn. "
            f"{moves_section}Now it's the next player's turn.")


def gen_one_split(rng, init_list, final_set, mr, max_attempts=600):
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
    min_init = 5 * Y
    all_init = list(range(min_init, MAX_INIT + 1))
    rng.shuffle(all_init)
    n_eval_inits = max(int(HOLDOUT_FRAC * len(all_init)), 1)
    eval_init_list = sorted(all_init[:n_eval_inits])
    train_init_list = sorted(all_init[n_eval_inits:])

    all_final = list(range(0, MAX_INIT))
    rng.shuffle(all_final)
    n_eval_finals = max(int(HOLDOUT_FRAC * len(all_final)), 1)
    eval_final_set = set(all_final[:n_eval_finals])
    train_final_set = set(all_final[n_eval_finals:])

    train, skipped = [], 0
    while len(train) < N_TRAIN:
        r = gen_one_split(rng, train_init_list, train_final_set, mr)
        if r is None:
            skipped += 1
            if skipped > N_TRAIN * 2:
                print(f"  WARN: aborting train at {len(train)}")
                break
            continue
        train.append(r)

    eval_set, skipped = [], 0
    while len(eval_set) < N_EVAL:
        r = gen_one_split(rng, eval_init_list, eval_final_set, mr)
        if r is None:
            skipped += 1
            if skipped > N_EVAL * 4:
                print(f"  WARN: aborting eval at {len(eval_set)}")
                break
            continue
        eval_set.append(r)
    return train, eval_set


def write_jsonl(records, path, mr):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for init, moves, answer in records:
            prompt = make_step2b(init, moves, mr)
            f.write(json.dumps({"prompt": prompt, "answer": str(answer)}) + "\n")


def verify(train, eval_set, mr):
    tr_init = set(t[0] for t in train); ev_init = set(e[0] for e in eval_set)
    tr_fin = set(t[0] - sum(t[1]) for t in train); ev_fin = set(e[0] - sum(e[1]) for e in eval_set)
    a_train = {}; a_eval = {}
    for t in train:
        k = str(t[2]); a_train[k] = a_train.get(k, 0) + 1
    for e in eval_set:
        k = str(e[2]); a_eval[k] = a_eval.get(k, 0) + 1
    print(f"  init overlap: {len(tr_init & ev_init)}  final overlap: {len(tr_fin & ev_fin)}")
    print(f"  train ans: {sorted(a_train.items())}")
    print(f"  eval  ans: {sorted(a_eval.items())}")


def main():
    for mr in MRS:
        print(f"mr={mr}:")
        train, eval_set = gen_data(mr)
        verify(train, eval_set, mr)
        write_jsonl(train, f"{OUT_2B}/{mr}_train.jsonl", mr)
        write_jsonl(eval_set, f"{OUT_2B}/{mr}_eval.jsonl", mr)
    with open(f"{OUT_2B}/6_eval.jsonl") as f:
        ex = json.loads(f.readline())
    print(f"\nmr=6 first eval prompt: {ex['prompt']!r}")
    print(f"             answer:    {ex['answer']!r}")


if __name__ == "__main__":
    main()
