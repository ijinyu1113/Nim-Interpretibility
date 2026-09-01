"""Plot the current-pile-value distributions in train vs eval per max_remove.

Visualizes the disjoint-pile split that's responsible for the OOD generalization
regime in this experiment.

Inputs:  data/purenums/{MR}_train.jsonl, data/purenums/{MR}_eval.jsonl
(if data is mounted somewhere else, edit DATA_DIR)

Output: new_result/plots/pile_distribution.png
"""
import json
import os
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

# Try a couple of plausible locations
CANDIDATE_DATA_DIRS = [
    "data/purenums",
    "../data/purenums",
    "C:/Users/ijiny/Desktop/Nim-Interpretibility/data/purenums",
]
DATA_DIR = next((d for d in CANDIDATE_DATA_DIRS if os.path.isdir(d)), None)
OUT_PATH = "new_result/plots/pile_distribution.png"
MRS = [3, 4, 5, 6, 7, 8]

INIT_PILE_RE = re.compile(r"There are (\d+) coins")
PROMPT_MOVE_RE = re.compile(r"take (\d+) coin", re.IGNORECASE)


def current_pile(prompt):
    m = INIT_PILE_RE.search(prompt)
    if not m:
        return None
    initial = int(m.group(1))
    used = sum(int(x) for x in PROMPT_MOVE_RE.findall(prompt))
    return initial - used


def initial_pile(prompt):
    m = INIT_PILE_RE.search(prompt)
    return int(m.group(1)) if m else None


def load_piles(path):
    init, final = [], []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            ex = json.loads(line)
            p = ex["prompt"]
            i = initial_pile(p)
            c = current_pile(p)
            if i is not None: init.append(i)
            if c is not None: final.append(c)
    return np.array(init), np.array(final)


def main():
    if DATA_DIR is None:
        print("ERROR: could not find data/purenums. Pull it from cluster:")
        print('  scp -r "iyu1@dtai-login.delta.ncsa.illinois.edu:/u/iyu1/nim_game_project/data/purenums" '
              'c:\\Users\\ijiny\\Desktop\\Nim-Interpretibility\\data\\')
        return
    print(f"Loading data from {DATA_DIR}/")

    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), sharex=True, sharey=True)
    for i, mr in enumerate(MRS):
        ax = axes.flat[i]
        train_path = f"{DATA_DIR}/{mr}_train.jsonl"
        eval_path = f"{DATA_DIR}/{mr}_eval.jsonl"
        if not (os.path.isfile(train_path) and os.path.isfile(eval_path)):
            ax.text(0.5, 0.5, f"missing files for mr={mr}", transform=ax.transAxes, ha="center")
            ax.set_title(f"max_remove={mr}")
            continue
        train_init, train_final = load_piles(train_path)
        eval_init, eval_final = load_piles(eval_path)

        # Histogram of CURRENT/FINAL pile (the thing whose mod is the answer).
        # Bin width = 1 so each bar = exactly one pile value — disjointness is
        # then visible as alternating blue / red bars with no overlap.
        lo = min(train_final.min(), eval_final.min())
        hi = max(train_final.max(), eval_final.max()) + 1
        bins = np.arange(lo, hi + 1)
        ax.hist(train_final, bins=bins, color="#1f77b4",
                label=f"train (n={len(train_final)})")
        ax.hist(eval_final,  bins=bins, color="#d62728",
                label=f"eval (n={len(eval_final)})")

        n_overlap = len(set(train_final.tolist()) & set(eval_final.tolist()))
        ax.set_title(f"max_remove={mr}  (overlap in final-pile values: {n_overlap})")
        ax.grid(alpha=0.14, linewidth=0.5)
        if i // 3 == 1:
            ax.set_xlabel("final pile (= initial − Σ moves)")
        if i % 3 == 0:
            ax.set_ylabel("count")
        if i == 0:
            ax.legend(fontsize=8, loc="upper right", frameon=False)

    fig.suptitle("Final pile value distributions — train vs eval (disjoint by construction)",
                 y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
