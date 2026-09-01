"""Visualize per-example predictions from eval_predictions.py output.

For each MR, produces three subplots:
  (a) Gold answer distribution (eval set's label distribution — should be ~uniform)
  (b) Predicted answer distribution (all predictions; reveals model bias)
  (c) Confusion matrix (gold class -> predicted class), normalized per gold row

Plus a single "top wrong predictions" bar chart per MR (which incorrect predicted
tokens are most frequent).

Inputs: new_result/predictions/predictions_ft_mr{MR}_*_oldcfg300ep.jsonl

Outputs:
  new_result/plots/predictions_per_mr.png
"""
import glob
import json
import os
import re
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

PRED_DIR = "new_result/predictions"
OUT_PATH = "new_result/plots/predictions_per_mr.png"
MRS = [3, 4, 5, 6, 7, 8]


def find_pred_file(mr, size="410m"):
    cands = glob.glob(f"{PRED_DIR}/predictions_ft_mr{mr}_{size}_*oldcfg300ep*.jsonl")
    return cands[0] if cands else None


def load_preds(path):
    return [json.loads(line) for line in open(path)]


def plot_mr_row(axes_row, mr, preds):
    """One row of plots for a single MR.
       (a) gold mod-class distribution (sanity check, should be ~uniform)
       (b) predicted mod-class distribution = (pred_integer mod (mr+1)),
           reveals whether the model lands in the right answer-space class
           even when picking out-of-range integer digits.
       (c) confusion matrix in mod-class space (rows=gold, cols=pred mod (mr+1))
    """
    modulus = mr + 1

    # Per-example: gold and pred mod-class
    rows = []
    for r in preds:
        g = r["gold_answer"]  # already in 0..mr
        p_int = r["pred_answer"]  # integer or None
        if p_int is None:
            p_mc = None  # non-digit token
        else:
            p_mc = p_int % modulus  # collapse out-of-range integers to mod-class
        rows.append({"gold_mc": g, "pred_mc": p_mc, "pred_int": p_int})

    # (a) gold mod-class distribution
    ax = axes_row[0]
    gold_counts = [sum(1 for r in rows if r["gold_mc"] == c) for c in range(modulus)]
    ax.bar(range(modulus), gold_counts, color="#444")
    ax.set_title(f"mr={mr}: gold mod-{modulus} dist")
    ax.set_xticks(range(modulus))

    # (b) predicted mod-class distribution (collapsed)
    ax = axes_row[1]
    pred_counts = [sum(1 for r in rows if r["pred_mc"] == c) for c in range(modulus)]
    nondigit = sum(1 for r in rows if r["pred_mc"] is None)
    ax.bar(range(modulus), pred_counts, color="#1f77b4")
    if nondigit > 0:
        ax.bar(modulus, nondigit, color="#aaaaaa")
        ax.set_xticks(list(range(modulus)) + [modulus])
        ax.set_xticklabels(list(range(modulus)) + [f"non-digit ({nondigit})"])
    else:
        ax.set_xticks(range(modulus))
    ax.set_title(f"mr={mr}: pred (mod {modulus}) dist")

    # (c) confusion matrix in mod-class space
    ax = axes_row[2]
    M = np.zeros((modulus, modulus), dtype=float)
    for r in rows:
        g, p = r["gold_mc"], r["pred_mc"]
        if g is None or p is None:
            continue
        M[g, p] += 1
    M_norm = M / M.sum(axis=1, keepdims=True).clip(min=1)
    ax.imshow(M_norm, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    # Mod-class accuracy = trace(M) / sum(M)
    mc_acc = float(np.trace(M) / max(M.sum(), 1))
    ax.set_title(f"mr={mr}: confusion (mod-acc={mc_acc:.3f})")
    ax.set_xticks(range(modulus))
    ax.set_yticks(range(modulus))
    ax.set_xlabel("pred mod-class")
    ax.set_ylabel("gold mod-class")
    for i in range(modulus):
        for j in range(modulus):
            ax.text(j, i, f"{M_norm[i,j]:.2f}", ha="center", va="center",
                    fontsize=6, color="white" if M_norm[i, j] < 0.6 else "black")


def main():
    rows_with_data = []
    for mr in MRS:
        p = find_pred_file(mr)
        if p is None:
            print(f"  no predictions file for mr={mr}; skipping")
            continue
        rows_with_data.append((mr, p))

    if not rows_with_data:
        print(f"No prediction files in {PRED_DIR}. Run eval_predictions.py first.")
        return

    n = len(rows_with_data)
    fig, axes = plt.subplots(n, 3, figsize=(11, 2.6 * n))
    if n == 1:
        axes = axes.reshape(1, 3)
    for row_idx, (mr, p) in enumerate(rows_with_data):
        preds = load_preds(p)
        plot_mr_row(axes[row_idx], mr, preds)

    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}  ({n} MRs)")


if __name__ == "__main__":
    main()
