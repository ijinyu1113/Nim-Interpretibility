"""Plot train+eval move_acc curves for all moduli mr ∈ {3,4,5,6,7,8} at
nimsimple max_pile=50000. One subplot per mr, linear x-axis starting at step 0.

Output: new_result/plots/nimsimple_mr_sweep.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/nimsimple_mr_sweep.png"
MRS = [3, 4, 5, 6, 7, 8]


def load(mr):
    """Merge available files in priority order: evalevery1 > evalevery10 > standard.
    Returns steps, train, eval AND the boundary step where eval_every=1 data ends."""
    candidates = [
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000_evalevery1.jsonl",
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000_evalevery10.jsonl",
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000.jsonl",
    ]
    by_step = defaultdict(dict)
    finegrain_max = 0
    for i, p in enumerate(candidates):
        if not os.path.isfile(p):
            continue
        with open(p) as h:
            for line in h:
                r = json.loads(line)
                st = r.get("step")
                if st is None: continue
                if "eval_eval_move_acc" in r and "eval" not in by_step[st]:
                    by_step[st]["eval"] = r["eval_eval_move_acc"]
                    if i == 0:  # from evalevery1 file
                        finegrain_max = max(finegrain_max, st)
                if "eval_train_move_acc" in r and "train" not in by_step[st]:
                    by_step[st]["train"] = r["eval_train_move_acc"]
    if not by_step:
        return np.array([]), np.array([]), np.array([]), 0
    steps = sorted(by_step.keys())
    return (
        np.array(steps),
        np.array([by_step[s].get("train", np.nan) for s in steps]),
        np.array([by_step[s].get("eval", np.nan) for s in steps]),
        finegrain_max,
    )


def main():
    # Per-mr x-axis range tuned to where the interesting dynamics live.
    XLIM = {
        3: (0, 1000),
        4: (0, 1000),
        5: (0, 1000),
        6: (0, 10000),   # long chance plateau, transition at ~5-7k
        7: (0, 1000),
        8: (0, 10000),   # long chance plateau, transition at ~3-5k
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 7), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        steps, tr, ev, finegrain_max = load(mr)
        chance = 1.0 / (mr + 1)
        m_t = ~np.isnan(tr)
        m_e = ~np.isnan(ev)
        # Plot eval first (solid blue), then train (dashed orange) on top so it stays visible
        ax.plot(steps[m_e], ev[m_e], color="#1f77b4", lw=1.3,
                marker="o", ms=2.5, label="eval")
        ax.plot(steps[m_t], tr[m_t], color="#ff7f0e", lw=1.1, ls="--",
                marker="s", ms=2, alpha=0.85, label="train")
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        # Mark cadence change: vertical dashed line at the actual step where
        # eval_every=1 finegrain data ends (auto-detected from data).
        if finegrain_max > 0 and XLIM[mr][0] <= finegrain_max <= XLIM[mr][1]:
            ax.axvline(finegrain_max, color="#888", ls="--", lw=0.8, alpha=0.6)
            ax.text(finegrain_max, 1.02, f"step {finegrain_max}\n(eval_every: 1 → 250)",
                    ha="center", va="bottom", fontsize=7, color="#555")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xlabel("Training step")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xlim(*XLIM[mr])
        ax.grid(alpha=0.14, linewidth=0.5)
        ax.legend(fontsize=8, loc="lower right", frameon=False)
    for ax in axes[:, 0]:
        ax.set_ylabel("move_acc")
    fig.suptitle("Nimsimple max_pile=50000, Pythia 410m — train vs eval per modulus", y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
