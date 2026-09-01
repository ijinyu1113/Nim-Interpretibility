"""Step 2b heldout, long cosine (stop 150k): max=500 vs max=50k per mr.
Uses eval_seq_acc.

Output: new_result/plots/ladder_2b_long_500_vs_50k.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/ladder_2b_long_500_vs_50k.png"
MRS = [3, 4, 5, 6, 7, 8]

SERIES = [
    ("",       "max=500", "#1f77b4", "o"),
    ("_max50k", "max=50k", "#d62728", "s"),
]


def load(path):
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            if "eval_eval_seq_acc" in r:
                by_step[st] = r["eval_eval_seq_acc"]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)
        for suffix, label, color, marker in SERIES:
            f = (f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_"
                 f"oldcfg1280ep_ladderC_step2b_heldout{suffix}_evalevery500.jsonl")
            s, v = load(f)
            if s.size == 0:
                continue
            ax.plot(s, v, color=color, lw=1.2, marker=marker, ms=2.6,
                    alpha=0.85, label=label)
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xscale("log")
        ax.set_xlim(400, 150000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5, which="both")
        ax.set_xlabel("Training step (log)")
        if mr in (3, 6):
            ax.set_ylabel("eval seq_acc")
        if mr == 3:
            ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle(
        "Step 2b heldout, long cosine (stop 150k): pile pool max=500 vs max=50k — Pythia 410m",
        y=1.01, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
