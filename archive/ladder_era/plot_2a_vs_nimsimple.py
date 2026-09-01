"""Confirm: step 2a (max=500) snaps uniformly fast; nimsimple (max=50000) is
slow only for mr=8 (mod 9, with a mod-3 coset plateau) and mr=6 (mod 7).
Both constant LR, correct holdout.

Output: new_result/plots/2a_vs_nimsimple.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS = "new_result/purenum_metrics"
OUT = "new_result/plots/2a_vs_nimsimple.png"
MRS = [3, 4, 5, 6, 7, 8]
COLORS = {3: "#1f77b4", 4: "#ff7f0e", 5: "#2ca02c", 6: "#d62728", 7: "#9467bd", 8: "#8c564b"}


def load(path):
    by = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    for ln in open(path):
        r = json.loads(ln); s = r.get("step")
        if s is None: continue
        if "eval_eval_seq_acc" in r: by[s] = r["eval_eval_seq_acc"]
        elif "eval_eval_move_acc" in r: by[s] = r["eval_eval_move_acc"]
    st = sorted(by)
    return np.array(st), np.array([by[s] for s in st])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5), sharey=True)

    ax = axes[0]
    for mr in MRS:
        s, v = load(f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr30000steps_ladderC_step2a_heldout.jsonl")
        if s.size:
            m = s <= 10000
            ax.plot(s[m], v[m], color=COLORS[mr], lw=1.4, marker="o", ms=2.5, label=f"mr={mr} (mod {mr+1})")
    ax.set_title("step 2a (symbolic '-', max=500)\nall snap ~step 250")
    ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Training step"); ax.set_ylabel("eval seq_acc")
    ax.grid(alpha=0.14, lw=0.5); ax.legend(fontsize=8, loc="lower right", frameon=False)

    ax = axes[1]
    for mr in MRS:
        s, v = load(f"{METRICS}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl")
        if s.size:
            m = s <= 10000
            ax.plot(s[m], v[m], color=COLORS[mr], lw=1.4, marker="o", ms=2.5, label=f"mr={mr} (mod {mr+1})")
    # mod-3 coset line for mod 9
    ax.axhline(1/3, color="grey", ls=":", lw=1.0, alpha=0.8)
    ax.text(9800, 1/3 + 0.01, "mod-3 coset (1/3)", ha="right", fontsize=8, color="grey")
    ax.set_title("nimsimple (pile given, max=50000)\nmr=8 dwells at mod-3 coset; mr=6 stalls")
    ax.set_xlim(0, 10000); ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Training step")
    ax.grid(alpha=0.14, lw=0.5); ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle("Pile MAGNITUDE drives the slowdown, not the prompt: 2a@max500 snaps; nimsimple@max50000 shows cosets",
                 y=1.02, fontsize=11)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
