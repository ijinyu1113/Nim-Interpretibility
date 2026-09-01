"""Standalone plot of the no-holdout bare-math subtraction run (max=400,
variable 2-4 moves, eval_every=1). All 6 mrs on one figure.

Output: new_result/plots/modarith_subtract_noholdout.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/modarith_subtract_noholdout.png"
MRS = [3, 4, 5, 6, 7, 8]


def load(path):
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            if "eval_eval_move_acc" in r:
                by_step[st] = r["eval_eval_move_acc"]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)
        f = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_noholdout_max400_evalevery1.jsonl"
        s, v = load(f)
        if s.size:
            mask = s <= 200
            ax.plot(s[mask], v[mask], color="#d62728", lw=1.4,
                    marker="o", ms=3, label="eval move_acc")
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xlim(0, 200)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        ax.set_xlabel("Training step")
        if mr in (3, 6):
            ax.set_ylabel("eval move_acc")
        ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle(
        "Bare-math subtraction NO HOLDOUT (max=400, vars 2-4 moves, ~100% pile overlap) — Pythia 410m oldcfg",
        y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
