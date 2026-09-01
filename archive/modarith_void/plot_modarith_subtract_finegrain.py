"""Finegrain (eval_every=1) plot for modarith_subtract mr=5 and mr=8.
Zoom in on the transition window (~step 95-115) to reveal heuristic plateaus.
Compare max=500 vs max=50k for each mr.

Output: new_result/plots/modarith_subtract_finegrain.png
"""
import json
import os

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/modarith_subtract_finegrain.png"


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


HEUR = {
    5: {"chance (1/6)": 1/6, "parity (2/6)": 0.333, "mod-3 (3/6)": 0.5, "full": 1.0},
    8: {"chance (1/9)": 1/9, "mod-3 (3/9)": 0.333, "full": 1.0},
}


def main():
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), sharey=True)
    cases = [
        (5, 500,   axes[0, 0], "mr=5 (mod 6), max=500"),
        (5, 50000, axes[0, 1], "mr=5 (mod 6), max=50k"),
        (8, 500,   axes[1, 0], "mr=8 (mod 9), max=500"),
        (8, 50000, axes[1, 1], "mr=8 (mod 9), max=50k"),
    ]
    for mr, mx, ax, title in cases:
        f = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_max{mx}_evalevery1.jsonl"
        s, v = load(f)
        if not s.size:
            ax.set_title(f"{title} (missing)")
            continue
        # Restrict to transition window
        mask = (s >= 90) & (s <= 140)
        ax.plot(s[mask], v[mask], color="#1f77b4", lw=1.4,
                marker="o", ms=4.5, label="eval (every 1 step)")
        # Heuristic reference lines
        for name, y in HEUR[mr].items():
            ax.axhline(y, color="#d62728" if "full" in name else ("#888" if "chance" in name else "#ff7f0e"),
                       ls=":", lw=1.0, alpha=0.7)
            ax.text(140, y, name, va="center", fontsize=7,
                    color="#d62728" if "full" in name else ("#888" if "chance" in name else "#ff7f0e"))
        ax.set_title(title)
        ax.set_xlabel("Training step")
        ax.set_xlim(90, 140)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        if mr == 5:
            ax.set_ylabel("eval move_acc")
        ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle(
        "Bare-math subtract — fine eval (every 1 step) reveals ~1-3 step heuristic hints before snap",
        y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
