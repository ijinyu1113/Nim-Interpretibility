"""Compare no-holdout (max=400, ~100% overlap) vs held-out (max=500) on the
bare-math subtraction prompt. Tests whether train/eval pile-value overlap
gives the model a learning shortcut.

Output: new_result/plots/modarith_subtract_holdout_vs_no.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/modarith_subtract_holdout_vs_no.png"
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


def load_held_out(mr):
    """Prefer finegrain (eval_every=1) when available, else coarse."""
    candidates = [
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_max500_evalevery1.jsonl",
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_max500_evalevery100.jsonl",
    ]
    for p in candidates:
        if os.path.isfile(p):
            return load(p), os.path.basename(p)
    return (np.array([]), np.array([])), None


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)

        # No-holdout max=400
        f_no = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_modarith_subtract_noholdout_max400_evalevery1.jsonl"
        s_no, v_no = load(f_no)
        if s_no.size:
            mask = s_no <= 200
            ax.plot(s_no[mask], v_no[mask], color="#d62728", lw=1.4,
                    marker="o", ms=3, label="no holdout (max=400)")

        # Held-out max=500
        (s_h, v_h), src = load_held_out(mr)
        if s_h.size:
            mask = s_h <= 200
            ax.plot(s_h[mask], v_h[mask], color="#1f77b4", lw=1.4,
                    marker="s", ms=3, alpha=0.85,
                    label="held out (max=500)")

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
        "Bare-math subtraction: no-holdout (100% pile overlap) vs held-out — Pythia 410m oldcfg",
        y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
