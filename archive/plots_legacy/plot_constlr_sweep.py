"""Plot constant-LR (no warmup) nimsimple max_pile=50000 sweep across mrs.
Overlay LR=1e-6 vs LR=3e-5 per mr, with cosine+warmup baseline as reference.

Output: new_result/plots/constlr_sweep.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/constlr_sweep.png"
MRS = [3, 4, 5, 6, 7, 8]
LRS = ["1e-6", "3e-5"]
COLOR_BY_LR = {"1e-6": "#1f77b4", "3e-5": "#d62728"}


def load(path, eval_key="eval_eval_move_acc"):
    by_step = {}
    if not os.path.isfile(path):
        return np.array([]), np.array([])
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            st = r.get("step")
            if st is None: continue
            if eval_key in r:
                by_step[st] = r[eval_key]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def load_baseline(mr):
    """Load the merged cosine+warmup baseline (evalevery1/10/standard fallback)."""
    candidates = [
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000_evalevery1.jsonl",
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000_evalevery10.jsonl",
        f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000.jsonl",
    ]
    by_step = {}
    for p in candidates:
        if not os.path.isfile(p):
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                st = r.get("step")
                if st is None or "eval_eval_move_acc" not in r:
                    continue
                if st not in by_step:
                    by_step[st] = r["eval_eval_move_acc"]
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def main():
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=True)
    for ax, mr in zip(axes.flat, MRS):
        chance = 1.0 / (mr + 1)

        # Cosine + warmup baseline (faded)
        b_steps, b_vals = load_baseline(mr)
        if b_steps.size:
            ax.plot(b_steps, b_vals, color="#888", lw=1.0, marker="^", ms=2,
                    alpha=0.55, label="cosine + warmup (lr=3e-5 peak)")

        # Constant-LR runs
        for lr in LRS:
            f = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr{lr}_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl"
            steps, vals = load(f)
            if steps.size:
                marker = "o" if lr == "1e-6" else "s"
                ax.plot(steps, vals, color=COLOR_BY_LR[lr], lw=1.2,
                        marker=marker, ms=3, label=f"const lr={lr}")

        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 30000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5, which="both")
        ax.set_xlabel("Training step (log)")
        if mr in (3, 6):
            ax.set_ylabel("eval move_acc")
        ax.legend(fontsize=7, loc="lower right", frameon=False)

    fig.suptitle(
        "Constant LR (no warmup) vs cosine+warmup baseline — nimsimple max_pile=50000",
        y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
