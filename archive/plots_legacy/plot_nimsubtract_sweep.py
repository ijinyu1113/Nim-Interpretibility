"""Plot nimsubtract (4-move subtraction prompt) vs nimsimple baseline.
One panel per mr, showing:
  - nimsubtract oldcfg (cosine+warmup, ~50k steps)
  - nimsubtract constlr=3e-5 (no warmup, 50k steps)
  - nimsimple constlr=3e-5 (reference from prior sweep, 25k steps for most, 50k for mr=6)

Output: new_result/plots/nimsubtract_sweep.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT = "new_result/plots/nimsubtract_sweep.png"
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

        # Nimsubtract oldcfg (cosine+warmup)
        f_old = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_oldcfg213ep_nimsubtract_max50000.jsonl"
        s, v = load(f_old)
        if s.size:
            ax.plot(s, v, color="#1f77b4", lw=1.2, marker="o", ms=3,
                    label="nimsubtract oldcfg")

        # Nimsubtract constlr=3e-5
        f_const = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr50000steps_nimsubtract_max50000.jsonl"
        s, v = load(f_const)
        if s.size:
            ax.plot(s, v, color="#d62728", lw=1.2, marker="s", ms=3,
                    label="nimsubtract constlr=3e-5")

        # Nimsimple constlr=3e-5 reference (prior sweep: 25k for most, 50k for mr=6)
        f_nimsimple_50k = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr50000steps_nimsimple_max50000_evalevery50.jsonl"
        f_nimsimple_25k = f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_constlr25000steps_nimsimple_max50000_evalevery50.jsonl"
        f_nimsimple = f_nimsimple_50k if os.path.isfile(f_nimsimple_50k) else f_nimsimple_25k
        s, v = load(f_nimsimple)
        if s.size:
            ax.plot(s, v, color="#888", lw=1.0, ls="--", marker="^", ms=2,
                    alpha=0.6, label="nimsimple constlr=3e-5 (ref)")

        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.set_title(f"mr={mr}  (mod {mr+1})")
        ax.set_xscale("symlog", linthresh=100)
        ax.set_xlim(0, 60000)
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5, which="both")
        ax.set_xlabel("Training step")
        if mr in (3, 6):
            ax.set_ylabel("eval move_acc")
        ax.legend(fontsize=7, loc="upper left", frameon=False)

    fig.suptitle(
        "Nimsubtract (4-move subtraction prompt) vs nimsimple reference — Pythia 410m, max_pile=50000",
        y=1.01)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight", dpi=160)
    plt.close(fig)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()
