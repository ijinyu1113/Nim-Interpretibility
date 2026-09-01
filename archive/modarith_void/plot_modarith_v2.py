"""Plot modarith v2 sweep: 3 configs (A/B/C) per MR, all evaluating on the
same shared eval set per MR.

Output: new_result/plots/modarith_v2_sweep.png  (1x2: mr=3, mr=6)
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT_PATH = "new_result/plots/modarith_v2_sweep.png"
MRS = [3, 6]

CONFIGS = [
    ("A: max_X=500, full (~400 train)",       "maxX500",            "0",   "#1f77b4"),
    ("B: max_X=10000, sub=400 (~400 train)",  "maxX10000_subN400", "400", "#2ca02c"),
    ("C: max_X=10000, full (~9900 train)",    "maxX10000",          "0",   "#d62728"),
]


def file_for(mr, suffix):
    return f"{METRICS_DIR}/mr{mr}_410m_seed42_lr3e-5_wd0.05_modarith_v2_{suffix}.jsonl"


def load_run(path):
    by_step = defaultdict(lambda: None)
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if "eval_eval_move_acc" in r:
                step = r.get("step")
                if step is not None:
                    by_step[step] = float(r["eval_eval_move_acc"])
    steps = sorted(by_step.keys())
    return np.array(steps), np.array([by_step[s] for s in steps])


def main():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.0), sharey=True)
    for col, mr in enumerate(MRS):
        ax = axes[col]
        chance = 1.0 / (mr + 1)
        for label, suffix, _, color in CONFIGS:
            path = file_for(mr, suffix)
            d = load_run(path)
            if d is None:
                print(f"  missing: {path}")
                continue
            steps, evals = d
            ax.plot(steps / 1000.0, evals, color=color, lw=1.5, label=label)
        ax.axhline(chance, color="grey", ls=":", lw=0.9, alpha=0.7,
                   label=f"chance ({chance:.3f})")
        ax.axhline(1.0, color="#888", lw=0.5, ls="-", alpha=0.4)
        ax.set_title(f"mr={mr}  (mod {mr + 1})")
        ax.set_xlabel(r"Training step (${\times}10^3$)")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        if col == 0:
            ax.set_ylabel("eval move_acc (shared eval, 100 ex/MR)")
            ax.legend(fontsize=8, loc="lower right", frameon=False)

    fig.suptitle(
        "Modarith v2: 'X mod Y = Z' prompt. All 3 configs reach ~1.0 by first eval (step 250).",
        y=1.02,
    )
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
