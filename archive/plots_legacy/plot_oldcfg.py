"""Plot train + eval move_acc for the old-config runs (cosine + warmup + 300ep,
lr=3e-5, wd=0.05), and compare against the best prior constant-LR runs.

Output: new_result/plots/oldcfg_vs_constant.png  (2x2 grid)
   rows: eval / train  cols: mr=3 / mr=6
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT_PATH = "new_result/plots/oldcfg_vs_constant.png"

# Runs to compare. Tuple = (label, path, color, linestyle)
RUNS = {
    3: [
        ("oldcfg (cos+warm, 300ep, lr=3e-5, wd=0.05)",
         f"{METRICS_DIR}/mr3_410m_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl",
         "#d62728", "-"),
        ("const lr=5e-6, wd=0.1 (best prior)",
         f"{METRICS_DIR}/mr3_410m_seed42_lr5e-6_wd0.1.jsonl",
         "#1f77b4", "-"),
    ],
    6: [
        ("oldcfg (cos+warm, 300ep, lr=3e-5, wd=0.05)",
         f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl",
         "#d62728", "-"),
        ("const lr=5e-6, wd=0.5 (best prior)",
         f"{METRICS_DIR}/mr6_410m_seed42_lr5e-6_wd0.5.jsonl",
         "#1f77b4", "-"),
    ],
}

# Train→eval probe ceiling per mr (from probe_modulo train→eval results)
PROBE_CEILING = {3: 0.595, 6: 0.183}


def load_run(path):
    by_step = defaultdict(lambda: {"train_acc": None, "eval_acc": None})
    with open(path, "r") as f:
        for line in f:
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            if "eval_eval_move_acc" in row:
                by_step[step]["eval_acc"] = float(row["eval_eval_move_acc"])
            if "eval_train_move_acc" in row:
                by_step[step]["train_acc"] = float(row["eval_train_move_acc"])
    steps = sorted(by_step.keys())
    return {
        "step": np.array(steps),
        "train_acc": np.array(
            [by_step[s]["train_acc"] if by_step[s]["train_acc"] is not None else np.nan
             for s in steps], dtype=float),
        "eval_acc": np.array(
            [by_step[s]["eval_acc"] if by_step[s]["eval_acc"] is not None else np.nan
             for s in steps], dtype=float),
    }


def plot_panel(ax, mr, kind):
    """kind: 'eval' or 'train'."""
    for label, path, color, ls in RUNS[mr]:
        if not os.path.isfile(path):
            print(f"  missing: {path}")
            continue
        d = load_run(path)
        arr = d["eval_acc"] if kind == "eval" else d["train_acc"]
        mask = ~np.isnan(arr)
        if not mask.any():
            continue
        ax.plot(d["step"][mask] / 1000.0, arr[mask],
                color=color, ls=ls, label=label, lw=1.6)

    if kind == "eval":
        # chance line
        ax.axhline(1.0 / (mr + 1), color="grey", ls=":", lw=0.8, alpha=0.7)
        # probe ceiling (train→eval)
        ax.axhline(PROBE_CEILING[mr], color="#888", ls="--", lw=0.9, alpha=0.7)
    ax.set_title(f"mr={mr}  ({'eval' if kind == 'eval' else 'train'})")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.14, linewidth=0.5)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for col, mr in enumerate([3, 6]):
        plot_panel(axes[0, col], mr, kind="eval")
        plot_panel(axes[1, col], mr, kind="train")
    for ax in axes[1, :]:
        ax.set_xlabel(r"Training step (${\times}10^3$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("move_acc")

    # Build a single legend from any panel that has data
    handles, labels = axes[0, 0].get_legend_handles_labels()
    extra = [
        plt.Line2D([], [], color="#888", ls="--", lw=0.9, label="train→eval probe ceiling"),
        plt.Line2D([], [], color="grey", ls=":", lw=0.8, label="chance baseline"),
    ]
    fig.legend(handles=handles + extra, loc="upper center", ncol=2,
               bbox_to_anchor=(0.5, 1.03), frameon=False, fontsize=8.5)
    fig.suptitle("Old config (cosine+warmup+300ep) vs best constant-LR baseline",
                 y=1.10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
