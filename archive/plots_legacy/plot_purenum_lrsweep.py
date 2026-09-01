"""Plot train/eval move-acc curves for the LR x WD sweep at 410m, seed=42.

Two figures (train, eval). Each is a 2x3 grid — one subplot per max_remove —
with one line per (lr, wd) config.

Inputs: new_result/purenum_metrics/mr{MR}_410m_seed42_lr{LR}_wd{WD}.jsonl
Outputs:
  new_result/plots/purenum_lrsweep_eval.png
  new_result/plots/purenum_lrsweep_train.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT_PLOT_DIR = "new_result/plots"
MRS = [3, 4, 5, 6, 7, 8]
SIZE = "410m"
SEED = 42

# 2x2 grid of (lr, wd). Ordered so colors are visually distinguishable.
CONFIGS = [
    ("5e-6", "0.1"),
    ("1e-5", "0.1"),
    ("5e-6", "0.5"),
    ("1e-5", "0.5"),
]
CONFIG_COLORS = {
    cfg: c for cfg, c in zip(CONFIGS, plt.cm.viridis(np.linspace(0.15, 0.85, len(CONFIGS))))
}


def cfg_label(lr, wd):
    return f"lr={lr}, wd={wd}"


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


def filepath(mr, lr, wd):
    return os.path.join(
        METRICS_DIR, f"mr{mr}_{SIZE}_seed{SEED}_lr{lr}_wd{wd}.jsonl"
    )


def plot_grid(metric, title, outpath):
    fig, axes = plt.subplots(2, 3, figsize=(11, 6), sharex=True, sharey=True)
    any_data = False
    for i, mr in enumerate(MRS):
        ax = axes.flat[i]
        for cfg in CONFIGS:
            lr, wd = cfg
            p = filepath(mr, lr, wd)
            if not os.path.isfile(p):
                print(f"  missing: {p}")
                continue
            run = load_run(p)
            mask = ~np.isnan(run[metric])
            if not mask.any():
                continue
            any_data = True
            ax.plot(run["step"][mask] / 1000.0, run[metric][mask],
                    color=CONFIG_COLORS[cfg], label=cfg_label(lr, wd))
        ax.axhline(1.0 / (mr + 1), **HLINE_KW)
        ax.set_title(f"max_remove={mr}")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        if i // 3 == 1:
            ax.set_xlabel(r"Training step (${\times}10^3$)")
        if i % 3 == 0:
            ax.set_ylabel("Accuracy")

    if not any_data:
        print(f"No data found for {metric}; skipping {outpath}")
        plt.close(fig)
        return

    handles = [
        plt.Line2D([], [], color=CONFIG_COLORS[cfg], lw=2.0, label=cfg_label(*cfg))
        for cfg in CONFIGS
    ]
    fig.legend(handles=handles, loc="upper center", ncol=len(CONFIGS),
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle(title, y=1.07)
    fig.tight_layout()
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    fig.savefig(outpath, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {outpath}")


def main():
    plot_grid("eval_acc",  "Eval move-acc (410m, seed=42) — LR x WD sweep",
              os.path.join(OUT_PLOT_DIR, "purenum_lrsweep_eval.png"))
    plot_grid("train_acc", "Train move-acc (410m, seed=42) — LR x WD sweep",
              os.path.join(OUT_PLOT_DIR, "purenum_lrsweep_train.png"))


if __name__ == "__main__":
    main()
