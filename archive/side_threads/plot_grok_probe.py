"""Plot the grokking-probe experiment for mr=3 and mr=6.

Compares two configs over 150k effective training steps:
  - wd=0.5 stitched: original 0-50k + resume 50k-150k (resume file's step+50000)
  - wd=1.0 fresh:    0-150k from scratch

Produces 2x2 grid (rows = eval/train acc, cols = mr).

Inputs from new_result/purenum_metrics/:
  mr{MR}_410m_seed42_lr5e-6_wd0.5.jsonl              (original 50k)
  mr{MR}_410m_seed42_lr5e-6_wd0.5_resumefrom50k.jsonl (resume +100k)
  mr{MR}_410m_seed42_lr5e-6_wd1.0.jsonl              (fresh 150k)

Output:
  new_result/plots/grok_probe.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT_PATH = "new_result/plots/grok_probe.png"
MRS = [3, 6]
SIZE = "410m"
SEED = 42
LR = "5e-6"
STEP_OFFSET_RESUME = 50000  # resume runs restart their step counter at 0

WD_COLORS = {
    "0.5": "#1f77b4",   # blue (stitched)
    "1.0": "#d62728",   # red  (fresh wd=1.0)
}


def load_run(path):
    by_step = defaultdict(lambda: {"train_acc": None, "eval_acc": None})
    if not os.path.isfile(path):
        return {}
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
    return dict(by_step)


def to_arrays(by_step, step_offset=0):
    steps = sorted(by_step.keys())
    eval_arr = np.array(
        [by_step[s]["eval_acc"] if by_step[s]["eval_acc"] is not None else np.nan
         for s in steps], dtype=float)
    train_arr = np.array(
        [by_step[s]["train_acc"] if by_step[s]["train_acc"] is not None else np.nan
         for s in steps], dtype=float)
    step_arr = np.array(steps) + step_offset
    return step_arr, train_arr, eval_arr


def stitched_wd05(mr):
    """Concatenate original (0-50k) with resume (offset +50k)."""
    orig = load_run(f"{METRICS_DIR}/mr{mr}_{SIZE}_seed{SEED}_lr{LR}_wd0.5.jsonl")
    rsm = load_run(f"{METRICS_DIR}/mr{mr}_{SIZE}_seed{SEED}_lr{LR}_wd0.5_resumefrom50k.jsonl")
    s_o, t_o, e_o = to_arrays(orig, step_offset=0)
    s_r, t_r, e_r = to_arrays(rsm, step_offset=STEP_OFFSET_RESUME)
    return (np.concatenate([s_o, s_r]),
            np.concatenate([t_o, t_r]),
            np.concatenate([e_o, e_r]))


def fresh_wd10(mr):
    run = load_run(f"{METRICS_DIR}/mr{mr}_{SIZE}_seed{SEED}_lr{LR}_wd1.0.jsonl")
    return to_arrays(run, step_offset=0)


def plot_panel(ax, mr, kind):
    """kind: 'eval' or 'train'."""
    idx = 2 if kind == "eval" else 1

    s05, t05, e05 = stitched_wd05(mr)
    arr05 = e05 if kind == "eval" else t05
    mask = ~np.isnan(arr05)
    if mask.any():
        ax.plot(s05[mask] / 1000.0, arr05[mask],
                color=WD_COLORS["0.5"], label="wd=0.5 (orig+resume)")

    s10, t10, e10 = fresh_wd10(mr)
    arr10 = e10 if kind == "eval" else t10
    mask = ~np.isnan(arr10)
    if mask.any():
        ax.plot(s10[mask] / 1000.0, arr10[mask],
                color=WD_COLORS["1.0"], label="wd=1.0 (fresh)")

    if kind == "eval":
        ax.axhline(1.0 / (mr + 1), **HLINE_KW)
    ax.axvline(50, color="grey", lw=0.7, ls="--", alpha=0.5)  # resume seam
    ax.set_title(f"max_remove={mr}  ({'eval' if kind == 'eval' else 'train'})")
    ax.set_ylim(-0.02, 1.05)
    ax.grid(alpha=0.14, linewidth=0.5)


def main():
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), sharex=True, sharey=True)
    for col, mr in enumerate(MRS):
        plot_panel(axes[0, col], mr, kind="eval")
        plot_panel(axes[1, col], mr, kind="train")
    for ax in axes[1, :]:
        ax.set_xlabel(r"Training step (${\times}10^3$)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Accuracy")

    handles = [
        plt.Line2D([], [], color=WD_COLORS["0.5"], lw=2.0, label="wd=0.5 (orig 0-50k + resume 50k-150k)"),
        plt.Line2D([], [], color=WD_COLORS["1.0"], lw=2.0, label="wd=1.0 (fresh 0-150k)"),
        plt.Line2D([], [], color="grey", lw=0.7, ls="--", label="resume seam (50k)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.02), frameon=False)
    fig.suptitle("Grokking probe (410m, seed=42, lr=5e-6)", y=1.07)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    fig.savefig(OUT_PATH, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_PATH}")


if __name__ == "__main__":
    main()
