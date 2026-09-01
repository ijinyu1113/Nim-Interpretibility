"""Plot the old-config sweep across MRs at 410m, plus the 1b mr=6 check.

Two figures:
  1. Training curves per MR (2x3 grid, eval move_acc vs step, chance line per panel,
     overlay 1b for mr=6).
  2. Summary bar chart: best eval vs MR, comparing 410m old-config / 1b mr=6 /
     410m constant-LR baseline, with chance line.

Outputs:
  new_result/plots/oldcfg_sweep_curves.png
  new_result/plots/oldcfg_sweep_summary.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
OUT_CURVES = "new_result/plots/oldcfg_sweep_curves.png"
OUT_SUMMARY = "new_result/plots/oldcfg_sweep_summary.png"

MRS = [3, 4, 5, 6, 7]   # mr=8 not yet pulled


def load_run(path):
    by_step = defaultdict(lambda: {"train": None, "eval": None})
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            step = row.get("step")
            if step is None:
                continue
            if "eval_eval_move_acc" in row:
                by_step[step]["eval"] = float(row["eval_eval_move_acc"])
            if "eval_train_move_acc" in row:
                by_step[step]["train"] = float(row["eval_train_move_acc"])
    steps = sorted(by_step.keys())
    return {
        "step": np.array(steps),
        "train": np.array([by_step[s]["train"] if by_step[s]["train"] is not None else np.nan for s in steps]),
        "eval":  np.array([by_step[s]["eval"]  if by_step[s]["eval"]  is not None else np.nan for s in steps]),
    }


def oldcfg_path(mr, size="410m"):
    return f"{METRICS_DIR}/mr{mr}_{size}_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl"


# Best constant-LR config per MR, restricted to runs that completed the full
# 50k steps (so curves end at the same training duration in plots). Differs
# by at most 0.01 from the absolute best across all constant-LR variants.
CONSTANT_LR_PATHS = {
    3: f"{METRICS_DIR}/mr3_410m_seed42_lr5e-6_wd0.1.jsonl",   # 0.653 (complete 50k)
    4: f"{METRICS_DIR}/mr4_410m_seed42_lr5e-6_wd0.5.jsonl",   # 0.732 (complete 50k)
    5: f"{METRICS_DIR}/mr5_410m_seed42_lr1e-5_wd0.5.jsonl",   # 0.313 (complete 50k)
    6: f"{METRICS_DIR}/mr6_410m_seed42_lr5e-6_wd0.5.jsonl",   # 0.189 (complete 50k)
    7: f"{METRICS_DIR}/mr7_410m_seed42_lr5e-6_wd0.1.jsonl",   # 0.202 (complete 50k)
}


def plot_curves():
    fig, axes = plt.subplots(2, 3, figsize=(12, 6.5), sharex=True, sharey=True)
    for i, mr in enumerate(MRS):
        ax = axes.flat[i]
        chance = 1.0 / (mr + 1)
        # Old-config 410m
        d = load_run(oldcfg_path(mr, "410m"))
        if d is not None:
            ax.plot(d["step"] / 1000.0, d["eval"], color="#d62728",
                    lw=1.6, label="old-cfg 410m")
        # Old-config 1b for mr=6
        if mr == 6:
            d1 = load_run(oldcfg_path(mr, "1b"))
            if d1 is not None:
                ax.plot(d1["step"] / 1000.0, d1["eval"], color="#9467bd",
                        lw=1.6, label="old-cfg 1b")
        # Constant-LR baseline (best of prior runs)
        p_const = CONSTANT_LR_PATHS.get(mr)
        if p_const:
            d_const = load_run(p_const)
            if d_const is not None:
                ax.plot(d_const["step"] / 1000.0, d_const["eval"],
                        color="#1f77b4", lw=1.4, alpha=0.85,
                        label="best constant-LR baseline")
        ax.axhline(chance, color="grey", ls=":", lw=0.8, alpha=0.7)
        ax.set_title(f"mr={mr}  (mod {mr + 1}, chance={chance:.3f})")
        ax.set_ylim(-0.02, 1.05)
        ax.grid(alpha=0.14, linewidth=0.5)
        if i // 3 == 1:
            ax.set_xlabel(r"Training step (${\times}10^3$)")
        if i % 3 == 0:
            ax.set_ylabel("eval move_acc")
        if i == 0:
            ax.legend(fontsize=8, loc="lower right", frameon=False)
    # Hide the unused panel (last cell if 5 MRs in 2x3)
    if len(MRS) < 6:
        for j in range(len(MRS), 6):
            axes.flat[j].axis("off")

    fig.suptitle("Eval move-acc by max_remove — old config vs constant-LR baseline", y=1.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(OUT_CURVES), exist_ok=True)
    fig.savefig(OUT_CURVES, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_CURVES}")


def plot_summary():
    """Bar chart: best eval per (mr, regime), with chance line per mr."""
    fig, ax = plt.subplots(figsize=(9, 4.5))
    width = 0.30
    x = np.arange(len(MRS))

    def best_eval(path):
        d = load_run(path)
        if d is None:
            return np.nan
        ev = d["eval"]
        ev = ev[~np.isnan(ev)]
        return float(ev.max()) if ev.size else np.nan

    old_410 = [best_eval(oldcfg_path(mr, "410m")) for mr in MRS]
    old_1b = [best_eval(oldcfg_path(mr, "1b")) if mr == 6 else np.nan for mr in MRS]
    const = [best_eval(CONSTANT_LR_PATHS.get(mr, "")) for mr in MRS]
    chance = [1.0 / (mr + 1) for mr in MRS]

    ax.bar(x - width, const, width, color="#1f77b4", label="best constant-LR baseline")
    ax.bar(x,         old_410, width, color="#d62728", label="old-cfg 410m")
    ax.bar(x + width, old_1b, width, color="#9467bd", label="old-cfg 1b")

    # Chance markers
    for i, c in enumerate(chance):
        ax.plot([x[i] - 1.4 * width, x[i] + 1.4 * width],
                [c, c], color="grey", ls="--", lw=1.0, alpha=0.8)
    ax.plot([], [], color="grey", ls="--", lw=1.0, label="chance per mr")

    ax.set_xticks(x)
    ax.set_xticklabels([f"mr={mr}\n(mod {mr+1})" for mr in MRS])
    ax.set_ylabel("best eval move_acc")
    ax.set_ylim(0, 1.0)
    ax.set_title("Best eval move_acc by max_remove — regime comparison")
    ax.grid(axis="y", alpha=0.14, linewidth=0.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    # Annotate bars
    for xi, v in zip(x - width, const):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x, old_410):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)
    for xi, v in zip(x + width, old_1b):
        if not np.isnan(v):
            ax.text(xi, v + 0.01, f"{v:.2f}", ha="center", fontsize=7.5)

    fig.tight_layout()
    fig.savefig(OUT_SUMMARY, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {OUT_SUMMARY}")


if __name__ == "__main__":
    plot_curves()
    plot_summary()
