"""Plot nimsimple (no past moves) eval/train curves for mr ∈ {3,6} and
max_pile ∈ {500, 50000}. Compare against original Nim to see if simpler prompt
unlocks algorithmic generalization.

Outputs:
  new_result/plots/nimsimple_mr3.png
  new_result/plots/nimsimple_mr6.png
  new_result/plots/nimsimple_vs_original.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style

setup_style()

METRICS_DIR = "new_result/purenum_metrics"

# Nimsimple runs
NIMSIMPLE = {
    3: {
        500:   f"{METRICS_DIR}/mr3_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max500.jsonl",
        50000: f"{METRICS_DIR}/mr3_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000.jsonl",
    },
    6: {
        500:   f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max500.jsonl",
        50000: f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_nimsimple_max50000.jsonl",
    },
}

# Original Nim (full prompt with past moves)
ORIGINAL_NIM = {
    3: {
        500:   f"{METRICS_DIR}/mr3_410m_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl",
        50000: f"{METRICS_DIR}/mr3_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenums_max50000.jsonl",
    },
    6: {
        500:   f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl",
        50000: f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenums_max50000.jsonl",
    },
}

CHANCE = {3: 0.25, 6: 1.0/7}
COLOR_BY_POOL = {500: "#1f77b4", 50000: "#d62728"}


def load_run(path):
    by_step = defaultdict(lambda: {"train": None, "eval": None})
    if not os.path.isfile(path):
        return None
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            step = r.get("step")
            if step is None:
                continue
            if "eval_eval_move_acc" in r:
                by_step[step]["eval"] = float(r["eval_eval_move_acc"])
            if "eval_train_move_acc" in r:
                by_step[step]["train"] = float(r["eval_train_move_acc"])
    steps = sorted(by_step.keys())
    return {
        "step": np.array(steps),
        "train": np.array([by_step[s]["train"] if by_step[s]["train"] is not None else np.nan for s in steps]),
        "eval":  np.array([by_step[s]["eval"]  if by_step[s]["eval"]  is not None else np.nan for s in steps]),
    }


def plot_nimsimple_by_mr(mr):
    """Plot nimsimple for a single mr, comparing max_pile=500 vs 50000."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    eval_ax, train_ax = axes

    for max_pile in [500, 50000]:
        path = NIMSIMPLE[mr][max_pile]
        d = load_run(path)
        if d is None:
            print(f"  missing: {path}")
            continue
        color = COLOR_BY_POOL[max_pile]
        label = f"max_pile={max_pile}"

        m_e = ~np.isnan(d["eval"])
        if m_e.any():
            eval_ax.plot(d["step"][m_e] / 1000.0, d["eval"][m_e],
                         color=color, lw=1.4, label=label)
        m_t = ~np.isnan(d["train"])
        if m_t.any():
            train_ax.plot(d["step"][m_t] / 1000.0, d["train"][m_t],
                          color=color, lw=1.4, label=label)

    eval_ax.axhline(CHANCE[mr], color="grey", ls=":", lw=0.9, alpha=0.7,
                    label=f"chance ({CHANCE[mr]:.3f})")
    eval_ax.set_title(f"Nimsimple mr={mr} — eval move_acc")
    eval_ax.set_xlabel(r"Training step (${\times}10^3$)")
    eval_ax.set_ylabel("move_acc")
    eval_ax.set_ylim(-0.02, 1.05)
    eval_ax.grid(alpha=0.14, linewidth=0.5)
    eval_ax.legend(fontsize=8, loc="upper right", frameon=False)

    train_ax.set_title(f"Nimsimple mr={mr} — train move_acc")
    train_ax.set_xlabel(r"Training step (${\times}10^3$)")
    train_ax.set_ylim(-0.02, 1.05)
    train_ax.grid(alpha=0.14, linewidth=0.5)

    fig.suptitle(f"Nimsimple (no past moves) mr={mr}", y=1.02)
    fig.tight_layout()
    out = f"new_result/plots/nimsimple_mr{mr}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


def plot_nimsimple_vs_original(mr):
    """Compare nimsimple vs original Nim at max_pile=500 to isolate prompt effect."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    eval_ax, train_ax = axes

    # Nimsimple (no past moves)
    path_simple = NIMSIMPLE[mr][500]
    d_simple = load_run(path_simple)
    if d_simple:
        m_e = ~np.isnan(d_simple["eval"])
        if m_e.any():
            eval_ax.plot(d_simple["step"][m_e] / 1000.0, d_simple["eval"][m_e],
                         color="#2ca02c", lw=1.4, label="nimsimple (no moves)")
        m_t = ~np.isnan(d_simple["train"])
        if m_t.any():
            train_ax.plot(d_simple["step"][m_t] / 1000.0, d_simple["train"][m_t],
                          color="#2ca02c", lw=1.4, label="nimsimple")

    # Original Nim (with past moves)
    path_orig = ORIGINAL_NIM[mr][500]
    d_orig = load_run(path_orig)
    if d_orig:
        m_e = ~np.isnan(d_orig["eval"])
        if m_e.any():
            eval_ax.plot(d_orig["step"][m_e] / 1000.0, d_orig["eval"][m_e],
                         color="#ff7f0e", lw=1.4, label="original nim (+ moves)")
        m_t = ~np.isnan(d_orig["train"])
        if m_t.any():
            train_ax.plot(d_orig["step"][m_t] / 1000.0, d_orig["train"][m_t],
                          color="#ff7f0e", lw=1.4, label="original nim")

    eval_ax.axhline(CHANCE[mr], color="grey", ls=":", lw=0.9, alpha=0.7,
                    label=f"chance ({CHANCE[mr]:.3f})")
    eval_ax.set_title(f"Eval: nimsimple vs original (mr={mr}, max_pile=500)")
    eval_ax.set_xlabel(r"Training step (${\times}10^3$)")
    eval_ax.set_ylabel("move_acc")
    eval_ax.set_ylim(-0.02, 1.05)
    eval_ax.grid(alpha=0.14, linewidth=0.5)
    eval_ax.legend(fontsize=8, loc="upper right", frameon=False)

    train_ax.set_title(f"Train: nimsimple vs original (mr={mr}, max_pile=500)")
    train_ax.set_xlabel(r"Training step (${\times}10^3$)")
    train_ax.set_ylim(-0.02, 1.05)
    train_ax.grid(alpha=0.14, linewidth=0.5)

    fig.suptitle(f"Prompt effect: nimsimple removes multi-step computation", y=1.02)
    fig.tight_layout()
    out = f"new_result/plots/nimsimple_vs_original_mr{mr}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    for mr in [3, 6]:
        plot_nimsimple_by_mr(mr)
        plot_nimsimple_vs_original(mr)
