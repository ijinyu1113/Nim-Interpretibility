"""Plot Nim eval/train curves across max_pile ∈ {500, 5000, 50000} for mr=6,
separately for Pythia 70m and 410m.

Outputs:
  new_result/plots/nim_pool_size_70m.png
  new_result/plots/nim_pool_size_410m.png
"""
import json
import os
from collections import defaultdict

import numpy as np
import matplotlib.pyplot as plt

from plot_style import setup_style, HLINE_KW

setup_style()

METRICS_DIR = "new_result/purenum_metrics"
MR = 6
CHANCE = 1.0 / (MR + 1)

# Per size: (max_pile -> file path)
RUNS = {
    "70m": {
        500:   f"{METRICS_DIR}/mr6_70m_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl",
        5000:  f"{METRICS_DIR}/mr6_70m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenums_max5000.jsonl",
        50000: f"{METRICS_DIR}/mr6_70m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenums_max50000.jsonl",
    },
    "410m": {
        500:   f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep.jsonl",
        5000:  f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenums_max5000.jsonl",
        50000: f"{METRICS_DIR}/mr6_410m_seed42_lr3e-5_wd0.05_oldcfg300ep_purenums_max50000.jsonl",
    },
}

COLOR_BY_POOL = {500: "#1f77b4", 5000: "#2ca02c", 50000: "#d62728"}


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


def plot_for_size(size):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), sharey=True)
    eval_ax, train_ax = axes

    for max_pile, path in RUNS[size].items():
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

    eval_ax.axhline(CHANCE, color="grey", ls=":", lw=0.9, alpha=0.7,
                    label=f"chance ({CHANCE:.3f})")
    eval_ax.set_title("held-out accuracy")
    eval_ax.set_xlabel(r"Training step (${\times}10^3$)")
    eval_ax.set_ylabel("move_acc")
    eval_ax.set_ylim(-0.02, 1.05)
    eval_ax.grid(alpha=0.14, linewidth=0.5)
    eval_ax.legend(fontsize=11, loc="upper right", frameon=False)

    train_ax.set_title("train accuracy")
    train_ax.set_xlabel(r"Training step (${\times}10^3$)")
    train_ax.set_ylim(-0.02, 1.05)
    train_ax.grid(alpha=0.14, linewidth=0.5)

    fig.tight_layout()
    out = f"new_result/plots/nim_pool_size_{size}.png"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    for size in ["70m", "410m"]:
        plot_for_size(size)
